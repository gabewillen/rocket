// SPDX-License-Identifier: Apache-2.0
#include "attention/full_attention_native.h"

#include "attention/qsa_preprocess.h"
#include "projection/cutlass_qkv.h"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace rocket::qwen38::attention {
namespace {
constexpr int kRows = 16, kHidden = 2560, kQkv = 6656;
constexpr int kQuery = 3072, kKv = 256, kIndexQuery = 512, kIndex = 128;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}
}  // namespace

struct FullAttentionNativeProgram::Impl {
  FullAttentionNativeConfig config;
  void* qkv_plan = nullptr;
  void* qsa_plan = nullptr;
  std::unique_ptr<QsaStateFork> state_fork;
  __nv_bfloat16 *query = nullptr, *key = nullptr, *value = nullptr,
                 *gate = nullptr, *index_query = nullptr,
                 *index_raw_key = nullptr, *index_scratch = nullptr,
                 *compressed_rows = nullptr, *projected = nullptr;
  std::uint8_t *main_rows = nullptr, *raw_rows = nullptr;
  cudaEvent_t phase[9]{};
  std::string error;

  explicit Impl(FullAttentionNativeConfig config_value) : config(config_value) {
    if (config.device < 0 || !config.q_weight || !config.q_scale ||
        !config.k_weight || !config.k_scale || !config.v_weight ||
        !config.v_scale || !config.o_weight || !config.o_scale ||
        !config.q_norm || !config.k_norm || !config.index_qk_first ||
        !config.index_qk_second || !config.index_q_norm ||
        !config.index_k_norm || !config.rope_positions ||
        !config.logical_positions || !config.sequence_lengths ||
        !config.token_to_request || !config.active_state_generation)
      throw std::invalid_argument("complete native attention bindings required");
    try {
    check(cudaSetDevice(config.device), "cudaSetDevice");
    for (auto& event : phase)
      check(cudaEventCreate(&event), "create phase event");
    if (qwen38_cutlass_qkv_create(
            config.q_weight, config.q_scale, config.q_global,
            config.k_weight, config.k_scale, config.k_global,
            config.v_weight, config.v_scale, config.v_global, config.device,
            &qkv_plan))
      throw std::runtime_error(qwen38_cutlass_qkv_last_error());
    if (qwen38_qsa_indexer_create(config.o_weight, config.o_scale,
                                  config.o_global, config.device, &qsa_plan))
      throw std::runtime_error(qwen38_cutlass_qkv_last_error());
    check(cudaMalloc(&query, kRows * kQuery * 2), "allocate native query");
    check(cudaMalloc(&key, kRows * kKv * 2), "allocate native key");
    check(cudaMalloc(&value, kRows * kKv * 2), "allocate native value");
    check(cudaMalloc(&gate, kRows * kQuery * 2), "allocate output gate");
    check(cudaMalloc(&index_query, kRows * kIndexQuery * 2),
          "allocate index query");
    check(cudaMalloc(&index_raw_key, kRows * kIndex * 2),
          "allocate raw index key");
    check(cudaMalloc(&index_scratch, kRows * 640 * 2),
          "allocate index projection scratch");
    check(cudaMalloc(&main_rows, kRows * 512), "allocate main state rows");
    check(cudaMalloc(&raw_rows, kRows * 280), "allocate raw state rows");
    check(cudaMalloc(&compressed_rows, kRows * 256),
          "allocate compressed state rows");
    check(cudaMalloc(&projected, 128ULL * kHidden * 2),
          "allocate verifier projected output");
    check(cudaMemset(projected, 0, 128ULL * kHidden * 2),
          "clear verifier projected output");
    state_fork = std::make_unique<QsaStateFork>(config.device,
                                                config.active_state);
    } catch (...) {
      release();
      throw;
    }
  }

  ~Impl() {
    release();
  }

  void release() noexcept {
    cudaSetDevice(config.device);
    state_fork.reset();
    cudaFree(projected); cudaFree(compressed_rows); cudaFree(raw_rows);
    cudaFree(main_rows); cudaFree(index_scratch); cudaFree(index_raw_key);
    cudaFree(index_query); cudaFree(gate); cudaFree(value); cudaFree(key);
    cudaFree(query);
    for (auto& event : phase) {
      if (event) cudaEventDestroy(event);
      event = nullptr;
    }
    qwen38_qsa_indexer_destroy(qsa_plan);
    qwen38_cutlass_qkv_destroy(qkv_plan);
    projected = nullptr; compressed_rows = nullptr; raw_rows = nullptr;
    main_rows = nullptr; index_scratch = nullptr; index_raw_key = nullptr;
    index_query = nullptr; gate = nullptr; value = nullptr; key = nullptr;
    query = nullptr; qsa_plan = nullptr; qkv_plan = nullptr;
  }

  int fail(const char* operation, const char* detail) noexcept {
    error = std::string(operation) + ": " +
            (detail && *detail ? detail : "unknown CUDA failure");
    return 1;
  }

  int stage(const __nv_bfloat16* hidden,
            decode::FullAttentionLaunchShape shape, cudaStream_t stream) {
    error.clear();
    if (!hidden || !stream || shape.verify_width != 1 ||
        shape.token_rows != shape.sequences || shape.token_rows < 1 ||
        shape.token_rows > kRows)
      return fail("stage", "native QKV body currently admits K0 rows only");
    cudaEventRecord(phase[0], stream);
    if (qwen38_cutlass_qkv_launch(qkv_plan, hidden, stream))
      return fail("QKV", qwen38_cutlass_qkv_last_error());
    cudaEventRecord(phase[1], stream);
    void* qkv = nullptr;
    std::size_t elements = 0;
    if (qwen38_cutlass_qkv_output(qkv_plan, &qkv, &elements) ||
        elements != static_cast<std::size_t>(kRows) * kQkv)
      return fail("QKV output", qwen38_cutlass_qkv_last_error());
    if (qwen38_qsa_preprocess(
            hidden, static_cast<const __nv_bfloat16*>(qkv), config.q_norm,
            config.k_norm, config.index_qk_first, config.index_qk_second,
            config.index_q_norm, config.index_k_norm, config.rope_positions,
            shape.token_rows, query, key, value, gate, index_query,
            index_raw_key, index_scratch, stream))
      return fail("preprocess", qwen38_qsa_preprocess_last_error());
    cudaEventRecord(phase[2], stream);
    if (qwen38_qsa_format_state_rows(
            key, value, index_raw_key, config.index_k_norm,
            config.rope_positions, config.logical_positions,
            config.token_to_request, config.active_state.raw_state,
            shape.token_rows, main_rows, raw_rows, compressed_rows, stream))
      return fail("state rows", qwen38_qsa_preprocess_last_error());
    cudaEventRecord(phase[3], stream);
    try {
      state_fork->stage(main_rows, raw_rows,
                        reinterpret_cast<std::uint8_t*>(compressed_rows),
                        config.logical_positions, config.token_to_request,
                        shape.sequences, shape.verify_width, shape.token_rows,
                        stream);
    } catch (const std::exception& exception) {
      return fail("state fork", exception.what());
    }
    cudaEventRecord(phase[4], stream);
    if (qwen38_qsa_indexer_score_external(
            qsa_plan, index_query, config.active_state.compressed_state,
            compressed_rows, config.logical_positions,
            config.sequence_lengths, config.token_to_request,
            shape.token_rows, stream))
      return fail("QSA score", qwen38_cutlass_qkv_last_error());
    cudaEventRecord(phase[5], stream);
    if (qwen38_qsa_indexer_select_expand(
            qsa_plan, config.logical_positions, config.sequence_lengths,
            config.token_to_request, stream))
      return fail("QSA select", qwen38_cutlass_qkv_last_error());
    cudaEventRecord(phase[6], stream);
    const int attention_status = config.use_scalar_attention_control
        ? qwen38_qsa_sparse_attention_external_control(
              qsa_plan, query, config.active_state.main_state, main_rows,
              config.logical_positions, config.token_to_request,
              shape.token_rows, stream)
        : qwen38_qsa_sparse_attention_external(
              qsa_plan, query, config.active_state.main_state, main_rows,
              config.logical_positions, config.token_to_request,
              shape.token_rows, stream);
    if (attention_status)
      return fail("QSA attention", qwen38_cutlass_qkv_last_error());
    cudaEventRecord(phase[7], stream);
    void* attention = nullptr;
    if (qwen38_qsa_attention_output(qsa_plan, &attention, &elements) ||
        qwen38_qsa_apply_output_gate(
            static_cast<__nv_bfloat16*>(attention), gate, shape.token_rows,
            stream) ||
        qwen38_qsa_output_project(qsa_plan, stream))
      return fail("gated output", qwen38_cutlass_qkv_last_error());
    void* internal_output = nullptr;
    if (qwen38_qsa_projected_output(qsa_plan, &internal_output, &elements) ||
        elements != static_cast<std::size_t>(kRows) * kHidden)
      return fail("projected output", qwen38_cutlass_qkv_last_error());
    const cudaError_t copy = cudaMemcpyAsync(
        projected, internal_output,
        static_cast<std::size_t>(shape.token_rows) * kHidden * 2,
        cudaMemcpyDeviceToDevice, stream);
    if (copy != cudaSuccess)
      return fail("projected output", cudaGetErrorString(copy));
    cudaEventRecord(phase[8], stream);
    return 0;
  }

  FullAttentionNativeProfile profile() const {
    FullAttentionNativeProfile result{};
    float* values[] = {&result.qkv_ms, &result.preprocess_ms,
                       &result.state_format_ms, &result.state_fork_ms,
                       &result.score_ms, &result.select_ms,
                       &result.attention_ms, &result.gate_output_ms};
    for (int index = 0; index < 8; ++index)
      check(cudaEventElapsedTime(values[index], phase[index], phase[index + 1]),
            "read native phase time");
    return result;
  }
};

FullAttentionNativeProgram::FullAttentionNativeProgram(
    FullAttentionNativeConfig config)
    : impl_(new Impl(config)) {}
FullAttentionNativeProgram::~FullAttentionNativeProgram() { delete impl_; }

decode::FullAttentionCudaProgram FullAttentionNativeProgram::callbacks() noexcept {
  return {this, stage_callback, accept_callback, reset_callback,
          output_callback, error_callback};
}
__nv_bfloat16* FullAttentionNativeProgram::projected_output() const noexcept {
  return impl_->projected;
}
const char* FullAttentionNativeProgram::last_error() const noexcept {
  return impl_->error.c_str();
}
FullAttentionNativeProfile FullAttentionNativeProgram::profile() const {
  return impl_->profile();
}

int FullAttentionNativeProgram::stage_callback(
    void* opaque, const __nv_bfloat16* input,
    decode::FullAttentionLaunchShape shape, cudaStream_t stream) {
  return static_cast<FullAttentionNativeProgram*>(opaque)->impl_->stage(
      input, shape, stream);
}
int FullAttentionNativeProgram::accept_callback(
    void* opaque, const std::int32_t* accepted, int count,
    std::uint64_t generation, cudaStream_t stream) {
  auto* self = static_cast<FullAttentionNativeProgram*>(opaque)->impl_;
  try {
    self->state_fork->accept(accepted, count, stream);
    *self->config.active_state_generation = generation;
    return 0;
  } catch (const std::exception& exception) {
    return self->fail("accept", exception.what());
  }
}
int FullAttentionNativeProgram::reset_callback(void* opaque,
                                                cudaStream_t stream) {
  auto* self = static_cast<FullAttentionNativeProgram*>(opaque)->impl_;
  try {
    self->state_fork->reset(stream);
    return 0;
  } catch (const std::exception& exception) {
    return self->fail("reset", exception.what());
  }
}
const __nv_bfloat16* FullAttentionNativeProgram::output_callback(void* opaque) {
  return static_cast<FullAttentionNativeProgram*>(opaque)->impl_->projected;
}
const char* FullAttentionNativeProgram::error_callback(void* opaque) {
  return static_cast<FullAttentionNativeProgram*>(opaque)->impl_->error.c_str();
}

}  // namespace rocket::qwen38::attention

extern "C" int qwen38_full_attention_native_create(
    const rocket::qwen38::attention::FullAttentionNativeConfig* config,
    void** program) {
  if (!config || !program) return 1;
  *program = nullptr;
  try {
    *program = new rocket::qwen38::attention::FullAttentionNativeProgram(*config);
    return 0;
  } catch (...) {
    return 1;
  }
}

extern "C" int qwen38_full_attention_native_stage(
    void* program, const __nv_bfloat16* hidden, int sequences,
    int verify_width, int token_rows, cudaStream_t stream) {
  if (!program) return 1;
  auto* native = static_cast<
      rocket::qwen38::attention::FullAttentionNativeProgram*>(program);
  const auto callbacks = native->callbacks();
  return callbacks.stage(callbacks.context, hidden,
                         {sequences, verify_width, token_rows}, stream);
}

extern "C" int qwen38_full_attention_native_accept(
    void* program, const std::int32_t* accepted_lengths, int count,
    std::uint64_t generation, cudaStream_t stream) {
  if (!program) return 1;
  auto* native = static_cast<
      rocket::qwen38::attention::FullAttentionNativeProgram*>(program);
  const auto callbacks = native->callbacks();
  return callbacks.accept(callbacks.context, accepted_lengths, count,
                          generation, stream);
}

extern "C" int qwen38_full_attention_native_reset(void* program,
                                                    cudaStream_t stream) {
  if (!program) return 1;
  auto* native = static_cast<
      rocket::qwen38::attention::FullAttentionNativeProgram*>(program);
  const auto callbacks = native->callbacks();
  return callbacks.reset(callbacks.context, stream);
}

extern "C" int qwen38_full_attention_native_output(void* program,
                                                     void** output) {
  if (!program || !output) return 1;
  *output = static_cast<
      rocket::qwen38::attention::FullAttentionNativeProgram*>(program)
                ->projected_output();
  return 0;
}

extern "C" const char* qwen38_full_attention_native_last_error(void* program) {
  return program ? static_cast<
                       rocket::qwen38::attention::FullAttentionNativeProgram*>(
                       program)
                       ->last_error()
                 : "null native full-attention program";
}

extern "C" int qwen38_full_attention_native_destroy(void* program) {
  delete static_cast<
      rocket::qwen38::attention::FullAttentionNativeProgram*>(program);
  return 0;
}

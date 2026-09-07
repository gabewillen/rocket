// SPDX-License-Identifier: Apache-2.0
// Model-specific speculative state fork for Qwen3.8 GDN. The prefix snapshots
// remove the general-engine rollback and token-replay paths from publication.
#include "linear_attention/gdn_verifier.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>

namespace rocket::qwen38::linear_attention {
namespace {

constexpr std::size_t kConvStride =
    static_cast<std::size_t>(kConvStateRows) * kQkvWidth;
constexpr std::size_t kRecurrentStride =
    static_cast<std::size_t>(kValueHeads) * kHeadDim * kHeadDim;
constexpr std::size_t kHidden = 2'560;

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T>
__global__ void gather_slots(const T* source, T* destination,
                             const std::int32_t* source_slots,
                             std::size_t slot_stride, int sequences) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(sequences) * slot_stride;
  if (index >= count) return;
  const int sequence = static_cast<int>(index / slot_stride);
  const std::size_t element = index % slot_stride;
  const int source_slot = source_slots[sequence];
  if (source_slot > 0) {
    destination[static_cast<std::size_t>(sequence + 1) * slot_stride + element] =
        source[static_cast<std::size_t>(source_slot) * slot_stride + element];
  } else {
    destination[static_cast<std::size_t>(sequence + 1) * slot_stride + element] =
        T{};
  }
}

template <typename T>
__global__ void publish_prefixes(const T* snapshots, T* accepted_state,
                                 const std::int32_t* accepted_slots,
                                 const std::int32_t* accepted_prefixes,
                                 std::size_t slot_stride, int sequences,
                                 int verify_width, int state_pool_slots) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(sequences) * slot_stride;
  if (index >= count) return;
  const int sequence = static_cast<int>(index / slot_stride);
  const std::size_t element = index % slot_stride;
  const int prefix = accepted_prefixes[sequence];
  const int slot = accepted_slots[sequence];
  if (prefix <= 0 || prefix > verify_width || slot <= 0 ||
      slot >= state_pool_slots) {
    return;
  }
  const std::size_t snapshot =
      (static_cast<std::size_t>(prefix - 1) * kMaxRows + sequence) * slot_stride;
  accepted_state[static_cast<std::size_t>(slot) * slot_stride + element] =
      snapshots[snapshot + element];
}

std::uint64_t elapsed_ns(std::chrono::steady_clock::time_point begin) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
}

}  // namespace

struct GdnVerifier::Impl {
  Impl(int selected_device, CutlassGdnGraph& selected_graph,
       int selected_pool_slots, pair_reduce::OtelStageSink& selected_telemetry)
      : device(selected_device), graph(selected_graph),
        state_pool_slots(selected_pool_slots), telemetry(selected_telemetry) {}

  int device;
  CutlassGdnGraph& graph;
  int state_pool_slots;
  pair_reduce::OtelStageSink& telemetry;
  VerifierState state = VerifierState::kReady;
  VerifierShape shape{};
  __nv_bfloat16* accepted_conv = nullptr;
  float* accepted_recurrent = nullptr;
  const std::int32_t* accepted_slots = nullptr;
  __nv_bfloat16* dense_conv = nullptr;
  float* dense_recurrent = nullptr;
  __nv_bfloat16* prefix_conv = nullptr;
  float* prefix_recurrent = nullptr;
  __nv_bfloat16* compact_input = nullptr;
  __nv_bfloat16* output = nullptr;
  std::int32_t* accepted_prefixes = nullptr;
  std::uint64_t stage_bytes = 0;
  cudaStream_t bound_stream = nullptr;

  void emit(std::string_view stage, pair_reduce::Outcome outcome,
            std::string_view trace_id, std::string_view request_id,
            std::uint64_t duration, std::uint64_t bytes) noexcept {
    const int bucket = allowed_m(shape.sequences) ? shape.sequences : 0;
    telemetry.emit_span_and_log({stage, trace_id, request_id, 0, bucket,
                                 "bf16_fp32", outcome, duration, bytes});
    telemetry.record_duration(
        {0, bucket, "bf16_fp32", outcome, duration});
  }
};

GdnVerifier::GdnVerifier(int device, CutlassGdnGraph& graph,
                         int state_pool_slots,
                         pair_reduce::OtelStageSink& telemetry)
    : impl_(new Impl(device, graph, state_pool_slots, telemetry)) {
  if (device < 0 || state_pool_slots < 2) {
    delete impl_;
    impl_ = nullptr;
    throw std::invalid_argument("GDN verifier device or state pool changed");
  }
  try {
    cuda_check(cudaSetDevice(device), "cudaSetDevice");
    cuda_check(cudaMalloc(&impl_->dense_conv,
                          (kMaxRows + 1ULL) * kConvStride *
                              sizeof(__nv_bfloat16)),
               "malloc verifier convolution fork");
    cuda_check(cudaMalloc(&impl_->dense_recurrent,
                          (kMaxRows + 1ULL) * kRecurrentStride * sizeof(float)),
               "malloc verifier recurrent fork");
    cuda_check(cudaMalloc(&impl_->prefix_conv,
                          kMaxVerifyWidth * kMaxRows * kConvStride *
                              sizeof(__nv_bfloat16)),
               "malloc verifier convolution prefixes");
    cuda_check(cudaMalloc(&impl_->prefix_recurrent,
                          kMaxVerifyWidth * kMaxRows * kRecurrentStride *
                              sizeof(float)),
               "malloc verifier recurrent prefixes");
    cuda_check(cudaMalloc(&impl_->compact_input,
                          kMaxVerifyRows * kHidden * sizeof(__nv_bfloat16)),
               "malloc verifier compact input");
    cuda_check(cudaMalloc(&impl_->output,
                          kMaxVerifyRows * kHidden * sizeof(__nv_bfloat16)),
               "malloc verifier output");
    cuda_check(cudaMalloc(&impl_->accepted_prefixes,
                          kMaxRows * sizeof(std::int32_t)),
               "malloc verifier accepted prefixes");
  } catch (...) {
    this->~GdnVerifier();
    throw;
  }
}

GdnVerifier::~GdnVerifier() {
  if (!impl_) return;
  cudaSetDevice(impl_->device);
  cudaFree(impl_->accepted_prefixes);
  cudaFree(impl_->output);
  cudaFree(impl_->compact_input);
  cudaFree(impl_->prefix_recurrent);
  cudaFree(impl_->prefix_conv);
  cudaFree(impl_->dense_recurrent);
  cudaFree(impl_->dense_conv);
  delete impl_;
}

void GdnVerifier::stage(const __nv_bfloat16* input,
                        __nv_bfloat16* accepted_conv, float* accepted_recurrent,
                        const std::int32_t* accepted_slots,
                        VerifierShape shape, std::string_view trace_id,
                        std::string_view request_id, cudaStream_t stream) {
  const auto begin = std::chrono::steady_clock::now();
  if (impl_->state != VerifierState::kReady || !input || !accepted_conv ||
      !accepted_recurrent || !accepted_slots || !stream ||
      (impl_->bound_stream && impl_->bound_stream != stream) ||
      !allowed_verifier_shape(shape)) {
    impl_->shape = shape;
    impl_->emit("rocket.qwen38.gdn.verifier.stage",
                pair_reduce::Outcome::kContractError, trace_id, request_id,
                elapsed_ns(begin), 0);
    throw std::invalid_argument("GDN verifier stage contract changed");
  }
  impl_->shape = shape;
  impl_->bound_stream = stream;
  impl_->accepted_conv = accepted_conv;
  impl_->accepted_recurrent = accepted_recurrent;
  impl_->accepted_slots = accepted_slots;
  try {
    cuda_check(cudaSetDevice(impl_->device), "cudaSetDevice");
    const std::size_t conv_count =
        static_cast<std::size_t>(shape.sequences) * kConvStride;
    const std::size_t recurrent_count =
        static_cast<std::size_t>(shape.sequences) * kRecurrentStride;
    gather_slots<<<(conv_count + 255) / 256, 256, 0, stream>>>(
        accepted_conv, impl_->dense_conv, accepted_slots, kConvStride,
        shape.sequences);
    gather_slots<<<(recurrent_count + 255) / 256, 256, 0, stream>>>(
        accepted_recurrent, impl_->dense_recurrent, accepted_slots,
        kRecurrentStride, shape.sequences);
    cuda_check(cudaMemsetAsync(impl_->compact_input, 0,
                               kMaxVerifyRows * kHidden *
                                   sizeof(__nv_bfloat16),
                               stream),
               "clear verifier input tail");
    cuda_check(cudaMemcpyAsync(
                   impl_->compact_input, input,
                   static_cast<std::size_t>(shape.token_rows()) * kHidden *
                       sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToDevice, stream),
               "compact verifier rows");
    impl_->graph.launch_verifier(
        impl_->compact_input, impl_->dense_conv, impl_->dense_recurrent,
        impl_->prefix_conv, impl_->prefix_recurrent, shape.sequences,
        shape.verify_width, stream);
    cuda_check(cudaMemcpyAsync(
                   impl_->output, impl_->graph.verifier_output(),
                   static_cast<std::size_t>(shape.token_rows()) * kHidden *
                       sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToDevice, stream),
               "retain verifier output");
    cuda_check(cudaPeekAtLastError(), "stage GDN verifier");
    const std::uint64_t fork_bytes =
        static_cast<std::uint64_t>(shape.sequences) *
        (kConvStride * sizeof(__nv_bfloat16) +
         kRecurrentStride * sizeof(float));
    constexpr std::uint64_t weight_bytes =
        static_cast<std::uint64_t>(8'240 + 48) * 2'560 * 9 / 16 +
        static_cast<std::uint64_t>(2'560) * 3'072 * 9 / 16 + 41'312;
    impl_->stage_bytes =
        static_cast<std::uint64_t>(shape.verify_width + 4) * fork_bytes +
        weight_bytes + static_cast<std::uint64_t>(shape.token_rows()) *
                           (2ULL * kHidden + 2ULL * kHidden);
    impl_->state = VerifierState::kStaged;
    impl_->emit("rocket.qwen38.gdn.verifier.stage",
                pair_reduce::Outcome::kOk, trace_id, request_id,
                elapsed_ns(begin), impl_->stage_bytes);
  } catch (...) {
    impl_->state = VerifierState::kFaulted;
    impl_->emit("rocket.qwen38.gdn.verifier.stage",
                pair_reduce::Outcome::kCudaError, trace_id, request_id,
                elapsed_ns(begin), 0);
    throw;
  }
}

void GdnVerifier::accept(const std::int32_t* prefixes,
                         std::string_view trace_id,
                         std::string_view request_id, cudaStream_t stream) {
  const auto begin = std::chrono::steady_clock::now();
  if (impl_->state != VerifierState::kStaged || !prefixes || !stream ||
      stream != impl_->bound_stream) {
    impl_->emit("rocket.qwen38.gdn.verifier.accept",
                pair_reduce::Outcome::kContractError, trace_id, request_id,
                elapsed_ns(begin), 0);
    throw std::invalid_argument("GDN verifier accept contract changed");
  }
  try {
    for (int sequence = 0; sequence < impl_->shape.sequences; ++sequence) {
      if (prefixes[sequence] < 0 ||
          prefixes[sequence] > impl_->shape.verify_width) {
        throw std::invalid_argument("GDN accepted prefix is out of range");
      }
    }
    cuda_check(cudaMemcpyAsync(impl_->accepted_prefixes, prefixes,
                               impl_->shape.sequences * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream),
               "copy GDN accepted prefixes");
    const std::size_t conv_count =
        static_cast<std::size_t>(impl_->shape.sequences) * kConvStride;
    const std::size_t recurrent_count =
        static_cast<std::size_t>(impl_->shape.sequences) * kRecurrentStride;
    publish_prefixes<<<(conv_count + 255) / 256, 256, 0, stream>>>(
        impl_->prefix_conv, impl_->accepted_conv, impl_->accepted_slots,
        impl_->accepted_prefixes, kConvStride, impl_->shape.sequences,
        impl_->shape.verify_width, impl_->state_pool_slots);
    publish_prefixes<<<(recurrent_count + 255) / 256, 256, 0, stream>>>(
        impl_->prefix_recurrent, impl_->accepted_recurrent,
        impl_->accepted_slots, impl_->accepted_prefixes, kRecurrentStride,
        impl_->shape.sequences, impl_->shape.verify_width,
        impl_->state_pool_slots);
    cuda_check(cudaPeekAtLastError(), "publish GDN verifier prefixes");
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(impl_->shape.sequences) *
        (kConvStride * sizeof(__nv_bfloat16) +
         kRecurrentStride * sizeof(float));
    impl_->emit("rocket.qwen38.gdn.verifier.accept",
                pair_reduce::Outcome::kOk, trace_id, request_id,
                elapsed_ns(begin), bytes);
    impl_->state = VerifierState::kReady;
    impl_->shape = {};
  } catch (...) {
    impl_->state = VerifierState::kFaulted;
    impl_->emit("rocket.qwen38.gdn.verifier.accept",
                pair_reduce::Outcome::kCudaError, trace_id, request_id,
                elapsed_ns(begin), 0);
    throw;
  }
}

void GdnVerifier::reset(std::string_view trace_id,
                        std::string_view request_id) noexcept {
  const auto begin = std::chrono::steady_clock::now();
  if (impl_->state == VerifierState::kStaged) {
    impl_->emit("rocket.qwen38.gdn.verifier.reset", pair_reduce::Outcome::kOk,
                trace_id, request_id, elapsed_ns(begin), 0);
    impl_->state = VerifierState::kReady;
    impl_->shape = {};
    impl_->accepted_conv = nullptr;
    impl_->accepted_recurrent = nullptr;
    impl_->accepted_slots = nullptr;
    impl_->stage_bytes = 0;
    return;
  }
  impl_->emit("rocket.qwen38.gdn.verifier.reset",
              pair_reduce::Outcome::kContractError, trace_id, request_id,
              elapsed_ns(begin), 0);
}

const __nv_bfloat16* GdnVerifier::staged_output() const noexcept {
  return impl_ && impl_->state == VerifierState::kStaged ? impl_->output
                                                         : nullptr;
}

VerifierState GdnVerifier::state() const noexcept {
  return impl_ ? impl_->state : VerifierState::kFaulted;
}

VerifierShape GdnVerifier::staged_shape() const noexcept {
  return impl_ ? impl_->shape : VerifierShape{};
}

std::uint64_t GdnVerifier::logical_stage_bytes() const noexcept {
  return impl_ ? impl_->stage_bytes : 0;
}

}  // namespace rocket::qwen38::linear_attention

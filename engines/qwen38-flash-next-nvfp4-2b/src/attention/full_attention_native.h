// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "attention/qsa_state_fork.h"
#include "decode/full_attention_graph.h"

namespace rocket::qwen38::attention {

struct FullAttentionNativeConfig {
  int device;
  const std::uint8_t *q_weight, *q_scale, *k_weight, *k_scale,
      *v_weight, *v_scale, *o_weight, *o_scale;
  float q_global, k_global, v_global, o_global;
  const __nv_bfloat16 *q_norm, *k_norm, *index_qk_first,
      *index_qk_second, *index_q_norm, *index_k_norm;
  QsaStateExtents active_state;
  const std::int64_t *rope_positions, *logical_positions;
  const std::int32_t *sequence_lengths, *token_to_request;
  std::uint64_t* active_state_generation;
};

struct FullAttentionNativeProfile {
  float qkv_ms, preprocess_ms, state_format_ms, state_fork_ms, score_ms,
      select_ms, attention_ms, gate_output_ms;
};

// Current native body proves K0 buckets (verify_width=1, rows<=16). The owner
// ABI remains sequences*verify_width<=128; wider verifier bodies fail closed
// until the QKV/output CUTLASS plans are generalized beyond their measured
// c16 shape.
class FullAttentionNativeProgram final {
 public:
  explicit FullAttentionNativeProgram(FullAttentionNativeConfig config);
  ~FullAttentionNativeProgram();
  FullAttentionNativeProgram(const FullAttentionNativeProgram&) = delete;
  FullAttentionNativeProgram& operator=(const FullAttentionNativeProgram&) = delete;

  decode::FullAttentionCudaProgram callbacks() noexcept;
  __nv_bfloat16* projected_output() const noexcept;
  FullAttentionNativeProfile profile() const;
  const char* last_error() const noexcept;

 private:
  static int stage_callback(void*, const __nv_bfloat16*,
                            decode::FullAttentionLaunchShape, cudaStream_t);
  static int accept_callback(void*, const std::int32_t*, int, std::uint64_t,
                             cudaStream_t);
  static int reset_callback(void*, cudaStream_t);
  static const __nv_bfloat16* output_callback(void*);
  static const char* error_callback(void*);

  struct Impl;
  Impl* impl_;
};

}  // namespace rocket::qwen38::attention

extern "C" {
int qwen38_full_attention_native_create(
    const rocket::qwen38::attention::FullAttentionNativeConfig* config,
    void** program);
int qwen38_full_attention_native_stage(void* program,
                                       const __nv_bfloat16* hidden,
                                       int sequences, int verify_width,
                                       int token_rows, cudaStream_t stream);
int qwen38_full_attention_native_accept(void* program,
                                        const std::int32_t* accepted_lengths,
                                        int count, std::uint64_t generation,
                                        cudaStream_t stream);
int qwen38_full_attention_native_reset(void* program, cudaStream_t stream);
int qwen38_full_attention_native_output(void* program, void** output);
const char* qwen38_full_attention_native_last_error(void* program);
int qwen38_full_attention_native_destroy(void* program);
}

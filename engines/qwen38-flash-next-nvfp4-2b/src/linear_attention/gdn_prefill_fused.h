// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "linear_attention/gdn_prefill_width4.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace rocket::qwen38::linear_attention::prefill {
inline constexpr int kKeyHeads = 8, kValueHeads = 24, kHeadDim = 128;
inline constexpr int kQkvWidth = 5'120, kGateWidth = 48, kMaxTokens = 8'192;
struct Outputs { __nv_bfloat16* q; __nv_bfloat16* k; __nv_bfloat16* v;
  float* g; float* beta; __nv_bfloat16* final_conv_state; };
// qkv is the contiguous logical [Q,K,V] projection prepared from Qwen's
// interleaved checkpoint layout. ba is contiguous [b,a]. A_log and dt_bias
// use the authenticated slab's BF16 source dtype and are promoted on load.
cudaError_t launch_fused(const __nv_bfloat16* qkv, const __nv_bfloat16* ba,
                         const __nv_bfloat16* conv_weight,
                         const __nv_bfloat16* initial_conv_state,
                         const __nv_bfloat16* a_log,
                         const __nv_bfloat16* dt_bias, Outputs, int tokens,
                         cudaStream_t) noexcept;
cudaError_t launch_materialized_reference(
    const __nv_bfloat16* qkv, const __nv_bfloat16* ba,
    const __nv_bfloat16* conv_weight,
    const __nv_bfloat16* initial_conv_state, const __nv_bfloat16* a_log,
    const __nv_bfloat16* dt_bias, __nv_bfloat16* conv_output, Outputs,
    int tokens, cudaStream_t) noexcept;
constexpr std::uint64_t eliminated_intermediate_bytes(int tokens) noexcept {
  return tokens > 0 ? 4ULL * tokens * kQkvWidth : 0;
}
}  // namespace rocket::qwen38::linear_attention::prefill

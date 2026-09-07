// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace rocket::qwen38::hyperconnection {

inline constexpr int kStreams = 4;
inline constexpr int kHidden = 2'560;
inline constexpr int kHyperHidden = kStreams * kHidden;
inline constexpr int kLowRank = 320;
inline constexpr int kMergedRows = 336;
inline constexpr float kEpsilon = 1.0e-6F;

struct Weights {
  const __nv_bfloat16* norm;
  const __nv_bfloat16* down;
  const __nv_bfloat16* injection;
  const __nv_bfloat16* up;
};

class Plan final {
 public:
  Plan(int device, Weights attention, Weights mlp);
  ~Plan();
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;

  // Exact pinned vLLM GatedResidual.mix. All buffers are fixed c16 extents;
  // m selects one immutable prefix bucket from {1,2,4,8,16}.
  void mix(const __nv_bfloat16* hidden, __nv_bfloat16* block_input,
           __nv_bfloat16* injection, int m, cudaStream_t stream);

  // Exact pinned combine_and_mix, except PairReduce's FP32 block output is
  // rounded to BF16 at the materialized combine boundary before normalization.
  void combine_and_mix(const __nv_bfloat16* hidden, const float* block_output,
                       const __nv_bfloat16* injection,
                       __nv_bfloat16* updated_hidden,
                       __nv_bfloat16* next_block_input,
                       __nv_bfloat16* next_injection, int m,
                       cudaStream_t stream);

 private:
  struct Impl;
  Impl* impl_;
};

constexpr bool allowed_m(int m) noexcept {
  return m == 1 || m == 2 || m == 4 || m == 8 || m == 16;
}

}  // namespace rocket::qwen38::hyperconnection

extern "C" {
int qwen38_hc_create(int device,
                     const __nv_bfloat16* attn_norm,
                     const __nv_bfloat16* attn_down,
                     const __nv_bfloat16* attn_injection,
                     const __nv_bfloat16* attn_up,
                     const __nv_bfloat16* mlp_norm,
                     const __nv_bfloat16* mlp_down,
                     const __nv_bfloat16* mlp_injection,
                     const __nv_bfloat16* mlp_up,
                     void** plan);
int qwen38_hc_mix(void* plan, const __nv_bfloat16* hidden,
                  __nv_bfloat16* block_input, __nv_bfloat16* injection,
                  int m, cudaStream_t stream);
int qwen38_hc_combine_and_mix(
    void* plan, const __nv_bfloat16* hidden, const float* block_output,
    const __nv_bfloat16* injection, __nv_bfloat16* updated_hidden,
    __nv_bfloat16* next_block_input, __nv_bfloat16* next_injection,
    int m, cudaStream_t stream);
int qwen38_hc_destroy(void* plan);
const char* qwen38_hc_last_error();
}

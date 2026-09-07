// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace rocket::qwen38::linear_attention {

inline constexpr int kMaxRows = 16;
inline constexpr int kMaxVerifierRows = 128;
inline constexpr int kMaxVerifyWidth = 8;
inline constexpr int kKeyHeads = 8;
inline constexpr int kValueHeads = 24;
inline constexpr int kHeadDim = 128;
inline constexpr int kQkvWidth = 5'120;
inline constexpr int kGateWidth = 3'072;
inline constexpr int kConvStateRows = 6;
inline constexpr int kConvKernel = 4;

constexpr bool allowed_m(int m) noexcept {
  return m == 1 || m == 2 || m == 4 || m == 8 || m == 16;
}

// Stable buffers for the exact Qwen3.8 TP2 GDN core after QKVZ/BA projection.
// The caller owns weights and accepted state. Plan owns only graph-stable
// intermediates. State pool slot 0 is null; indices <=0 produce zero output.
class CorePlan final {
 public:
  CorePlan(int device, const __nv_bfloat16* conv_weight,
           const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
           const __nv_bfloat16* norm_weight);
  ~CorePlan();
  CorePlan(const CorePlan&) = delete;
  CorePlan& operator=(const CorePlan&) = delete;

  void launch(const __nv_bfloat16* qkvz, const __nv_bfloat16* ba,
              __nv_bfloat16* conv_state, std::size_t conv_slot_stride,
              float* recurrent_state, std::size_t recurrent_slot_stride,
              const std::int32_t* state_indices, int m,
              cudaStream_t stream);
  void launch_verifier(
      const __nv_bfloat16* position_major_qkvz,
      const __nv_bfloat16* position_major_ba,
      const __nv_bfloat16* accepted_conv_state,
      const float* accepted_recurrent_state,
      const std::int32_t* accepted_state_indices, int sequences,
      int verify_width, cudaStream_t stream);
  void accept_verifier(const __nv_bfloat16* position_major_qkvz,
                       const __nv_bfloat16* position_major_ba,
                       __nv_bfloat16* accepted_conv_state,
                       float* accepted_recurrent_state,
                       const std::int32_t* accepted_state_indices,
                       const std::int32_t* accepted_prefixes, int sequences,
                       int verify_width, cudaStream_t stream);
  const __nv_bfloat16* output() const noexcept;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace rocket::qwen38::linear_attention

extern "C" {
int qwen38_gdn_core_create(int device, const __nv_bfloat16* conv_weight,
                           const __nv_bfloat16* a_log,
                           const __nv_bfloat16* dt_bias,
                           const __nv_bfloat16* norm_weight, void** plan);
int qwen38_gdn_core_launch(
    void* plan, const __nv_bfloat16* qkvz, const __nv_bfloat16* ba,
    __nv_bfloat16* conv_state, std::size_t conv_slot_stride,
    float* recurrent_state, std::size_t recurrent_slot_stride,
    const std::int32_t* state_indices, int m, cudaStream_t stream);
int qwen38_gdn_core_output(void* plan, void** output_bf16,
                           std::size_t* elements);
int qwen38_gdn_core_destroy(void* plan);
const char* qwen38_gdn_core_last_error();
}

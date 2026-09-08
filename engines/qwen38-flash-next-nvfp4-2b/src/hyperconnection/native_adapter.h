// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/full_attention_layer.h"
#include "hyperconnection/hyperconnection.h"

namespace rocket::qwen38::hyperconnection {

// Production adapter from the authenticated HC plan to TargetFullLayer's
// borrowed-stream interface. Plan creation and weight copies occur before
// graph capture. Every execution method only enqueues fixed-buffer work.
class NativeFullAttentionHyperConnection final
    : public decode::FullAttentionHyperConnection {
 public:
  explicit NativeFullAttentionHyperConnection(Plan& plan) noexcept
      : plan_(plan) {}

  void mix(const __nv_bfloat16* hidden, __nv_bfloat16* block_input,
           __nv_bfloat16* injection, int m,
           cudaStream_t stream) override;
  void combine_and_mix(const __nv_bfloat16* hidden,
                       const float* block_output,
                       const __nv_bfloat16* injection,
                       __nv_bfloat16* updated_hidden,
                       __nv_bfloat16* next_block_input,
                       __nv_bfloat16* next_injection, int m,
                       cudaStream_t stream) override;
  void combine(const __nv_bfloat16* hidden, const float* block_output,
               const __nv_bfloat16* injection,
               __nv_bfloat16* updated_hidden, int m,
               cudaStream_t stream) override;
  void synchronize(cudaStream_t stream) override;

 private:
  Plan& plan_;
};

}  // namespace rocket::qwen38::hyperconnection

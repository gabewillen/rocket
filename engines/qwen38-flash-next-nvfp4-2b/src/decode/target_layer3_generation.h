// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_layer3_prefill.h"

namespace rocket::qwen38::decode {

// Fixed graph-safe owner for QSA row mappings plus the composed target-MoE
// requested generation. Construction validates pointers without launching.
class NativeTargetQsaGenerationOwner final
    : public TargetLayer3GenerationOwner {
 public:
  NativeTargetQsaGenerationOwner(
      int rank, int layer, const attention::TargetQsaStateView& storage,
      std::uint64_t* target_moe_requested_generation, int max_rows);

  int rank() const noexcept override { return rank_; }
  int layer() const noexcept override { return layer_; }
  bool authenticated() const noexcept override { return authenticated_; }
  const attention::TargetQsaStateView& view(
      int row, std::uint64_t generation) const override;
  void enqueue_prepare(int row, std::uint64_t generation,
                       cudaStream_t stream) override;

 private:
  int rank_;
  int layer_;
  int max_rows_;
  attention::TargetQsaStateView storage_{};
  mutable attention::TargetQsaStateView view_{};
  std::uint64_t* target_moe_requested_generation_;
  bool authenticated_ = false;
  bool faulted_ = false;
  mutable bool view_pending_ = false;
  mutable int pending_row_ = -1;
  mutable std::uint64_t pending_generation_ = 0;
  int next_row_ = 0;
  std::uint64_t last_generation_ = 0;
};

}  // namespace rocket::qwen38::decode

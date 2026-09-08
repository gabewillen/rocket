// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>

#include "decode/target_layer3_prefill.h"

namespace rocket::qwen38::decode {

// Fixed graph-safe owner for QSA row mappings plus the composed target-MoE
// requested generation. Construction validates pointers without launching.
class NativeTargetLayer3GenerationOwner final
    : public TargetLayer3GenerationOwner {
 public:
  NativeTargetLayer3GenerationOwner(
      int rank, const attention::TargetQsaStateView& storage,
      std::uint64_t* target_moe_requested_generation);

  int rank() const noexcept override { return rank_; }
  int layer() const noexcept override { return 3; }
  bool authenticated() const noexcept override { return authenticated_; }
  const attention::TargetQsaStateView& view(
      int row, std::uint64_t generation) const override;
  void enqueue_prepare(int row, std::uint64_t generation,
                       cudaStream_t stream) override;

 private:
  int rank_;
  std::array<attention::TargetQsaStateView, kTargetLayer3OracleRows> views_;
  std::uint64_t* target_moe_requested_generation_;
  bool authenticated_ = false;
  bool faulted_ = false;
  int next_row_ = 0;
};

}  // namespace rocket::qwen38::decode

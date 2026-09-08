// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_k0_executor.h"

#include <array>
#include <memory>

namespace rocket::qwen38::decode {

[[nodiscard]] constexpr TargetK0AttentionKind
target_k0_attention_kind(int layer) noexcept {
  return is_qsa_layer(layer) ? TargetK0AttentionKind::kQsa
                             : TargetK0AttentionKind::kGdn;
}

// Process-local owner of the complete rank-local decoder layer inventory.
// Construction consumes exactly 48 already-constructed production owners and
// rejects any missing, unauthenticated, cross-rank, cross-layer, cross-kind,
// or cross-wired reduction participant before publishing the borrowed port
// array. The inventory is single-owner and not thread-safe; execution remains
// serialized by TargetK0Executor on its bound stream.
class TargetK0LayerOwnerInventory final {
 public:
  TargetK0LayerOwnerInventory(
      int rank,
      std::array<std::unique_ptr<TargetK0LayerPort>, kDecoderLayers> owners,
      TargetK0PairReduceSchedule& reductions,
      pair_reduce::OtelStageSink& telemetry);
  TargetK0LayerOwnerInventory(const TargetK0LayerOwnerInventory&) = delete;
  TargetK0LayerOwnerInventory& operator=(
      const TargetK0LayerOwnerInventory&) = delete;

  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] bool authenticated() const noexcept { return authenticated_; }
  [[nodiscard]] const std::array<TargetK0LayerPort*, kDecoderLayers>& ports()
      const noexcept {
    return ports_;
  }

 private:
  int rank_ = -1;
  bool authenticated_ = false;
  std::array<std::unique_ptr<TargetK0LayerPort>, kDecoderLayers> owners_{};
  std::array<TargetK0LayerPort*, kDecoderLayers> ports_{};
};

}  // namespace rocket::qwen38::decode

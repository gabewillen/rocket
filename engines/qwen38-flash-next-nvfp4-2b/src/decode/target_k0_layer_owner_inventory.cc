// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_layer_owner_inventory.h"

#include <set>
#include <stdexcept>
#include <utility>

namespace rocket::qwen38::decode {

TargetK0LayerOwnerInventory::TargetK0LayerOwnerInventory(
    int rank,
    std::array<std::unique_ptr<TargetK0LayerPort>, kDecoderLayers> owners,
    TargetK0PairReduceSchedule& reductions,
    pair_reduce::OtelStageSink& telemetry)
    : rank_(rank), owners_(std::move(owners)) {
  const auto emit = [&](pair_reduce::Outcome outcome) noexcept {
    telemetry.emit_span_and_log({
        "rocket.qwen38.k0.layer_owner_inventory", "owner-init",
        "all48-owner-inventory", (rank == 0 || rank == 1) ? rank : -1, 1,
        pair_reduce::kDtype, outcome, 0, 0});
    telemetry.record_duration({(rank == 0 || rank == 1) ? rank : -1, 1,
                               pair_reduce::kDtype, outcome, 0});
  };
  try {
    if (rank != 0 && rank != 1)
      throw std::invalid_argument("target K0 layer owner rank changed");
    std::set<const HiddenPartialReducer*> reduction_identities;
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      auto* owner = owners_[static_cast<std::size_t>(layer)].get();
      auto* attention = owner ? owner->attention_reducer_identity() : nullptr;
      auto* moe = owner ? owner->moe_reducer_identity() : nullptr;
      if (!owner || owner->rank() != rank || owner->layer() != layer ||
          owner->attention_kind() != target_k0_attention_kind(layer) ||
          !owner->authenticated() ||
          attention != &reductions.attention_port(layer) ||
          moe != &reductions.moe_port(layer) || attention == moe ||
          !reduction_identities.insert(attention).second ||
          !reduction_identities.insert(moe).second)
        throw std::invalid_argument(
            "target K0 layer owner inventory binding changed");
      ports_[static_cast<std::size_t>(layer)] = owner;
    }
  } catch (...) {
    emit(pair_reduce::Outcome::kContractError);
    throw;
  }
  authenticated_ = true;
  emit(pair_reduce::Outcome::kOk);
}

}  // namespace rocket::qwen38::decode

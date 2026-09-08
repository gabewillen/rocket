// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_c1.h"

#include <cmath>

namespace rocket::qwen38::moe {
namespace {

bool valid_identity(int rank, int layer) noexcept {
  return (rank == 0 || rank == 1) && layer >= 0 && layer < 48;
}

bool valid_route(std::int32_t expert, float weight) noexcept {
  return expert >= 0 && expert < kTargetMoeGlobalExperts &&
         std::isfinite(weight) && weight >= 0.0F && weight <= 1.0F;
}

}  // namespace

TargetMoeOutcome validate_target_moe_c1_summary(
    const TargetMoeC1Summary& summary,
    std::uint64_t requested_generation, int rank, int layer) noexcept {
  if (!valid_identity(rank, layer) || requested_generation == 0)
    return TargetMoeOutcome::kContractError;
  if (summary.outcome != TargetMoeOutcome::kOk)
    return summary.outcome;
  if (summary.generation != requested_generation)
    return TargetMoeOutcome::kStaleGeneration;
  if (summary.local_routes < 0 || summary.local_routes > kTargetMoeC1TopK)
    return TargetMoeOutcome::kContractError;
  return TargetMoeOutcome::kOk;
}

TargetMoeC1RouteReference target_moe_c1_route_reference(
    int rank, int layer,
    const std::array<std::int32_t, kTargetMoeC1TopK>& global_expert_ids,
    const std::array<float, kTargetMoeC1TopK>& routing_weights,
    std::uint64_t source_generation,
    std::uint64_t requested_generation) noexcept {
  TargetMoeC1RouteReference result;
  result.summary.outcome = TargetMoeOutcome::kContractError;
  if (!valid_identity(rank, layer) || requested_generation == 0) return result;
  if (source_generation != requested_generation) {
    result.summary.outcome = TargetMoeOutcome::kStaleGeneration;
    return result;
  }
  for (int slot = 0; slot < kTargetMoeC1TopK; ++slot) {
    if (!valid_route(global_expert_ids[slot], routing_weights[slot])) return result;
  }
  const int first = rank * kTargetMoeLocalExperts;
  for (int slot = 0; slot < kTargetMoeC1TopK; ++slot) {
    const int global = global_expert_ids[slot];
    const bool local = global >= first && global < first + kTargetMoeLocalExperts;
    result.local_expert_ids[slot] = local ? global - first : 0;
    result.local_routing_weights[slot] = local ? routing_weights[slot] : 0.0F;
    result.summary.local_routes += local ? 1 : 0;
  }
  result.summary.generation = requested_generation;
  result.summary.outcome = TargetMoeOutcome::kOk;
  return result;
}

void export_target_moe_c1_otel_after_fence(
    const TargetMoeC1Summary& summary,
    std::uint64_t requested_generation, int rank, int layer,
    TargetMoeOtelSink& sink) noexcept {
  const auto outcome = validate_target_moe_c1_summary(
      summary, requested_generation, rank, layer);
  const std::uint64_t local_routes = outcome == TargetMoeOutcome::kOk
                                         ? summary.local_routes
                                         : 0;
  sink.add_counter({TargetMoeCounter::kLaunch, outcome, rank, layer, 1});
  sink.add_counter(
      {TargetMoeCounter::kLocalRoutes, outcome, rank, layer, local_routes});
}

}  // namespace rocket::qwen38::moe

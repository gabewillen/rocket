// SPDX-License-Identifier: Apache-2.0
#include "moe/route_compaction.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocket::qwen38::moe {
namespace {

struct Failure {
  RouteCompactionOutcome outcome;
};

struct RoutePair {
  std::int32_t expert;
  float weight;
};

CpuRouteCompactionResult failure(Failure failure) {
  CpuRouteCompactionResult result;
  result.summary.outcome = failure.outcome;
  return result;
}

bool valid_route(RoutePair route) noexcept {
  return route.expert >= 0 && route.expert < kGlobalExperts &&
         std::isfinite(route.weight) && route.weight >= 0.0F &&
         route.weight <= 1.0F;
}

}  // namespace

RouteCompactionOutcome validate_route_compaction_summary(
    const RouteCompactionValidation& validation) noexcept {
  const auto& summary = validation.summary;
  const auto requested_generation = validation.requested_generation;
  const auto shape = validation.shape;
  if (!allowed_shape(shape) || requested_generation == 0)
    return RouteCompactionOutcome::kContractError;
  if (summary.outcome != RouteCompactionOutcome::kOk) return summary.outcome;
  if (summary.generation != requested_generation)
    return RouteCompactionOutcome::kStaleGeneration;
  if (summary.active_experts < 0 || summary.active_experts > kLocalExperts ||
      summary.active_rows < 0 || summary.active_rows > shape.rows ||
      summary.active_routes < 0 ||
      summary.active_routes > shape.rows * kTopK ||
      summary.active_weight_bytes !=
          static_cast<std::uint64_t>(summary.active_experts) *
              kNvfp4BytesPerExpert)
    return RouteCompactionOutcome::kContractError;
  return RouteCompactionOutcome::kOk;
}

CpuRouteCompactionResult compact_owner_routes_reference(
    const CpuRouteCompactionInput& input) {
  if (!allowed_shape(input.shape) || !allowed_capacity(input.capacity) ||
      input.shape.rows > input.capacity.rows) {
    return failure({.outcome = RouteCompactionOutcome::kContractError});
  }
  if (input.source_generation == 0 ||
      input.source_generation != input.requested_generation) {
    return failure({.outcome = RouteCompactionOutcome::kStaleGeneration});
  }
  const auto route_count = static_cast<std::size_t>(input.shape.rows) * kTopK;
  if (input.global_expert_ids.size() != route_count ||
      input.routing_weights.size() != route_count) {
    return failure({.outcome = RouteCompactionOutcome::kContractError});
  }

  CpuRouteCompactionResult result;
  std::vector<std::int32_t> local_to_active(kLocalExperts, -1);
  const int first_expert = input.shape.rank * kLocalExperts;
  int active_rows = 0;

  for (int row = 0; row < input.shape.rows; ++row) {
    bool owner_row = false;
    for (int slot = 0; slot < kTopK; ++slot) {
      const auto route = static_cast<std::size_t>(row * kTopK + slot);
      const std::int32_t expert = input.global_expert_ids[route];
      const float weight = input.routing_weights[route];
      if (!valid_route({.expert = expert, .weight = weight})) {
        return failure({.outcome = RouteCompactionOutcome::kContractError});
      }
      for (int prior = 0; prior < slot; ++prior) {
        if (input.global_expert_ids[static_cast<std::size_t>(row * kTopK + prior)] ==
            expert) {
          return failure({.outcome = RouteCompactionOutcome::kContractError});
        }
      }
      if (expert < first_expert || expert >= first_expert + kLocalExperts) {
        continue;
      }
      if (result.owner_route_global_expert_ids.size() >=
          static_cast<std::size_t>(input.capacity.routes)) {
        return failure({.outcome = RouteCompactionOutcome::kOverflow});
      }
      owner_row = true;
      const int local_expert = expert - first_expert;
      int active_expert = local_to_active[static_cast<std::size_t>(local_expert)];
      if (active_expert < 0) {
        if (result.active_global_expert_ids.size() >=
            static_cast<std::size_t>(input.capacity.experts)) {
          return failure({.outcome = RouteCompactionOutcome::kOverflow});
        }
        active_expert = static_cast<int>(result.active_global_expert_ids.size());
        local_to_active[static_cast<std::size_t>(local_expert)] = active_expert;
        result.active_global_expert_ids.push_back(expert);
        result.expert_row_counts.push_back(0);
      }
      ++result.expert_row_counts[static_cast<std::size_t>(active_expert)];
      result.owner_route_global_expert_ids.push_back(expert);
      result.owner_route_weights.push_back(weight);
      result.owner_route_rows.push_back(row);
      result.owner_route_slots.push_back(static_cast<std::uint8_t>(slot));
    }
    if (owner_row) ++active_rows;
  }

  result.expert_route_offsets.resize(result.active_global_expert_ids.size() + 1);
  result.expert_route_offsets[0] = 0;
  for (std::size_t expert = 0; expert < result.expert_row_counts.size(); ++expert) {
    result.expert_route_offsets[expert + 1] =
        result.expert_route_offsets[expert] + result.expert_row_counts[expert];
  }
  result.expert_route_indices.resize(result.owner_route_global_expert_ids.size());
  auto cursors = result.expert_route_offsets;
  cursors.pop_back();
  for (std::size_t route = 0;
       route < result.owner_route_global_expert_ids.size(); ++route) {
    const int local_expert =
        result.owner_route_global_expert_ids[route] - first_expert;
    const int active_expert = local_to_active[static_cast<std::size_t>(local_expert)];
    result.expert_route_indices[static_cast<std::size_t>(
        cursors[static_cast<std::size_t>(active_expert)]++)] =
        static_cast<std::int32_t>(route);
  }

  result.summary.generation = input.requested_generation;
  result.summary.active_experts =
      static_cast<std::int32_t>(result.active_global_expert_ids.size());
  result.summary.active_rows = active_rows;
  result.summary.active_routes =
      static_cast<std::int32_t>(result.owner_route_global_expert_ids.size());
  result.summary.active_weight_bytes =
      static_cast<std::uint64_t>(result.summary.active_experts) *
      kNvfp4BytesPerExpert;
  result.summary.outcome = RouteCompactionOutcome::kOk;
  return result;
}

void export_route_compaction_otel_after_fence(
    const RouteCompactionOtelExport& export_request) noexcept {
  const auto& snapshot = export_request.snapshot;
  const auto shape = export_request.shape;
  auto& sink = export_request.sink;
  RouteCompactionOutcome outcome = validate_route_compaction_summary(
      {.summary = snapshot,
       .requested_generation = export_request.requested_generation,
       .shape = shape});
  std::uint64_t experts = 0;
  std::uint64_t rows = 0;
  std::uint64_t bytes = 0;
  if (outcome == RouteCompactionOutcome::kOk) {
    experts = static_cast<std::uint64_t>(snapshot.active_experts);
    rows = static_cast<std::uint64_t>(snapshot.active_rows);
    bytes = snapshot.active_weight_bytes;
  }
  const RouteCompactionOtelAttributes attributes{
      .outcome = outcome,
      .rank = allowed_shape(shape) ? shape.rank : 0,
      .sequence_bucket = allowed_shape(shape) ? shape.sequences : 1,
  };
  sink.add_counter({.counter = RouteCompactionCounter::kActiveExperts,
                    .attributes = attributes,
                    .value = experts});
  sink.add_counter({.counter = RouteCompactionCounter::kActiveRows,
                    .attributes = attributes,
                    .value = rows});
  sink.add_counter({.counter = RouteCompactionCounter::kActiveWeightBytes,
                    .attributes = attributes,
                    .value = bytes});
}

}  // namespace rocket::qwen38::moe

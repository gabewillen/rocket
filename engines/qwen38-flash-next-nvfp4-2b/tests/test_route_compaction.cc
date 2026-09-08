// SPDX-License-Identifier: Apache-2.0
#include "moe/route_compaction.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace moe = rocket::qwen38::moe;

namespace {

void check(bool condition) {
  if (!condition) std::abort();
}

struct Generations {
  std::uint64_t source = 7;
  std::uint64_t requested = 7;
};

struct Fixture {
  moe::RouteCompactionShape shape{.rank = 0, .sequences = 2, .rows = 5};
  moe::RouteCompactionCapacity capacity{};
  std::vector<std::int32_t> ids;
  std::vector<float> weights;

  Fixture() {
    ids.reserve(static_cast<std::size_t>(shape.rows * moe::kTopK));
    weights.reserve(static_cast<std::size_t>(shape.rows * moe::kTopK));
    for (int row = 0; row < shape.rows; ++row) {
      for (int slot = 0; slot < moe::kTopK; ++slot) {
        const int owner_id = (row * 3 + slot * 7) % 64;
        ids.push_back(slot % 2 == 0 ? owner_id : 256 + owner_id);
        weights.push_back(std::bit_cast<float>(
            std::uint32_t{0x3f000000U} +
            static_cast<std::uint32_t>(row * moe::kTopK + slot)));
      }
    }
  }

  [[nodiscard]] moe::CpuRouteCompactionInput input(
      Generations generations = {}) const {
    return {
        .shape = shape,
        .capacity = capacity,
        .global_expert_ids = ids,
        .routing_weights = weights,
        .source_generation = generations.source,
        .requested_generation = generations.requested,
    };
  }
};

void check_success_and_order() {
  Fixture fixture;
  const auto first = moe::compact_owner_routes_reference(fixture.input());
  const auto second = moe::compact_owner_routes_reference(fixture.input());
  check(first.summary.outcome == moe::RouteCompactionOutcome::kOk);
  check(first.summary.generation == 7);
  check(first.summary.active_rows == fixture.shape.rows);
  check(first.summary.active_routes == fixture.shape.rows * 5);
  check(first.summary.active_experts ==
        static_cast<int>(first.active_global_expert_ids.size()));
  check(first.summary.active_weight_bytes ==
        static_cast<std::uint64_t>(first.summary.active_experts) *
            moe::kNvfp4BytesPerExpert);

  std::size_t owner_route = 0;
  std::vector<std::int32_t> first_seen;
  for (int row = 0; row < fixture.shape.rows; ++row) {
    for (int slot = 0; slot < moe::kTopK; slot += 2) {
      const auto input_route = static_cast<std::size_t>(row * moe::kTopK + slot);
      const auto expert = fixture.ids[input_route];
      if (std::find(first_seen.begin(), first_seen.end(), expert) ==
          first_seen.end()) {
        first_seen.push_back(expert);
      }
      check(first.owner_route_global_expert_ids[owner_route] == expert);
      check(first.owner_route_rows[owner_route] == row);
      check(first.owner_route_slots[owner_route] == slot);
      check(std::bit_cast<std::uint32_t>(first.owner_route_weights[owner_route]) ==
            std::bit_cast<std::uint32_t>(fixture.weights[input_route]));
      ++owner_route;
    }
  }
  check(first.active_global_expert_ids == first_seen);
  check(first.expert_route_offsets.front() == 0);
  check(first.expert_route_offsets.back() == first.summary.active_routes);
  for (int active = 0; active < first.summary.active_experts; ++active) {
    int previous_owner_route = -1;
    for (int position = first.expert_route_offsets[static_cast<std::size_t>(active)];
         position < first.expert_route_offsets[static_cast<std::size_t>(active + 1)];
         ++position) {
      const int route = first.expert_route_indices[static_cast<std::size_t>(position)];
      check(route > previous_owner_route);
      check(first.owner_route_global_expert_ids[static_cast<std::size_t>(route)] ==
            first.active_global_expert_ids[static_cast<std::size_t>(active)]);
      previous_owner_route = route;
    }
  }
  check(first.active_global_expert_ids == second.active_global_expert_ids);
  check(first.owner_route_global_expert_ids ==
        second.owner_route_global_expert_ids);
  check(first.expert_route_indices == second.expert_route_indices);
  for (std::size_t index = 0; index < first.owner_route_weights.size(); ++index) {
    check(std::bit_cast<std::uint32_t>(first.owner_route_weights[index]) ==
          std::bit_cast<std::uint32_t>(second.owner_route_weights[index]));
  }
}

void check_empty_owner_prefix() {
  Fixture fixture;
  for (auto& expert : fixture.ids) expert += expert < 256 ? 256 : 0;
  const auto result = moe::compact_owner_routes_reference(fixture.input());
  check(result.summary.outcome == moe::RouteCompactionOutcome::kOk);
  check(result.summary.active_experts == 0 && result.summary.active_rows == 0 &&
        result.summary.active_routes == 0 &&
        result.summary.active_weight_bytes == 0);
  check(result.expert_route_offsets.size() == 1 &&
        result.expert_route_offsets.front() == 0);
}

void check_failures() {
  Fixture fixture;
  const auto stale = moe::compact_owner_routes_reference(fixture.input(
      {.source = 6, .requested = 7}));
  check(stale.summary.outcome == moe::RouteCompactionOutcome::kStaleGeneration &&
        stale.summary.generation == 0 && stale.summary.active_routes == 0);

  auto invalid = fixture;
  invalid.ids[0] = -1;
  check(moe::compact_owner_routes_reference(invalid.input()).summary.outcome ==
        moe::RouteCompactionOutcome::kContractError);

  auto nonfinite = fixture;
  nonfinite.weights[0] = std::numeric_limits<float>::quiet_NaN();
  check(moe::compact_owner_routes_reference(nonfinite.input()).summary.outcome ==
        moe::RouteCompactionOutcome::kContractError);

  auto negative_weight = fixture;
  negative_weight.weights[0] = -0.25F;
  check(moe::compact_owner_routes_reference(negative_weight.input())
            .summary.outcome == moe::RouteCompactionOutcome::kContractError);

  auto duplicate = fixture;
  duplicate.ids[1] = duplicate.ids[0];
  check(moe::compact_owner_routes_reference(duplicate.input()).summary.outcome ==
        moe::RouteCompactionOutcome::kContractError);

  auto expert_overflow = fixture;
  expert_overflow.capacity.experts = 1;
  check(moe::compact_owner_routes_reference(expert_overflow.input())
            .summary.outcome == moe::RouteCompactionOutcome::kOverflow);

  auto route_overflow = fixture;
  route_overflow.capacity.routes = 1;
  check(moe::compact_owner_routes_reference(route_overflow.input())
            .summary.outcome == moe::RouteCompactionOutcome::kOverflow);

  auto row_capacity = fixture;
  row_capacity.capacity.rows = 4;
  check(moe::compact_owner_routes_reference(row_capacity.input())
            .summary.outcome == moe::RouteCompactionOutcome::kContractError);
}

void check_post_fence_summary_validation() {
  const moe::RouteCompactionShape shape{.rank = 0, .sequences = 16, .rows = 16};
  moe::RouteCompactionDeviceSummary summary{
      .generation = 9,
      .active_weight_bytes = 19 * moe::kNvfp4BytesPerExpert,
      .active_experts = 19,
      .active_rows = 16,
      .active_routes = 80,
      .outcome = moe::RouteCompactionOutcome::kOk,
  };
  check(moe::validate_route_compaction_summary(
            {.summary = summary, .requested_generation = 9, .shape = shape}) ==
        moe::RouteCompactionOutcome::kOk);
  check(moe::validate_route_compaction_summary(
            {.summary = summary, .requested_generation = 10, .shape = shape}) ==
        moe::RouteCompactionOutcome::kStaleGeneration);
  summary.generation = 10;
  summary.active_experts = 257;
  check(moe::validate_route_compaction_summary(
            {.summary = summary, .requested_generation = 10, .shape = shape}) ==
        moe::RouteCompactionOutcome::kContractError);
  summary.active_experts = 19;
  summary.active_weight_bytes = 0;
  check(moe::validate_route_compaction_summary(
            {.summary = summary, .requested_generation = 10, .shape = shape}) ==
        moe::RouteCompactionOutcome::kContractError);
  summary.active_weight_bytes = 19 * moe::kNvfp4BytesPerExpert;
  summary.outcome = moe::RouteCompactionOutcome::kOverflow;
  check(moe::validate_route_compaction_summary(
            {.summary = summary, .requested_generation = 10, .shape = shape}) ==
        moe::RouteCompactionOutcome::kOverflow);
}

class CaptureSink final : public moe::RouteCompactionOtelSink {
 public:
  void add_counter(const moe::RouteCompactionOtelPoint& point) noexcept override {
    points.push_back(point);
  }
  std::vector<moe::RouteCompactionOtelPoint> points;
};

void check_otel_cardinality() {
  static_assert(3 * 5 * 2 * 5 == 150);
  const moe::RouteCompactionDeviceSummary summary{
      .generation = 9,
      .active_weight_bytes = 19 * moe::kNvfp4BytesPerExpert,
      .active_experts = 19,
      .active_rows = 8,
      .active_routes = 40,
      .outcome = moe::RouteCompactionOutcome::kOk,
  };
  CaptureSink sink;
  moe::export_route_compaction_otel_after_fence({
      .snapshot = summary,
      .requested_generation = 9,
      .shape = {.rank = 1, .sequences = 8, .rows = 8},
      .sink = sink,
  });
  check(sink.points.size() == 3);
  check(sink.points[0].counter == moe::RouteCompactionCounter::kActiveExperts &&
        sink.points[0].value == 19);
  check(sink.points[1].counter == moe::RouteCompactionCounter::kActiveRows &&
        sink.points[1].value == 8);
  check(sink.points[2].counter == moe::RouteCompactionCounter::kActiveWeightBytes &&
        sink.points[2].value == 19 * moe::kNvfp4BytesPerExpert);
  for (const auto& point : sink.points) {
    check(point.attributes.outcome == moe::RouteCompactionOutcome::kOk &&
          point.attributes.rank == 1 && point.attributes.sequence_bucket == 8);
  }

  CaptureSink corrupt_sink;
  auto corrupt = summary;
  corrupt.active_experts = 257;
  moe::export_route_compaction_otel_after_fence({
      .snapshot = corrupt,
      .requested_generation = 9,
      .shape = {.rank = 1, .sequences = 8, .rows = 8},
      .sink = corrupt_sink,
  });
  check(corrupt_sink.points.size() == 3);
  for (const auto& point : corrupt_sink.points) {
    check(point.attributes.outcome ==
              moe::RouteCompactionOutcome::kContractError &&
          point.value == 0);
  }

  CaptureSink stale_sink;
  moe::export_route_compaction_otel_after_fence({
      .snapshot = summary,
      .requested_generation = 10,
      .shape = {.rank = 1, .sequences = 8, .rows = 8},
      .sink = stale_sink,
  });
  check(stale_sink.points.size() == 3);
  for (const auto& point : stale_sink.points) {
    check(point.attributes.outcome ==
              moe::RouteCompactionOutcome::kStaleGeneration &&
          point.value == 0);
  }
}

}  // namespace

int main() {
  static_assert(moe::allowed_shape({.rank = 0, .sequences = 16, .rows = 128}));
  static_assert(!moe::allowed_shape({.rank = 0, .sequences = 16, .rows = 129}));
  check_success_and_order();
  check_empty_owner_prefix();
  check_failures();
  check_post_fence_summary_validation();
  check_otel_cardinality();
  return 0;
}

// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_c1.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace moe = rocket::qwen38::moe;

void check(bool value) {
  if (!value) std::abort();
}

class Sink final : public moe::TargetMoeOtelSink {
 public:
  void add_counter(const moe::TargetMoeOtelPoint& point) noexcept override {
    points.push_back(point);
  }
  std::vector<moe::TargetMoeOtelPoint> points;
};

int main() {
  const std::array<std::int32_t, 10> ids{0, 255, 256, 511, 7,
                                         300, 3, 400, 9, 500};
  const std::array<float, 10> weights{0.11F, 0.12F, 0.13F, 0.14F, 0.15F,
                                      0.16F, 0.17F, 0.18F, 0.19F, 0.20F};
  const auto rank0 = moe::target_moe_c1_route_reference(
      0, 47, ids, weights, 9, 9);
  const auto rank1 = moe::target_moe_c1_route_reference(
      1, 47, ids, weights, 9, 9);
  check(rank0.summary.outcome == moe::TargetMoeOutcome::kOk &&
        rank1.summary.outcome == moe::TargetMoeOutcome::kOk);
  check(rank0.local_expert_ids[0] == 0 && rank0.local_routing_weights[0] == weights[0]);
  check(rank0.local_expert_ids[2] == 0 && rank0.local_routing_weights[2] == 0.0F);
  check(rank1.local_expert_ids[2] == 0 && rank1.local_routing_weights[2] == weights[2]);
  check(rank1.local_expert_ids[3] == 255 && rank1.local_routing_weights[3] == weights[3]);
  for (int slot = 0; slot < 10; ++slot)
    check((rank0.local_routing_weights[slot] == weights[slot]) !=
          (rank1.local_routing_weights[slot] == weights[slot]));

  const auto stale = moe::target_moe_c1_route_reference(
      0, 0, ids, weights, 8, 9);
  check(stale.summary.outcome == moe::TargetMoeOutcome::kStaleGeneration);
  for (float weight : stale.local_routing_weights) check(weight == 0.0F);
  auto invalid_weights = weights;
  invalid_weights[4] = NAN;
  const auto invalid = moe::target_moe_c1_route_reference(
      0, 0, ids, invalid_weights, 9, 9);
  check(invalid.summary.outcome == moe::TargetMoeOutcome::kContractError);
  check(moe::validate_target_moe_c1_summary(rank0.summary, 9, 0, 47) ==
        moe::TargetMoeOutcome::kOk);
  check(moe::validate_target_moe_c1_summary(rank0.summary, 10, 0, 47) ==
        moe::TargetMoeOutcome::kStaleGeneration);

  Sink sink;
  moe::export_target_moe_c1_otel_after_fence(rank0.summary, 9, 0, 47, sink);
  check(sink.points.size() == 2 && sink.points[0].rank == 0 &&
        sink.points[0].layer == 47 &&
        sink.points[1].value == static_cast<std::uint64_t>(rank0.summary.local_routes));
}

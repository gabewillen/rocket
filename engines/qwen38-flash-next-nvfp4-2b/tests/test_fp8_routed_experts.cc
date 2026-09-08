// SPDX-License-Identifier: Apache-2.0
#include "moe/fp8_routed_experts.h"

#include <cstdlib>
#include <string_view>
#include <vector>

#undef assert
#define assert(condition) do { if (!(condition)) std::abort(); } while (false)

namespace moe = rocket::qwen38::moe;

namespace {
class Sink final : public moe::RoutedExpertOtelSink {
 public:
  void add_routed_expert_counter(
      const moe::RoutedExpertOtelPoint& point) noexcept override {
    points.push_back(point);
  }
  std::vector<moe::RoutedExpertOtelPoint> points;
};
}

int main() {
  static_assert(moe::kGlobalExperts == 512);
  static_assert(moe::kLocalExperts == 256);
  static_assert(moe::kTopK == 10);
  static_assert(moe::kHidden == 2560);
  static_assert(moe::kLogicalIntermediate == 640);
  static_assert(moe::kPhysicalIntermediate == 768);
  static_assert(moe::kFp8Block == 128);
  static_assert(moe::kFp8BytesPerExpert == 4'915'800);
  assert(std::string_view(moe::kMtpExpertSourceAbi) ==
         "fp8_e4m3_block_128x128");
  assert(std::string_view(moe::kMtpExpertServingAbi) ==
         moe::kMtpExpertSourceAbi);

  moe::RoutedExpertDeviceSummary valid{
      .generation = 9,
      .active_weight_bytes = 12 * moe::kFp8BytesPerExpert,
      .fc1_tiles = 60 * 5,
      .fc2_tiles = 60 * 20,
      .active_experts = 12,
      .active_routes = 60,
      .outcome = moe::RoutedExpertOutcome::kOk,
  };
  const moe::RouteCompactionShape shape{0, 8, 8};
  assert(moe::validate_routed_expert_summary({valid, 9, shape}) ==
         moe::RoutedExpertOutcome::kOk);
  auto stale = valid;
  stale.generation = 8;
  assert(moe::validate_routed_expert_summary({stale, 9, shape}) ==
         moe::RoutedExpertOutcome::kStaleGeneration);
  auto invalid = valid;
  ++invalid.fc1_tiles;
  assert(moe::validate_routed_expert_summary({invalid, 9, shape}) ==
         moe::RoutedExpertOutcome::kContractError);

  Sink sink;
  moe::export_routed_expert_otel_after_fence(valid, 9, shape, sink);
  assert(sink.points.size() == 5);
  assert(sink.points[0].rank == 0 && sink.points[0].sequence_bucket == 8);
  assert(sink.points[0].value == 12);
  assert(sink.points[1].value == 60);
  assert(sink.points[2].value == 12 * moe::kFp8BytesPerExpert);
  assert(sink.points[3].value == 300);
  assert(sink.points[4].value == 1200);
  return 0;
}

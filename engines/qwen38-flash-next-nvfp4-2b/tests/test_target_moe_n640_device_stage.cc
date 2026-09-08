// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_n640_device_stage.h"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace moe = rocket::qwen38::moe;

namespace {
struct Sink final : moe::TargetMoeStageOtelSink {
  void add_counter(const moe::TargetMoeStageOtelPoint& point) noexcept override {
    points[count++] = point;
  }
  std::array<moe::TargetMoeStageOtelPoint, 8> points{};
  int count = 0;
};

void require(bool value) {
  if (!value) throw std::runtime_error("target MoE device stage contract changed");
}
}  // namespace

int main() {
  static_assert(moe::kTargetMoeStagedExperts == 10);
  static_assert(moe::kTargetMoeStagedW13PackedBytes == 19'660'800);
  static_assert(moe::kTargetMoeStagedW13ScaleBytes == 2'457'600);
  static_assert(moe::kTargetMoeStagedDownPackedBytes == 9'830'400);
  static_assert(moe::kTargetMoeStagedDownScaleBytes == 1'228'800);
  static_assert(moe::kTargetMoeStageRawPlaneBytes == 33'177'760);
  static_assert(moe::kTargetMoeStageScratchBytes == 33'177'896);

  moe::TargetMoeN768StageScratch scratch{};
  scratch.w13_packed = reinterpret_cast<std::uint8_t*>(0x1000);
  scratch.w13_packed_bytes = moe::kTargetMoeStagedW13PackedBytes;
  scratch.w13_scale = reinterpret_cast<std::uint8_t*>(0x2000);
  scratch.w13_scale_bytes = moe::kTargetMoeStagedW13ScaleBytes;
  scratch.down_packed = reinterpret_cast<std::uint8_t*>(0x3000);
  scratch.down_packed_bytes = moe::kTargetMoeStagedDownPackedBytes;
  scratch.down_scale = reinterpret_cast<std::uint8_t*>(0x4000);
  scratch.down_scale_bytes = moe::kTargetMoeStagedDownScaleBytes;
  scratch.input_global_scale = reinterpret_cast<float*>(0x5000);
  scratch.folded_w1_alpha = reinterpret_cast<float*>(0x6000);
  scratch.w2_alpha = reinterpret_cast<float*>(0x7000);
  scratch.down_input_scale = reinterpret_cast<float*>(0x8000);
  scratch.source_expert_ids = reinterpret_cast<std::int32_t*>(0x9000);
  scratch.compact_expert_ids = reinterpret_cast<std::int32_t*>(0xa000);
  scratch.compact_routing_weights = reinterpret_cast<float*>(0xb000);
  scratch.evidence = reinterpret_cast<moe::TargetMoeN640StageEvidence*>(0xc000);
  scratch.scalar_capacity = 10;
  scratch.route_capacity = 10;
  require(moe::validate_target_moe_stage_scratch(scratch));
  auto short_scratch = scratch;
  --short_scratch.down_scale_bytes;
  require(!moe::validate_target_moe_stage_scratch(short_scratch));
  const auto weights = moe::target_moe_staged_weights(scratch);
  require(weights.w13_packed == scratch.w13_packed &&
          weights.w13_scale == scratch.w13_scale &&
          weights.down_packed == scratch.down_packed &&
          weights.down_scale == scratch.down_scale &&
          weights.input_global_scale == scratch.input_global_scale &&
          weights.folded_w1_alpha == scratch.folded_w1_alpha &&
          weights.w2_alpha == scratch.w2_alpha &&
          weights.down_input_scale == scratch.down_input_scale);

  std::array<std::int32_t, 10> ids{3, 7, 9, 11, 13, 0, 0, 0, 0, 0};
  std::array<float, 10> routes{0.4F, 0.3F, 0.2F, 0.1F};
  const auto mapped = moe::target_moe_n640_stage_reference(ids, routes, 7, 7);
  require(mapped.evidence.outcome == moe::TargetMoeOutcome::kOk &&
          mapped.evidence.active_experts == 4 &&
          mapped.source_expert_ids[0] == 3 &&
          mapped.source_expert_ids[3] == 11 &&
          mapped.compact_expert_ids[3] == 3 &&
          mapped.compact_routing_weights[4] == 0.0F);
  ids[1] = 3;
  const auto duplicate =
      moe::target_moe_n640_stage_reference(ids, routes, 7, 7);
  require(duplicate.evidence.outcome == moe::TargetMoeOutcome::kContractError &&
          duplicate.compact_routing_weights[0] == 0.0F &&
          duplicate.source_expert_ids[0] == -1);
  ids[1] = 7;
  routes[1] = std::numeric_limits<float>::quiet_NaN();
  const auto invalid = moe::target_moe_n640_stage_reference(ids, routes, 7, 7);
  require(invalid.evidence.outcome == moe::TargetMoeOutcome::kContractError &&
          invalid.compact_routing_weights[0] == 0.0F &&
          invalid.source_expert_ids[0] == -1);
  routes[1] = 0.3F;
  require(moe::target_moe_n640_stage_reference(ids, routes, 6, 7)
              .evidence.outcome == moe::TargetMoeOutcome::kStaleGeneration);

  Sink sink;
  const moe::TargetMoeN640StageEvidence good{
      7, 3, moe::TargetMoeOutcome::kOk};
  moe::export_target_moe_stage_otel_after_fence(good, 7, 1, 3, sink);
  require(sink.count == 4 && sink.points[0].counter ==
          moe::TargetMoeStageCounter::kLaunch &&
          sink.points[1].value == 3 &&
          sink.points[3].value == moe::kTargetMoeStageScratchBytes);
  const moe::TargetMoeN640StageEvidence stale{
      6, 3, moe::TargetMoeOutcome::kOk};
  moe::export_target_moe_stage_otel_after_fence(stale, 7, 1, 3, sink);
  require(sink.count == 8 &&
          sink.points[4].outcome == moe::TargetMoeOutcome::kContractError);
}

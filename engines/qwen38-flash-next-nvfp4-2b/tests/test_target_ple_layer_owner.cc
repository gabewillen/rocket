// SPDX-License-Identifier: Apache-2.0
#include "decode/target_ple_layer_owner.h"

#include <array>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace decode = rocket::qwen38::decode;

namespace {

void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

class Sink final : public decode::TargetPleStageSink {
 public:
  void emit(const decode::TargetPleStageEvent& event) noexcept override {
    events.push_back(event);
  }
  std::vector<decode::TargetPleStageEvent> events;
};

void check_plan(const std::filesystem::path& path, int expected_rank) {
  auto plan = decode::load_target_layer_native_plan(path);
  check(plan.rank == expected_rank && plan.layer == decode::kTargetPleLayer,
        "PLE descriptor identity changed");
  check(plan.extents.size() == 3'191 &&
            decode::validate_target_ple_layer_plan(plan),
        "PLE plan did not authenticate");
  Sink sink;
  auto owner = decode::TargetPleLayerOwnerContract::create(plan, sink);
  check(owner->authenticated() && owner->rank() == expected_rank &&
            owner->layer() == decode::kTargetPleLayer &&
            owner->binding().embedding_shards.size() == 64 &&
            owner->binding().key_projection.input_scale.length_bytes == 4 &&
            owner->binding().value_projection.input_scale.length_bytes == 4,
        "PLE owner binding changed");
  for (const auto stage : decode::kTargetPleStageOrder)
    check(owner->record_stage(stage, decode::TargetPleStageOutcome::kOk),
          "PLE stage order rejected");
  check(owner->complete() && !owner->faulted() && sink.events.size() == 10,
        "PLE telemetry completion changed");
  for (std::size_t index = 0; index < sink.events.size(); ++index)
    check(sink.events[index].rank == expected_rank &&
              sink.events[index].layer == decode::kTargetPleLayer &&
              sink.events[index].stage == decode::kTargetPleStageOrder[index],
          "PLE telemetry identity changed");

  Sink invalid_sink;
  auto invalid = decode::TargetPleLayerOwnerContract::create(plan, invalid_sink);
  check(!invalid->record_stage(decode::TargetPleStage::kKeyProjection,
                               decode::TargetPleStageOutcome::kOk) &&
            invalid->faulted() && invalid_sink.events.size() == 1 &&
            invalid_sink.events.front().outcome ==
                decode::TargetPleStageOutcome::kContractError,
        "out-of-order PLE stage was accepted");

  auto mutated = plan;
  for (auto& extent : mutated.extents)
    if (extent.name ==
        "model.language_model.layers.1.ple.key_proj.input_scale") {
      extent.length_bytes += 1;
      break;
    }
  check(!decode::validate_target_ple_layer_plan(mutated),
        "mutated PLE plan was accepted");
}

}  // namespace

int main(int argc, char** argv) {
  static_assert(decode::kTargetPleLayer == 1);
  static_assert(decode::kTargetPleEmbeddingShards == 64);
  static_assert(decode::kTargetPleStageOrder.front() ==
                decode::TargetPleStage::kEmbeddingLookup);
  static_assert(decode::kTargetPleStageOrder.back() ==
                decode::TargetPleStage::kResidual);
  if (argc != 3) return 2;
  check_plan(argv[1], 0);
  check_plan(argv[2], 1);
  return 0;
}

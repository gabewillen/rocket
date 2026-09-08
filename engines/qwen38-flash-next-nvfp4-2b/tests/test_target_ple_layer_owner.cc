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

class ContextSink final : public decode::TargetPleRowContextOtelSink {
 public:
  void emit(const decode::TargetPleRowContextEvent& event) noexcept override {
    events.push_back(event);
  }
  std::vector<decode::TargetPleRowContextEvent> events;
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

  const decode::TargetPleConvStateLayout conv_state{
      8, decode::kTargetPleConvChannels,
      decode::kTargetPleConvHistoryTokens};
  ContextSink context_sink;
  auto context = decode::TargetPleRowContextProvider::create(
      plan, 35, conv_state, context_sink);
  check(context->authenticated() && context->rank() == expected_rank &&
            context->layer() == decode::kTargetPleLayer &&
            context->conv_state_layout().history_tokens == 9 &&
            context->conv_state_layout().channels == 10'240,
        "PLE row context identity changed");
  decode::TargetPleRowContextInput input{};
  input.generation = 17;
  input.input_ids = {11, 12, 13};
  input.query_start_loc = {0, 2, 3};
  input.ngram_context = {101, 102, 201, 202};
  input.conv_state_indices = {2, 4};
  input.has_initial_state = {1, 0};
  const auto& published = context->publish(std::move(input));
  check(published.generation == 17 && published.rows == 3 &&
            published.requests == 2 &&
            &context->acquire(17) == &published &&
            published.input_ids[2] == 13 &&
            published.ngram_context.size() == 4 &&
            context_sink.events.size() == 2 &&
            context_sink.events[0].operation ==
                decode::TargetPleRowContextOperation::kPublish &&
            context_sink.events[1].operation ==
                decode::TargetPleRowContextOperation::kAcquire,
        "PLE row context publication changed");

  ContextSink malformed_sink;
  auto malformed = decode::TargetPleRowContextProvider::create(
      plan, 35, conv_state, malformed_sink);
  decode::TargetPleRowContextInput malformed_input{};
  malformed_input.generation = 1;
  malformed_input.input_ids = {11, 12};
  malformed_input.query_start_loc = {0, 2};
  malformed_input.ngram_context = {101};
  malformed_input.conv_state_indices = {0};
  malformed_input.has_initial_state = {1};
  bool malformed_rejected = false;
  try {
    (void)malformed->publish(std::move(malformed_input));
  } catch (const std::invalid_argument&) {
    malformed_rejected = true;
  }
  check(malformed_rejected && malformed->faulted() &&
            !malformed->authenticated() && malformed_sink.events.size() == 1 &&
            malformed_sink.events.front().outcome ==
                decode::TargetPleStageOutcome::kContractError,
        "malformed PLE context did not fail closed");

  ContextSink stale_sink;
  auto stale = decode::TargetPleRowContextProvider::create(
      plan, 35, conv_state, stale_sink);
  decode::TargetPleRowContextInput stale_input{};
  stale_input.generation = 4;
  stale_input.input_ids = {21};
  stale_input.query_start_loc = {0, 1};
  stale_input.ngram_context = {19, 20};
  stale_input.conv_state_indices = {1};
  stale_input.has_initial_state = {0};
  (void)stale->publish(std::move(stale_input));
  bool stale_rejected = false;
  try {
    (void)stale->acquire(3);
  } catch (const std::invalid_argument&) {
    stale_rejected = true;
  }
  check(stale_rejected && stale->faulted() && !stale->authenticated(),
        "stale PLE generation did not fail closed");

  bool layout_rejected = false;
  try {
    (void)decode::TargetPleRowContextProvider::create(
        plan, 35,
        {8, decode::kTargetPleConvChannels,
         decode::kTargetPleConvHistoryTokens - 1},
        context_sink);
  } catch (const std::invalid_argument&) {
    layout_rejected = true;
  }
  check(layout_rejected, "non-nine-token PLE history was accepted");
}

}  // namespace

int main(int argc, char** argv) {
  static_assert(decode::kTargetPleLayer == 1);
  static_assert(decode::kTargetPleEmbeddingShards == 64);
  static_assert(decode::kTargetPleNgramContextTokens == 2);
  static_assert(decode::kTargetPleConvChannels == 10'240);
  static_assert(decode::kTargetPleConvHistoryTokens == 9);
  static_assert(decode::kTargetPleStageOrder.front() ==
                decode::TargetPleStage::kEmbeddingLookup);
  static_assert(decode::kTargetPleStageOrder.back() ==
                decode::TargetPleStage::kResidual);
  if (argc != 3) return 2;
  check_plan(argv[1], 0);
  check_plan(argv[2], 1);
  return 0;
}

// SPDX-License-Identifier: Apache-2.0
#include "decode/target_ple_layer_owner.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocket::qwen38::decode {
namespace {

constexpr std::string_view kNvfp4Abi =
    "modelopt_nvfp4_group16_cutlass_sm121_sfb";

[[noreturn]] void fail(std::string_view reason) {
  throw std::invalid_argument("target PLE layer owner: " +
                              std::string(reason));
}

TargetLayerNativeExtent required(
    const TargetLayerNativePlan& plan, std::string_view suffix,
    std::uint64_t bytes, std::string_view dtype, std::string_view layout,
    std::string_view abi, std::initializer_list<std::uint64_t> shape) {
  const std::string name = "model.language_model.layers.1.ple." +
                           std::string(suffix);
  const auto found = std::find_if(
      plan.extents.begin(), plan.extents.end(),
      [&](const auto& item) { return item.name == name; });
  const std::vector<std::uint64_t> expected_shape(shape);
  if (found == plan.extents.end() || found->storage != "target_slab" ||
      found->length_bytes != bytes || found->dtype != dtype ||
      found->layout != layout || found->abi != abi ||
      found->shape != expected_shape)
    fail("required extent changed: " + name);
  return *found;
}

TargetPleNvfp4MatrixBinding matrix(
    const TargetLayerNativePlan& plan, std::string_view root,
    std::uint64_t weight_bytes, std::uint64_t scale_bytes,
    std::initializer_list<std::uint64_t> weight_shape,
    std::initializer_list<std::uint64_t> scale_shape) {
  return {
      required(plan, std::string(root) + ".weight", weight_bytes, "U8",
               "packed_e2m1_row_major", kNvfp4Abi, weight_shape),
      required(plan, std::string(root) + ".weight_scale", scale_bytes,
               "F8_E4M3", "cutlass_sm121_sfb", kNvfp4Abi, scale_shape),
      required(plan, std::string(root) + ".weight_scale_2", 4, "F32",
               "scalar", kNvfp4Abi, {1}),
      required(plan, std::string(root) + ".input_scale", 4, "F32",
               "scalar", kNvfp4Abi, {1}),
  };
}

}  // namespace

TargetPleLayerBinding bind_target_ple_layer_plan(
    const TargetLayerNativePlan& plan) {
  validate_target_layer_native_plan_binding(plan);
  if (plan.layer != kTargetPleLayer) fail("PLE is not attached to layer index 1");
  TargetPleLayerBinding result{};
  result.rank = plan.rank;
  result.layer = plan.layer;
  result.artifact_key = plan.artifact_key;
  result.descriptor_sha256 = plan.descriptor_sha256;
  result.binding_inventory_sha256 = plan.native_binding_inventory_sha256;
  result.publication_layout_sha256 = plan.slab_publication_layout_sha256;
  result.key_projection = matrix(plan, "key_proj", 13'107'200, 1'638'400,
                                 {10'240, 1'280}, {1'638'400});
  result.value_projection = matrix(plan, "value_proj", 3'276'800, 409'600,
                                   {2'560, 1'280}, {409'600});
  result.norm_key = required(plan, "norm_key.weight", 20'480, "BF16",
                             "checkpoint", "native", {10'240});
  result.norm_query = required(plan, "norm_query.weight", 20'480, "BF16",
                               "checkpoint", "native", {10'240});
  result.norm_conv = required(plan, "norm_conv.weight", 20'480, "BF16",
                              "checkpoint", "native", {10'240});
  result.convolution = required(plan, "conv1d.weight", 81'920, "BF16",
                                "checkpoint", "native", {10'240, 1, 4});
  result.layer_multipliers = required(
      plan, "ple_embedding.layer_multipliers", 24, "I64", "checkpoint",
      "native", {3});
  result.ngram_head_offsets = required(
      plan, "ple_embedding.ngram_heads_offsets", 128, "I64", "checkpoint",
      "native", {16});
  result.ngram_head_vocab_sizes = required(
      plan, "ple_embedding.ngram_heads_vocab_sizes", 128, "I64",
      "checkpoint", "native", {16});
  result.embedding_scale = required(
      plan, "ple_embedding.ngram_embedding.weight_scale", 2, "BF16",
      "checkpoint", "native", {1});
  for (int index = 0; index < kTargetPleEmbeddingShards; ++index) {
    const int global_shard = plan.rank * kTargetPleEmbeddingShards + index;
    result.embedding_shards[index] = required(
        plan, "ple_embedding.ngram_embedding.shard_" +
                  std::to_string(global_shard) + ".weight",
        400'001'920, "F8_E4M3", "checkpoint", "native",
        {2'500'012, 160});
  }
  return result;
}

bool validate_target_ple_layer_plan(
    const TargetLayerNativePlan& plan) noexcept {
  try {
    (void)bind_target_ple_layer_plan(plan);
    return true;
  } catch (...) {
    return false;
  }
}

std::unique_ptr<TargetPleLayerOwnerContract>
TargetPleLayerOwnerContract::create(
    const TargetLayerNativePlan& plan, TargetPleStageSink& telemetry) {
  return std::unique_ptr<TargetPleLayerOwnerContract>(
      new TargetPleLayerOwnerContract(bind_target_ple_layer_plan(plan),
                                      telemetry));
}

TargetPleLayerOwnerContract::TargetPleLayerOwnerContract(
    TargetPleLayerBinding binding, TargetPleStageSink& telemetry)
    : binding_(std::move(binding)), telemetry_(&telemetry),
      authenticated_(true) {}

bool TargetPleLayerOwnerContract::record_stage(
    TargetPleStage stage, TargetPleStageOutcome outcome) noexcept {
  if (!authenticated_ || faulted_ || next_stage_ >= kTargetPleStageOrder.size() ||
      stage != kTargetPleStageOrder[next_stage_]) {
    if (authenticated_ && !faulted_)
      telemetry_->emit({binding_.rank, binding_.layer, stage,
                        TargetPleStageOutcome::kContractError});
    faulted_ = true;
    return false;
  }
  telemetry_->emit({binding_.rank, binding_.layer, stage, outcome});
  ++next_stage_;
  if (outcome != TargetPleStageOutcome::kOk) faulted_ = true;
  return !faulted_;
}

std::unique_ptr<TargetPleRowContextProvider>
TargetPleRowContextProvider::create(const TargetLayerNativePlan& plan,
                                    int max_rows,
                                    TargetPleConvStateLayout conv_state,
                                    TargetPleRowContextOtelSink& telemetry) {
  if (max_rows <= 0) fail("PLE row capacity must be positive");
  if (conv_state.slots <= 0 || conv_state.channels != kTargetPleConvChannels ||
      conv_state.history_tokens != kTargetPleConvHistoryTokens)
    fail("PLE convolution state layout changed");
  return std::unique_ptr<TargetPleRowContextProvider>(
      new TargetPleRowContextProvider(bind_target_ple_layer_plan(plan),
                                      max_rows, conv_state, telemetry));
}

TargetPleRowContextProvider::TargetPleRowContextProvider(
    TargetPleLayerBinding binding, int max_rows,
    TargetPleConvStateLayout conv_state,
    TargetPleRowContextOtelSink& telemetry)
    : binding_(std::move(binding)), max_rows_(max_rows),
      conv_state_(conv_state), authenticated_(true), telemetry_(&telemetry) {}

[[noreturn]] void TargetPleRowContextProvider::fault(
    TargetPleRowContextOperation operation, std::string_view reason) const {
  telemetry_->emit({binding_.rank, binding_.layer, operation,
                    TargetPleStageOutcome::kContractError});
  faulted_ = true;
  throw std::invalid_argument("target PLE row context: " +
                              std::string(reason));
}

const TargetPleRowContextSnapshot& TargetPleRowContextProvider::publish(
    TargetPleRowContextInput input) {
  constexpr auto operation = TargetPleRowContextOperation::kPublish;
  if (!authenticated_ || faulted_)
    fault(operation, "provider is not authenticated");
  if (published_ && input.generation <= snapshot_.generation)
    fault(operation, "generation did not advance");
  if (input.input_ids.empty() ||
      input.input_ids.size() > static_cast<std::size_t>(max_rows_))
    fault(operation, "input_ids row count is outside capacity");
  if (std::any_of(input.input_ids.begin(), input.input_ids.end(),
                  [](std::int32_t token) { return token < 0; }))
    fault(operation, "input_ids contains a negative token");
  if (input.query_start_loc.size() < 2 ||
      input.query_start_loc.front() != 0 ||
      input.query_start_loc.back() !=
          static_cast<std::int32_t>(input.input_ids.size()))
    fault(operation, "query_start_loc does not cover input_ids");
  if (!std::is_sorted(input.query_start_loc.begin(),
                      input.query_start_loc.end()) ||
      std::adjacent_find(input.query_start_loc.begin(),
                         input.query_start_loc.end()) !=
          input.query_start_loc.end())
    fault(operation, "query_start_loc contains an empty or reversed request");

  const std::size_t requests = input.query_start_loc.size() - 1;
  if (input.ngram_context.size() !=
      requests * static_cast<std::size_t>(kTargetPleNgramContextTokens))
    fault(operation, "ngram_context shape changed");
  if (std::any_of(input.ngram_context.begin(), input.ngram_context.end(),
                  [](std::int32_t token) { return token < 0; }))
    fault(operation, "ngram_context contains a negative token");
  if (input.conv_state_indices.size() != requests ||
      input.has_initial_state.size() != requests)
    fault(operation, "convolution request metadata shape changed");
  if (std::any_of(input.has_initial_state.begin(),
                  input.has_initial_state.end(),
                  [](std::uint8_t value) { return value > 1; }))
    fault(operation, "has_initial_state is not canonical");

  std::vector<std::int32_t> sorted_indices = input.conv_state_indices;
  for (const auto index : sorted_indices)
    if (index < 0 || index >= conv_state_.slots)
      fault(operation, "convolution state index is outside capacity");
  std::sort(sorted_indices.begin(), sorted_indices.end());
  if (std::adjacent_find(sorted_indices.begin(), sorted_indices.end()) !=
      sorted_indices.end())
    fault(operation, "convolution state indices alias within a publication");

  TargetPleRowContextSnapshot next{};
  next.generation = input.generation;
  next.rows = static_cast<int>(input.input_ids.size());
  next.requests = static_cast<int>(requests);
  next.input_ids = std::move(input.input_ids);
  next.query_start_loc = std::move(input.query_start_loc);
  next.ngram_context = std::move(input.ngram_context);
  next.conv_state_indices = std::move(input.conv_state_indices);
  next.has_initial_state = std::move(input.has_initial_state);
  snapshot_ = std::move(next);
  published_ = true;
  telemetry_->emit({binding_.rank, binding_.layer, operation,
                    TargetPleStageOutcome::kOk});
  return snapshot_;
}

const TargetPleRowContextSnapshot& TargetPleRowContextProvider::acquire(
    std::uint64_t generation) const {
  constexpr auto operation = TargetPleRowContextOperation::kAcquire;
  if (!authenticated_ || faulted_)
    fault(operation, "provider is not authenticated");
  if (!published_) fault(operation, "no row context has been published");
  if (generation != snapshot_.generation)
    fault(operation, "requested generation is stale or unpublished");
  telemetry_->emit({binding_.rank, binding_.layer, operation,
                    TargetPleStageOutcome::kOk});
  return snapshot_;
}

}  // namespace rocket::qwen38::decode

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

}  // namespace rocket::qwen38::decode

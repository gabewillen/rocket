// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer_native_bindings.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace rocket::qwen38::decode {
namespace {

[[noreturn]] void fail(std::string_view reason) {
  throw std::invalid_argument("target layer native weight binding: " +
                              std::string(reason));
}

const TargetLayerNativeExtent& extent(
    const TargetLayerNativePlan& plan, std::string_view suffix,
    std::uint64_t bytes) {
  const std::string name = "model.language_model.layers." +
                           std::to_string(plan.layer) + "." +
                           std::string(suffix);
  const auto found = std::find_if(
      plan.extents.begin(), plan.extents.end(),
      [&](const auto& item) { return item.name == name; });
  if (found == plan.extents.end() || found->storage != "target_slab" ||
      found->length_bytes != bytes)
    fail("required extent changed: " + name);
  return *found;
}

const TargetLayerNativeExtent& storage_extent(
    const TargetLayerNativePlan& plan, std::string_view suffix,
    std::string_view storage, std::uint64_t bytes) {
  const std::string name = "model.language_model.layers." +
                           std::to_string(plan.layer) + "." +
                           std::string(suffix);
  const auto found = std::find_if(
      plan.extents.begin(), plan.extents.end(),
      [&](const auto& item) { return item.name == name; });
  if (found == plan.extents.end() || found->storage != storage ||
      found->length_bytes != bytes)
    fail("required extent changed: " + name);
  return *found;
}

const TargetLayerNativeExtent& typed_extent(
    const TargetLayerNativePlan& plan, std::string_view suffix,
    std::uint64_t bytes, std::string_view dtype, std::string_view layout,
    std::initializer_list<std::uint64_t> shape,
    std::initializer_list<std::uint64_t> strides) {
  const auto& item = extent(plan, suffix, bytes);
  constexpr std::string_view abi =
      "modelopt_nvfp4_group16_cutlass_sm121_sfb";
  if (item.dtype != dtype || item.layout != layout || item.abi != abi ||
      item.shape != std::vector<std::uint64_t>(shape) ||
      item.strides != std::vector<std::uint64_t>(strides))
    fail("routed source dtype, layout, shape, stride, or ABI changed");
  return item;
}

template <class T>
const T* address(const std::uint8_t* base,
                 const TargetLayerNativeExtent& item) {
  const auto value = reinterpret_cast<std::uintptr_t>(base);
  if (!value || item.offset_bytes >
                    std::numeric_limits<std::uintptr_t>::max() - value)
    fail("device address overflow");
  return reinterpret_cast<const T*>(
      value + static_cast<std::uintptr_t>(item.offset_bytes));
}

}  // namespace

TargetLayerNativeMoeWeights bind_target_layer_native_moe_weights(
    const TargetLayerNativePlan& plan,
    const model::TargetSlabPublication& slab) {
  validate_target_layer_native_plan_binding(plan);
  if ((plan.rank != 0 && plan.rank != 1) || plan.peer_rank != 1 - plan.rank ||
      !slab.device_base || !slab.ready_event || slab.bytes != plan.slab_bytes ||
      slab.device < 0 || slab.rank != plan.rank ||
      slab.artifact_key != model::kTargetSlabArtifactKey ||
      slab.artifact_key != plan.artifact_key ||
      slab.slab_key != plan.slab_key ||
      slab.manifest_sha256 != model::kTargetSlabManifestSha256 ||
      slab.layout_sha256 != plan.slab_publication_layout_sha256 ||
      slab.open_to_publish_ns == 0 ||
      slab.chunks_authenticated != model::kTargetSlabChunks ||
      slab.peak_host_pinned_bytes != model::kTargetSlabPeakPinnedBytes)
    fail("MoE slab publication identity changed");
  const auto target = [&](std::string_view name, std::uint64_t bytes) {
    return extent(plan, name, bytes);
  };
  const auto* base = slab.device_base;
  TargetLayerNativeMoeWeights result{};
  result.router = {
      address<std::uint8_t>(base, target("mlp.gate.weight", 655'360)),
      address<std::uint8_t>(base, target("mlp.gate.weight_scale", 81'920)),
      address<float>(base, target("mlp.gate.weight_scale_2", 4))};
  result.shared = {
      address<__nv_bfloat16>(base,
          target("mlp.shared_expert.gate_proj.weight", 1'638'400)),
      address<__nv_bfloat16>(base,
          target("mlp.shared_expert.up_proj.weight", 1'638'400)),
      address<__nv_bfloat16>(base,
          target("mlp.shared_expert.down_proj.weight", 1'638'400)),
      address<__nv_bfloat16>(base,
          target("mlp.shared_expert_gate.weight", 5'120))};
  const int first = plan.rank * moe::kTargetMoeLocalExperts;
  for (int local = 0; local < moe::kTargetMoeLocalExperts; ++local) {
    const auto root = "mlp.experts." + std::to_string(first + local) + ".";
    const auto packed = [&](std::string_view projection) {
      return address<std::uint8_t>(base, typed_extent(
          plan, root + std::string(projection) + ".weight", 819'200,
          "U8", "checkpoint",
          projection == "down_proj"
              ? std::initializer_list<std::uint64_t>{2'560, 320}
              : std::initializer_list<std::uint64_t>{640, 1'280},
          projection == "down_proj"
              ? std::initializer_list<std::uint64_t>{320, 1}
              : std::initializer_list<std::uint64_t>{1'280, 1}));
    };
    const auto scale = [&](std::string_view projection) {
      return address<std::uint8_t>(base, typed_extent(
          plan, root + std::string(projection) + ".weight_scale", 102'400,
          "F8_E4M3", "cutlass_sm121_sfb", {102'400}, {1}));
    };
    const auto scalar = [&](std::string_view projection,
                            std::string_view leaf) {
      return address<float>(base, typed_extent(
          plan, root + std::string(projection) + "." + std::string(leaf),
          4, "F32", "checkpoint", {}, {}));
    };
    result.routed_source[local] = {
        packed("up_proj"), scale("up_proj"),
        scalar("up_proj", "input_scale"),
        scalar("up_proj", "weight_scale_2"),
        packed("gate_proj"), scale("gate_proj"),
        scalar("gate_proj", "input_scale"),
        scalar("gate_proj", "weight_scale_2"),
        packed("down_proj"), scale("down_proj"),
        scalar("down_proj", "input_scale"),
        scalar("down_proj", "weight_scale_2")};
  }
  result.routed_identity = moe::TargetMoeCompactRuntimeIdentity{
      plan.rank, plan.descriptor_sha256,
      plan.native_binding_inventory_sha256,
      plan.slab_publication_layout_sha256,
      "modelopt_nvfp4_group16_cutlass_sm121_sfb",
      std::string(moe::kTargetMoeDeviceStageAbi),
      "route_position_iota10_unique_positive_remote_zero_v1"};
  return result;
}

TargetQsaLayerNativeWeights bind_target_qsa_layer_native_weights(
    const TargetLayerNativePlan& plan,
    const model::TargetSlabPublication& slab,
    const attention::QsaSidecarPublication& sidecar,
    const attention::Layer3RopeIdentity& rope_identity,
    const attention::Layer3RopeView& rope) {
  validate_target_layer_native_plan_binding(plan);
  const auto expected_sidecar =
      attention::target_qsa_sidecar_identity(plan.rank, plan.layer);
  const auto expected_rope =
      attention::target_qsa_rope_identity(plan.rank, plan.layer);
  if (plan.attention_kind != TargetK0AttentionKind::kQsa ||
      !slab.device_base || !slab.ready_event || slab.rank != plan.rank ||
      slab.bytes != plan.slab_bytes || slab.artifact_key != plan.artifact_key ||
      slab.slab_key != plan.slab_key ||
      slab.layout_sha256 != plan.slab_publication_layout_sha256 ||
      !sidecar.device_base || sidecar.device != slab.device ||
      sidecar.bytes != attention::kQsaSidecarBytes ||
      sidecar.identity.artifact_key != expected_sidecar.artifact_key ||
      sidecar.identity.payload_sha256 != expected_sidecar.payload_sha256 ||
      sidecar.identity.layer3_sha256 != expected_sidecar.layer3_sha256 ||
      sidecar.identity.rank != plan.rank || sidecar.identity.layer != plan.layer ||
      rope_identity.checkpoint_revision != expected_rope.checkpoint_revision ||
      rope_identity.config_sha256 != expected_rope.config_sha256 ||
      rope_identity.vllm_revision != expected_rope.vllm_revision ||
      rope_identity.rank != plan.rank || rope_identity.layer != plan.layer ||
      !rope.cos_sin || !rope.ready ||
      rope.payload_sha256 != attention::kLayer3RopePayloadSha256 ||
      rope.rows != attention::kLayer3RopeRows ||
      rope.columns != attention::kLayer3RopeColumns ||
      rope.row_stride != attention::kLayer3RopeColumns)
    fail("QSA publication identity changed");
  const auto target = [&](std::string_view suffix, std::uint64_t bytes) ->
      const TargetLayerNativeExtent& {
    return storage_extent(plan, suffix, "target_slab", bytes);
  };
  const auto side = [&](std::string_view suffix, std::uint64_t bytes) ->
      const TargetLayerNativeExtent& {
    return storage_extent(plan, suffix, "indexer_sidecar", bytes);
  };
  const auto* base = slab.device_base;
  TargetQsaLayerNativeWeights result{};
  result.projection = {
      address<std::uint8_t>(base, target("self_attn.q_proj.weight", 7'864'320)),
      address<std::uint8_t>(base, target("self_attn.q_proj.weight_scale", 983'040)),
      plan.attention_projection_globals[0],
      address<std::uint8_t>(base, target("self_attn.k_proj.weight", 327'680)),
      address<std::uint8_t>(base, target("self_attn.k_proj.weight_scale", 40'960)),
      plan.attention_projection_globals[1],
      address<std::uint8_t>(base, target("self_attn.v_proj.weight", 327'680)),
      address<std::uint8_t>(base, target("self_attn.v_proj.weight_scale", 40'960)),
      plan.attention_projection_globals[2],
      address<std::uint8_t>(base, target("self_attn.o_proj.weight", 3'932'160)),
      address<std::uint8_t>(base, target("self_attn.o_proj.weight_scale", 491'520)),
      plan.attention_projection_globals[3]};
  result.preprocess = {
      address<__nv_bfloat16>(base, target("self_attn.q_norm.weight", 512)),
      address<__nv_bfloat16>(base, target("self_attn.k_norm.weight", 512)),
      address<__nv_bfloat16>(sidecar.device_base,
          side("self_attn.indexer.index_qk_proj.weight", 3'276'800)),
      address<__nv_bfloat16>(base,
          target("self_attn.indexer.q_layernorm.weight", 256)),
      address<__nv_bfloat16>(base,
          target("self_attn.indexer.k_layernorm.weight", 256)),
      rope.cos_sin};
  const auto hyper = [&](std::string_view family) {
    const std::string prefix = std::string(family) + "_hyper_connection.";
    return hyperconnection::Weights{
        address<__nv_bfloat16>(base, target(prefix + "hc_norm.weight", 20'480)),
        address<__nv_bfloat16>(base, target(prefix + "input_mix_weight_down.weight", 6'553'600)),
        address<__nv_bfloat16>(base, target(prefix + "block_inject_weight.weight", 81'920)),
        address<__nv_bfloat16>(base, target(prefix + "input_mix_weight_up.weight", 6'553'600))};
  };
  result.rope_ready_event = rope.ready;
  result.attention_hyperconnection = hyper("attn");
  result.mlp_hyperconnection = hyper("mlp");
  return result;
}

}  // namespace rocket::qwen38::decode

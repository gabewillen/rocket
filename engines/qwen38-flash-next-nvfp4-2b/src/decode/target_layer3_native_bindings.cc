// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_native_bindings.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocket::qwen38::decode {
namespace {

constexpr std::string_view kPrefix = "model.language_model.layers.3.";

[[noreturn]] void fail(std::string_view reason) {
  throw std::invalid_argument("layer-3 native weight binding: " +
                              std::string(reason));
}

void validate_plan_storage(const TargetLayer3NativePlan& plan) {
  if (plan.extents.size() != 3'108) fail("extent inventory changed");
  std::set<std::string_view> names;
  std::array<std::vector<std::pair<std::uint64_t, std::uint64_t>>, 2> ranges;
  for (const auto& item : plan.extents) {
    const int storage = item.storage == "target_slab" ? 0 :
                        item.storage == "indexer-sidecar" ? 1 : -1;
    const std::uint64_t bound = storage == 0
                                    ? plan.slab_bytes
                                    : attention::kQsaSidecarBytes;
    if (storage < 0 || !names.insert(item.name).second ||
        item.length_bytes == 0 || item.offset_bytes % 256 != 0 ||
        item.offset_bytes > bound ||
        item.length_bytes > bound - item.offset_bytes)
      fail("extent bounds, alignment, or identity changed");
    ranges[storage].push_back(
        {item.offset_bytes, item.offset_bytes + item.length_bytes});
  }
  for (auto& storage : ranges) {
    std::sort(storage.begin(), storage.end());
    for (std::size_t index = 1; index < storage.size(); ++index)
      if (storage[index].first < storage[index - 1].second)
        fail("extent overlap changed");
  }
}

const TargetLayer3NativeExtent& extent(
    const TargetLayer3NativePlan& plan, std::string_view suffix,
    std::string_view storage, std::uint64_t bytes) {
  const std::string name = std::string(kPrefix) + std::string(suffix);
  const auto found = std::find_if(
      plan.extents.begin(), plan.extents.end(),
      [&](const auto& item) { return item.name == name; });
  if (found == plan.extents.end() || found->storage != storage ||
      found->length_bytes != bytes)
    fail("required extent changed: " + name);
  return *found;
}

template <class T>
const T* address(const std::uint8_t* base,
                 const TargetLayer3NativeExtent& item) {
  const auto value = reinterpret_cast<std::uintptr_t>(base);
  if (!value || item.offset_bytes >
                    std::numeric_limits<std::uintptr_t>::max() - value)
    fail("device address overflow");
  return reinterpret_cast<const T*>(
      value + static_cast<std::uintptr_t>(item.offset_bytes));
}

void emit(pair_reduce::OtelStageSink& telemetry, int rank,
          pair_reduce::Outcome outcome) noexcept {
  telemetry.emit_span_and_log({
      "rocket.qwen38.layer3.native_weight_binding", "layer3-init",
      "oracle-05ea3af", rank, 1, pair_reduce::kDtype, outcome, 0, 0});
  telemetry.record_duration({rank, 1, pair_reduce::kDtype, outcome, 0});
}

}  // namespace

bool CudaTargetLayer3ReadyEventProbe::complete(cudaEvent_t event) noexcept {
  return event && cudaEventQuery(event) == cudaSuccess;
}

TargetLayer3NativeWeightBindings bind_target_layer3_native_weights(
    const TargetLayer3NativePlan& plan,
    const model::TargetSlabPublication& slab,
    const attention::QsaSidecarPublication& sidecar,
    const __nv_bfloat16* rope_cos_sin,
    TargetLayer3ReadyEventProbe& ready_event_probe,
    pair_reduce::OtelStageSink& telemetry) {
  try {
    validate_target_layer3_native_plan_binding(plan);
    validate_plan_storage(plan);
    const auto expected_sidecar =
        attention::layer3_qsa_sidecar_identity(plan.rank);
    if ((plan.rank != 0 && plan.rank != 1) || plan.layer != 3 ||
        plan.peer_rank != 1 - plan.rank ||
        !slab.device_base || !slab.ready_event ||
        !ready_event_probe.complete(slab.ready_event) ||
        slab.bytes != plan.slab_bytes ||
        slab.device < 0 || slab.rank != plan.rank ||
        slab.artifact_key != model::kTargetSlabArtifactKey ||
        slab.artifact_key != plan.artifact_key ||
        slab.slab_key != plan.slab_key ||
        slab.manifest_sha256 != model::kTargetSlabManifestSha256 ||
        slab.layout_sha256 != plan.slab_publication_layout_sha256 ||
        slab.open_to_publish_ns == 0 ||
        slab.chunks_authenticated != model::kTargetSlabChunks ||
        slab.peak_host_pinned_bytes !=
            model::kTargetSlabPeakPinnedBytes ||
        !sidecar.device_base || sidecar.bytes != attention::kQsaSidecarBytes ||
        sidecar.device != slab.device ||
        sidecar.identity.artifact_key != expected_sidecar.artifact_key ||
        sidecar.identity.payload_sha256 != expected_sidecar.payload_sha256 ||
        sidecar.identity.layer3_sha256 != expected_sidecar.layer3_sha256 ||
        sidecar.identity.rank != plan.rank || sidecar.identity.layer != 3 ||
        !rope_cos_sin)
      fail("publication identity changed");

    const auto target = [&](std::string_view name, std::uint64_t bytes) {
      return extent(plan, name, "target_slab", bytes);
    };
    const auto side = [&](std::string_view name, std::uint64_t bytes) {
      return extent(plan, name, "indexer-sidecar", bytes);
    };
    const auto* target_base = slab.device_base;
    const auto* sidecar_base = sidecar.device_base;
    const auto qsa_projection = attention::TargetQsaProjectionWeights{
        address<std::uint8_t>(target_base, target("self_attn.q_proj.weight", 7'864'320)),
        address<std::uint8_t>(target_base, target("self_attn.q_proj.weight_scale", 983'040)),
        plan.qsa_projection_globals[0],
        address<std::uint8_t>(target_base, target("self_attn.k_proj.weight", 327'680)),
        address<std::uint8_t>(target_base, target("self_attn.k_proj.weight_scale", 40'960)),
        plan.qsa_projection_globals[1],
        address<std::uint8_t>(target_base, target("self_attn.v_proj.weight", 327'680)),
        address<std::uint8_t>(target_base, target("self_attn.v_proj.weight_scale", 40'960)),
        plan.qsa_projection_globals[2],
        address<std::uint8_t>(target_base, target("self_attn.o_proj.weight", 3'932'160)),
        address<std::uint8_t>(target_base, target("self_attn.o_proj.weight_scale", 491'520)),
        plan.qsa_projection_globals[3]};
    const auto qsa_preprocess = attention::TargetQsaPreprocessWeights{
        address<__nv_bfloat16>(target_base, target("self_attn.q_norm.weight", 512)),
        address<__nv_bfloat16>(target_base, target("self_attn.k_norm.weight", 512)),
        address<__nv_bfloat16>(sidecar_base,
            side("self_attn.indexer.index_qk_proj.weight", 3'276'800)),
        address<__nv_bfloat16>(target_base,
            target("self_attn.indexer.q_layernorm.weight", 256)),
        address<__nv_bfloat16>(target_base,
            target("self_attn.indexer.k_layernorm.weight", 256)),
        rope_cos_sin};
    const auto hyper = [&](std::string_view family) {
      const std::string prefix = std::string(family) + "_hyper_connection.";
      return hyperconnection::Weights{
          address<__nv_bfloat16>(target_base,
              target(prefix + "hc_norm.weight", 20'480)),
          address<__nv_bfloat16>(target_base,
              target(prefix + "input_mix_weight_down.weight", 6'553'600)),
          address<__nv_bfloat16>(target_base,
              target(prefix + "block_inject_weight.weight", 81'920)),
          address<__nv_bfloat16>(target_base,
              target(prefix + "input_mix_weight_up.weight", 6'553'600))};
    };
    const auto router = moe::TargetRouterNvfp4Weights{
        address<std::uint8_t>(target_base, target("mlp.gate.weight", 655'360)),
        address<std::uint8_t>(target_base,
            target("mlp.gate.weight_scale", 81'920)),
        address<float>(target_base, target("mlp.gate.weight_scale_2", 4))};
    const auto shared = moe::TargetSharedBf16Weights{
        address<__nv_bfloat16>(target_base,
            target("mlp.shared_expert.gate_proj.weight", 1'638'400)),
        address<__nv_bfloat16>(target_base,
            target("mlp.shared_expert.up_proj.weight", 1'638'400)),
        address<__nv_bfloat16>(target_base,
            target("mlp.shared_expert.down_proj.weight", 1'638'400)),
        address<__nv_bfloat16>(target_base,
            target("mlp.shared_expert_gate.weight", 5'120))};
    TargetLayer3NativeWeightBindings result{
        qsa_projection, qsa_preprocess, hyper("attn"), hyper("mlp"),
        router, shared};
    emit(telemetry, plan.rank, pair_reduce::Outcome::kOk);
    return result;
  } catch (...) {
    emit(telemetry, (plan.rank == 0 || plan.rank == 1) ? plan.rank : -1,
         pair_reduce::Outcome::kContractError);
    throw;
  }
}

}  // namespace rocket::qwen38::decode

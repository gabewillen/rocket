// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_native_bindings.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace decode = rocket::qwen38::decode;
namespace attention = rocket::qwen38::attention;
namespace model = rocket::qwen38::model;
namespace pair_reduce = rocket::qwen38::pair_reduce;

struct Sink final : pair_reduce::OtelStageSink {
  void emit_span_and_log(const pair_reduce::SpanRecord& value) noexcept override {
    ++spans;
    outcome = value.outcome;
  }
  void record_duration(const pair_reduce::MetricPoint& value) noexcept override {
    ++metrics;
    outcome = value.outcome;
  }
  int spans = 0;
  int metrics = 0;
  pair_reduce::Outcome outcome = pair_reduce::Outcome::kCudaError;
};

struct ReadyProbe final : decode::TargetLayer3ReadyEventProbe {
  bool complete(cudaEvent_t event) noexcept override {
    return accept && event != nullptr;
  }
  bool accept = true;
};

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    const auto plan = decode::load_target_layer3_native_plan(argv[1]);
    const int rank = argv[2][0] == '0' && argv[2][1] == '\0' ? 0 :
                     argv[2][0] == '1' && argv[2][1] == '\0' ? 1 : -1;
    if (rank != plan.rank) throw std::invalid_argument("rank changed");
    const auto sidecar_identity = attention::layer3_qsa_sidecar_identity(rank);
    model::TargetSlabPublication slab{
        reinterpret_cast<const std::uint8_t*>(0x100000000000ULL),
        reinterpret_cast<cudaEvent_t>(0x1000ULL), plan.slab_bytes, rank, rank,
        plan.artifact_key, plan.slab_key,
        model::kTargetSlabManifestSha256,
        plan.slab_publication_layout_sha256, 1, model::kTargetSlabChunks,
        model::kTargetSlabPeakPinnedBytes};
    attention::QsaSidecarPublication sidecar{
        reinterpret_cast<const std::uint8_t*>(0x200000000000ULL),
        attention::kQsaSidecarBytes, rank, sidecar_identity};
    auto rope_identity = attention::layer3_rope_identity(rank);
    attention::Layer3RopeView rope{
        reinterpret_cast<const __nv_bfloat16*>(0x300000000000ULL),
        reinterpret_cast<cudaEvent_t>(0x2000ULL),
        attention::kLayer3RopePayloadSha256,
        attention::kLayer3RopeRows, attention::kLayer3RopeColumns,
        attention::kLayer3RopeColumns};
    Sink sink;
    ReadyProbe probe;
    const auto binding = decode::bind_target_layer3_native_weights(
        plan, slab, sidecar, rope_identity, rope, probe, sink);
    if (!binding.qsa_projection.q_weight ||
        !binding.qsa_preprocess.index_qk ||
        binding.rope_ready_event != rope.ready ||
        !binding.attention_hyperconnection.norm ||
        !binding.mlp_hyperconnection.norm || !binding.router.packed_e2m1 ||
        !binding.shared.gate || sink.spans != 1 || sink.metrics != 1 ||
        sink.outcome != pair_reduce::Outcome::kOk)
      throw std::logic_error("native binding publication changed");

    const auto reject = [&](const model::TargetSlabPublication& slab_value,
                            const attention::QsaSidecarPublication& sidecar_value,
                            const char* label) {
      try {
        (void)decode::bind_target_layer3_native_weights(
            plan, slab_value, sidecar_value, rope_identity, rope, probe, sink);
        throw std::logic_error(std::string(label) + " unexpectedly bound");
      } catch (const std::invalid_argument&) {
      }
    };
    auto bad_slab = slab;
    bad_slab.layout_sha256 = "wrong";
    reject(bad_slab, sidecar, "layout mutation");
    bad_slab = slab;
    bad_slab.ready_event = nullptr;
    reject(bad_slab, sidecar, "ready-event mutation");
    bad_slab = slab;
    bad_slab.chunks_authenticated -= 1;
    reject(bad_slab, sidecar, "chunk-count mutation");
    auto bad_sidecar = sidecar;
    bad_sidecar.identity.rank = 1 - rank;
    reject(slab, bad_sidecar, "sidecar-rank mutation");
    probe.accept = false;
    reject(slab, sidecar, "incomplete-event mutation");
    probe.accept = true;
    auto bad_plan = plan;
    for (auto& item : bad_plan.extents) {
      if (item.name ==
          "model.language_model.layers.3.self_attn.q_proj.weight") {
        item.offset_bytes = bad_plan.slab_bytes + 256;
        break;
      }
    }
    try {
      (void)decode::bind_target_layer3_native_weights(
          bad_plan, slab, sidecar, rope_identity, rope, probe, sink);
      throw std::logic_error("in-memory offset mutation unexpectedly bound");
    } catch (const std::invalid_argument&) {
    }
    bad_plan = plan;
    auto* k_weight = static_cast<decode::TargetLayer3NativeExtent*>(nullptr);
    auto* v_weight = static_cast<decode::TargetLayer3NativeExtent*>(nullptr);
    for (auto& item : bad_plan.extents) {
      if (item.name == "model.language_model.layers.3.self_attn.k_proj.weight")
        k_weight = &item;
      if (item.name == "model.language_model.layers.3.self_attn.v_proj.weight")
        v_weight = &item;
    }
    if (!k_weight || !v_weight) throw std::logic_error("test extents changed");
    std::swap(k_weight->offset_bytes, v_weight->offset_bytes);
    try {
      (void)decode::bind_target_layer3_native_weights(
          bad_plan, slab, sidecar, rope_identity, rope, probe, sink);
      throw std::logic_error("equal-size offset swap unexpectedly bound");
    } catch (const std::invalid_argument&) {
    }
    bad_plan = plan;
    std::swap(bad_plan.qsa_projection_globals[0],
              bad_plan.qsa_projection_globals[1]);
    try {
      (void)decode::bind_target_layer3_native_weights(
          bad_plan, slab, sidecar, rope_identity, rope, probe, sink);
      throw std::logic_error("projection scalar swap unexpectedly bound");
    } catch (const std::invalid_argument&) {
    }
    bad_plan = plan;
    bad_plan.artifact_key = "evil";
    bad_slab = slab;
    bad_slab.artifact_key = "evil";
    try {
      (void)decode::bind_target_layer3_native_weights(
          bad_plan, bad_slab, sidecar, rope_identity, rope, probe, sink);
      throw std::logic_error("artifact co-mutation unexpectedly bound");
    } catch (const std::invalid_argument&) {
    }
    bad_plan = plan;
    bad_plan.slab_publication_layout_sha256 = "evil";
    bad_slab = slab;
    bad_slab.layout_sha256 = "evil";
    try {
      (void)decode::bind_target_layer3_native_weights(
          bad_plan, bad_slab, sidecar, rope_identity, rope, probe, sink);
      throw std::logic_error("layout co-mutation unexpectedly bound");
    } catch (const std::invalid_argument&) {
    }
    auto bad_rope_identity = rope_identity;
    bad_rope_identity.first_position = 35;
    try {
      (void)decode::bind_target_layer3_native_weights(
          plan, slab, sidecar, bad_rope_identity, rope, probe, sink);
      throw std::logic_error("decode RoPE identity unexpectedly bound");
    } catch (const std::invalid_argument&) {
    }
    bad_rope_identity = rope_identity;
    bad_rope_identity.uses_mrope = true;
    try {
      (void)decode::bind_target_layer3_native_weights(
          plan, slab, sidecar, bad_rope_identity, rope, probe, sink);
      throw std::logic_error("mRoPE identity unexpectedly bound");
    } catch (const std::invalid_argument&) {
    }
    auto bad_rope = rope;
    bad_rope.rows = 34;
    try {
      (void)decode::bind_target_layer3_native_weights(
          plan, slab, sidecar, rope_identity, bad_rope, probe, sink);
      throw std::logic_error("short RoPE view unexpectedly bound");
    } catch (const std::invalid_argument&) {
    }
    bad_rope = rope;
    bad_rope.payload_sha256 = "wrong";
    try {
      (void)decode::bind_target_layer3_native_weights(
          plan, slab, sidecar, rope_identity, bad_rope, probe, sink);
      throw std::logic_error("wrong RoPE payload unexpectedly bound");
    } catch (const std::invalid_argument&) {
    }
    bad_rope = rope;
    bad_rope.ready = nullptr;
    try {
      (void)decode::bind_target_layer3_native_weights(
          plan, slab, sidecar, rope_identity, bad_rope, probe, sink);
      throw std::logic_error("unpublished RoPE view unexpectedly bound");
    } catch (const std::invalid_argument&) {
    }
    if (sink.spans != 16 || sink.metrics != 16 ||
        sink.outcome != pair_reduce::Outcome::kContractError)
      throw std::logic_error("binding failure telemetry changed");
    std::printf("rank=%d extents=%zu native_weight_binding=1\n", rank,
                plan.extents.size());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

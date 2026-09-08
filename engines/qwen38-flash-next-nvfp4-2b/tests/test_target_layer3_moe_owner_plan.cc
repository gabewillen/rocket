// SPDX-License-Identifier: Apache-2.0
#include "moe/target_layer3_moe_owner.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace decode = rocket::qwen38::decode;
namespace model = rocket::qwen38::model;
namespace moe = rocket::qwen38::moe;

int main(int argc, char** argv) {
  static_assert(!std::is_constructible_v<
      moe::TargetLayer3MoeDeviceOwner, int,
      const decode::TargetLayer3NativePlan&,
      std::shared_ptr<const model::TargetSlabLease>,
      std::shared_ptr<moe::TargetFullMoeOtelSink>,
      std::shared_ptr<moe::TargetMoeStageOtelSink>>);
  static_assert(!std::is_constructible_v<
      model::ProcessLifetimeTargetSlabLease,
      model::TargetSlabPublication>);
  if (argc != 1 && argc != 3) return 2;
  try {
    decode::TargetLayer3NativePlan plan{};
    int rank = 0;
    if (argc == 3) {
      plan = decode::load_target_layer3_native_plan(argv[1]);
      rank = argv[2][0] == '0' && argv[2][1] == '\0' ? 0 :
             argv[2][0] == '1' && argv[2][1] == '\0' ? 1 : -1;
    } else {
      plan.rank = 0;
      plan.peer_rank = 1;
      plan.layer = 3;
      plan.slab_bytes = model::kTargetSlabBytes;
      plan.descriptor_sha256 =
          "4b603bda78cb22f5a6d59b4d97f9b07a0e5ec3a824e5e08f563a715dbd2a1e46";
      plan.native_binding_inventory_sha256 =
          "0f88201c81e6ace3991969dc397fca345773b7a28f317e08a6727c7903ee30c5";
      plan.artifact_key = model::kTargetSlabArtifactKey;
      plan.slab_key = "rank0-target";
      plan.slab_publication_layout_sha256 =
          "4f03ccc90c9020ff2e87f044867f2ac9896ac20c0d97c85055beef0b125ce6d6";
    }
    model::TargetSlabPublication slab{
        reinterpret_cast<const std::uint8_t*>(0x100000000000ULL),
        reinterpret_cast<cudaEvent_t>(0x1000ULL), plan.slab_bytes, rank, rank,
        plan.artifact_key, plan.slab_key, model::kTargetSlabManifestSha256,
        plan.slab_publication_layout_sha256, 1, model::kTargetSlabChunks,
        model::kTargetSlabPeakPinnedBytes};
    if (argc == 3) {
      if (rank != plan.rank ||
          !moe::validate_target_layer3_moe_owner_plan(plan, slab))
        throw std::runtime_error("authentic owner plan rejected");
      const auto weights =
          decode::bind_target_layer3_native_moe_weights(plan, slab);
      const auto begin = reinterpret_cast<std::uintptr_t>(slab.device_base);
      const auto end = begin + slab.bytes;
      const auto inside_slab = [&](const void* pointer) {
        const auto value = reinterpret_cast<std::uintptr_t>(pointer);
        return value >= begin && value < end;
      };
      if (!inside_slab(weights.router.packed_e2m1) ||
          !inside_slab(weights.shared.down) ||
          !inside_slab(weights.routed_source.front().up_packed) ||
          !inside_slab(weights.routed_source.back().down_alpha))
        throw std::runtime_error("owner accepted null or cross-wired weights");
      const auto owned_identity = weights.routed_identity;
      const auto descriptor = plan.descriptor_sha256;
      const auto inventory = plan.native_binding_inventory_sha256;
      const auto publication = plan.slab_publication_layout_sha256;
      plan.descriptor_sha256.assign(64, '0');
      plan.native_binding_inventory_sha256.assign(64, '0');
      plan.slab_publication_layout_sha256.assign(64, '0');
      if (!moe::authenticate_target_moe_compact_runtime_identity(
              owned_identity) ||
          owned_identity.descriptor_sha256 != descriptor)
        throw std::runtime_error("compact runtime identity borrowed plan storage");
      plan.descriptor_sha256 = descriptor;
      plan.native_binding_inventory_sha256 = inventory;
      plan.slab_publication_layout_sha256 = publication;
      if (moe::validate_target_layer3_moe_owner_handoff(
              plan, reinterpret_cast<void*>(0x1234)))
        throw std::runtime_error("accepted-loader owner handoff changed");
      auto bad_slab = slab;
      bad_slab.layout_sha256 = "0";
      if (moe::validate_target_layer3_moe_owner_plan(plan, bad_slab))
        throw std::runtime_error("publication layout mutation accepted");
      bad_slab = slab;
      bad_slab.device_base = nullptr;
      if (moe::validate_target_layer3_moe_owner_plan(plan, bad_slab))
        throw std::runtime_error("null publication accepted");
      bad_slab = slab;
      bad_slab.rank = 1 - rank;
      if (moe::validate_target_layer3_moe_owner_plan(plan, bad_slab))
        throw std::runtime_error("cross-rank publication accepted");
    }
    static_assert(moe::kTargetMoeWeightExperts == 10);
    static_assert(moe::kTargetMoeStateExperts == 11);
    static_assert(moe::kTargetMoeE11WorkspaceBytes == 522'496);
    static_assert(moe::kTargetLayer3MoeRuntimeBytes == 532'992);
    static_assert(moe::kTargetMoeStageScratchBytes == 33'177'896);
    static_assert(moe::kTargetMoeStageDeviceBytes == 33'177'880);
    std::vector<std::uint8_t> stage(
        moe::kTargetMoeStageDeviceBytes + 512, 0xA5);
    std::vector<std::uint8_t> runtime(
        moe::kTargetLayer3MoeRuntimeBytes + 512, 0x5A);
    const auto aligned = [](std::vector<std::uint8_t>& value) {
      const auto address = reinterpret_cast<std::uintptr_t>(value.data() + 1);
      return reinterpret_cast<std::uint8_t*>((address + 255) & ~255ULL);
    };
    auto* stage_base = aligned(stage);
    auto* runtime_base = aligned(runtime);
    moe::TargetMoeN640StageEvidence evidence{};
    std::uint64_t requested = 1;
    const auto storage = moe::bind_target_layer3_moe_storage(
        stage_base, moe::kTargetMoeStageDeviceBytes, &evidence,
        &evidence, runtime_base, moe::kTargetLayer3MoeRuntimeBytes,
        &requested);
    const auto inside = [](const void* pointer, const std::uint8_t* begin,
                           std::size_t bytes) {
      const auto value = reinterpret_cast<std::uintptr_t>(pointer);
      const auto first = reinterpret_cast<std::uintptr_t>(begin);
      return value >= first && value < first + bytes;
    };
    if (!inside(storage.stage.w13_packed, stage_base,
                moe::kTargetMoeStageDeviceBytes) ||
        !inside(storage.stage.compact_routing_weights, stage_base,
                moe::kTargetMoeStageDeviceBytes) ||
        !inside(storage.runtime.router_logits_f32, runtime_base,
                moe::kTargetLayer3MoeRuntimeBytes) ||
        !inside(storage.runtime.routed.token_weights, runtime_base,
                moe::kTargetLayer3MoeRuntimeBytes) ||
        !inside(storage.rank_local_output, runtime_base,
                moe::kTargetLayer3MoeRuntimeBytes) ||
        stage_base[-1] != 0xA5 ||
        stage_base[moe::kTargetMoeStageDeviceBytes] != 0xA5 ||
        runtime_base[-1] != 0x5A ||
        runtime_base[moe::kTargetLayer3MoeRuntimeBytes] != 0x5A)
      throw std::runtime_error("owner storage cursor or canary changed");
    try {
      (void)moe::bind_target_layer3_moe_storage(
          stage_base, moe::kTargetMoeStageDeviceBytes - 1, &evidence,
          &evidence, runtime_base, moe::kTargetLayer3MoeRuntimeBytes,
          &requested);
      throw std::runtime_error("short stage storage accepted");
    } catch (const std::invalid_argument&) {
    }
    std::printf("rank=%d stage_bytes=%zu e11_workspace_bytes=%zu owner_plan=1\n",
                rank, moe::kTargetMoeStageScratchBytes,
                moe::kTargetMoeE11WorkspaceBytes);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

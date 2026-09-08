// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "attention/target_k0_qsa_state_owner.h"
#include "decode/target_gdn_layer_owner.h"
#include "decode/target_k0_layer_owner_inventory.h"
#include "decode/target_k0_native_plan_inventory.h"
#include "decode/target_k0_physical_layers.h"
#include "decode/target_qsa_k0_layer_owner.h"

#include <filesystem>
#include <memory>

namespace rocket::qwen38::decode {

enum class TargetK0PhysicalLayerConstructionStage : std::int32_t {
  kUnknown = 0,
  kPlanInventory = 1,
  kQsaArena = 2,
  kSidecar = 3,
  kRope = 4,
  kGdnOwner = 5,
  kQsaOwner = 6,
  kInventoryAssembly = 7,
};

struct TargetK0PhysicalLayerConstructionProgress {
  TargetK0PhysicalLayerConstructionStage stage =
      TargetK0PhysicalLayerConstructionStage::kUnknown;
  int layer = -1;
};

bool validate_target_k0_physical_layer_plans(
    const TargetK0NativePlanInventory& plans, int rank) noexcept;

// Oracle35-only owner of the complete rank-local layer graph. Shared QSA
// sidecar, RoPE, and state owners are retained above all 48 layer owners, so
// twelve QSA layers alias one authenticated sidecar allocation and one RoPE
// publication. The accepted target slab remains owned by its process-lifetime
// loader capability.
class TargetK0PhysicalLayerOwners final : public TargetK0PhysicalLayers {
 public:
  static std::unique_ptr<TargetK0PhysicalLayerOwners> create(
      int device, int rank, void* accepted_loader_lease_handle,
      std::unique_ptr<TargetK0NativePlanInventory> plans,
      const std::filesystem::path& sidecar_payload,
      TargetK0PairReduceSchedule& reductions,
      std::shared_ptr<pair_reduce::OtelStageSink> layer_telemetry,
      std::shared_ptr<moe::TargetFullMoeOtelSink> moe_telemetry,
      std::shared_ptr<moe::TargetMoeStageOtelSink> stage_telemetry,
      std::shared_ptr<attention::TargetK0OracleQsaStateOtelSink>
          state_telemetry,
      TargetK0PhysicalLayerConstructionProgress* progress = nullptr);

  TargetK0LayerOwnerInventory& inventory() noexcept override {
    return *inventory_;
  }
  const TargetK0NativePlanInventory& plans() const noexcept { return *plans_; }
  int rank() const noexcept override { return rank_; }
  bool authenticated() const noexcept override { return authenticated_; }

 private:
  TargetK0PhysicalLayerOwners() = default;
  int rank_ = -1;
  bool authenticated_ = false;
  std::unique_ptr<TargetK0NativePlanInventory> plans_;
  std::unique_ptr<attention::TargetK0OracleQsaStateOwner> qsa_state_;
  std::unique_ptr<attention::QsaSidecarDeviceOwner> qsa_sidecar_;
  std::unique_ptr<attention::Layer3RopeDeviceOwner> qsa_rope_;
  std::unique_ptr<TargetK0LayerOwnerInventory> inventory_;
};

}  // namespace rocket::qwen38::decode

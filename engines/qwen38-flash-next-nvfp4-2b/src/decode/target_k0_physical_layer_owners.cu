// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_physical_layer_owners.h"

#include <array>
#include <stdexcept>
#include <utility>

namespace rocket::qwen38::decode {

bool validate_target_k0_physical_layer_plans(
    const TargetK0NativePlanInventory& plans, int rank) noexcept {
  try {
    if (rank != 0 && rank != 1) return false;
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      const auto& plan = plans.at(rank, layer);
      validate_target_layer_native_plan_binding(plan);
      if (plan.rank != rank || plan.layer != layer ||
          plan.attention_kind != target_k0_attention_kind(layer))
        return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

std::unique_ptr<TargetK0PhysicalLayerOwners>
TargetK0PhysicalLayerOwners::create(
    int device, int rank, void* accepted_loader_lease_handle,
    std::unique_ptr<TargetK0NativePlanInventory> plans,
    const std::filesystem::path& sidecar_payload,
    TargetK0PairReduceSchedule& reductions,
    std::shared_ptr<pair_reduce::OtelStageSink> layer_telemetry,
    std::shared_ptr<moe::TargetFullMoeOtelSink> moe_telemetry,
    std::shared_ptr<moe::TargetMoeStageOtelSink> stage_telemetry,
    std::shared_ptr<attention::TargetK0OracleQsaStateOtelSink>
        state_telemetry,
    TargetK0PhysicalLayerConstructionProgress* progress) {
  const auto mark = [progress](TargetK0PhysicalLayerConstructionStage stage,
                               int layer = -1) {
    if (progress) *progress = {stage, layer,
        TargetGdnOwnerConstructionStage::kUnknown,
        moe::TargetMoeAotCudaFailure::kSuccess};
  };
  if (device < 0 || !accepted_loader_lease_handle || !plans ||
      !validate_target_k0_physical_layer_plans(*plans, rank) ||
      reductions.rank() != rank || !layer_telemetry || !moe_telemetry ||
      !stage_telemetry || !state_telemetry)
    throw std::invalid_argument("K0 physical layer factory identity changed");

  auto result = std::unique_ptr<TargetK0PhysicalLayerOwners>(
      new TargetK0PhysicalLayerOwners);
  result->rank_ = rank;
  result->plans_ = std::move(plans);
  mark(TargetK0PhysicalLayerConstructionStage::kQsaArena);
  result->qsa_state_ = attention::TargetK0OracleQsaStateOwner::create(
      device, rank, 35, std::move(state_telemetry));
  mark(TargetK0PhysicalLayerConstructionStage::kSidecar);
  result->qsa_sidecar_ =
      std::make_unique<attention::QsaSidecarDeviceOwner>(
          device, sidecar_payload,
          attention::target_qsa_sidecar_identity(rank, 3));
  mark(TargetK0PhysicalLayerConstructionStage::kRope);
  result->qsa_rope_ = std::make_unique<attention::Layer3RopeDeviceOwner>(
      device, attention::target_qsa_rope_identity(rank, 3));

  std::array<std::unique_ptr<TargetK0LayerPort>, kDecoderLayers> owners{};
  for (int layer = 0; layer < kDecoderLayers; ++layer) {
    const auto& plan = result->plans_->at(rank, layer);
    if (is_qsa_layer(layer)) {
      mark(TargetK0PhysicalLayerConstructionStage::kQsaOwner, layer);
      auto sidecar = result->qsa_sidecar_->publication();
      sidecar.identity = attention::target_qsa_sidecar_identity(rank, layer);
      owners[static_cast<std::size_t>(layer)] =
          TargetQsaK0LayerOwner::create(
              device, plan, accepted_loader_lease_handle, sidecar,
              attention::target_qsa_rope_identity(rank, layer),
              result->qsa_rope_->view(), result->qsa_state_->view(layer),
              result->qsa_state_->ready_event(),
              reductions.attention_port(layer), reductions.moe_port(layer),
              layer_telemetry, moe_telemetry, stage_telemetry, 35);
    } else {
      mark(TargetK0PhysicalLayerConstructionStage::kGdnOwner, layer);
      moe::TargetLayerMoeConstructionStage moe_stage =
          moe::TargetLayerMoeConstructionStage::kUnknown;
      moe::TargetMoeAotConstructionStage aot_stage{};
      moe::TargetMoeAotCudaFailure cuda_failure =
          moe::TargetMoeAotCudaFailure::kSuccess;
      std::unique_ptr<moe::TargetLayerMoeDeviceOwner> moe;
      try {
        moe = moe::TargetLayerMoeDeviceOwner::create(
            device, plan, accepted_loader_lease_handle, moe_telemetry,
            stage_telemetry, &moe_stage, &aot_stage, &cuda_failure);
      } catch (const moe::TargetMoeAotConstructionError& error) {
        if (progress) {
          progress->moe_aot_cuda_failure = cuda_failure;
          progress->gdn_stage =
              error.stage() == moe::TargetMoeAotConstructionStage::kIdentity
                  ? TargetGdnOwnerConstructionStage::kMoeAotIdentity
                  : (error.stage() ==
                             moe::TargetMoeAotConstructionStage::kModuleData
                         ? TargetGdnOwnerConstructionStage::kMoeAotModuleData
                         : TargetGdnOwnerConstructionStage::kMoeAotModuleLoad);
        }
        throw;
      } catch (const moe::TargetFullMoeConstructionError&) {
        if (progress)
          progress->gdn_stage =
              TargetGdnOwnerConstructionStage::kMoeParticipantContract;
        throw;
      } catch (...) {
        if (progress) {
          progress->moe_aot_cuda_failure = cuda_failure;
          progress->gdn_stage = aot_stage ==
                                        moe::TargetMoeAotConstructionStage::kIdentity
                                    ? TargetGdnOwnerConstructionStage::kMoeAotIdentity
                                : aot_stage == moe::TargetMoeAotConstructionStage::kModuleData
                                    ? TargetGdnOwnerConstructionStage::kMoeAotModuleData
                                : aot_stage == moe::TargetMoeAotConstructionStage::kModuleLoad
                                    ? TargetGdnOwnerConstructionStage::kMoeAotModuleLoad
                                : moe_stage == moe::TargetLayerMoeConstructionStage::kAot
                                    ? TargetGdnOwnerConstructionStage::kMoeParticipantContract
                                : (moe_stage == moe::TargetLayerMoeConstructionStage::kStage
                         ? TargetGdnOwnerConstructionStage::kMoeStage
                         : TargetGdnOwnerConstructionStage::kPlanBinder);
        }
        throw;
      }
      owners[static_cast<std::size_t>(layer)] =
          TargetGdnLayerDeviceOwner::create(
              device, plan, accepted_loader_lease_handle, std::move(moe),
              reductions, *layer_telemetry,
              progress ? &progress->gdn_stage : nullptr);
    }
  }
  mark(TargetK0PhysicalLayerConstructionStage::kInventoryAssembly);
  result->inventory_ = std::make_unique<TargetK0LayerOwnerInventory>(
      rank, std::move(owners), reductions, *layer_telemetry);
  result->authenticated_ = result->qsa_state_->authenticated() &&
                           result->inventory_->authenticated();
  if (!result->authenticated_)
    throw std::runtime_error("K0 physical layer factory did not publish");
  return result;
}

}  // namespace rocket::qwen38::decode

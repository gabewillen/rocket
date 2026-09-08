// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_gdn_layer.h"
#include "decode/target_k0_executor.h"
#include "decode/target_layer_native_plan.h"
#include "hyperconnection/hyperconnection.h"
#include "linear_attention/gdn_cutlass.h"
#include "model/target_slab_owner.h"
#include "moe/target_layer_moe_owner.h"

#include <memory>

namespace rocket::qwen38::decode {

enum class TargetGdnOwnerConstructionStage : std::int32_t {
  kUnknown = 0,
  kLease = 1,
  kPlanBinder = 2,
  kGlobals = 3,
  kMoeStage = 4,
  kMoeAot = 5,
  kStorage = 6,
  kCutlassGraph = 7,
  kHyperconnection = 8,
  kComposite = 9,
  kMoeAotIdentity = 10,
  kMoeAotModuleData = 11,
  kMoeAotModuleLoad = 12,
  kMoeParticipantContract = 13,
};

struct TargetGdnNativeWeightBindings {
  linear_attention::GdnWeights attention;
  hyperconnection::Weights attention_hyperconnection;
  hyperconnection::Weights mlp_hyperconnection;
};

// CPU-only bounded validation of the authenticated plan's GDN and HC subset.
// It performs no pointer arithmetic or CUDA work.
bool validate_target_gdn_layer_plan(
    const TargetLayerNativePlan& plan) noexcept;

// Resolves borrowed device aliases only after the complete generic plan and
// accepted target-slab publication have been authenticated. The returned
// aliases remain valid only while the slab lease remains alive.
TargetGdnNativeWeightBindings bind_target_gdn_native_weights(
    const TargetLayerNativePlan& plan,
    const model::TargetSlabPublication& slab);

constexpr std::size_t target_gdn_owner_append(
    std::size_t offset, std::size_t bytes,
    std::size_t alignment = 256) noexcept {
  return ((offset + alignment - 1) & ~(alignment - 1)) + bytes;
}

constexpr std::size_t target_gdn_owner_storage_bytes() noexcept {
  std::size_t offset = 0;
  for (const auto bytes : {
           kTargetGdnStateSlots * kTargetGdnConvSlotElements *
               sizeof(__nv_bfloat16),
           kTargetGdnStateSlots * kTargetGdnRecurrentSlotElements *
               sizeof(float),
           sizeof(std::int32_t),
           static_cast<std::size_t>(kTargetK0Hidden) * sizeof(__nv_bfloat16),
           static_cast<std::size_t>(kTargetK0HiddenStreams) *
               sizeof(__nv_bfloat16),
           static_cast<std::size_t>(kTargetK0Hidden) * sizeof(float),
           static_cast<std::size_t>(kTargetK0HyperHidden) *
               sizeof(__nv_bfloat16),
           static_cast<std::size_t>(kTargetK0Hidden) * sizeof(__nv_bfloat16),
           static_cast<std::size_t>(kTargetK0HiddenStreams) *
               sizeof(__nv_bfloat16),
           static_cast<std::size_t>(kTargetK0Hidden) * sizeof(float),
       })
    offset = target_gdn_owner_append(offset, bytes);
  return target_gdn_owner_append(offset, 0);
}

inline constexpr std::size_t kTargetGdnOwnerStorageBytes =
    target_gdn_owner_storage_bytes();

struct TargetGdnOwnerStorageBinding {
  TargetGdnC1State state;
  TargetGdnC1Buffers buffers;
  std::int32_t* mutable_state_index;
};

// Allocation-free layout binder shared by production and CPU contract tests.
// Storage is borrowed and must be 256-byte aligned with the exact byte count.
TargetGdnOwnerStorageBinding bind_target_gdn_owner_storage(
    void* storage, std::size_t bytes);

// Production owner for one of the 36 rank-local GDN target layers. create()
// consumes the already-authenticated generic MoE owner and retains the
// process-lifetime slab lease before allocating or constructing CUDA owners.
// The object is single-owner and single-stream. Every returned port identity
// remains stable for its lifetime. A failed execution quarantines all device
// dependencies rather than freeing storage that may still be referenced.
class TargetGdnLayerDeviceOwner final : public TargetK0LayerPort {
 public:
  static std::unique_ptr<TargetGdnLayerDeviceOwner> create(
      int device, const TargetLayerNativePlan& plan,
      void* accepted_loader_lease_handle,
      std::unique_ptr<moe::TargetLayerMoeDeviceOwner> moe_owner,
      TargetK0PairReduceSchedule& reductions,
      pair_reduce::OtelStageSink& telemetry,
      TargetGdnOwnerConstructionStage* construction_stage = nullptr);
  ~TargetGdnLayerDeviceOwner() override;
  TargetGdnLayerDeviceOwner(const TargetGdnLayerDeviceOwner&) = delete;
  TargetGdnLayerDeviceOwner& operator=(const TargetGdnLayerDeviceOwner&) =
      delete;

  int rank() const noexcept override { return rank_; }
  int layer() const noexcept override { return layer_; }
  TargetK0AttentionKind attention_kind() const noexcept override {
    return TargetK0AttentionKind::kGdn;
  }
  bool authenticated() const noexcept override { return authenticated_; }
  const HiddenPartialReducer* attention_reducer_identity()
      const noexcept override;
  const HiddenPartialReducer* moe_reducer_identity() const noexcept override;
  void wait_source(cudaStream_t stream) override;
  void execute_row(std::uint64_t generation,
                   const __nv_bfloat16* replicated_pre_layer,
                   __nv_bfloat16* replicated_post_layer,
                   cudaStream_t stream) override;

 private:
  struct Bundle;
  TargetGdnLayerDeviceOwner(
      int device, const TargetLayerNativePlan& plan,
      std::shared_ptr<const model::TargetSlabLease> slab_lease,
      std::unique_ptr<moe::TargetLayerMoeDeviceOwner> moe_owner,
      TargetK0PairReduceSchedule& reductions,
      pair_reduce::OtelStageSink& telemetry,
      TargetGdnOwnerConstructionStage* construction_stage);

  int device_ = -1;
  int rank_ = -1;
  int layer_ = -1;
  bool authenticated_ = false;
  std::unique_ptr<Bundle> bundle_;
};

}  // namespace rocket::qwen38::decode

// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_layer_native_bindings.h"
#include "model/target_slab_owner.h"
#include "moe/native_target_moe_graph.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rocket::qwen38::moe {

enum class TargetLayerMoeConstructionStage : std::int32_t {
  kUnknown = 0,
  kPlanBinder = 1,
  kStage = 2,
  kAot = 3,
};

constexpr std::size_t target_layer_moe_owner_append(
    std::size_t offset, std::size_t bytes,
    std::size_t alignment = 256) noexcept {
  return ((offset + alignment - 1) & ~(alignment - 1)) + bytes;
}

constexpr std::size_t target_layer_moe_e11_workspace_bytes() noexcept {
  std::size_t o = 0;
  for (const auto bytes : {140'800ULL, 225'280ULL, 153'600ULL, 4ULL, 4ULL,
                           44ULL, 4ULL, 44ULL, 40ULL, 112ULL, 440ULL,
                           440ULL})
    o = target_layer_moe_owner_append(o, bytes);
  return target_layer_moe_owner_append(o, 0);
}

constexpr std::size_t target_layer_moe_runtime_bytes() noexcept {
  std::size_t o = 0;
  for (const auto bytes : {2'048ULL, 40ULL, 40ULL, 40ULL, 40ULL, 8ULL, 16ULL,
                           140'800ULL, 225'280ULL, 153'600ULL, 4ULL, 4ULL,
                           44ULL, 4ULL, 44ULL, 40ULL, 112ULL, 440ULL, 440ULL,
                           640ULL, 640ULL, 4ULL, 5'120ULL})
    o = target_layer_moe_owner_append(o, bytes);
  return target_layer_moe_owner_append(o, 0);
}

inline constexpr std::size_t kTargetLayerMoeE11WorkspaceBytes =
    target_layer_moe_e11_workspace_bytes();
inline constexpr std::size_t kTargetLayerMoeRuntimeBytes =
    target_layer_moe_runtime_bytes();
inline constexpr std::size_t kTargetLayerMoeStageDeviceBytes =
    kTargetMoeStageRawPlaneBytes +
    3ULL * kTargetMoeStagedExperts * sizeof(std::uint32_t);
static_assert(kTargetLayerMoeStageDeviceBytes + sizeof(TargetMoeN640StageEvidence) ==
              kTargetMoeStageScratchBytes);

// CPU-only, allocation-free gate used by --preflight-only. It authenticates
// every identity consumed by the compact AOT owner before any CUDA call.
bool validate_target_layer_moe_owner_plan(
    const decode::TargetLayerNativePlan& plan,
    const model::TargetSlabPublication& slab) noexcept;
bool validate_target_layer_moe_owner_handoff(
    const decode::TargetLayerNativePlan& plan,
    void* accepted_loader_lease_handle) noexcept;

struct TargetLayerMoeStorageBinding {
  TargetMoeN768StageScratch stage;
  TargetFullMoeC1Workspace runtime;
  __nv_bfloat16* rank_local_output;
};

// Allocation-free layout binder shared by the production owner and CPU
// canary proof. It rejects short storage before returning any pointers.
TargetLayerMoeStorageBinding bind_target_layer_moe_storage(
    void* stage_storage, std::size_t stage_bytes,
    TargetMoeN640StageEvidence* evidence_host,
    TargetMoeN640StageEvidence* evidence_device,
    void* runtime_storage, std::size_t runtime_bytes,
    const std::uint64_t* requested_generation);

// Rank-local target-layer owner. Its preallocated dependency bundle retains the
// target-slab lease, one reusable E10 N768 staging allocation, E11
// AOT/router/shared workspace, requested-generation word, mapped terminal
// evidence, both telemetry owners, stage, participant, and graph adapter.
// A failed terminal fence releases that entire bundle into process-lifetime
// quarantine without allocating or freeing any referenced object.
class TargetLayerMoeDeviceOwner final {
 public:
  static std::unique_ptr<TargetLayerMoeDeviceOwner> create(
      int device, const decode::TargetLayerNativePlan& plan,
      void* accepted_loader_lease_handle,
      std::shared_ptr<TargetFullMoeOtelSink> telemetry,
      std::shared_ptr<TargetMoeStageOtelSink> stage_telemetry,
      TargetLayerMoeConstructionStage* construction_stage = nullptr,
      TargetMoeAotConstructionStage* aot_stage = nullptr);
  ~TargetLayerMoeDeviceOwner();
  TargetLayerMoeDeviceOwner(const TargetLayerMoeDeviceOwner&) = delete;
  TargetLayerMoeDeviceOwner& operator=(const TargetLayerMoeDeviceOwner&) = delete;

  void wait_source(cudaStream_t stream);
  int rank() const noexcept { return rank_; }
  int layer() const noexcept { return layer_; }
  bool authenticated() const noexcept { return authenticated_; }
  NativeTargetMoeGraph& graph() noexcept;
  const TargetFullMoeC1Workspace& workspace() const noexcept;
  std::uint64_t* requested_generation() noexcept;

 private:
  struct Bundle;
  TargetLayerMoeDeviceOwner(
      int device, const decode::TargetLayerNativePlan& plan,
      std::shared_ptr<const model::TargetSlabLease> slab_lease,
      std::shared_ptr<TargetFullMoeOtelSink> telemetry,
      std::shared_ptr<TargetMoeStageOtelSink> stage_telemetry,
      TargetLayerMoeConstructionStage* construction_stage,
      TargetMoeAotConstructionStage* aot_stage);
  int device_ = -1;
  int rank_ = -1;
  int layer_ = -1;
  bool authenticated_ = false;
  std::unique_ptr<Bundle> bundle_;
};

}  // namespace rocket::qwen38::moe

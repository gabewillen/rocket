// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_layer3_native_bindings.h"
#include "model/target_slab_owner.h"
#include "moe/native_target_moe_graph.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rocket::qwen38::moe {

constexpr std::size_t target_moe_owner_append(
    std::size_t offset, std::size_t bytes,
    std::size_t alignment = 256) noexcept {
  return ((offset + alignment - 1) & ~(alignment - 1)) + bytes;
}

constexpr std::size_t target_moe_e11_workspace_bytes() noexcept {
  std::size_t o = 0;
  for (const auto bytes : {140'800ULL, 225'280ULL, 153'600ULL, 4ULL, 4ULL,
                           44ULL, 4ULL, 44ULL, 40ULL, 112ULL, 440ULL,
                           440ULL})
    o = target_moe_owner_append(o, bytes);
  return target_moe_owner_append(o, 0);
}

constexpr std::size_t target_layer3_moe_runtime_bytes() noexcept {
  std::size_t o = 0;
  for (const auto bytes : {2'048ULL, 40ULL, 40ULL, 40ULL, 40ULL, 8ULL, 16ULL,
                           140'800ULL, 225'280ULL, 153'600ULL, 4ULL, 4ULL,
                           44ULL, 4ULL, 44ULL, 40ULL, 112ULL, 440ULL, 440ULL,
                           640ULL, 640ULL, 4ULL, 5'120ULL})
    o = target_moe_owner_append(o, bytes);
  return target_moe_owner_append(o, 0);
}

inline constexpr std::size_t kTargetMoeE11WorkspaceBytes =
    target_moe_e11_workspace_bytes();
inline constexpr std::size_t kTargetLayer3MoeRuntimeBytes =
    target_layer3_moe_runtime_bytes();
inline constexpr std::size_t kTargetMoeStageDeviceBytes =
    kTargetMoeStageRawPlaneBytes +
    3ULL * kTargetMoeStagedExperts * sizeof(std::uint32_t);
static_assert(kTargetMoeStageDeviceBytes + sizeof(TargetMoeN640StageEvidence) ==
              kTargetMoeStageScratchBytes);

// CPU-only, allocation-free gate used by --preflight-only. It authenticates
// every identity consumed by the compact AOT owner before any CUDA call.
bool validate_target_layer3_moe_owner_plan(
    const decode::TargetLayer3NativePlan& plan,
    const model::TargetSlabPublication& slab) noexcept;
bool validate_target_layer3_moe_owner_handoff(
    const decode::TargetLayer3NativePlan& plan,
    void* accepted_loader_lease_handle) noexcept;

struct TargetLayer3MoeStorageBinding {
  TargetMoeN768StageScratch stage;
  TargetFullMoeC1Workspace runtime;
  __nv_bfloat16* rank_local_output;
};

// Allocation-free layout binder shared by the production owner and CPU
// canary proof. It rejects short storage before returning any pointers.
TargetLayer3MoeStorageBinding bind_target_layer3_moe_storage(
    void* stage_storage, std::size_t stage_bytes,
    TargetMoeN640StageEvidence* evidence_host,
    TargetMoeN640StageEvidence* evidence_device,
    void* runtime_storage, std::size_t runtime_bytes,
    const std::uint64_t* requested_generation);

// Rank-local layer3 owner. Its preallocated dependency bundle retains the
// target-slab lease, one reusable E10 N768 staging allocation, E11
// AOT/router/shared workspace, requested-generation word, mapped terminal
// evidence, both telemetry owners, stage, participant, and graph adapter.
// A failed terminal fence releases that entire bundle into process-lifetime
// quarantine without allocating or freeing any referenced object.
class TargetLayer3MoeDeviceOwner final {
 public:
  static std::unique_ptr<TargetLayer3MoeDeviceOwner> create(
      int device, const decode::TargetLayer3NativePlan& plan,
      void* accepted_loader_lease_handle,
      std::shared_ptr<TargetFullMoeOtelSink> telemetry,
      std::shared_ptr<TargetMoeStageOtelSink> stage_telemetry);
  ~TargetLayer3MoeDeviceOwner();
  TargetLayer3MoeDeviceOwner(const TargetLayer3MoeDeviceOwner&) = delete;
  TargetLayer3MoeDeviceOwner& operator=(const TargetLayer3MoeDeviceOwner&) = delete;

  void wait_source(cudaStream_t stream);
  NativeTargetMoeGraph& graph() noexcept;
  const TargetFullMoeC1Workspace& workspace() const noexcept;
  std::uint64_t* requested_generation() noexcept;

 private:
  struct Bundle;
  TargetLayer3MoeDeviceOwner(
      int device, const decode::TargetLayer3NativePlan& plan,
      std::shared_ptr<const model::TargetSlabLease> slab_lease,
      std::shared_ptr<TargetFullMoeOtelSink> telemetry,
      std::shared_ptr<TargetMoeStageOtelSink> stage_telemetry);
  int device_ = -1;
  std::unique_ptr<Bundle> bundle_;
};

}  // namespace rocket::qwen38::moe

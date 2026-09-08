// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "attention/layer3_rope_owner.h"
#include "attention/qsa_sidecar_owner.h"
#include "decode/target_k0_executor.h"
#include "decode/target_layer_native_plan.h"
#include "model/target_slab_owner.h"
#include "moe/target_layer_moe_owner.h"

#include <filesystem>
#include <memory>

namespace rocket::qwen38::decode {

// Physical QSA layer port. Construction binds one authenticated plan, slab,
// sidecar, RoPE table, state view, both fixed reducers, HC, and compact-E10
// MoE into a single lifetime domain.
class TargetQsaK0LayerOwner final : public TargetK0LayerPort {
 public:
  static std::unique_ptr<TargetQsaK0LayerOwner> create(
      int device, const TargetLayerNativePlan& plan,
      void* accepted_loader_lease_handle,
      const attention::QsaSidecarPublication& sidecar,
      const attention::Layer3RopeIdentity& rope_identity,
      const attention::Layer3RopeView& rope,
      const attention::TargetQsaStateView& state,
      cudaEvent_t state_ready,
      HiddenPartialReducer& attention_reducer,
      HiddenPartialReducer& moe_reducer,
      std::shared_ptr<pair_reduce::OtelStageSink> layer_telemetry,
      std::shared_ptr<moe::TargetFullMoeOtelSink> moe_telemetry,
      std::shared_ptr<moe::TargetMoeStageOtelSink> stage_telemetry,
      int max_rows = 35);
  ~TargetQsaK0LayerOwner() override;
  TargetQsaK0LayerOwner(const TargetQsaK0LayerOwner&) = delete;
  TargetQsaK0LayerOwner& operator=(const TargetQsaK0LayerOwner&) = delete;

  int rank() const noexcept override { return rank_; }
  int layer() const noexcept override { return layer_; }
  TargetK0AttentionKind attention_kind() const noexcept override {
    return TargetK0AttentionKind::kQsa;
  }
  bool authenticated() const noexcept override { return authenticated_; }
  const HiddenPartialReducer* attention_reducer_identity()
      const noexcept override { return attention_reducer_; }
  const HiddenPartialReducer* moe_reducer_identity()
      const noexcept override { return moe_reducer_; }
  void wait_source(cudaStream_t stream) override;
  void execute_row(std::uint64_t generation,
                   const __nv_bfloat16* replicated_pre_layer,
                   __nv_bfloat16* replicated_post_layer,
                   cudaStream_t stream) override;

 private:
  struct Bundle;
  TargetQsaK0LayerOwner(
      int device, const TargetLayerNativePlan& plan,
      void* accepted_loader_lease_handle,
      const attention::QsaSidecarPublication& sidecar,
      const attention::Layer3RopeIdentity& rope_identity,
      const attention::Layer3RopeView& rope,
      const attention::TargetQsaStateView& state,
      cudaEvent_t state_ready,
      HiddenPartialReducer& attention_reducer,
      HiddenPartialReducer& moe_reducer,
      std::shared_ptr<pair_reduce::OtelStageSink> layer_telemetry,
      std::shared_ptr<moe::TargetFullMoeOtelSink> moe_telemetry,
      std::shared_ptr<moe::TargetMoeStageOtelSink> stage_telemetry,
      int max_rows);
  int device_ = -1;
  int rank_ = -1;
  int layer_ = -1;
  int row_ = 0;
  bool authenticated_ = false;
  HiddenPartialReducer* attention_reducer_ = nullptr;
  HiddenPartialReducer* moe_reducer_ = nullptr;
  std::unique_ptr<Bundle> bundle_;
};

}  // namespace rocket::qwen38::decode

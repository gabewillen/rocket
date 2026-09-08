// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "moe/target_moe_b12x_aot.h"
#include "moe/target_moe_c1.h"
#include "moe/target_moe_n640_device_stage.h"
#include "moe/target_router_shared_c1.h"

namespace rocket::qwen38::moe {

class TargetFullMoeConstructionError final : public std::runtime_error {
 public:
  TargetFullMoeConstructionError()
      : std::runtime_error("target full MoE participant contract changed") {}
};

struct TargetFullMoeC1Weights {
  TargetRouterNvfp4Weights router;
  TargetMoeCompactRuntimeIdentity routed_identity;
  TargetMoeN640DeviceStage* routed_stage;
  TargetMoeN768StageScratch routed_stage_scratch;
  TargetMoeStageOtelSink* routed_stage_telemetry;
  TargetSharedBf16Weights shared;
};

struct TargetFullMoeC1Workspace {
  float* router_logits_f32;
  std::int32_t* global_ids_i32;
  float* routing_weights_f32;
  std::int32_t* local_ids_i32;
  float* local_weights_f32;
  std::uint64_t* source_generation;
  const std::uint64_t* requested_generation;
  TargetMoeC1Summary* route_summary;
  TargetMoeB12xWorkspace routed;
  TargetMoeN768StageScratch routed_stage;
  float* shared_gate_scratch_f32;
  float* shared_up_scratch_f32;
  float* shared_gate_scalar_f32;
};

struct TargetFullMoeC1Launch {
  const __nv_bfloat16* hidden_bf16;
  __nv_bfloat16* rank_local_partial_bf16;
  TargetFullMoeC1Workspace workspace;
  cudaStream_t stream;
};

enum class TargetFullMoeComponent : std::uint8_t {
  kRouter,
  kLocalization,
  kStaging,
  kRoutedExperts,
  kSharedExpert,
};

struct TargetFullMoeOtelPoint {
  TargetFullMoeComponent component;
  TargetDenseOutcome outcome;
  int rank;
  int layer;
};

class TargetFullMoeOtelSink {
 public:
  virtual ~TargetFullMoeOtelSink() = default;
  virtual void emit(const TargetFullMoeOtelPoint& point) noexcept = 0;
};

class TargetFullMoeC1Port {
 public:
  virtual ~TargetFullMoeC1Port() = default;
  [[nodiscard]] virtual TargetDenseOutcome enqueue(
      const TargetFullMoeC1Launch& launch) const noexcept = 0;
  [[nodiscard]] virtual TargetDenseOutcome enqueue_with_telemetry(
      const TargetFullMoeC1Launch& launch,
      TargetFullMoeOtelSink& telemetry) const noexcept = 0;
  [[nodiscard]] virtual const TargetDenseIdentity& identity() const noexcept = 0;
  virtual void wait_source(cudaStream_t stream) = 0;
  [[nodiscard]] virtual TargetDenseOutcome publish_after_fence(
      std::uint64_t generation,
      TargetFullMoeOtelSink& telemetry) noexcept = 0;
};

// Fixed rank/layer3/K0 participant. Construction loads the generated B12X
// module outside capture. Enqueue only submits work to the borrowed stream and
// writes the caller-owned rank-local BF16 partial. wait_source() must run once
// on the borrowed graph stream before capture. publish_after_fence() reads only
// the caller-owned mapped stage evidence after that stream has been fenced.
class TargetFullMoeC1 final : public TargetFullMoeC1Port {
 public:
  TargetFullMoeC1(int device, TargetDenseIdentity identity,
                  TargetFullMoeC1Weights weights);
  ~TargetFullMoeC1();
  TargetFullMoeC1(const TargetFullMoeC1&) = delete;
  TargetFullMoeC1& operator=(const TargetFullMoeC1&) = delete;

  [[nodiscard]] TargetDenseOutcome enqueue(
      const TargetFullMoeC1Launch& launch) const noexcept override;
  [[nodiscard]] TargetDenseOutcome enqueue_with_telemetry(
      const TargetFullMoeC1Launch& launch,
      TargetFullMoeOtelSink& telemetry) const noexcept override;
  [[nodiscard]] const TargetDenseIdentity& identity() const noexcept override;
  void wait_source(cudaStream_t stream) override;
  [[nodiscard]] TargetDenseOutcome publish_after_fence(
      std::uint64_t generation,
      TargetFullMoeOtelSink& telemetry) noexcept override;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace rocket::qwen38::moe

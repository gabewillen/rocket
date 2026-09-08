// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "attention/native_qsa_graph.h"
#include "attention/qsa_sidecar_owner.h"
#include "decode/target_layer3_native_plan.h"
#include "hyperconnection/hyperconnection.h"
#include "model/target_slab_owner.h"
#include "moe/target_router_shared_c1.h"

namespace rocket::qwen38::decode {

struct TargetLayer3NativeWeightBindings {
  attention::TargetQsaProjectionWeights qsa_projection;
  attention::TargetQsaPreprocessWeights qsa_preprocess;
  hyperconnection::Weights attention_hyperconnection;
  hyperconnection::Weights mlp_hyperconnection;
  moe::TargetRouterNvfp4Weights router;
  moe::TargetSharedBf16Weights shared;
};

class TargetLayer3ReadyEventProbe {
 public:
  virtual ~TargetLayer3ReadyEventProbe() = default;
  virtual bool complete(cudaEvent_t event) noexcept = 0;
};

// Production readiness probe. It performs a nonblocking query and accepts only
// cudaSuccess; an invalid or incomplete event fails the binding transaction.
class CudaTargetLayer3ReadyEventProbe final
    : public TargetLayer3ReadyEventProbe {
 public:
  bool complete(cudaEvent_t event) noexcept override;
};

// Validates the complete plan, slab publication, sidecar publication, rank,
// device, and immutable identities before resolving any device address. All
// arguments are borrowed for this synchronous call. The returned aggregate
// contains borrowed device pointers whose lifetimes remain bounded by the two
// publication owners and the caller-owned RoPE owner. Reentrant for immutable
// publications; throws std::invalid_argument and publishes no partial binding
// on any mismatch. The telemetry sink is synchronous and must not throw.
TargetLayer3NativeWeightBindings bind_target_layer3_native_weights(
    const TargetLayer3NativePlan& plan,
    const model::TargetSlabPublication& slab,
    const attention::QsaSidecarPublication& sidecar,
    const __nv_bfloat16* rope_cos_sin,
    TargetLayer3ReadyEventProbe& ready_event_probe,
    pair_reduce::OtelStageSink& telemetry);

}  // namespace rocket::qwen38::decode

// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "attention/layer3_rope_owner.h"
#include "attention/native_qsa_graph.h"
#include "attention/qsa_sidecar_owner.h"
#include "decode/target_layer_native_plan.h"
#include "hyperconnection/hyperconnection.h"
#include "model/target_slab_owner.h"
#include "moe/target_moe_n640_device_stage.h"
#include "moe/target_router_shared_c1.h"

namespace rocket::qwen38::decode {

struct TargetLayerNativeMoeWeights {
  moe::TargetRouterNvfp4Weights router;
  moe::TargetSharedBf16Weights shared;
  std::array<moe::TargetMoeN640DeviceExpert,
             moe::kTargetMoeLocalExperts> routed_source;
  moe::TargetMoeCompactRuntimeIdentity routed_identity;
};

struct TargetQsaLayerNativeWeights {
  attention::TargetQsaProjectionWeights projection;
  attention::TargetQsaPreprocessWeights preprocess;
  cudaEvent_t rope_ready_event = nullptr;
  hyperconnection::Weights attention_hyperconnection;
  hyperconnection::Weights mlp_hyperconnection;
};

TargetLayerNativeMoeWeights bind_target_layer_native_moe_weights(
    const TargetLayerNativePlan& plan,
    const model::TargetSlabPublication& slab);

TargetQsaLayerNativeWeights bind_target_qsa_layer_native_weights(
    const TargetLayerNativePlan& plan,
    const model::TargetSlabPublication& slab,
    const attention::QsaSidecarPublication& sidecar,
    const attention::Layer3RopeIdentity& rope_identity,
    const attention::Layer3RopeView& rope);

}  // namespace rocket::qwen38::decode

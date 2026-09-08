// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_layer_native_plan.h"
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

TargetLayerNativeMoeWeights bind_target_layer_native_moe_weights(
    const TargetLayerNativePlan& plan,
    const model::TargetSlabPublication& slab);

}  // namespace rocket::qwen38::decode

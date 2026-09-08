// SPDX-License-Identifier: Apache-2.0
#include "moe/native_target_moe_graph.h"

#include <stdexcept>

namespace rocket::qwen38::moe {

NativeTargetMoeGraph::NativeTargetMoeGraph(
    TargetFullMoeC1Port& participant, const TargetFullMoeC1Workspace& workspace,
    __nv_bfloat16* rank_local_partial_bf16, TargetFullMoeOtelSink& telemetry)
    : participant_(participant),
      workspace_(workspace),
      rank_local_partial_bf16_(rank_local_partial_bf16),
      telemetry_(telemetry),
      rank_(participant.identity().rank),
      layer_(participant.identity().layer) {
  if ((rank_ != 0 && rank_ != 1) || layer_ != kTargetCompositionLayer ||
      !rank_local_partial_bf16_)
    throw std::invalid_argument("native target MoE graph identity changed");
}

void NativeTargetMoeGraph::launch(
    const __nv_bfloat16* block_input, std::uint64_t generation, int m,
    cudaStream_t stream) {
  if (faulted_ || enqueued_generation_ != published_generation_ ||
      !block_input || !stream || m != 1 || generation == 0 ||
      generation != published_generation_ + 1) {
    faulted_ = true;
    throw std::invalid_argument("native target MoE graph launch changed");
  }
  const TargetFullMoeC1Launch launch{
      block_input, rank_local_partial_bf16_, workspace_, stream};
  if (participant_.enqueue_with_telemetry(launch, telemetry_) !=
      TargetDenseOutcome::kOk) {
    faulted_ = true;
    throw std::runtime_error("native target MoE graph enqueue failed");
  }
  enqueued_generation_ = generation;
}

void NativeTargetMoeGraph::publish_after_fence(std::uint64_t generation) {
  if (faulted_ || generation == 0 || generation != enqueued_generation_ ||
      published_generation_ + 1 != generation) {
    faulted_ = true;
    throw std::invalid_argument("native target MoE fence generation changed");
  }
  published_generation_ = generation;
}

TargetMoeComponentIdentity NativeTargetMoeGraph::component_identity(
    TargetFullMoeComponent component) const noexcept {
  const auto dtype = component == TargetFullMoeComponent::kSharedExpert
                         ? TargetMoeServingDtype::kBfloat16
                         : TargetMoeServingDtype::kNvfp4;
  return {component, participant_.identity(), dtype};
}

}  // namespace rocket::qwen38::moe

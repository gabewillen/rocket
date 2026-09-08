// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_c1.h"

#include <cmath>

namespace rocket::qwen38::moe {
namespace {

__global__ void localize_target_moe_c1(TargetMoeC1RouteLaunch launch) {
  __shared__ int invalid;
  __shared__ int stale;
  __shared__ int local_routes;
  const int slot = threadIdx.x;
  if (slot == 0) {
    invalid = 0;
    stale = *launch.source_generation != *launch.requested_generation ||
            *launch.requested_generation == 0;
    local_routes = 0;
    *launch.summary = {0, 0, TargetMoeOutcome::kContractError};
  }
  if (slot < kTargetMoeC1TopK) {
    launch.local_expert_ids[slot] = 0;
    launch.local_routing_weights[slot] = 0.0F;
    const int expert = launch.global_expert_ids[slot];
    const float weight = launch.routing_weights[slot];
    if (expert < 0 || expert >= kTargetMoeGlobalExperts || !isfinite(weight) ||
        weight < 0.0F || weight > 1.0F)
      atomicExch(&invalid, 1);
  }
  __syncthreads();
  if (stale || invalid) {
    if (slot == 0)
      launch.summary->outcome = stale ? TargetMoeOutcome::kStaleGeneration
                                      : TargetMoeOutcome::kContractError;
    return;
  }
  if (slot < kTargetMoeC1TopK) {
    const int first = launch.rank * kTargetMoeLocalExperts;
    const int global = launch.global_expert_ids[slot];
    const bool local = global >= first && global < first + kTargetMoeLocalExperts;
    if (local) {
      launch.local_expert_ids[slot] = global - first;
      launch.local_routing_weights[slot] = launch.routing_weights[slot];
      atomicAdd(&local_routes, 1);
    }
  }
  __syncthreads();
  if (slot == 0) {
    launch.summary->generation = *launch.requested_generation;
    launch.summary->local_routes = local_routes;
    launch.summary->outcome = TargetMoeOutcome::kOk;
  }
}

}  // namespace

TargetMoeOutcome enqueue_target_moe_c1_routes(
    const TargetMoeC1RouteLaunch& launch) noexcept {
  if ((launch.rank != 0 && launch.rank != 1) || launch.layer < 0 ||
      launch.layer >= 48 || !launch.global_expert_ids ||
      !launch.routing_weights || !launch.source_generation ||
      !launch.requested_generation || !launch.local_expert_ids ||
      !launch.local_routing_weights || !launch.summary || !launch.stream)
    return TargetMoeOutcome::kContractError;
  localize_target_moe_c1<<<1, 32, 0, launch.stream>>>(launch);
  return cudaPeekAtLastError() == cudaSuccess ? TargetMoeOutcome::kOk
                                              : TargetMoeOutcome::kCudaError;
}

}  // namespace rocket::qwen38::moe

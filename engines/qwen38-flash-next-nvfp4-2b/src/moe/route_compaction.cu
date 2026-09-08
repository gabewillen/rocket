// SPDX-License-Identifier: Apache-2.0
#include "moe/route_compaction.h"

#include <cuda_runtime.h>

#include <cmath>

namespace rocket::qwen38::moe {
namespace {

// Reference lineage:
//   vLLM 8e685d198 and 6b5a12c0 group row-major routes by expert and publish
//   a device-resident active prefix for GEMM.
//   FlashInfer 91bda04c assigns dense experts by first occurrence and drives
//   SM12x work from active_expert_count and per-expert row counts.
// This Qwen3.8-only kernel fixes E512/EP2/top-k10/max-128-row geometry and keeps
// all storage caller-owned so graph replay launches one fixed CTA.
struct RouteCompactionKernelArgs {
  RouteCompactionShape shape;
  RouteCompactionCapacity capacity;
  RouteCompactionInput input;
  RouteCompactionBuffers output;
};

__global__ void compact_owner_routes(RouteCompactionKernelArgs args) {
  const auto shape = args.shape;
  const auto capacity = args.capacity;
  const auto input = args.input;
  const auto output = args.output;
  const int lane = threadIdx.x;
  for (int local = lane; local < kLocalExperts; local += blockDim.x) {
    output.local_to_active[local] = -1;
  }
  for (int expert = lane; expert < capacity.experts; expert += blockDim.x) {
    output.active_global_expert_ids[expert] = -1;
    output.expert_row_counts[expert] = 0;
    output.expert_route_offsets[expert] = 0;
    output.expert_route_cursors[expert] = 0;
  }
  if (lane == 0) {
    output.expert_route_offsets[capacity.experts] = 0;
    *output.summary = {
        .generation = 0,
        .active_weight_bytes = 0,
        .active_experts = 0,
        .active_rows = 0,
        .active_routes = 0,
        .outcome = RouteCompactionOutcome::kContractError,
    };
  }
  __syncthreads();
  if (lane != 0) return;

  const std::uint64_t source_generation = *input.source_generation;
  const std::uint64_t requested_generation = *input.requested_generation;
  if (source_generation == 0 || source_generation != requested_generation) {
    output.summary->outcome = RouteCompactionOutcome::kStaleGeneration;
    return;
  }

  const int first_expert = shape.rank * kLocalExperts;
  int active_experts = 0;
  int active_rows = 0;
  int active_routes = 0;
  for (int row = 0; row < shape.rows; ++row) {
    bool owner_row = false;
    for (int slot = 0; slot < kTopK; ++slot) {
      const int input_route = row * kTopK + slot;
      const std::int32_t expert = input.global_expert_ids[input_route];
      const float weight = input.routing_weights[input_route];
      if (expert < 0 || expert >= kGlobalExperts || !isfinite(weight) ||
          weight < 0.0F || weight > 1.0F) {
        return;
      }
      for (int prior = 0; prior < slot; ++prior) {
        if (input.global_expert_ids[row * kTopK + prior] == expert) return;
      }
      if (expert < first_expert || expert >= first_expert + kLocalExperts) {
        continue;
      }
      if (active_routes == capacity.routes) {
        output.summary->outcome = RouteCompactionOutcome::kOverflow;
        return;
      }
      owner_row = true;
      const int local_expert = expert - first_expert;
      int active_expert = output.local_to_active[local_expert];
      if (active_expert < 0) {
        if (active_experts == capacity.experts) {
          output.summary->outcome = RouteCompactionOutcome::kOverflow;
          return;
        }
        active_expert = active_experts++;
        output.local_to_active[local_expert] = active_expert;
        output.active_global_expert_ids[active_expert] = expert;
      }
      ++output.expert_row_counts[active_expert];
      output.owner_route_global_expert_ids[active_routes] = expert;
      output.owner_route_weights[active_routes] = weight;
      output.owner_route_rows[active_routes] = row;
      output.owner_route_slots[active_routes] = static_cast<std::uint8_t>(slot);
      ++active_routes;
    }
    if (owner_row) ++active_rows;
  }

  int prefix = 0;
  for (int expert = 0; expert < active_experts; ++expert) {
    output.expert_route_offsets[expert] = prefix;
    output.expert_route_cursors[expert] = prefix;
    prefix += output.expert_row_counts[expert];
  }
  output.expert_route_offsets[active_experts] = prefix;
  for (int route = 0; route < active_routes; ++route) {
    const int local_expert =
        output.owner_route_global_expert_ids[route] - first_expert;
    const int active_expert = output.local_to_active[local_expert];
    output.expert_route_indices[output.expert_route_cursors[active_expert]++] =
        route;
  }

  *output.summary = {
      .generation = requested_generation,
      .active_weight_bytes =
          static_cast<std::uint64_t>(active_experts) * kNvfp4BytesPerExpert,
      .active_experts = active_experts,
      .active_rows = active_rows,
      .active_routes = active_routes,
      .outcome = RouteCompactionOutcome::kOk,
  };
}

bool bound(const RouteCompactionBuffers& output) noexcept {
  return output.active_global_expert_ids && output.local_to_active &&
         output.expert_row_counts && output.expert_route_offsets &&
         output.expert_route_cursors && output.owner_route_global_expert_ids &&
         output.owner_route_weights && output.owner_route_rows &&
         output.owner_route_slots && output.expert_route_indices &&
         output.summary;
}

}  // namespace

RouteCompactionOutcome enqueue_route_compaction(
    const RouteCompactionLaunch& launch) noexcept {
  if (!allowed_shape(launch.shape) || !allowed_capacity(launch.capacity) ||
      launch.shape.rows > launch.capacity.rows ||
      !launch.input.global_expert_ids || !launch.input.routing_weights ||
      !launch.input.source_generation || !launch.input.requested_generation ||
      !bound(launch.output) || !launch.stream) {
    return RouteCompactionOutcome::kContractError;
  }
  constexpr int kThreads = 256;
  compact_owner_routes<<<1, kThreads, 0, launch.stream>>>({
      .shape = launch.shape,
      .capacity = launch.capacity,
      .input = launch.input,
      .output = launch.output,
  });
  return cudaPeekAtLastError() == cudaSuccess
             ? RouteCompactionOutcome::kOk
             : RouteCompactionOutcome::kCudaError;
}

}  // namespace rocket::qwen38::moe

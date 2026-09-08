// SPDX-License-Identifier: Apache-2.0
#include "moe/fp8_routed_experts.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>

namespace rocket::qwen38::moe {
namespace {

// Dataflow lineage: vLLM 8e685d198
// model_executor/layers/fused_moe/experts/triton_moe.py and fused_moe.py,
// Apache-2.0. The pinned implementation sorts and pads routes, launches gate/up,
// fuses SiLU*up with block quantization, launches down, then applies route
// weights. Rocket's preceding compactor already supplies the stable grouped
// permutation and exact active prefix, so this fixed-shape port removes the
// duplicate sort and padding. vLLM main 6b5a12c0 retains the same modular
// sequence; its DeepGEMM path has no pure-FP8 SM120 runtime and falls back to
// this Triton path for the authenticated MTP slab.
constexpr int kThreads = 128;
constexpr int kFc1Tiles =
    static_cast<int>(kLogicalIntermediate) / kThreads;
constexpr int kFc2Tiles = static_cast<int>(kHidden) / kThreads;
static_assert(kLogicalIntermediate % kThreads == 0);
static_assert(kHidden % kThreads == 0);

struct KernelArgs {
  RouteCompactionShape shape;
  RouteCompactionCapacity capacity;
  RouteCompactionBuffers routes;
  Fp8ExpertTables experts;
  RoutedExpertBuffers buffers;
};

__device__ float fp8_value(const std::uint8_t* values, int index) {
  const auto* typed = reinterpret_cast<const __nv_fp8_e4m3*>(values);
  return static_cast<float>(typed[index]);
}

__device__ int active_expert_for_grouped_route(const KernelArgs& args,
                                                int grouped_route,
                                                int active_experts) {
  int low = 0;
  int high = active_experts;
  while (low + 1 < high) {
    const int middle = (low + high) / 2;
    if (args.routes.expert_route_offsets[middle] <= grouped_route)
      low = middle;
    else
      high = middle;
  }
  return low;
}

__device__ bool valid_generation(const KernelArgs& args) {
  return args.buffers.summary->outcome == RoutedExpertOutcome::kOk &&
         args.buffers.summary->generation != 0 &&
         args.routes.summary->outcome == RouteCompactionOutcome::kOk &&
         args.buffers.summary->generation == args.routes.summary->generation;
}

__global__ void prepare_consumer(KernelArgs args) {
  for (int index = threadIdx.x;
       index < args.shape.rows * static_cast<int>(kHidden);
       index += blockDim.x)
    args.buffers.rank_output[index] = 0.0F;
  __syncthreads();

  if (threadIdx.x == 0) {
    auto result = RoutedExpertDeviceSummary{
        .generation = 0,
        .active_weight_bytes = 0,
        .fc1_tiles = 0,
        .fc2_tiles = 0,
        .active_experts = 0,
        .active_routes = 0,
        .outcome = RoutedExpertOutcome::kContractError,
    };
    const auto route = *args.routes.summary;
    if (route.outcome == RouteCompactionOutcome::kStaleGeneration) {
      result.outcome = RoutedExpertOutcome::kStaleGeneration;
    } else if (route.outcome == RouteCompactionOutcome::kOk &&
               route.generation != 0 && route.active_experts >= 0 &&
               route.active_experts <= args.capacity.experts &&
               route.active_routes >= 0 &&
               route.active_routes <= args.capacity.routes) {
      bool valid = args.routes.expert_route_offsets[0] == 0 &&
                   args.routes.expert_route_offsets[route.active_experts] ==
                       route.active_routes;
      const int first_expert = args.shape.rank * kLocalExperts;
      for (int active = 0; valid && active < route.active_experts; ++active) {
        const int begin = args.routes.expert_route_offsets[active];
        const int end = args.routes.expert_route_offsets[active + 1];
        const int global = args.routes.active_global_expert_ids[active];
        valid = begin >= 0 && end >= begin && end <= route.active_routes &&
                end - begin == args.routes.expert_row_counts[active] &&
                global >= first_expert && global < first_expert + kLocalExperts;
        const int local = global - first_expert;
        valid = valid && args.experts.gate_weights[local] &&
                args.experts.gate_scale_inv[local] &&
                args.experts.up_weights[local] &&
                args.experts.up_scale_inv[local] &&
                args.experts.down_weights[local] &&
                args.experts.down_scale_inv[local];
        for (int grouped = begin; valid && grouped < end; ++grouped) {
          const int owner = args.routes.expert_route_indices[grouped];
          valid = owner >= 0 && owner < route.active_routes &&
                  args.routes.owner_route_rows[owner] >= 0 &&
                  args.routes.owner_route_rows[owner] < args.shape.rows &&
                  args.routes.owner_route_global_expert_ids[owner] == global &&
                  isfinite(args.routes.owner_route_weights[owner]) &&
                  args.routes.owner_route_weights[owner] >= 0.0F &&
                  args.routes.owner_route_weights[owner] <= 1.0F;
        }
      }
      if (valid) {
        result = {
            .generation = route.generation,
            .active_weight_bytes = route.active_weight_bytes,
            .fc1_tiles = static_cast<std::uint32_t>(route.active_routes) *
                         kFc1Tiles,
            .fc2_tiles = static_cast<std::uint32_t>(route.active_routes) *
                         kFc2Tiles,
            .active_experts = route.active_experts,
            .active_routes = route.active_routes,
            .outcome = RoutedExpertOutcome::kOk,
        };
      }
    }
    *args.buffers.summary = result;
  }
}

__global__ void gate_up_silu(KernelArgs args) {
  if (!valid_generation(args)) return;
  const int grouped_route = blockIdx.y;
  const int active_routes = args.buffers.summary->active_routes;
  if (grouped_route >= active_routes) return;
  const int active = active_expert_for_grouped_route(
      args, grouped_route, args.buffers.summary->active_experts);
  const int group_begin = args.routes.expert_route_offsets[active];
  const int group_end = args.routes.expert_route_offsets[active + 1];
  if (grouped_route < group_begin || grouped_route >= group_end ||
      group_end - group_begin != args.routes.expert_row_counts[active])
    return;
  const int owner_route = args.routes.expert_route_indices[grouped_route];
  if (owner_route < 0 || owner_route >= active_routes) return;
  const int row = args.routes.owner_route_rows[owner_route];
  if (row < 0 || row >= args.shape.rows) return;
  const int global_expert = args.routes.active_global_expert_ids[active];
  const int local_expert = global_expert - args.shape.rank * kLocalExperts;
  if (local_expert < 0 || local_expert >= kLocalExperts) return;

  const int output = blockIdx.x * blockDim.x + threadIdx.x;
  if (output >= static_cast<int>(kLogicalIntermediate)) return;
  const auto* gate = args.experts.gate_weights[local_expert];
  const auto* up = args.experts.up_weights[local_expert];
  const auto* gate_scale = args.experts.gate_scale_inv[local_expert];
  const auto* up_scale = args.experts.up_scale_inv[local_expert];
  if (!gate || !up || !gate_scale || !up_scale) return;

  float gate_sum = 0.0F;
  float up_sum = 0.0F;
  const int output_block = output / static_cast<int>(kFp8Block);
  const int weight_row = output * static_cast<int>(kHidden);
  const int hidden_row = row * static_cast<int>(kHidden);
  for (int input = 0; input < static_cast<int>(kHidden); ++input) {
    const int scale_index = output_block * 20 + input / kFp8Block;
    const float x = static_cast<float>(args.buffers.hidden[hidden_row + input]);
    gate_sum += x * fp8_value(gate, weight_row + input) *
                static_cast<float>(gate_scale[scale_index]);
    up_sum += x * fp8_value(up, weight_row + input) *
              static_cast<float>(up_scale[scale_index]);
  }
  const float silu = gate_sum / (1.0F + expf(-gate_sum));
  args.buffers.activated[
      grouped_route * static_cast<int>(kPhysicalIntermediate) + output] =
      static_cast<__nv_bfloat16>(silu * up_sum);
}

__global__ void down_and_reduce(KernelArgs args) {
  if (!valid_generation(args)) return;
  const int grouped_route = blockIdx.y;
  const int active_routes = args.buffers.summary->active_routes;
  if (grouped_route >= active_routes) return;
  const int active = active_expert_for_grouped_route(
      args, grouped_route, args.buffers.summary->active_experts);
  if (grouped_route < args.routes.expert_route_offsets[active] ||
      grouped_route >= args.routes.expert_route_offsets[active + 1])
    return;
  const int owner_route = args.routes.expert_route_indices[grouped_route];
  if (owner_route < 0 || owner_route >= active_routes) return;
  const int row = args.routes.owner_route_rows[owner_route];
  if (row < 0 || row >= args.shape.rows) return;
  const int local_expert = args.routes.active_global_expert_ids[active] -
                           args.shape.rank * kLocalExperts;
  if (local_expert < 0 || local_expert >= kLocalExperts) return;

  const int output = blockIdx.x * blockDim.x + threadIdx.x;
  if (output >= static_cast<int>(kHidden)) return;
  const auto* down = args.experts.down_weights[local_expert];
  const auto* down_scale = args.experts.down_scale_inv[local_expert];
  if (!down || !down_scale) return;
  float sum = 0.0F;
  const int output_block = output / static_cast<int>(kFp8Block);
  const int weight_row = output * static_cast<int>(kLogicalIntermediate);
  const int activation_row =
      grouped_route * static_cast<int>(kPhysicalIntermediate);
  for (int input = 0; input < static_cast<int>(kLogicalIntermediate); ++input) {
    const int scale_index = output_block * 5 + input / kFp8Block;
    sum += static_cast<float>(args.buffers.activated[activation_row + input]) *
           fp8_value(down, weight_row + input) *
           static_cast<float>(down_scale[scale_index]);
  }
  atomicAdd(args.buffers.rank_output +
                row * static_cast<int>(kHidden) + output,
            sum * args.routes.owner_route_weights[owner_route]);
}

bool bound(const RoutedExpertLaunch& launch) noexcept {
  const auto& e = launch.experts;
  const auto& b = launch.buffers;
  const auto& r = launch.routes;
  return r.active_global_expert_ids && r.expert_row_counts &&
         r.expert_route_offsets && r.owner_route_weights &&
         r.owner_route_rows && r.expert_route_indices && r.summary &&
         e.gate_weights && e.gate_scale_inv && e.up_weights &&
         e.up_scale_inv && e.down_weights && e.down_scale_inv && b.hidden &&
         b.activated && b.rank_output && b.summary;
}

}  // namespace

RoutedExpertOutcome enqueue_fp8_routed_experts(
    const RoutedExpertLaunch& launch) noexcept {
  if (!allowed_shape(launch.shape) || !allowed_capacity(launch.capacity) ||
      launch.shape.rows > launch.capacity.rows || !bound(launch) ||
      !launch.stream)
    return RoutedExpertOutcome::kContractError;
  const KernelArgs args{launch.shape, launch.capacity, launch.routes,
                        launch.experts, launch.buffers};
  prepare_consumer<<<1, 256, 0, launch.stream>>>(args);
  gate_up_silu<<<dim3(kFc1Tiles, launch.capacity.routes), kThreads, 0,
                 launch.stream>>>(args);
  down_and_reduce<<<dim3(kFc2Tiles, launch.capacity.routes), kThreads, 0,
                    launch.stream>>>(args);
  return cudaPeekAtLastError() == cudaSuccess ? RoutedExpertOutcome::kOk
                                               : RoutedExpertOutcome::kCudaError;
}

}  // namespace rocket::qwen38::moe

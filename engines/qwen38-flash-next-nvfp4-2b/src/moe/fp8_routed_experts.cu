// SPDX-License-Identifier: Apache-2.0
#include "moe/fp8_routed_experts.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifndef ROCKET_QWEN38_MTP_FP8_TRITON_DIR
#error "ROCKET_QWEN38_MTP_FP8_TRITON_DIR must name the vendored cubin directory"
#endif

namespace rocket::qwen38::moe {
namespace {

// Kernel/dataflow lineage: vLLM 8e685d198, Apache-2.0,
// model_executor/layers/fused_moe/experts/triton_moe.py and fused_moe.py.
// The pinned implementation selects block-scaled Triton FP8 on this device
// because DeepGEMM has no pure-FP8 SM121 runtime. Current upstream main at
// 869f78732b64454293d2ba42ae0008386fdaa6a6 retains the fallback. Rocket
// AOT-specializes that tl.dot path for H=2560, logical
// I=640, physical I=768, E=512/rank E=256/top10/c1-c16. Route compaction
// supplies the stable active prefix, avoiding vLLM's second sort/pad pass.
constexpr unsigned kTritonThreads = 128;
constexpr unsigned kMetadataSharedBytes = 16;
constexpr unsigned kDotSharedBytes = 16384;
constexpr int kHiddenBlocks = static_cast<int>(kHidden / kFp8Block);
constexpr int kIntermediateBlocks =
    static_cast<int>(kLogicalIntermediate / kFp8Block);

struct KernelArgs {
  RouteCompactionShape shape;
  RouteCompactionCapacity capacity;
  RouteCompactionBuffers routes;
  Fp8ExpertTables experts;
  RoutedExpertBuffers buffers;
};

__global__ void prepare_consumer(KernelArgs args) {
  for (int index = threadIdx.x;
       index < args.shape.rows * static_cast<int>(kHidden);
       index += blockDim.x) {
    args.buffers.rank_output[index] = 0.0F;
  }
  __syncthreads();

  if (threadIdx.x != 0) return;
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
              global >= first_expert &&
              global < first_expert + kLocalExperts;
      const int local = global - first_expert;
      valid = valid && args.routes.local_to_active[local] == active &&
              args.experts.gate_weights[local] &&
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
                       kIntermediateBlocks,
          .fc2_tiles = static_cast<std::uint32_t>(route.active_routes) *
                       kHiddenBlocks,
          .active_experts = route.active_experts,
          .active_routes = route.active_routes,
          .outcome = RoutedExpertOutcome::kOk,
      };
    }
  }
  *args.buffers.summary = result;
}

bool bound(const RoutedExpertLaunch& launch) noexcept {
  const auto& e = launch.experts;
  const auto& b = launch.buffers;
  const auto& r = launch.routes;
  return r.active_global_expert_ids && r.local_to_active &&
         r.expert_row_counts && r.expert_route_offsets &&
         r.owner_route_global_expert_ids && r.owner_route_weights &&
         r.owner_route_rows && r.expert_route_indices && r.summary &&
         e.gate_weights && e.gate_scale_inv && e.up_weights &&
         e.up_scale_inv && e.down_weights && e.down_scale_inv && b.hidden &&
         b.quantized_hidden && b.hidden_scale_inv && b.gate_up && b.activated &&
         b.activated_scale_inv && b.rank_output && b.summary;
}

void require_driver(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* detail = nullptr;
  cuGetErrorString(result, &detail);
  throw std::runtime_error(std::string(operation) + ": " +
                           (detail ? detail : "unknown CUDA driver error"));
}

struct ModuleFunction {
  CUmodule module{};
  CUfunction function{};
};

ModuleFunction load_kernel(const std::filesystem::path& path,
                           const char* function_name) {
  ModuleFunction loaded{};
  require_driver(cuModuleLoad(&loaded.module, path.c_str()), "cuModuleLoad");
  try {
    require_driver(cuModuleGetFunction(&loaded.function, loaded.module,
                                       function_name),
                   "cuModuleGetFunction");
  } catch (...) {
    cuModuleUnload(loaded.module);
    throw;
  }
  return loaded;
}

CUresult launch_kernel(const ModuleFunction& kernel, dim3 grid,
                       unsigned shared_bytes, cudaStream_t stream,
                       void** parameters) noexcept {
  return cuLaunchKernel(kernel.function, grid.x, grid.y, grid.z,
                        kTritonThreads, 1, 1, shared_bytes,
                        reinterpret_cast<CUstream>(stream), parameters,
                        nullptr);
}

}  // namespace

struct Fp8RoutedExperts::Impl {
  explicit Impl(int device, int rank, const char* module_directory)
      : device(device), rank(rank) {
    if (device < 0 || rank < 0 || rank > 1)
      throw std::invalid_argument("invalid routed-expert device or rank");
    const cudaError_t selected = cudaSetDevice(device);
    if (selected != cudaSuccess)
      throw std::runtime_error(std::string("cudaSetDevice: ") +
                               cudaGetErrorString(selected));
    const std::filesystem::path root = module_directory
                                           ? module_directory
                                           : ROCKET_QWEN38_MTP_FP8_TRITON_DIR;
    try {
      quantize =
          load_kernel(root / "quantize_hidden.cubin", "quantize_hidden");
      silu = load_kernel(root / "silu_mul_quantize.cubin",
                         "silu_mul_quantize");
      const auto rank_root = root / (rank == 0 ? "rank0" : "rank1");
      gate_up =
          load_kernel(rank_root / "gate_up_silu.cubin", "gate_up_silu");
      down = load_kernel(rank_root / "down_weighted_reduce.cubin",
                         "down_weighted_reduce");
    } catch (...) {
      unload();
      throw;
    }
  }

  ~Impl() { unload(); }

  void unload() noexcept {
    if (down.module) cuModuleUnload(down.module);
    if (gate_up.module) cuModuleUnload(gate_up.module);
    if (silu.module) cuModuleUnload(silu.module);
    if (quantize.module) cuModuleUnload(quantize.module);
    down = {};
    gate_up = {};
    silu = {};
    quantize = {};
  }

  int device;
  int rank;
  ModuleFunction quantize;
  ModuleFunction gate_up;
  ModuleFunction silu;
  ModuleFunction down;
};

Fp8RoutedExperts::Fp8RoutedExperts(int device, int rank,
                                   const char* module_directory)
    : impl_(std::make_unique<Impl>(device, rank, module_directory)) {}

Fp8RoutedExperts::~Fp8RoutedExperts() = default;

RoutedExpertOutcome Fp8RoutedExperts::enqueue(
    const RoutedExpertLaunch& launch) const noexcept {
  if (!impl_ || launch.shape.rank != impl_->rank ||
      !allowed_shape(launch.shape) || !allowed_capacity(launch.capacity) ||
      launch.shape.rows > launch.capacity.rows || !bound(launch) ||
      !launch.stream) {
    return RoutedExpertOutcome::kContractError;
  }
  if (cudaSetDevice(impl_->device) != cudaSuccess)
    return RoutedExpertOutcome::kCudaError;

  const KernelArgs args{launch.shape, launch.capacity, launch.routes,
                        launch.experts, launch.buffers};
  prepare_consumer<<<1, 256, 0, launch.stream>>>(args);
  if (cudaPeekAtLastError() != cudaSuccess)
    return RoutedExpertOutcome::kCudaError;

  auto* hidden = launch.buffers.hidden;
  auto* quantized_hidden = launch.buffers.quantized_hidden;
  auto* hidden_scales = launch.buffers.hidden_scale_inv;
  // Triton 3.7.1 appends global_scratch and profile_scratch pointers to its
  // raw cubin ABI even when metadata reports both scratch sizes as zero.
  CUdeviceptr global_scratch = 0;
  CUdeviceptr profile_scratch = 0;
  std::array<void*, 5> quantize_args{&hidden, &quantized_hidden,
                                     &hidden_scales, &global_scratch,
                                     &profile_scratch};
  if (launch_kernel(impl_->quantize,
                    dim3(launch.capacity.rows, kHiddenBlocks),
                    kMetadataSharedBytes, launch.stream,
                    quantize_args.data()) != CUDA_SUCCESS) {
    return RoutedExpertOutcome::kCudaError;
  }

  auto* gate_up = launch.buffers.gate_up;
  auto* active_routes = &launch.buffers.summary->active_routes;
  auto* active_ids = launch.routes.active_global_expert_ids;
  auto* local_to_active = launch.routes.local_to_active;
  auto* owner_ids = launch.routes.owner_route_global_expert_ids;
  auto* owner_rows = launch.routes.owner_route_rows;
  auto* permutation = launch.routes.expert_route_indices;
  auto* gate_tables = launch.experts.gate_weights;
  auto* gate_scales = launch.experts.gate_scale_inv;
  auto* up_tables = launch.experts.up_weights;
  auto* up_scales = launch.experts.up_scale_inv;
  std::array<void*, 15> gate_args{
      &quantized_hidden, &hidden_scales, &gate_up,       &active_routes,
      &active_ids,       &local_to_active, &owner_ids,   &owner_rows,
      &permutation,      &gate_tables,     &gate_scales, &up_tables,
      &up_scales,        &global_scratch,  &profile_scratch};
  if (launch_kernel(impl_->gate_up,
                    dim3(launch.capacity.routes, kIntermediateBlocks),
                    kDotSharedBytes, launch.stream, gate_args.data()) !=
      CUDA_SUCCESS) {
    return RoutedExpertOutcome::kCudaError;
  }

  auto* activated = launch.buffers.activated;
  auto* activated_scales = launch.buffers.activated_scale_inv;
  std::array<void*, 6> silu_args{&gate_up, &activated, &activated_scales,
                                 &active_routes, &global_scratch,
                                 &profile_scratch};
  if (launch_kernel(impl_->silu,
                    dim3(launch.capacity.routes, kIntermediateBlocks),
                    kMetadataSharedBytes, launch.stream, silu_args.data()) !=
      CUDA_SUCCESS) {
    return RoutedExpertOutcome::kCudaError;
  }

  auto* output = launch.buffers.rank_output;
  auto* owner_weights = launch.routes.owner_route_weights;
  auto* down_tables = launch.experts.down_weights;
  auto* down_scales = launch.experts.down_scale_inv;
  std::array<void*, 14> down_args{
      &activated,      &activated_scales, &output,         &active_routes,
      &active_ids,     &local_to_active,  &owner_ids,      &owner_weights,
      &owner_rows,     &permutation,      &down_tables,    &down_scales,
      &global_scratch, &profile_scratch};
  if (launch_kernel(impl_->down,
                    dim3(launch.capacity.routes, kHiddenBlocks),
                    kDotSharedBytes, launch.stream, down_args.data()) !=
      CUDA_SUCCESS) {
    return RoutedExpertOutcome::kCudaError;
  }
  return RoutedExpertOutcome::kOk;
}

}  // namespace rocket::qwen38::moe

// SPDX-License-Identifier: Apache-2.0
#include "moe/fp8_routed_experts.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#undef assert
#define assert(condition) do { if (!(condition)) std::abort(); } while (false)

namespace moe = rocket::qwen38::moe;

namespace {
template <typename T>
T* allocate(std::size_t elements) {
  T* pointer = nullptr;
  assert(cudaMalloc(&pointer, elements * sizeof(T)) == cudaSuccess);
  return pointer;
}
}

int main() {
  constexpr int kActive = 10;
  constexpr int kRows = 1;
  constexpr int kRoutes = kActive;
  constexpr std::size_t kWeightElements =
      moe::kHidden * moe::kLogicalIntermediate;
  constexpr std::size_t kScaleElements =
      (moe::kHidden / moe::kFp8Block) *
      (moe::kLogicalIntermediate / moe::kFp8Block);
  cudaStream_t stream = nullptr;
  assert(cudaStreamCreate(&stream) == cudaSuccess);

  auto* gate = allocate<std::uint8_t>(kActive * kWeightElements);
  auto* up = allocate<std::uint8_t>(kActive * kWeightElements);
  auto* down = allocate<std::uint8_t>(kActive * kWeightElements);
  auto* gate_scale = allocate<__nv_bfloat16>(kActive * kScaleElements);
  auto* up_scale = allocate<__nv_bfloat16>(kActive * kScaleElements);
  auto* down_scale = allocate<__nv_bfloat16>(kActive * kScaleElements);
  assert(cudaMemset(gate, 0, kActive * kWeightElements) == cudaSuccess);
  assert(cudaMemset(up, 0, kActive * kWeightElements) == cudaSuccess);
  assert(cudaMemset(down, 0, kActive * kWeightElements) == cudaSuccess);
  assert(cudaMemset(gate_scale, 0, kActive * kScaleElements * 2) == cudaSuccess);
  assert(cudaMemset(up_scale, 0, kActive * kScaleElements * 2) == cudaSuccess);
  assert(cudaMemset(down_scale, 0, kActive * kScaleElements * 2) == cudaSuccess);

  std::vector<const std::uint8_t*> gate_ptrs(moe::kLocalExperts, nullptr);
  std::vector<const std::uint8_t*> up_ptrs(moe::kLocalExperts, nullptr);
  std::vector<const std::uint8_t*> down_ptrs(moe::kLocalExperts, nullptr);
  std::vector<const __nv_bfloat16*> gate_scale_ptrs(moe::kLocalExperts, nullptr);
  std::vector<const __nv_bfloat16*> up_scale_ptrs(moe::kLocalExperts, nullptr);
  std::vector<const __nv_bfloat16*> down_scale_ptrs(moe::kLocalExperts, nullptr);
  for (int expert = 0; expert < kActive; ++expert) {
    gate_ptrs[expert] = gate + expert * kWeightElements;
    up_ptrs[expert] = up + expert * kWeightElements;
    down_ptrs[expert] = down + expert * kWeightElements;
    gate_scale_ptrs[expert] = gate_scale + expert * kScaleElements;
    up_scale_ptrs[expert] = up_scale + expert * kScaleElements;
    down_scale_ptrs[expert] = down_scale + expert * kScaleElements;
  }
  auto* d_gate_ptrs = allocate<const std::uint8_t*>(moe::kLocalExperts);
  auto* d_up_ptrs = allocate<const std::uint8_t*>(moe::kLocalExperts);
  auto* d_down_ptrs = allocate<const std::uint8_t*>(moe::kLocalExperts);
  auto* d_gate_scale_ptrs = allocate<const __nv_bfloat16*>(moe::kLocalExperts);
  auto* d_up_scale_ptrs = allocate<const __nv_bfloat16*>(moe::kLocalExperts);
  auto* d_down_scale_ptrs = allocate<const __nv_bfloat16*>(moe::kLocalExperts);
  assert(cudaMemcpy(d_gate_ptrs, gate_ptrs.data(), gate_ptrs.size() * 8,
                    cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMemcpy(d_up_ptrs, up_ptrs.data(), up_ptrs.size() * 8,
                    cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMemcpy(d_down_ptrs, down_ptrs.data(), down_ptrs.size() * 8,
                    cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMemcpy(d_gate_scale_ptrs, gate_scale_ptrs.data(),
                    gate_scale_ptrs.size() * 8,
                    cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMemcpy(d_up_scale_ptrs, up_scale_ptrs.data(),
                    up_scale_ptrs.size() * 8,
                    cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMemcpy(d_down_scale_ptrs, down_scale_ptrs.data(),
                    down_scale_ptrs.size() * 8,
                    cudaMemcpyHostToDevice) == cudaSuccess);

  std::array<std::int32_t, kRoutes> route_ids{};
  std::array<float, kRoutes> route_weights{};
  for (int route = 0; route < kRoutes; ++route) {
    route_ids[route] = route;
    route_weights[route] = 0.1F;
  }
  auto* d_route_ids = allocate<std::int32_t>(kRoutes);
  auto* d_route_weights = allocate<float>(kRoutes);
  auto* generation = allocate<std::uint64_t>(1);
  const std::uint64_t one = 1;
  assert(cudaMemcpy(d_route_ids, route_ids.data(), sizeof(route_ids),
                    cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMemcpy(d_route_weights, route_weights.data(), sizeof(route_weights),
                    cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMemcpy(generation, &one, sizeof(one), cudaMemcpyHostToDevice) ==
         cudaSuccess);

  moe::RouteCompactionBuffers routes{
      .active_global_expert_ids = allocate<std::int32_t>(kActive),
      .local_to_active = allocate<std::int32_t>(moe::kLocalExperts),
      .expert_row_counts = allocate<std::int32_t>(kActive),
      .expert_route_offsets = allocate<std::int32_t>(kActive + 1),
      .expert_route_cursors = allocate<std::int32_t>(kActive),
      .owner_route_global_expert_ids = allocate<std::int32_t>(kRoutes),
      .owner_route_weights = allocate<float>(kRoutes),
      .owner_route_rows = allocate<std::int32_t>(kRoutes),
      .owner_route_slots = allocate<std::uint8_t>(kRoutes),
      .expert_route_indices = allocate<std::int32_t>(kRoutes),
      .summary = allocate<moe::RouteCompactionDeviceSummary>(1),
  };
  const moe::RouteCompactionShape shape{0, kRows, kRows};
  const moe::RouteCompactionCapacity capacity{kActive, kRows, kRoutes};
  assert(moe::enqueue_route_compaction({
             shape, capacity, {d_route_ids, d_route_weights, generation, generation},
             routes, stream}) == moe::RouteCompactionOutcome::kOk);

  auto* hidden = allocate<__nv_bfloat16>(moe::kHidden);
  auto* quantized_hidden = allocate<std::uint8_t>(moe::kHidden);
  auto* hidden_scale = allocate<float>(moe::kHidden / moe::kFp8Block);
  auto* gate_up = allocate<__nv_bfloat16>(
      kRoutes * 2 * moe::kPhysicalIntermediate);
  auto* activated = allocate<std::uint8_t>(
      kRoutes * moe::kPhysicalIntermediate);
  auto* activated_scale = allocate<float>(
      kRoutes * moe::kLogicalIntermediate / moe::kFp8Block);
  auto* output = allocate<float>(moe::kHidden);
  auto* summary = allocate<moe::RoutedExpertDeviceSummary>(1);
  assert(cudaMemset(hidden, 0, moe::kHidden * 2) == cudaSuccess);
  moe::Fp8RoutedExperts consumer(0, 0);
  assert(consumer.enqueue({
             shape, capacity, routes,
             {d_gate_ptrs, d_gate_scale_ptrs, d_up_ptrs, d_up_scale_ptrs,
              d_down_ptrs, d_down_scale_ptrs},
             {hidden, quantized_hidden, hidden_scale, gate_up, activated,
              activated_scale, output, summary}, stream}) ==
         moe::RoutedExpertOutcome::kOk);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  moe::RoutedExpertDeviceSummary snapshot{};
  assert(cudaMemcpy(&snapshot, summary, sizeof(snapshot), cudaMemcpyDeviceToHost) ==
         cudaSuccess);
  assert(moe::validate_routed_expert_summary({snapshot, 1, shape}) ==
         moe::RoutedExpertOutcome::kOk);
  std::vector<float> host_output(moe::kHidden, 1.0F);
  assert(cudaMemcpy(host_output.data(), output, moe::kHidden * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess);
  for (float value : host_output) assert(value == 0.0F);

  cudaFree(summary); cudaFree(output); cudaFree(activated_scale);
  cudaFree(activated); cudaFree(gate_up); cudaFree(hidden_scale);
  cudaFree(quantized_hidden); cudaFree(hidden);
  cudaFree(routes.summary); cudaFree(routes.expert_route_indices);
  cudaFree(routes.owner_route_slots); cudaFree(routes.owner_route_rows);
  cudaFree(routes.owner_route_weights); cudaFree(routes.owner_route_global_expert_ids);
  cudaFree(routes.expert_route_cursors); cudaFree(routes.expert_route_offsets);
  cudaFree(routes.expert_row_counts); cudaFree(routes.local_to_active);
  cudaFree(routes.active_global_expert_ids); cudaFree(generation);
  cudaFree(d_route_weights); cudaFree(d_route_ids); cudaFree(d_down_scale_ptrs);
  cudaFree(d_up_scale_ptrs); cudaFree(d_gate_scale_ptrs); cudaFree(d_down_ptrs);
  cudaFree(d_up_ptrs); cudaFree(d_gate_ptrs); cudaFree(down_scale);
  cudaFree(up_scale); cudaFree(gate_scale); cudaFree(down); cudaFree(up);
  cudaFree(gate); cudaStreamDestroy(stream);
  return 0;
}

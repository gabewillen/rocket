// SPDX-License-Identifier: Apache-2.0
#include "linear_attention/gdn_prefill_fused.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace prefill = rocket::qwen38::linear_attention::prefill;

namespace {

void check(bool condition) {
  if (!condition) {
    std::abort();
  }
}

template <class T>
T* allocate(std::size_t elements) {
  T* pointer = nullptr;
  check(cudaMalloc(&pointer, elements * sizeof(T)) == cudaSuccess);
  return pointer;
}

struct Buffers {
  __nv_bfloat16* q;
  __nv_bfloat16* k;
  __nv_bfloat16* v;
  __nv_bfloat16* state;
  float* g;
  float* beta;
};

Buffers make_buffers(int tokens) {
  return {
      allocate<__nv_bfloat16>(tokens * prefill::kKeyHeads * prefill::kHeadDim),
      allocate<__nv_bfloat16>(tokens * prefill::kKeyHeads * prefill::kHeadDim),
      allocate<__nv_bfloat16>(tokens * prefill::kValueHeads * prefill::kHeadDim),
      allocate<__nv_bfloat16>(prefill::kQkvWidth * prefill::kConvStateWidth),
      allocate<float>(tokens * prefill::kValueHeads),
      allocate<float>(tokens * prefill::kValueHeads),
  };
}

void release(Buffers buffers) {
  cudaFree(buffers.beta);
  cudaFree(buffers.g);
  cudaFree(buffers.state);
  cudaFree(buffers.v);
  cudaFree(buffers.k);
  cudaFree(buffers.q);
}

bool equal_device_bytes(const void* left, const void* right, std::size_t bytes) {
  std::vector<std::byte> left_host(bytes);
  std::vector<std::byte> right_host(bytes);
  check(cudaMemcpy(left_host.data(), left, bytes, cudaMemcpyDeviceToHost) ==
        cudaSuccess);
  check(cudaMemcpy(right_host.data(), right, bytes, cudaMemcpyDeviceToHost) ==
        cudaSuccess);
  return left_host == right_host;
}

}  // namespace

int main() {
  constexpr int kTokens = 5;
  const std::size_t qkv_elements = kTokens * prefill::kQkvWidth;
  std::vector<__nv_bfloat16> qkv(qkv_elements);
  std::vector<__nv_bfloat16> ba(kTokens * prefill::kGateWidth);
  std::vector<__nv_bfloat16> weight(prefill::kQkvWidth * prefill::kConvWidth);
  std::vector<__nv_bfloat16> initial(prefill::kQkvWidth *
                                     prefill::kConvStateWidth);
  std::vector<__nv_bfloat16> scales(prefill::kValueHeads);
  for (std::size_t index = 0; index < qkv.size(); ++index) {
    qkv[index] =
        __float2bfloat16((static_cast<int>(index % 31) - 15) / 32.0F);
  }
  for (std::size_t index = 0; index < ba.size(); ++index) {
    ba[index] = __float2bfloat16((static_cast<int>(index % 9) - 4) / 16.0F);
  }
  for (std::size_t index = 0; index < weight.size(); ++index) {
    weight[index] =
        __float2bfloat16((static_cast<int>(index % 7) - 3) / 16.0F);
  }
  for (std::size_t index = 0; index < initial.size(); ++index) {
    initial[index] =
        __float2bfloat16((static_cast<int>(index % 13) - 6) / 16.0F);
  }
  std::fill(scales.begin(), scales.end(), __float2bfloat16(0.0F));

  auto* device_qkv = allocate<__nv_bfloat16>(qkv.size());
  auto* device_ba = allocate<__nv_bfloat16>(ba.size());
  auto* device_weight = allocate<__nv_bfloat16>(weight.size());
  auto* device_initial = allocate<__nv_bfloat16>(initial.size());
  auto* device_scales = allocate<__nv_bfloat16>(scales.size());
  auto* materialized = allocate<__nv_bfloat16>(qkv_elements);
  check(cudaMemcpy(device_qkv, qkv.data(), qkv.size() * sizeof(qkv.front()),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  check(cudaMemcpy(device_ba, ba.data(), ba.size() * sizeof(ba.front()),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  check(cudaMemcpy(device_weight, weight.data(),
                   weight.size() * sizeof(weight.front()),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  check(cudaMemcpy(device_initial, initial.data(),
                   initial.size() * sizeof(initial.front()),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  check(cudaMemcpy(device_scales, scales.data(),
                   scales.size() * sizeof(scales.front()),
                   cudaMemcpyHostToDevice) == cudaSuccess);

  Buffers fused = make_buffers(kTokens);
  Buffers reference = make_buffers(kTokens);
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream) == cudaSuccess);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) ==
        cudaSuccess);
  check(prefill::launch_fused(
            device_qkv, device_ba, device_weight, device_initial,
            device_scales, device_scales,
            {fused.q, fused.k, fused.v, fused.g, fused.beta, fused.state},
            kTokens, stream) == cudaSuccess);
  check(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  check(cudaGraphInstantiate(&executable, graph, 0) == cudaSuccess);
  check(cudaGraphLaunch(executable, stream) == cudaSuccess);
  check(prefill::launch_materialized_reference(
            device_qkv, device_ba, device_weight, device_initial,
            device_scales, device_scales, materialized,
            {reference.q, reference.k, reference.v, reference.g,
             reference.beta, reference.state},
            kTokens, stream) == cudaSuccess);
  check(cudaStreamSynchronize(stream) == cudaSuccess);

  check(equal_device_bytes(
      fused.q, reference.q,
      kTokens * prefill::kKeyHeads * prefill::kHeadDim * sizeof(*fused.q)));
  check(equal_device_bytes(
      fused.k, reference.k,
      kTokens * prefill::kKeyHeads * prefill::kHeadDim * sizeof(*fused.k)));
  check(equal_device_bytes(
      fused.v, reference.v,
      kTokens * prefill::kValueHeads * prefill::kHeadDim * sizeof(*fused.v)));
  check(equal_device_bytes(
      fused.g, reference.g,
      kTokens * prefill::kValueHeads * sizeof(*fused.g)));
  check(equal_device_bytes(
      fused.beta, reference.beta,
      kTokens * prefill::kValueHeads * sizeof(*fused.beta)));
  check(equal_device_bytes(fused.state, reference.state,
                           prefill::kQkvWidth * prefill::kConvStateWidth *
                               sizeof(*fused.state)));

  std::vector<__nv_bfloat16> final_state(prefill::kQkvWidth *
                                         prefill::kConvStateWidth);
  check(cudaMemcpy(final_state.data(), fused.state,
                   final_state.size() * sizeof(final_state.front()),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int channel = 0; channel < prefill::kQkvWidth; ++channel) {
    for (int slot = 0; slot < prefill::kConvStateWidth; ++slot) {
      check(__bfloat162float(
                final_state[channel * prefill::kConvStateWidth + slot]) ==
            __bfloat162float(qkv[(kTokens - prefill::kConvStateWidth + slot) *
                                     prefill::kQkvWidth +
                                 channel]));
    }
  }
  check(prefill::eliminated_intermediate_bytes(300) == 6'144'000);
  check(prefill::eliminated_intermediate_bytes(8192) == 167'772'160);

  cudaGraphExecDestroy(executable);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  release(reference);
  release(fused);
  cudaFree(materialized);
  cudaFree(device_scales);
  cudaFree(device_initial);
  cudaFree(device_weight);
  cudaFree(device_ba);
  cudaFree(device_qkv);
  return 0;
}

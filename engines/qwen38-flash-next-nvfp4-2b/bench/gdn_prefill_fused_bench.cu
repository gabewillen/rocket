// SPDX-License-Identifier: Apache-2.0
#include "linear_attention/gdn_prefill_fused.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace prefill = rocket::qwen38::linear_attention::prefill;

namespace {

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}

template <class T>
T* allocate(std::size_t elements) {
  T* pointer = nullptr;
  check(cudaMalloc(&pointer, elements * sizeof(T)));
  return pointer;
}

__global__ void fill_bf16(__nv_bfloat16* output, std::size_t elements,
                          int modulus) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __float2bfloat16(
        static_cast<float>(static_cast<int>(index % modulus) - modulus / 2) /
        static_cast<float>(modulus));
  }
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

cudaGraphExec_t capture(bool fused, const __nv_bfloat16* qkv,
                        const __nv_bfloat16* ba,
                        const __nv_bfloat16* weight,
                        const __nv_bfloat16* initial,
                        const __nv_bfloat16* scales,
                        __nv_bfloat16* materialized, Buffers outputs,
                        int tokens, cudaStream_t stream) {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  const prefill::Outputs output_view{outputs.q, outputs.k, outputs.v, outputs.g,
                                     outputs.beta, outputs.state};
  check(fused ? prefill::launch_fused(qkv, ba, weight, initial, scales, scales,
                                      output_view, tokens, stream)
              : prefill::launch_materialized_reference(
                    qkv, ba, weight, initial, scales, scales, materialized,
                    output_view, tokens, stream));
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&executable, graph, 0));
  check(cudaGraphDestroy(graph));
  return executable;
}

std::pair<float, float> measure(cudaGraphExec_t graph, cudaStream_t stream) {
  for (int iteration = 0; iteration < 10; ++iteration) {
    check(cudaGraphLaunch(graph, stream));
  }
  check(cudaStreamSynchronize(stream));
  std::vector<float> samples;
  samples.reserve(50);
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check(cudaEventCreate(&begin));
  check(cudaEventCreate(&end));
  for (int iteration = 0; iteration < 50; ++iteration) {
    check(cudaEventRecord(begin, stream));
    check(cudaGraphLaunch(graph, stream));
    check(cudaEventRecord(end, stream));
    check(cudaEventSynchronize(end));
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, begin, end));
    samples.push_back(milliseconds * 1000.0F);
  }
  check(cudaEventDestroy(end));
  check(cudaEventDestroy(begin));
  std::sort(samples.begin(), samples.end());
  return {samples[24], samples[47]};
}

bool equal_bytes(const void* left, const void* right, std::size_t bytes) {
  std::vector<std::byte> left_host(bytes);
  std::vector<std::byte> right_host(bytes);
  check(cudaMemcpy(left_host.data(), left, bytes, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(right_host.data(), right, bytes, cudaMemcpyDeviceToHost));
  return left_host == right_host;
}

void run(int tokens) {
  const std::size_t qkv_elements =
      static_cast<std::size_t>(tokens) * prefill::kQkvWidth;
  const std::size_t ba_elements =
      static_cast<std::size_t>(tokens) * prefill::kGateWidth;
  auto* qkv = allocate<__nv_bfloat16>(qkv_elements);
  auto* ba = allocate<__nv_bfloat16>(ba_elements);
  auto* weight =
      allocate<__nv_bfloat16>(prefill::kQkvWidth * prefill::kConvWidth);
  auto* initial = allocate<__nv_bfloat16>(prefill::kQkvWidth *
                                          prefill::kConvStateWidth);
  auto* scales = allocate<__nv_bfloat16>(prefill::kValueHeads);
  auto* materialized = allocate<__nv_bfloat16>(qkv_elements);
  fill_bf16<<<(qkv_elements + 255) / 256, 256>>>(qkv, qkv_elements, 31);
  fill_bf16<<<(ba_elements + 255) / 256, 256>>>(ba, ba_elements, 9);
  fill_bf16<<<(prefill::kQkvWidth * prefill::kConvWidth + 255) / 256, 256>>>(
      weight, prefill::kQkvWidth * prefill::kConvWidth, 7);
  fill_bf16<<<(prefill::kQkvWidth * prefill::kConvStateWidth + 255) / 256,
               256>>>(initial,
                      prefill::kQkvWidth * prefill::kConvStateWidth, 13);
  fill_bf16<<<1, 32>>>(scales, prefill::kValueHeads, 11);
  check(cudaDeviceSynchronize());

  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream));
  Buffers fused_outputs = make_buffers(tokens);
  Buffers materialized_outputs = make_buffers(tokens);
  cudaGraphExec_t fused =
      capture(true, qkv, ba, weight, initial, scales, materialized,
              fused_outputs, tokens, stream);
  cudaGraphExec_t reference =
      capture(false, qkv, ba, weight, initial, scales, materialized,
              materialized_outputs, tokens, stream);
  const auto [fused_p50, fused_p95] = measure(fused, stream);
  const auto [reference_p50, reference_p95] = measure(reference, stream);
  check(cudaGraphLaunch(fused, stream));
  check(cudaGraphLaunch(reference, stream));
  check(cudaStreamSynchronize(stream));

  const bool output_equal =
      equal_bytes(fused_outputs.q, materialized_outputs.q,
                  tokens * prefill::kKeyHeads * prefill::kHeadDim * 2ULL) &&
      equal_bytes(fused_outputs.k, materialized_outputs.k,
                  tokens * prefill::kKeyHeads * prefill::kHeadDim * 2ULL) &&
      equal_bytes(fused_outputs.v, materialized_outputs.v,
                  tokens * prefill::kValueHeads * prefill::kHeadDim * 2ULL) &&
      equal_bytes(fused_outputs.g, materialized_outputs.g,
                  tokens * prefill::kValueHeads * sizeof(float)) &&
      equal_bytes(fused_outputs.beta, materialized_outputs.beta,
                  tokens * prefill::kValueHeads * sizeof(float));
  const bool state_equal = equal_bytes(
      fused_outputs.state, materialized_outputs.state,
      prefill::kQkvWidth * prefill::kConvStateWidth * 2ULL);
  std::printf(
      "tokens=%d materialized_p50_us=%.3f materialized_p95_us=%.3f "
      "fused_p50_us=%.3f fused_p95_us=%.3f "
      "materialized_intermediate_bytes=%llu fused_intermediate_bytes=0 "
      "output_max_error=%s state_max_error=%s graph_capture=pass\n",
      tokens, reference_p50, reference_p95, fused_p50, fused_p95,
      static_cast<unsigned long long>(
          prefill::eliminated_intermediate_bytes(tokens)),
      output_equal ? "0" : "nonzero", state_equal ? "0" : "nonzero");

  cudaGraphExecDestroy(reference);
  cudaGraphExecDestroy(fused);
  release(materialized_outputs);
  release(fused_outputs);
  cudaStreamDestroy(stream);
  cudaFree(materialized);
  cudaFree(scales);
  cudaFree(initial);
  cudaFree(weight);
  cudaFree(ba);
  cudaFree(qkv);
  if (!output_equal || !state_equal) {
    std::exit(2);
  }
}

}  // namespace

int main() {
  run(300);
  run(8192);
  return 0;
}

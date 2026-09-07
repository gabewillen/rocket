// SPDX-License-Identifier: Apache-2.0
// Grouped normalization follows the Apache-2.0 vLLM Qwen3.8 HC kernel.
// Argmax reduction follows the pair-with-index structure used by FlashInfer
// sampling and TensorRT-LLM top-k kernels, specialized here to greedy TP2.
#include "output/token_output.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace rocket::qwen38::output {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ Winner better(Winner a, Winner b) {
  if (b.token < 0) return a;
  if (a.token < 0 || b.value > a.value ||
      (b.value == a.value && b.token < a.token))
    return b;
  return a;
}

__global__ void embedding_kernel(const std::int32_t* token_ids,
                                 const __nv_bfloat16* weight,
                                 __nv_bfloat16* output,
                                 std::int32_t* invalid_token, int rank) {
  const int row = blockIdx.x;
  const int token = token_ids[row];
  const bool valid = token >= 0 && token < kVocab;
  if (!valid && threadIdx.x == 0) atomicExch(invalid_token, 1);
  const int local = token - rank * kLocalVocab;
  const bool owned = valid && local >= 0 && local < kLocalVocab;
  for (int column = threadIdx.x; column < kHidden; column += blockDim.x) {
    output[row * kHidden + column] =
        owned ? weight[static_cast<std::int64_t>(local) * kHidden + column]
              : __float2bfloat16(0.0F);
  }
}

__device__ float block_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  __shared__ float warps[8];
  if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = value;
  __syncthreads();
  value = threadIdx.x < 8 ? warps[threadIdx.x] : 0.0F;
  if (threadIdx.x < 32) {
    for (int offset = 16; offset > 0; offset >>= 1)
      value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  if (threadIdx.x == 0) warps[0] = value;
  __syncthreads();
  return warps[0];
}

__global__ void grouped_norm_kernel(const __nv_bfloat16* input,
                                    const __nv_bfloat16* weight,
                                    __nv_bfloat16* output) {
  const int row_group = blockIdx.x;
  const int row = row_group / kHyperConnections;
  const int group = row_group % kHyperConnections;
  const int base = row * kHyperHidden + group * kHidden;
  float square_sum = 0.0F;
  for (int column = threadIdx.x; column < kHidden; column += blockDim.x) {
    const float value = __bfloat162float(input[base + column]);
    square_sum = fmaf(value, value, square_sum);
  }
  const float inverse_rms =
      rsqrtf(block_sum(square_sum) / static_cast<float>(kHidden) + kRmsEpsilon);
  for (int column = threadIdx.x; column < kHidden; column += blockDim.x) {
    const float value = __bfloat162float(input[base + column]);
    const float affine = 1.0F + __bfloat162float(weight[group * kHidden + column]);
    output[base + column] = __float2bfloat16(value * inverse_rms * affine);
  }
}

// Adapted from myllmbox-runner's Apache-2.0 mbx_vocab_gemv.py, itself
// attributed to b12x. The reference is TP1 [248320,2560] and writes BF16.
// This pinned engine is TP2 [124160,2560] and retains FP32 logits so greedy
// ordering is not perturbed by an extra narrowing conversion.
__global__ __launch_bounds__(kThreads) void vocab_gemv_kernel(
    const __nv_bfloat16* hidden, const __nv_bfloat16* weight, float* logits) {
  const int vocab_row = blockIdx.x;
  const __nv_bfloat16* row = weight + static_cast<std::int64_t>(vocab_row) * kHidden;
  float accumulator = 0.0F;
  // Five BF16x2 loads per lane cover K=2560. This is the CUDA equivalent of
  // the reference's BLOCK_K=1024 loop with eight warps.
  for (int column = threadIdx.x * 2; column < kHidden; column += kThreads * 2) {
    const __nv_bfloat162 x = __halves2bfloat162(hidden[column], hidden[column + 1]);
    const __nv_bfloat162 w = __halves2bfloat162(row[column], row[column + 1]);
    const float2 xf = __bfloat1622float2(x);
    const float2 wf = __bfloat1622float2(w);
    accumulator = fmaf(xf.x, wf.x, accumulator);
    accumulator = fmaf(xf.y, wf.y, accumulator);
  }
  const float sum = block_sum(accumulator);
  if (threadIdx.x == 0) logits[vocab_row] = sum;
}

__global__ void argmax_kernel(const float* logits, Winner* winners, int rank) {
  const int row = blockIdx.x;
  Winner candidate{-INFINITY, -1};
  for (int local = threadIdx.x; local < kLocalVocab; local += blockDim.x) {
    const float value = logits[static_cast<std::int64_t>(row) * kLocalVocab + local];
    if (isfinite(value)) candidate = better(candidate, {value, rank * kLocalVocab + local});
  }
  __shared__ Winner shared[kThreads];
  shared[threadIdx.x] = candidate;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width)
      shared[threadIdx.x] = better(shared[threadIdx.x], shared[threadIdx.x + width]);
    __syncthreads();
  }
  if (threadIdx.x == 0) winners[row] = shared[0];
}

__global__ void global_greedy_kernel(const Winner* rank_winners,
                                     std::int32_t* tokens, int m) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m) return;
  const Winner a = rank_winners[row * kTpSize];
  const Winner b = rank_winners[row * kTpSize + 1];
  if (a.token < 0 || a.token >= kLocalVocab || !isfinite(a.value) ||
      b.token < kLocalVocab || b.token >= kVocab || !isfinite(b.value)) {
    tokens[row] = -1;
    return;
  }
  tokens[row] = better(a, b).token;
}

bool valid(const void* a, const void* b, const void* c, int m) {
  return a != nullptr && b != nullptr && c != nullptr && allowed_m(m);
}

}  // namespace

cudaError_t embedding_lookup_rank(const std::int32_t* token_ids,
                                  const __nv_bfloat16* rank_weight,
                                  __nv_bfloat16* rank_output,
                                  std::int32_t* invalid_token, int m, int rank,
                                  cudaStream_t stream) noexcept {
  if (!valid(token_ids, rank_weight, rank_output, m) || invalid_token == nullptr ||
      !allowed_rank(rank))
    return cudaErrorInvalidValue;
  embedding_kernel<<<m, kThreads, 0, stream>>>(token_ids, rank_weight, rank_output,
                                               invalid_token, rank);
  return cudaPeekAtLastError();
}

cudaError_t final_grouped_rms_norm(const __nv_bfloat16* input,
                                   const __nv_bfloat16* weight,
                                   __nv_bfloat16* output, int m,
                                   cudaStream_t stream) noexcept {
  if (!valid(input, weight, output, m)) return cudaErrorInvalidValue;
  grouped_norm_kernel<<<m * kHyperConnections, kThreads, 0, stream>>>(input, weight,
                                                                      output);
  return cudaPeekAtLastError();
}

cublasStatus_t lm_head(cublasHandle_t handle, const __nv_bfloat16* hidden,
                       const __nv_bfloat16* rank_weight, float* rank_logits,
                       int m, int rank, cudaStream_t stream) noexcept {
  if (handle == nullptr || !valid(hidden, rank_weight, rank_logits, m) ||
      !allowed_rank(rank))
    return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 1) {
    vocab_gemv_kernel<<<kLocalVocab, kThreads, 0, stream>>>(hidden, rank_weight,
                                                            rank_logits);
    return cudaPeekAtLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS
                                                : CUBLAS_STATUS_EXECUTION_FAILED;
  }
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  // Row-major [M,K] x [V,K]^T is column-major [V,K] x [K,M].
  return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kLocalVocab, m, kHidden,
                      &alpha, rank_weight, CUDA_R_16BF, kHidden, hidden,
                      CUDA_R_16BF, kHidden, &beta, rank_logits, CUDA_R_32F,
                      kLocalVocab, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

cudaError_t local_argmax(const float* rank_logits, Winner* winners, int m,
                         int rank, cudaStream_t stream) noexcept {
  if (rank_logits == nullptr || winners == nullptr || !allowed_m(m) ||
      !allowed_rank(rank))
    return cudaErrorInvalidValue;
  argmax_kernel<<<m, kThreads, 0, stream>>>(rank_logits, winners, rank);
  return cudaPeekAtLastError();
}

cudaError_t global_greedy(const Winner* rank_winners, std::int32_t* tokens,
                          int m, float temperature, float top_p,
                          cudaStream_t stream) noexcept {
  if (rank_winners == nullptr || tokens == nullptr || !allowed_m(m) ||
      !sampling_supported(temperature, top_p))
    return cudaErrorInvalidValue;
  constexpr int threads = 32;
  global_greedy_kernel<<<(m + threads - 1) / threads, threads, 0, stream>>>(
      rank_winners, tokens, m);
  return cudaPeekAtLastError();
}

}  // namespace rocket::qwen38::output

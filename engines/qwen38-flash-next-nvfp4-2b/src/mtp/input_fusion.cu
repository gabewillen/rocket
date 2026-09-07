// SPDX-License-Identifier: Apache-2.0
// Specialized from the pinned Qwen3.8 Flash-Next MTP pre-FC topology.
#include "mtp/input_fusion.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "hyperconnection/hyperconnection.h"

namespace rocket::qwen38::mtp {
namespace {

constexpr int kThreads = 256;
constexpr float kEpsilon = 1.0e-6F;

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("MTP input fusion ") + operation +
                             ": " + cudaGetErrorString(status));
}

void cublas_check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("MTP input fusion ") + operation +
                             " failed: " + std::to_string(status));
}

__device__ float block_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  __shared__ float warp_sums[8];
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = value;
  __syncthreads();
  value = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
  if (threadIdx.x < 32)
    for (int offset = 16; offset > 0; offset >>= 1)
      value += __shfl_down_sync(0xffffffffU, value, offset);
  if (threadIdx.x == 0) warp_sums[0] = value;
  __syncthreads();
  return warp_sums[0];
}

__global__ void gemma_norm(const __nv_bfloat16* input,
                           const __nv_bfloat16* weight,
                           __nv_bfloat16* output, int width) {
  const int row = blockIdx.x;
  const int base = row * width;
  float sum = 0.0F;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(input[base + column]);
    sum = fmaf(value, value, sum);
  }
  const float inverse = rsqrtf(block_sum(sum) / width + kEpsilon);
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(input[base + column]) * inverse;
    const float affine = __bfloat162float(weight[column]);
    output[base + column] = __float2bfloat16(fmaf(value, affine, value));
  }
}

__global__ void scatter_rank(const __nv_bfloat16* local,
                             __nv_bfloat16* partial, int rows, int rank) {
  const int row = blockIdx.x;
  for (int column = threadIdx.x; column < kFusionHidden;
       column += blockDim.x) {
    const int local_column = column - rank * kFusionLocalHidden;
    partial[row * kFusionHidden + column] =
        local_column >= 0 && local_column < kFusionLocalHidden
            ? local[row * kFusionLocalHidden + local_column]
            : __float2bfloat16(0.0F);
  }
}

__global__ void add_reduced(const float* embedding, const float* hidden,
                            __nv_bfloat16* output) {
  const int row_stream = blockIdx.x;
  const int row = row_stream / kFusionStreams;
  for (int column = threadIdx.x; column < kFusionHidden;
       column += blockDim.x) {
    const float value = hidden[row_stream * kFusionHidden + column] +
                        embedding[row * kFusionHidden + column];
    output[row_stream * kFusionHidden + column] = __float2bfloat16(value);
  }
}

void gemm(cublasHandle_t handle, const __nv_bfloat16* input,
          const __nv_bfloat16* weight, __nv_bfloat16* output, int m) {
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  cublas_check(cublasGemmEx(
                   handle, CUBLAS_OP_T, CUBLAS_OP_N, kFusionLocalHidden, m,
                   kFusionHidden, &alpha, weight, CUDA_R_16BF, kFusionHidden,
                   input, CUDA_R_16BF, kFusionHidden, &beta, output,
                   CUDA_R_16BF, kFusionLocalHidden, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "projection GEMM");
}

}  // namespace

struct InputFusionPlan::Impl {
  int rank = 0;
  InputFusionWeights weights{};
  cublasHandle_t handle = nullptr;
  __nv_bfloat16* normalized_embedding = nullptr;
  __nv_bfloat16* normalized_hidden = nullptr;
  __nv_bfloat16* local_embedding = nullptr;
  __nv_bfloat16* local_hidden = nullptr;

  ~Impl() {
    cudaFree(local_hidden);
    cudaFree(local_embedding);
    cudaFree(normalized_hidden);
    cudaFree(normalized_embedding);
    if (handle) cublasDestroy(handle);
  }
};

InputFusionPlan::InputFusionPlan(int device, int rank,
                                 InputFusionWeights weights)
    : impl_(new Impl) {
  if (device < 0 || (rank != 0 && rank != 1) || !weights.embedding_norm ||
      !weights.hidden_norm || !weights.embedding_projection ||
      !weights.hidden_projection)
    throw std::invalid_argument("MTP input fusion binding contract drift");
  try {
    cuda_check(cudaSetDevice(device), "set device");
    impl_->rank = rank;
    impl_->weights = weights;
    cublas_check(cublasCreate(&impl_->handle), "create cuBLAS handle");
    cuda_check(cudaMalloc(&impl_->normalized_embedding,
                          16 * kFusionHidden * sizeof(__nv_bfloat16)),
               "allocate normalized embedding");
    cuda_check(cudaMalloc(&impl_->normalized_hidden,
                          16 * kFusionHyperHidden * sizeof(__nv_bfloat16)),
               "allocate normalized hidden");
    cuda_check(cudaMalloc(&impl_->local_embedding,
                          16 * kFusionLocalHidden * sizeof(__nv_bfloat16)),
               "allocate local embedding");
    cuda_check(cudaMalloc(&impl_->local_hidden,
                          16 * kFusionStreams * kFusionLocalHidden *
                              sizeof(__nv_bfloat16)),
               "allocate local hidden");
  } catch (...) {
    delete impl_;
    impl_ = nullptr;
    throw;
  }
}

InputFusionPlan::~InputFusionPlan() { delete impl_; }

void InputFusionPlan::local_project(
    const __nv_bfloat16* embedding, const __nv_bfloat16* multi_hidden,
    __nv_bfloat16* embedding_partial, __nv_bfloat16* hidden_partial, int m,
    cudaStream_t stream) {
  if (!embedding || !multi_hidden || !embedding_partial || !hidden_partial ||
      !hyperconnection::allowed_m(m) || !stream)
    throw std::invalid_argument("MTP input fusion local projection drift");
  cublas_check(cublasSetStream(impl_->handle, stream), "bind stream");
  gemma_norm<<<m, kThreads, 0, stream>>>(
      embedding, impl_->weights.embedding_norm, impl_->normalized_embedding,
      kFusionHidden);
  gemma_norm<<<m, kThreads, 0, stream>>>(
      multi_hidden, impl_->weights.hidden_norm, impl_->normalized_hidden,
      kFusionHyperHidden);
  gemm(impl_->handle, impl_->normalized_embedding,
       impl_->weights.embedding_projection, impl_->local_embedding, m);
  gemm(impl_->handle, impl_->normalized_hidden,
       impl_->weights.hidden_projection, impl_->local_hidden,
       m * kFusionStreams);
  scatter_rank<<<m, kThreads, 0, stream>>>(
      impl_->local_embedding, embedding_partial, m, impl_->rank);
  scatter_rank<<<m * kFusionStreams, kThreads, 0, stream>>>(
      impl_->local_hidden, hidden_partial, m * kFusionStreams, impl_->rank);
  cuda_check(cudaPeekAtLastError(), "launch local projection");
}

void InputFusionPlan::finish(const float* reduced_embedding,
                             const float* reduced_hidden,
                             __nv_bfloat16* fused_multi_hidden, int m,
                             cudaStream_t stream) {
  if (!reduced_embedding || !reduced_hidden || !fused_multi_hidden ||
      !hyperconnection::allowed_m(m) || !stream)
    throw std::invalid_argument("MTP input fusion finish drift");
  add_reduced<<<m * kFusionStreams, kThreads, 0, stream>>>(
      reduced_embedding, reduced_hidden, fused_multi_hidden);
  cuda_check(cudaPeekAtLastError(), "launch reduced fusion");
}

}  // namespace rocket::qwen38::mtp

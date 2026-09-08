// SPDX-License-Identifier: Apache-2.0
// Structure follows vLLM 8e685d198
// vllm/models/qwen3_8_flash_next/nvidia/ops/hc.py (Apache-2.0).
// Shapes and dispatch are fixed to Qwen3.8 TP2 c1/c2/c4/c8/c16.
#include "hyperconnection/hyperconnection.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace rocket::qwen38::hyperconnection {
namespace {

constexpr int kThreads = 256;

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("Qwen HC ") + operation + ": " +
                             cudaGetErrorString(status));
}
void cublas_check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("Qwen HC ") + operation +
                             " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
}

__device__ float block_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  __shared__ float warps[8];
  if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = value;
  __syncthreads();
  value = threadIdx.x < 8 ? warps[threadIdx.x] : 0.0F;
  if (threadIdx.x < 32)
    for (int offset = 16; offset > 0; offset >>= 1)
      value += __shfl_down_sync(0xffffffffU, value, offset);
  if (threadIdx.x == 0) warps[0] = value;
  __syncthreads();
  return warps[0];
}

__global__ void grouped_gemma_rmsnorm(const __nv_bfloat16* input,
                                      const __nv_bfloat16* weight,
                                      __nv_bfloat16* output) {
  const int row = blockIdx.x / kStreams;
  const int stream = blockIdx.x % kStreams;
  const int base = row * kHyperHidden + stream * kHidden;
  float sum = 0.0F;
  for (int column = threadIdx.x; column < kHidden; column += blockDim.x) {
    const float value = __bfloat162float(input[base + column]);
    sum = fmaf(value, value, sum);
  }
  const float inverse = rsqrtf(block_sum(sum) / kHidden + kEpsilon);
  for (int column = threadIdx.x; column < kHidden; column += blockDim.x) {
    const float value = __bfloat162float(input[base + column]) * inverse;
    const float affine = __bfloat162float(weight[stream * kHidden + column]);
    output[base + column] = __float2bfloat16(fmaf(value, affine, value));
  }
}

__global__ void hc_silu_and_injection(const __nv_bfloat16* merged,
                                      __nv_bfloat16* lora,
                                      __nv_bfloat16* injection) {
  const int row = blockIdx.x;
  for (int column = threadIdx.x; column < kLowRank; column += blockDim.x) {
    const float value = __bfloat162float(merged[row * kMergedRows + column]) /
                        static_cast<float>(kStreams);
    lora[row * kLowRank + column] =
        __float2bfloat16(value / (1.0F + expf(-value)));
  }
  if (threadIdx.x < kStreams)
    injection[row * kStreams + threadIdx.x] =
        merged[row * kMergedRows + kLowRank + threadIdx.x];
}

__global__ void hc_silu(const __nv_bfloat16* projected,
                        __nv_bfloat16* lora) {
  const int row = blockIdx.x;
  for (int column = threadIdx.x; column < kLowRank; column += blockDim.x) {
    const float value = __bfloat162float(projected[row * kLowRank + column]) /
                        static_cast<float>(kStreams);
    lora[row * kLowRank + column] =
        __float2bfloat16(value / (1.0F + expf(-value)));
  }
}

__global__ void gate_mix(const __nv_bfloat16* normalized,
                         const __nv_bfloat16* gate,
                         __nv_bfloat16* block_input) {
  const int row = blockIdx.x;
  for (int column = threadIdx.x; column < kHidden; column += blockDim.x) {
    float value = 0.0F;
#pragma unroll
    for (int stream = 0; stream < kStreams; ++stream) {
      const int index = row * kHyperHidden + stream * kHidden + column;
      const float g = __bfloat162float(gate[index]);
      value += __bfloat162float(normalized[index]) / (1.0F + expf(-g));
    }
    block_input[row * kHidden + column] =
        __float2bfloat16(value / static_cast<float>(kStreams));
  }
}

__global__ void combine_norm(const __nv_bfloat16* residual,
                             const float* block_output,
                             const __nv_bfloat16* injection,
                             const __nv_bfloat16* weight,
                             __nv_bfloat16* updated,
                             __nv_bfloat16* normalized) {
  extern __shared__ float materialized[];
  const int row = blockIdx.x / kStreams;
  const int stream = blockIdx.x % kStreams;
  const int base = row * kHyperHidden + stream * kHidden;
  const float inj = 2.0F /
      (1.0F + expf(-__bfloat162float(injection[row * kStreams + stream]) /
                         static_cast<float>(kStreams)));
  float sum = 0.0F;
  for (int column = threadIdx.x; column < kHidden; column += blockDim.x) {
    const float block = __bfloat162float(
        __float2bfloat16(block_output[row * kHidden + column]));
    const __nv_bfloat16 rounded = __float2bfloat16(
        __bfloat162float(residual[base + column]) + block * inj);
    updated[base + column] = rounded;
    const float value = __bfloat162float(rounded);
    materialized[column] = value;
    sum = fmaf(value, value, sum);
  }
  const float inverse = rsqrtf(block_sum(sum) / kHidden + kEpsilon);
  for (int column = threadIdx.x; column < kHidden; column += blockDim.x) {
    const float value = materialized[column] * inverse;
    const float affine = __bfloat162float(weight[stream * kHidden + column]);
    normalized[base + column] = __float2bfloat16(fmaf(value, affine, value));
  }
}

__global__ void combine_only(const __nv_bfloat16* residual,
                             const float* block_output,
                             const __nv_bfloat16* injection,
                             __nv_bfloat16* updated) {
  const int row = blockIdx.x / kStreams;
  const int stream = blockIdx.x % kStreams;
  const int base = row * kHyperHidden + stream * kHidden;
  const float inj = 2.0F /
      (1.0F + expf(-__bfloat162float(injection[row * kStreams + stream]) /
                         static_cast<float>(kStreams)));
  for (int column = threadIdx.x; column < kHidden; column += blockDim.x) {
    const float block = __bfloat162float(
        __float2bfloat16(block_output[row * kHidden + column]));
    updated[base + column] = __float2bfloat16(
        __bfloat162float(residual[base + column]) + block * inj);
  }
}

void gemm(cublasHandle_t handle, const __nv_bfloat16* input,
          const __nv_bfloat16* weight, __nv_bfloat16* output,
          int m, int n, int k) {
  const float alpha = 1.0F, beta = 0.0F;
  cublas_check(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, weight,
      CUDA_R_16BF, k, input, CUDA_R_16BF, k, &beta, output, CUDA_R_16BF, n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "fixed BF16 GEMM");
}

}  // namespace

struct Plan::Impl {
  cublasHandle_t handle = nullptr;
  __nv_bfloat16 *attention_norm = nullptr, *attention_merged = nullptr,
                 *attention_up = nullptr, *mlp_norm = nullptr,
                 *mlp_merged = nullptr, *mlp_up = nullptr;
  __nv_bfloat16 *normalized = nullptr, *merged_output = nullptr,
                 *lora = nullptr, *gate = nullptr;

  ~Impl() {
    cudaFree(gate); cudaFree(lora); cudaFree(merged_output); cudaFree(normalized);
    cudaFree(mlp_up); cudaFree(mlp_merged); cudaFree(mlp_norm);
    cudaFree(attention_up); cudaFree(attention_merged); cudaFree(attention_norm);
    if (handle) cublasDestroy(handle);
  }
};

Plan::Plan(int device, Weights attention, Weights mlp) : impl_(new Impl) {
  if (device < 0 || !attention.norm || !attention.down || !attention.injection ||
      !attention.up || !mlp.norm || !mlp.down || !mlp.injection || !mlp.up)
    throw std::invalid_argument("Qwen HC requires exact device weights");
  try {
    cuda_check(cudaSetDevice(device), "set device");
    cublas_check(cublasCreate(&impl_->handle), "create handle");
    auto allocate = [](auto** pointer, std::size_t elements) {
      cuda_check(cudaMalloc(pointer, elements * sizeof(__nv_bfloat16)), "allocate buffer");
    };
    allocate(&impl_->attention_norm, kHyperHidden);
    allocate(&impl_->attention_merged,
             static_cast<std::size_t>(kMergedRows) * kHyperHidden);
    allocate(&impl_->attention_up,
             static_cast<std::size_t>(kHyperHidden) * kLowRank);
    allocate(&impl_->mlp_norm, kHyperHidden);
    allocate(&impl_->mlp_merged,
             static_cast<std::size_t>(kMergedRows) * kHyperHidden);
    allocate(&impl_->mlp_up,
             static_cast<std::size_t>(kHyperHidden) * kLowRank);
    allocate(&impl_->normalized, 16 * kHyperHidden);
    allocate(&impl_->merged_output, 16 * kMergedRows);
    allocate(&impl_->lora, 16 * kLowRank);
    allocate(&impl_->gate, 16 * kHyperHidden);
    auto copy_weights = [&](Weights source, __nv_bfloat16* norm,
                            __nv_bfloat16* merged, __nv_bfloat16* up) {
      cuda_check(cudaMemcpy(norm, source.norm, kHyperHidden * 2,
                            cudaMemcpyDeviceToDevice), "copy norm");
      cuda_check(cudaMemcpy(merged, source.down,
                            static_cast<std::size_t>(kLowRank) * kHyperHidden * 2,
                            cudaMemcpyDeviceToDevice), "copy down");
      cuda_check(cudaMemcpy(merged + kLowRank * kHyperHidden, source.injection,
                            static_cast<std::size_t>(kStreams) * kHyperHidden * 2,
                            cudaMemcpyDeviceToDevice), "copy injection");
      cuda_check(cudaMemset(merged + (kLowRank + kStreams) * kHyperHidden, 0,
                            static_cast<std::size_t>(kMergedRows-kLowRank-kStreams) *
                                kHyperHidden * 2), "clear merged padding");
      cuda_check(cudaMemcpy(up, source.up,
                            static_cast<std::size_t>(kHyperHidden) * kLowRank * 2,
                            cudaMemcpyDeviceToDevice), "copy up");
    };
    copy_weights(attention, impl_->attention_norm, impl_->attention_merged,
                 impl_->attention_up);
    copy_weights(mlp, impl_->mlp_norm, impl_->mlp_merged, impl_->mlp_up);
  } catch (...) {
    delete impl_;
    impl_ = nullptr;
    throw;
  }
}

Plan::~Plan() { delete impl_; }

struct FinalPlan::Impl {
  cublasHandle_t handle = nullptr;
  const __nv_bfloat16* norm = nullptr;
  const __nv_bfloat16* down = nullptr;
  const __nv_bfloat16* up = nullptr;
  __nv_bfloat16* normalized = nullptr;
  __nv_bfloat16* projected = nullptr;
  __nv_bfloat16* lora = nullptr;
  __nv_bfloat16* gate = nullptr;

  ~Impl() {
    cudaFree(gate);
    cudaFree(lora);
    cudaFree(projected);
    cudaFree(normalized);
    if (handle) cublasDestroy(handle);
  }
};

FinalPlan::FinalPlan(int device, const __nv_bfloat16* norm,
                     const __nv_bfloat16* down,
                     const __nv_bfloat16* up)
    : impl_(new Impl) {
  if (device < 0 || !norm || !down || !up)
    throw std::invalid_argument("Qwen final HC requires exact device weights");
  try {
    cuda_check(cudaSetDevice(device), "set final HC device");
    cublas_check(cublasCreate(&impl_->handle), "create final HC handle");
    impl_->norm = norm;
    impl_->down = down;
    impl_->up = up;
    auto allocate = [](auto** pointer, std::size_t elements) {
      cuda_check(cudaMalloc(pointer, elements * sizeof(__nv_bfloat16)),
                 "allocate final HC buffer");
    };
    allocate(&impl_->normalized, 16 * kHyperHidden);
    allocate(&impl_->projected, 16 * kLowRank);
    allocate(&impl_->lora, 16 * kLowRank);
    allocate(&impl_->gate, 16 * kHyperHidden);
  } catch (...) {
    delete impl_;
    impl_ = nullptr;
    throw;
  }
}

FinalPlan::~FinalPlan() { delete impl_; }

void FinalPlan::combine_and_collapse(
    const __nv_bfloat16* hidden, const float* block_output,
    const __nv_bfloat16* injection, __nv_bfloat16* updated_hidden,
    __nv_bfloat16* token_hidden, int m, cudaStream_t stream) {
  if (!hidden || !block_output || !injection || !updated_hidden ||
      !token_hidden || !allowed_m(m) || !stream)
    throw std::invalid_argument("Qwen final HC contract drift");
  cublas_check(cublasSetStream(impl_->handle, stream),
               "bind final HC stream");
  combine_norm<<<m * kStreams, kThreads, kHidden * sizeof(float), stream>>>(
      hidden, block_output, injection, impl_->norm, updated_hidden,
      impl_->normalized);
  gemm(impl_->handle, impl_->normalized, impl_->down, impl_->projected, m,
       kLowRank, kHyperHidden);
  hc_silu<<<m, kThreads, 0, stream>>>(impl_->projected, impl_->lora);
  gemm(impl_->handle, impl_->lora, impl_->up, impl_->gate, m, kHyperHidden,
       kLowRank);
  gate_mix<<<m, kThreads, 0, stream>>>(impl_->normalized, impl_->gate,
                                      token_hidden);
  cuda_check(cudaPeekAtLastError(), "launch final HC");
}

void Plan::mix(const __nv_bfloat16* hidden, __nv_bfloat16* block_input,
               __nv_bfloat16* injection, int m, cudaStream_t stream) {
  if (!hidden || !block_input || !injection || !allowed_m(m) || !stream)
    throw std::invalid_argument("Qwen HC mix contract drift");
  cublas_check(cublasSetStream(impl_->handle, stream), "bind mix stream");
  grouped_gemma_rmsnorm<<<m * kStreams, kThreads, 0, stream>>>(
      hidden, impl_->attention_norm, impl_->normalized);
  gemm(impl_->handle, impl_->normalized, impl_->attention_merged,
       impl_->merged_output, m, kMergedRows, kHyperHidden);
  hc_silu_and_injection<<<m, kThreads, 0, stream>>>(
      impl_->merged_output, impl_->lora, injection);
  gemm(impl_->handle, impl_->lora, impl_->attention_up, impl_->gate,
       m, kHyperHidden, kLowRank);
  gate_mix<<<m, kThreads, 0, stream>>>(impl_->normalized, impl_->gate,
                                      block_input);
  cuda_check(cudaPeekAtLastError(), "launch mix");
}

void Plan::combine_and_mix(
    const __nv_bfloat16* hidden, const float* block_output,
    const __nv_bfloat16* injection, __nv_bfloat16* updated_hidden,
    __nv_bfloat16* next_block_input, __nv_bfloat16* next_injection, int m,
    cudaStream_t stream) {
  if (!hidden || !block_output || !injection || !updated_hidden ||
      !next_block_input || !next_injection || !allowed_m(m) || !stream)
    throw std::invalid_argument("Qwen HC combine-and-mix contract drift");
  cublas_check(cublasSetStream(impl_->handle, stream), "bind combine stream");
  combine_norm<<<m * kStreams, kThreads, kHidden * sizeof(float), stream>>>(
      hidden, block_output, injection, impl_->mlp_norm, updated_hidden,
      impl_->normalized);
  gemm(impl_->handle, impl_->normalized, impl_->mlp_merged,
       impl_->merged_output, m, kMergedRows, kHyperHidden);
  hc_silu_and_injection<<<m, kThreads, 0, stream>>>(
      impl_->merged_output, impl_->lora, next_injection);
  gemm(impl_->handle, impl_->lora, impl_->mlp_up, impl_->gate,
       m, kHyperHidden, kLowRank);
  gate_mix<<<m, kThreads, 0, stream>>>(impl_->normalized, impl_->gate,
                                      next_block_input);
  cuda_check(cudaPeekAtLastError(), "launch combine-and-mix");
}

void Plan::combine(const __nv_bfloat16* hidden, const float* block_output,
                   const __nv_bfloat16* injection,
                   __nv_bfloat16* updated_hidden, int m,
                   cudaStream_t stream) {
  if (!hidden || !block_output || !injection || !updated_hidden ||
      !allowed_m(m) || !stream)
    throw std::invalid_argument("Qwen HC combine contract drift");
  combine_only<<<m * kStreams, kThreads, 0, stream>>>(
      hidden, block_output, injection, updated_hidden);
  cuda_check(cudaPeekAtLastError(), "launch combine");
}

}  // namespace rocket::qwen38::hyperconnection

namespace {
thread_local std::string hc_last_error;

template <typename Operation>
int hc_call(Operation operation) noexcept {
  hc_last_error.clear();
  try {
    operation();
    return 0;
  } catch (const std::exception& error) {
    hc_last_error = error.what();
  } catch (...) {
    hc_last_error = "unknown Qwen HyperConnection failure";
  }
  return 1;
}
}  // namespace

extern "C" int qwen38_hc_create(
    int device, const __nv_bfloat16* attn_norm,
    const __nv_bfloat16* attn_down, const __nv_bfloat16* attn_injection,
    const __nv_bfloat16* attn_up, const __nv_bfloat16* mlp_norm,
    const __nv_bfloat16* mlp_down, const __nv_bfloat16* mlp_injection,
    const __nv_bfloat16* mlp_up, void** result) {
  if (!result) return 1;
  *result = nullptr;
  return hc_call([&] {
    *result = new rocket::qwen38::hyperconnection::Plan(
        device, {attn_norm, attn_down, attn_injection, attn_up},
        {mlp_norm, mlp_down, mlp_injection, mlp_up});
  });
}

extern "C" int qwen38_hc_mix(
    void* opaque, const __nv_bfloat16* hidden, __nv_bfloat16* block_input,
    __nv_bfloat16* injection, int m, cudaStream_t stream) {
  if (!opaque) return 1;
  return hc_call([&] {
    static_cast<rocket::qwen38::hyperconnection::Plan*>(opaque)->mix(
        hidden, block_input, injection, m, stream);
  });
}

extern "C" int qwen38_hc_combine_and_mix(
    void* opaque, const __nv_bfloat16* hidden, const float* block_output,
    const __nv_bfloat16* injection, __nv_bfloat16* updated_hidden,
    __nv_bfloat16* next_block_input, __nv_bfloat16* next_injection, int m,
    cudaStream_t stream) {
  if (!opaque) return 1;
  return hc_call([&] {
    static_cast<rocket::qwen38::hyperconnection::Plan*>(opaque)
        ->combine_and_mix(hidden, block_output, injection, updated_hidden,
                          next_block_input, next_injection, m, stream);
  });
}

extern "C" int qwen38_hc_combine(
    void* opaque, const __nv_bfloat16* hidden, const float* block_output,
    const __nv_bfloat16* injection, __nv_bfloat16* updated_hidden, int m,
    cudaStream_t stream) {
  if (!opaque) return 1;
  return hc_call([&] {
    static_cast<rocket::qwen38::hyperconnection::Plan*>(opaque)->combine(
        hidden, block_output, injection, updated_hidden, m, stream);
  });
}

extern "C" int qwen38_hc_destroy(void* opaque) {
  delete static_cast<rocket::qwen38::hyperconnection::Plan*>(opaque);
  return 0;
}

extern "C" int qwen38_final_hc_create(
    int device, const __nv_bfloat16* norm, const __nv_bfloat16* down,
    const __nv_bfloat16* up, void** result) {
  if (!result) return 1;
  *result = nullptr;
  return hc_call([&] {
    *result = new rocket::qwen38::hyperconnection::FinalPlan(
        device, norm, down, up);
  });
}

extern "C" int qwen38_final_hc_combine_and_collapse(
    void* opaque, const __nv_bfloat16* hidden, const float* block_output,
    const __nv_bfloat16* injection, __nv_bfloat16* updated_hidden,
    __nv_bfloat16* token_hidden, int m, cudaStream_t stream) {
  if (!opaque) return 1;
  return hc_call([&] {
    static_cast<rocket::qwen38::hyperconnection::FinalPlan*>(opaque)
        ->combine_and_collapse(hidden, block_output, injection, updated_hidden,
                               token_hidden, m, stream);
  });
}

extern "C" int qwen38_final_hc_destroy(void* opaque) {
  delete static_cast<rocket::qwen38::hyperconnection::FinalPlan*>(opaque);
  return 0;
}

extern "C" const char* qwen38_hc_last_error() {
  return hc_last_error.c_str();
}

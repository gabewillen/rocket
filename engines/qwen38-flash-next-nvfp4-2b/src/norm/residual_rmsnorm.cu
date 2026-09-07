// SPDX-License-Identifier: Apache-2.0
// Reduction and float-cache structure follows FlashInfer norm.cuh and
// TensorRT-LLM rmsnormKernels.cu, both Apache-2.0. Fixed Qwen widths remove
// their generic shape, dtype, quantization, stride, and dispatch branches.
#include "norm/residual_rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace rocket::qwen38::norm {
namespace {

constexpr int kThreads = 256;
constexpr int kValuesPerVector = 8;

struct alignas(16) Bf16x8 {
  std::uint32_t words[4];
};
static_assert(sizeof(Bf16x8) == 16);

__device__ __forceinline__ Bf16x8 load_bf16x8(const __nv_bfloat16* source) {
  Bf16x8 result;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(result.words[0]), "=r"(result.words[1]),
                 "=r"(result.words[2]), "=r"(result.words[3])
               : "l"(source));
  return result;
}

__device__ __forceinline__ void store_bf16x8(__nv_bfloat16* destination,
                                              const Bf16x8& value) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::
                   "l"(destination), "r"(value.words[0]), "r"(value.words[1]),
                   "r"(value.words[2]), "r"(value.words[3]) : "memory");
}

__device__ __forceinline__ __nv_bfloat16 get_bf16(const Bf16x8& value,
                                                   int lane) {
  const std::uint32_t word = value.words[lane >> 1];
  return __ushort_as_bfloat16(static_cast<unsigned short>(
      lane & 1 ? word >> 16 : word & 0xffffU));
}

__device__ __forceinline__ void set_bf16(Bf16x8& value, int lane,
                                          __nv_bfloat16 element) {
  const auto bits = static_cast<std::uint32_t>(__bfloat16_as_ushort(element));
  auto& word = value.words[lane >> 1];
  word = lane & 1 ? (word & 0xffffU) | (bits << 16)
                  : (word & 0xffff0000U) | bits;
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

template <int Hidden>
__global__ void residual_add_kernel(const __nv_bfloat16* input,
                                    const __nv_bfloat16* residual,
                                    __nv_bfloat16* output) {
  const int row = blockIdx.x;
  const int base = row * Hidden;
  constexpr int vectors = Hidden / kValuesPerVector;
  for (int vector = threadIdx.x; vector < vectors; vector += blockDim.x) {
    const int offset = base + vector * kValuesPerVector;
    const Bf16x8 input_values = load_bf16x8(input + offset);
    const Bf16x8 residual_values = load_bf16x8(residual + offset);
    Bf16x8 output_values{};
#pragma unroll
    for (int lane = 0; lane < kValuesPerVector; ++lane) {
      set_bf16(output_values, lane, __float2bfloat16(
          __bfloat162float(get_bf16(input_values, lane)) +
          __bfloat162float(get_bf16(residual_values, lane))));
    }
    store_bf16x8(output + offset, output_values);
  }
}

template <int Hidden, bool Fused>
__global__ void rms_kernel(const __nv_bfloat16* input,
                           const __nv_bfloat16* residual,
                           const __nv_bfloat16* weight,
                           __nv_bfloat16* residual_output,
                           __nv_bfloat16* norm_output) {
  extern __shared__ float values[];
  const int row = blockIdx.x;
  const int base = row * Hidden;
  constexpr int vectors = Hidden / kValuesPerVector;
  float square_sum = 0.0F;
  for (int vector = threadIdx.x; vector < vectors; vector += blockDim.x) {
    const int offset = base + vector * kValuesPerVector;
    const Bf16x8 input_values = load_bf16x8(input + offset);
    Bf16x8 residual_values{};
    Bf16x8 output_values{};
    if constexpr (Fused) residual_values = load_bf16x8(residual + offset);
#pragma unroll
    for (int lane = 0; lane < kValuesPerVector; ++lane) {
      const int column = vector * kValuesPerVector + lane;
      float value = __bfloat162float(get_bf16(input_values, lane));
      if constexpr (Fused) {
        value += __bfloat162float(get_bf16(residual_values, lane));
        set_bf16(output_values, lane, __float2bfloat16(value));
      }
      values[column] = value;
      square_sum = fmaf(value, value, square_sum);
    }
    if constexpr (Fused) store_bf16x8(residual_output + offset, output_values);
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum) /
                                   static_cast<float>(Hidden) + kEpsilon);
  for (int vector = threadIdx.x; vector < vectors; vector += blockDim.x) {
    const int offset = vector * kValuesPerVector;
    const Bf16x8 weight_values = load_bf16x8(weight + offset);
    Bf16x8 output_values{};
#pragma unroll
    for (int lane = 0; lane < kValuesPerVector; ++lane) {
      const int column = vector * kValuesPerVector + lane;
      const float scaled = values[column] * inverse_rms *
                           __bfloat162float(get_bf16(weight_values, lane));
      set_bf16(output_values, lane, __float2bfloat16(scaled));
    }
    store_bf16x8(norm_output + base + offset, output_values);
  }
}

template <int Hidden>
cudaError_t launch_add(const __nv_bfloat16* input, const __nv_bfloat16* residual,
                       __nv_bfloat16* output, int m, cudaStream_t stream) {
  residual_add_kernel<Hidden><<<m, kThreads, 0, stream>>>(input, residual, output);
  return cudaPeekAtLastError();
}

template <int Hidden, bool Fused>
cudaError_t launch_rms(const __nv_bfloat16* input,
                       const __nv_bfloat16* residual,
                       const __nv_bfloat16* weight,
                       __nv_bfloat16* residual_output,
                       __nv_bfloat16* norm_output, int m,
                       cudaStream_t stream) {
  rms_kernel<Hidden, Fused><<<m, kThreads, Hidden * sizeof(float), stream>>>(
      input, residual, weight, residual_output, norm_output);
  return cudaPeekAtLastError();
}

bool valid(const void* a, const void* b, const void* c, int m, int hidden) {
  return a != nullptr && b != nullptr && c != nullptr && allowed_m(m) &&
         allowed_hidden(hidden);
}

}  // namespace

cudaError_t residual_add(const __nv_bfloat16* input,
                         const __nv_bfloat16* residual,
                         __nv_bfloat16* output, int m, int hidden,
                         cudaStream_t stream) noexcept {
  if (!valid(input, residual, output, m, hidden)) return cudaErrorInvalidValue;
  return hidden == kHiddenFull ? launch_add<kHiddenFull>(input, residual, output, m, stream)
                               : launch_add<kHiddenTp>(input, residual, output, m, stream);
}

cudaError_t rms_norm(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                     __nv_bfloat16* output, int m, int hidden,
                     cudaStream_t stream) noexcept {
  if (!valid(input, weight, output, m, hidden)) return cudaErrorInvalidValue;
  return hidden == kHiddenFull
             ? launch_rms<kHiddenFull, false>(input, nullptr, weight, nullptr, output, m, stream)
             : launch_rms<kHiddenTp, false>(input, nullptr, weight, nullptr, output, m, stream);
}

cudaError_t fused_add_rms_norm(const __nv_bfloat16* input,
                               const __nv_bfloat16* residual,
                               const __nv_bfloat16* weight,
                               __nv_bfloat16* residual_output,
                               __nv_bfloat16* norm_output, int m, int hidden,
                               cudaStream_t stream) noexcept {
  if (!valid(input, residual, weight, m, hidden) || residual_output == nullptr ||
      norm_output == nullptr || residual_output == norm_output)
    return cudaErrorInvalidValue;
  return hidden == kHiddenFull
             ? launch_rms<kHiddenFull, true>(input, residual, weight,
                                             residual_output, norm_output, m, stream)
             : launch_rms<kHiddenTp, true>(input, residual, weight,
                                           residual_output, norm_output, m, stream);
}

}  // namespace rocket::qwen38::norm

// SPDX-License-Identifier: Apache-2.0
#include "linear_attention/gdn_prefill_owner.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>

namespace rocket::qwen38::linear_attention::prefill {
namespace {
constexpr float kAttentionScale = 0.08838834764831845F;
constexpr int kThreads = kNativePrefillHeadDim;
constexpr std::size_t kOutputElements =
    static_cast<std::size_t>(kNativePrefillRows) * kNativePrefillValueHeads *
    kNativePrefillHeadDim;
constexpr std::size_t kStateElements =
    static_cast<std::size_t>(kNativePrefillValueHeads) *
    kNativePrefillHeadDim * kNativePrefillHeadDim;

// This is the same sequential FP32 recurrence used by gdn_core.cu after its
// projection/normalization stages. The authenticated bundle already contains
// normalized Q/K, V, log-decay, and promoted beta, so this owner begins at the
// pinned FlashInfer chunk-prefill call boundary.

__device__ float block_sum(float value, float* warp_sums) {
  for (int delta = 16; delta != 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = value;
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = threadIdx.x < 4 ? warp_sums[lane] : 0.0F;
  if (warp == 0) {
    for (int delta = 16; delta != 0; delta >>= 1) {
      value += __shfl_down_sync(0xffffffffU, value, delta);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) warp_sums[0] = value;
  __syncthreads();
  const float result = warp_sums[0];
  __syncthreads();
  return result;
}

__global__ __launch_bounds__(kThreads) void recurrent_m35(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, const float* log_decay, const float* beta,
    float* state, __nv_bfloat16* output) {
  const int value_head = blockIdx.x;
  const int key_head = value_head / 3;
  const int dimension = threadIdx.x;
  __shared__ float query[kNativePrefillHeadDim];
  __shared__ float key[kNativePrefillHeadDim];
  __shared__ float warp_sums[4];

  float* head_state =
      state + static_cast<std::size_t>(value_head) * kNativePrefillHeadDim *
                  kNativePrefillHeadDim;
  for (int row = 0; row < kNativePrefillRows; ++row) {
    const std::size_t qk_index =
        (static_cast<std::size_t>(row) * kNativePrefillKeyHeads + key_head) *
            kNativePrefillHeadDim +
        dimension;
    query[dimension] = __bfloat162float(q[qk_index]);
    key[dimension] = __bfloat162float(k[qk_index]);
    __syncthreads();
    const float decay =
        expf(log_decay[row * kNativePrefillValueHeads + value_head]);
    const float update = beta[row * kNativePrefillValueHeads + value_head];
    for (int value_dimension = 0;
         value_dimension < kNativePrefillHeadDim; ++value_dimension) {
      float current =
          head_state[value_dimension * kNativePrefillHeadDim + dimension] *
          decay;
      const float prediction = block_sum(current * key[dimension], warp_sums);
      const std::size_t v_index =
          (static_cast<std::size_t>(row) * kNativePrefillValueHeads +
           value_head) *
              kNativePrefillHeadDim +
          value_dimension;
      current += update * (__bfloat162float(v[v_index]) - prediction) *
                 key[dimension];
      head_state[value_dimension * kNativePrefillHeadDim + dimension] =
          current;
      const float projected =
          block_sum(current * query[dimension], warp_sums) * kAttentionScale;
      if (dimension == 0) output[v_index] = __float2bfloat16(projected);
    }
    __syncthreads();
  }
}

bool overlaps(const void* left, std::size_t left_bytes, const void* right,
              std::size_t right_bytes) {
  const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
  const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
  return left_begin < right_begin + right_bytes &&
         right_begin < left_begin + left_bytes;
}

void require_device_pointer(const void* pointer, int device) {
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess ||
      attributes.type != cudaMemoryTypeDevice || attributes.device != device) {
    throw std::invalid_argument("GDN M35 device pointer changed");
  }
}
}  // namespace

NativeGdnM35PrefillOwner::NativeGdnM35PrefillOwner(
    const NativePrefillConfig& config)
    : device_(config.device),
      publish_(config.publish),
      publish_context_(config.publish_context) {
  if (config.device != 0 || config.rank != 0 || config.layer != 0 ||
      config.rows != kNativePrefillRows || config.publish == nullptr) {
    if (config.publish) {
      config.publish(config.publish_context,
              {NativePrefillPhase::kValidation,
               NativePrefillStatus::kIdentity,
               static_cast<std::int8_t>(config.rank),
               static_cast<std::int8_t>(config.layer),
               static_cast<std::uint8_t>(
                   config.rows == kNativePrefillRows ? config.rows : 0),
               false});
    }
    throw std::invalid_argument("GDN M35 owner identity changed");
  }
  emit(NativePrefillPhase::kValidation, NativePrefillStatus::kSuccess, true);
}

void NativeGdnM35PrefillOwner::emit(NativePrefillPhase phase,
                                    NativePrefillStatus status,
                                    bool success) const noexcept {
  publish_(publish_context_,
           {phase, status, 0, 0, kNativePrefillRows, success});
}

void NativeGdnM35PrefillOwner::launch(const NativePrefillTensors& tensors,
                                      cudaStream_t stream) {
  const void* pointers[] = {tensors.q,          tensors.k,
                            tensors.v,          tensors.log_decay,
                            tensors.beta,       tensors.initial_state,
                            tensors.output,     tensors.final_state};
  for (const void* pointer : pointers) {
    if (pointer == nullptr || (reinterpret_cast<std::uintptr_t>(pointer) & 15U)) {
      emit(NativePrefillPhase::kValidation, NativePrefillStatus::kPointer,
           false);
      throw std::invalid_argument("GDN M35 tensor pointer changed");
    }
  }
  unsigned int stream_flags = 0;
  if (cudaStreamGetFlags(stream, &stream_flags) != cudaSuccess) {
    emit(NativePrefillPhase::kValidation, NativePrefillStatus::kCuda, false);
    throw std::invalid_argument("GDN M35 stream changed");
  }
  const std::size_t output_bytes = kOutputElements * sizeof(__nv_bfloat16);
  const std::size_t state_bytes = kStateElements * sizeof(float);
  if (overlaps(tensors.output, output_bytes, tensors.final_state, state_bytes) ||
      overlaps(tensors.output, output_bytes, tensors.initial_state,
               state_bytes) ||
      (tensors.initial_state != tensors.final_state &&
       overlaps(tensors.initial_state, state_bytes, tensors.final_state,
                state_bytes))) {
    emit(NativePrefillPhase::kValidation, NativePrefillStatus::kAliasing,
         false);
    throw std::invalid_argument("GDN M35 tensor alias changed");
  }
  for (const void* pointer : pointers) {
    try {
      require_device_pointer(pointer, device_);
    } catch (...) {
      emit(NativePrefillPhase::kValidation, NativePrefillStatus::kDevice,
           false);
      throw;
    }
  }
  if (tensors.initial_state != tensors.final_state) {
    const auto copy = cudaMemcpyAsync(tensors.final_state, tensors.initial_state,
                                      state_bytes, cudaMemcpyDeviceToDevice,
                                      stream);
    if (copy != cudaSuccess) {
      emit(NativePrefillPhase::kStatePublication, NativePrefillStatus::kCuda,
           false);
      throw std::runtime_error("GDN M35 state publication failed");
    }
  }
  recurrent_m35<<<kNativePrefillValueHeads, kThreads, 0, stream>>>(
      tensors.q, tensors.k, tensors.v, tensors.log_decay, tensors.beta,
      tensors.final_state, tensors.output);
  if (cudaPeekAtLastError() != cudaSuccess) {
    emit(NativePrefillPhase::kLaunch, NativePrefillStatus::kCuda, false);
    throw std::runtime_error("GDN M35 launch failed");
  }
  emit(NativePrefillPhase::kLaunch, NativePrefillStatus::kSuccess, true);
}

}  // namespace rocket::qwen38::linear_attention::prefill

extern "C" int qwen38_gdn_m35_prefill_create(
    const rocket::qwen38::linear_attention::prefill::NativePrefillConfig* config,
    void** owner) noexcept {
  if (!config || !owner) return 1;
  *owner = nullptr;
  try {
    auto value = std::make_unique<
        rocket::qwen38::linear_attention::prefill::NativeGdnM35PrefillOwner>(
        *config);
    *owner = value.release();
    return 0;
  } catch (...) {
    return 1;
  }
}

extern "C" int qwen38_gdn_m35_prefill_launch(
    void* owner,
    const rocket::qwen38::linear_attention::prefill::NativePrefillTensors* tensors,
    cudaStream_t stream) noexcept {
  if (!owner || !tensors) return 1;
  try {
    static_cast<rocket::qwen38::linear_attention::prefill::NativeGdnM35PrefillOwner*>(
        owner)->launch(*tensors, stream);
    return 0;
  } catch (...) {
    return 1;
  }
}

extern "C" int qwen38_gdn_m35_prefill_destroy(void* owner) noexcept {
  const std::unique_ptr<
      rocket::qwen38::linear_attention::prefill::NativeGdnM35PrefillOwner>
      value(static_cast<
            rocket::qwen38::linear_attention::prefill::NativeGdnM35PrefillOwner*>(
          owner));
  return 0;
}

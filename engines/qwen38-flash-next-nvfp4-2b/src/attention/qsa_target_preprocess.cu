// SPDX-License-Identifier: Apache-2.0
#include "attention/qsa_target_preprocess.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace rocket::qwen38::attention {
namespace {

// Dataflow provenance, Apache-2.0:
// - pinned vLLM 8e685d198, models/qwen3_8_flash_next/nvidia/
//   {qsa.py,indexer_qsa.py,ops/qsa_pre_indexer.py}
// - current vLLM 9ea8f3ffc354901b740f0b31988900897b7221d7,
//   models/qwen4_exp/nvidia/{qsa.py,indexer_qsa.py,ops/qsa_pre_indexer.py,
//   ops/qsa_indexer.py}

constexpr int kHidden = 2560;
constexpr int kMainHeads = 12;
constexpr int kMainDim = 256;
constexpr int kMainKOffset = 6144;
constexpr int kMainVOffset = 6400;
constexpr int kIndexHeads = 4;
constexpr int kIndexDim = 128;
constexpr int kIndexOutputs = 640;
constexpr int kRotaryDim = 64;
constexpr int kRawStride = 140;
constexpr float kEpsilon = 1.0e-6f;

__device__ float warp_sum(float value) {
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

__global__ void index_qk_gemv(const __nv_bfloat16* hidden,
                              const __nv_bfloat16* weight,
                              __nv_bfloat16* output) {
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int column = threadIdx.x; column < kHidden; column += blockDim.x)
    sum = fmaf(__bfloat162float(hidden[column]),
               __bfloat162float(weight[row * kHidden + column]), sum);
  sum = warp_sum(sum);
  __shared__ float warps[8];
  if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warps[threadIdx.x] : 0.0f;
    sum = warp_sum(sum);
    if (threadIdx.x == 0) output[row] = __float2bfloat16(sum);
  }
}

__device__ float rrms_row(const __nv_bfloat16* row, int width) {
  float square = 0.0f;
  for (int index = threadIdx.x; index < width; index += blockDim.x) {
    const float value = __bfloat162float(row[index]);
    square = fmaf(value, value, square);
  }
  square = warp_sum(square);
  __shared__ float warps[8];
  if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = square;
  __syncthreads();
  if (threadIdx.x < 32) {
    square = threadIdx.x < 8 ? warps[threadIdx.x] : 0.0f;
    square = warp_sum(square);
    if (threadIdx.x == 0) warps[0] = rsqrtf(square / width + kEpsilon);
  }
  __syncthreads();
  return warps[0];
}

__device__ float norm_value(const __nv_bfloat16* row,
                            const __nv_bfloat16* weight, int index,
                            float rrms) {
  return __bfloat162float(row[index]) * rrms *
         (__bfloat162float(weight[index]) + 1.0f);
}

__device__ void rope_pair(float x0, float x1,
                          const __nv_bfloat16* cos_sin,
                          const std::int64_t* positions, int pair,
                          int uses_mrope,
                          float& y0, float& y1) {
  int axis = 0;
  if (uses_mrope && pair % 3 == 1 && pair <= 33) axis = 1;
  if (uses_mrope && pair % 3 == 2 && pair <= 30) axis = 2;
  const std::int64_t position = positions[axis];
  const __nv_bfloat16* row = cos_sin + position * kRotaryDim;
  const float cosine = __bfloat162float(row[pair]);
  const float sine = __bfloat162float(row[32 + pair]);
  y0 = x0 * cosine - x1 * sine;
  y1 = x1 * cosine + x0 * sine;
}

__global__ void main_qk_norm_rope_gate(
    const __nv_bfloat16* raw, const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight, const __nv_bfloat16* cos_sin,
    const std::int64_t* positions, int uses_mrope,
    __nv_bfloat16* query, __nv_bfloat16* gate, __nv_bfloat16* main_key,
    __nv_bfloat16* main_value, const std::int32_t* slot_mapping,
    int main_blocks) {
  const int owner = blockIdx.x;
  const bool is_query = owner < kMainHeads;
  const __nv_bfloat16* row = is_query
      ? raw + owner * (2 * kMainDim)
      : raw + kMainKOffset;
  const __nv_bfloat16* norm_weight = is_query ? q_weight : k_weight;
  const float rrms = rrms_row(row, kMainDim);
  __shared__ float rotated[kRotaryDim];
  if (threadIdx.x < kRotaryDim / 2) {
    const int pair = threadIdx.x;
    rope_pair(norm_value(row, norm_weight, pair, rrms),
              norm_value(row, norm_weight, pair + kRotaryDim / 2, rrms),
              cos_sin, positions, pair, uses_mrope, rotated[pair],
              rotated[pair + kRotaryDim / 2]);
  }
  __syncthreads();
  if (is_query) {
    for (int dim = threadIdx.x; dim < kMainDim; dim += blockDim.x) {
      const float value = dim < kRotaryDim
          ? rotated[dim]
          : norm_value(row, norm_weight, dim, rrms);
      query[owner * kMainDim + dim] = __float2bfloat16(value);
      gate[owner * kMainDim + dim] = raw[owner * 2 * kMainDim + kMainDim + dim];
    }
    return;
  }
  const int slot = slot_mapping[0];
  if (slot < 0 || slot >= main_blocks * 1600) return;
  for (int dim = threadIdx.x; dim < kMainDim; dim += blockDim.x) {
    const float value = dim < kRotaryDim
        ? rotated[dim]
        : norm_value(row, norm_weight, dim, rrms);
    main_key[static_cast<std::size_t>(slot) * kMainDim + dim] =
        __float2bfloat16(value);
    main_value[static_cast<std::size_t>(slot) * kMainDim + dim] =
        raw[kMainVOffset + dim];
  }
}

__global__ void index_norm_rope_and_raw_store(
    const __nv_bfloat16* projected, const __nv_bfloat16* q_weight,
    const __nv_bfloat16* cos_sin, __nv_bfloat16* query,
    __nv_bfloat16* raw_cache, const std::int32_t* raw_slots,
    const std::int64_t* positions, int uses_mrope) {
  const int head = blockIdx.x;
  const __nv_bfloat16* row = projected + head * kIndexDim;
  const float rrms = rrms_row(row, kIndexDim);
  __shared__ float rotated[kRotaryDim];
  if (threadIdx.x < kRotaryDim / 2) {
    const int pair = threadIdx.x;
    rope_pair(norm_value(row, q_weight, pair, rrms),
              norm_value(row, q_weight, pair + kRotaryDim / 2, rrms),
              cos_sin, positions, pair, uses_mrope, rotated[pair],
              rotated[pair + kRotaryDim / 2]);
  }
  __syncthreads();
  for (int dim = threadIdx.x; dim < kIndexDim; dim += blockDim.x) {
    query[head * kIndexDim + dim] = __float2bfloat16(
        dim < kRotaryDim ? rotated[dim] : norm_value(row, q_weight, dim, rrms));
  }
  if (head != 0) return;
  const int slot = raw_slots[0];
  if (slot < 0) return;
  __nv_bfloat16* target = raw_cache + static_cast<std::size_t>(slot) * kRawStride;
  for (int dim = threadIdx.x; dim < kIndexDim; dim += blockDim.x)
    target[dim] = projected[kIndexHeads * kIndexDim + dim];
  if (threadIdx.x == 0 && uses_mrope) {
    auto* tail = reinterpret_cast<std::int64_t*>(target + kIndexDim);
    tail[0] = positions[0];
    tail[1] = positions[1];
    tail[2] = positions[2];
  }
}

__global__ void compress_index_key(
    const __nv_bfloat16* raw_cache, __nv_bfloat16* compressed_cache,
    const std::int32_t* raw_block_table,
    const std::int32_t* compressed_slots,
    const std::int32_t* token_to_request,
    const std::int64_t* logical_positions,
    const __nv_bfloat16* norm_weight, const __nv_bfloat16* cos_sin,
    int compressed_blocks, int uses_mrope) {
  const int compressed_slot = compressed_slots[0];
  const int request = token_to_request[0];
  const std::int64_t logical = logical_positions[0];
  if (compressed_slot < 0 || compressed_slot >= compressed_blocks * 400 ||
      request < 0 || logical < 3 || logical % kTargetQsaCompressRatio != 3)
    return;
  const int raw_block = raw_block_table[request];
  if (raw_block < 0) return;
  float pooled = 0.0f;
  if (threadIdx.x < kIndexDim) {
    for (int item = 0; item < kTargetQsaCompressRatio; ++item) {
      const std::int64_t position = logical - 3 + item;
      const int slot = raw_block * kTargetQsaRawStateRows +
                       position % kTargetQsaRawStateRows;
      pooled += __bfloat162float(
          raw_cache[static_cast<std::size_t>(slot) * kRawStride + threadIdx.x]);
    }
    pooled = __bfloat162float(__float2bfloat16(pooled * 0.25f));
  }
  // rrms_row consumes BF16 storage, while pooled must be rounded to BF16
  // before normalization. Keep that exact intermediate in shared memory.
  __shared__ __nv_bfloat16 rounded[kIndexDim];
  rounded[threadIdx.x] = __float2bfloat16(pooled);
  __syncthreads();
  const float rounded_rrms = rrms_row(rounded, kIndexDim);
  __shared__ std::int64_t first_positions[3];
  if (threadIdx.x < 3) {
    if (uses_mrope) {
      const int first_slot = raw_block * kTargetQsaRawStateRows +
                             (logical - 3) % kTargetQsaRawStateRows;
      const auto* tail = reinterpret_cast<const std::int64_t*>(
          raw_cache + static_cast<std::size_t>(first_slot) * kRawStride +
          kIndexDim);
      first_positions[threadIdx.x] = tail[threadIdx.x];
    } else {
      first_positions[threadIdx.x] = logical - 3;
    }
  }
  __syncthreads();
  float value = norm_value(rounded, norm_weight, threadIdx.x, rounded_rrms);
  if (threadIdx.x < kRotaryDim) {
    const int pair = threadIdx.x % 32;
    float y0, y1;
    rope_pair(norm_value(rounded, norm_weight, pair, rounded_rrms),
              norm_value(rounded, norm_weight, pair + 32, rounded_rrms),
              cos_sin, first_positions, pair, uses_mrope, y0, y1);
    value = threadIdx.x < 32 ? y0 : y1;
  }
  compressed_cache[static_cast<std::size_t>(compressed_slot) * kIndexDim +
                   threadIdx.x] = __float2bfloat16(value);
}

void require(bool condition, const char* message) {
  if (!condition) throw TargetQsaPreprocessError(message);
}

}  // namespace

void launch_target_qsa_preprocess_c1(
    const __nv_bfloat16* hidden, const TargetQsaPreprocessWeights& weights,
    const TargetQsaPreprocessBuffers& buffers, const TargetQsaStateView& state,
    cudaStream_t stream) {
  require(hidden && weights.main_q_norm && weights.main_k_norm &&
              weights.index_qk && weights.index_q_norm &&
              weights.index_k_norm && weights.rope_cos_sin &&
              buffers.raw_main_qkv && buffers.index_projected_qk &&
              buffers.main_query && buffers.attention_gate &&
              buffers.index_query && stream,
          "complete target QSA preprocessing bindings are required");
  validate_target_qsa_state_view(state, state.rank, state.layer,
                                 state.generation);
  // Positions and cache slots remain device-resident and graph-stable.
  index_qk_gemv<<<kIndexOutputs, 256, 0, stream>>>(
      hidden, weights.index_qk, buffers.index_projected_qk);
  main_qk_norm_rope_gate<<<kMainHeads + 1, 256, 0, stream>>>(
      buffers.raw_main_qkv, weights.main_q_norm, weights.main_k_norm,
      weights.rope_cos_sin, state.positions, state.uses_mrope ? 1 : 0,
      buffers.main_query, buffers.attention_gate,
      state.main_key_cache, state.main_value_cache, state.main_slot_mapping,
      state.main_blocks);
  index_norm_rope_and_raw_store<<<kIndexHeads, 128, 0, stream>>>(
      buffers.index_projected_qk, weights.index_q_norm, weights.rope_cos_sin,
      buffers.index_query, state.raw_key_cache, state.raw_slot_mapping,
      state.positions, state.uses_mrope ? 1 : 0);
  compress_index_key<<<1, kIndexDim, 0, stream>>>(
      state.raw_key_cache, state.compressed_key_cache, state.raw_block_table,
      state.compressed_slot_mapping, state.token_to_request,
      state.logical_positions, weights.index_k_norm, weights.rope_cos_sin,
      state.compressed_blocks, state.uses_mrope ? 1 : 0);
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw TargetQsaPreprocessError(cudaGetErrorString(status));
  }
}

}  // namespace rocket::qwen38::attention

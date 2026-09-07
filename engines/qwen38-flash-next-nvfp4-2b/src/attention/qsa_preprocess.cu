// SPDX-License-Identifier: Apache-2.0
// Specialized from vLLM 8e685d198 fused_qk_norm_rope.py and FlashInfer
// 91bda04c include/flashinfer/norm/fused_qk_rmsnorm_rope.cuh (Apache-2.0).
#include "attention/qsa_preprocess.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <string>

namespace {

constexpr int kHidden = 2560;
constexpr int kHeads = 12;
constexpr int kHeadDim = 256;
constexpr int kQGate = kHeads * kHeadDim * 2;
constexpr int kQkvWidth = kQGate + 2 * kHeadDim;
constexpr int kIndexHeads = 4;
constexpr int kIndexDim = 128;
constexpr int kIndexWidth = (kIndexHeads + 1) * kIndexDim;
constexpr int kRotary = 64;
constexpr float kEpsilon = 1.0e-6F;
constexpr float kTheta = 1.0e7F;
thread_local std::string error;

bool ok(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  error = std::string(operation) + ": " + cudaGetErrorString(status);
  return false;
}

__device__ float warp_sum(float value) {
  for (int delta = 16; delta; delta >>= 1)
    value += __shfl_down_sync(0xffffffffU, value, delta);
  return value;
}

template <int Dim>
__device__ float block_sum(float value, float* scratch) {
  value = warp_sum(value);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  value = threadIdx.x < blockDim.x / 32 ? scratch[lane] : 0.0F;
  if (warp == 0) value = warp_sum(value);
  if (threadIdx.x == 0) scratch[0] = value;
  __syncthreads();
  return scratch[0];
}

__device__ int rope_axis(int frequency) {
  if (frequency % 3 == 1 && frequency < 33) return 1;
  if (frequency % 3 == 2 && frequency < 30) return 2;
  return 0;
}

template <int Dim>
__device__ void norm_rope_head(const __nv_bfloat16* input,
                               const __nv_bfloat16* weight,
                               const std::int64_t* positions, int rows,
                               int row, __nv_bfloat16* output) {
  __shared__ float scratch[8];
  const int dim = threadIdx.x;
  float value = dim < Dim ? __bfloat162float(input[dim]) : 0.0F;
  const float sum = block_sum<Dim>(dim < Dim ? value * value : 0.0F, scratch);
  const float inverse = rsqrtf(sum / Dim + kEpsilon);
  if (dim >= Dim) return;
  // GemmaRMSNorm stores the residual from one, hence weight + 1.
  value *= inverse * (1.0F + __bfloat162float(weight[dim]));
  value = __bfloat162float(__float2bfloat16(value));
  if (dim < kRotary) {
    const int frequency = dim % (kRotary / 2);
    const int axis = rope_axis(frequency);
    const float exponent = -2.0F * frequency / kRotary;
    const float angle = static_cast<float>(positions[axis * rows + row]) *
                        powf(kTheta, exponent);
    const float cosine = __bfloat162float(__float2bfloat16(cosf(angle)));
    const float sine = __bfloat162float(__float2bfloat16(sinf(angle)));
    const int pair = dim < kRotary / 2 ? dim + kRotary / 2
                                       : dim - kRotary / 2;
    float other = __bfloat162float(input[pair]);
    other *= inverse * (1.0F + __bfloat162float(weight[pair]));
    other = __bfloat162float(__float2bfloat16(other));
    value = dim < kRotary / 2 ? value * cosine - other * sine
                              : value * cosine + other * sine;
  }
  output[dim] = __float2bfloat16(value);
}

__global__ void postprocess_qkv(const __nv_bfloat16* qkv,
                                const __nv_bfloat16* q_norm,
                                const __nv_bfloat16* k_norm,
                                const std::int64_t* positions, int rows,
                                __nv_bfloat16* query, __nv_bfloat16* key,
                                __nv_bfloat16* value, __nv_bfloat16* gate) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  if (row >= rows) return;
  const __nv_bfloat16* row_input = qkv + static_cast<std::size_t>(row) * kQkvWidth;
  if (head < kHeads) {
    const __nv_bfloat16* q = row_input + head * 2 * kHeadDim;
    norm_rope_head<kHeadDim>(q, q_norm, positions, rows, row,
                             query + (row * kHeads + head) * kHeadDim);
    const int dim = threadIdx.x;
    if (dim < kHeadDim)
      gate[(row * kHeads + head) * kHeadDim + dim] = q[kHeadDim + dim];
  } else if (head == kHeads) {
    norm_rope_head<kHeadDim>(row_input + kQGate, k_norm, positions, rows, row,
                             key + row * kHeadDim);
  } else {
    const int dim = threadIdx.x;
    if (dim < kHeadDim)
      value[row * kHeadDim + dim] = row_input[kQGate + kHeadDim + dim];
  }
}

__global__ void index_projection(const __nv_bfloat16* hidden,
                                 const __nv_bfloat16* first,
                                 const __nv_bfloat16* second, int rows,
                                 __nv_bfloat16* output) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const int column = blockIdx.y;
  if (row >= rows || column >= kIndexWidth) return;
  const __nv_bfloat16* weight =
      column < 320 ? first + static_cast<std::size_t>(column) * kHidden
                   : second + static_cast<std::size_t>(column - 320) * kHidden;
  float sum = 0.0F;
  for (int k = threadIdx.x; k < kHidden; k += blockDim.x)
    sum = fmaf(__bfloat162float(hidden[row * kHidden + k]),
               __bfloat162float(weight[k]), sum);
  sum = block_sum<kHidden>(sum, scratch);
  if (threadIdx.x == 0) output[row * kIndexWidth + column] = __float2bfloat16(sum);
}

__global__ void postprocess_index(const __nv_bfloat16* projected,
                                  const __nv_bfloat16* q_norm,
                                  const __nv_bfloat16* k_norm,
                                  const std::int64_t* positions, int rows,
                                  __nv_bfloat16* query,
                                  __nv_bfloat16* raw_key) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  if (row >= rows) return;
  if (head < kIndexHeads)
    norm_rope_head<kIndexDim>(projected + (row * kIndexWidth + head * kIndexDim),
                              q_norm, positions, rows, row,
                              query + (row * kIndexHeads + head) * kIndexDim);
  else if (threadIdx.x < kIndexDim)
    raw_key[row * kIndexDim + threadIdx.x] =
        projected[row * kIndexWidth + kIndexHeads * kIndexDim + threadIdx.x];
}

__global__ void format_main_rows(const __nv_bfloat16* key,
                                 const __nv_bfloat16* value, int rows,
                                 __nv_fp8_e4m3* output) {
  const int row = blockIdx.x;
  const int dim = threadIdx.x;
  if (row >= rows || dim >= 2 * kHeadDim) return;
  const __nv_bfloat16 source = dim < kHeadDim
                                   ? key[row * kHeadDim + dim]
                                   : value[row * kHeadDim + dim - kHeadDim];
  output[row * 2 * kHeadDim + dim] =
      __nv_fp8_e4m3(__bfloat162float(source));
}

__global__ void format_side_rows(
    const __nv_bfloat16* raw_key, const __nv_bfloat16* k_norm,
    const std::int64_t* positions, const std::int64_t* logical_positions,
    const std::int32_t* token_to_request,
    const std::uint8_t* active_raw, int rows, std::uint8_t* raw_rows,
    __nv_bfloat16* compressed_rows) {
  __shared__ float scratch[4];
  __shared__ std::int64_t first_positions[3];
  const int row = blockIdx.x;
  const int dim = threadIdx.x;
  if (row >= rows || dim >= kIndexDim) return;
  auto* raw_output = reinterpret_cast<__nv_bfloat16*>(
      raw_rows + static_cast<std::size_t>(row) * 280);
  raw_output[dim] = raw_key[row * kIndexDim + dim];
  if (dim < 3) {
    reinterpret_cast<std::int64_t*>(raw_output + kIndexDim)[dim] =
        positions[dim * rows + row];
  }
  const std::int64_t end = logical_positions[row];
  const int request = token_to_request[row];
  if (request < 0 || request >= 16 || end < 0 || end >= 262144) {
    compressed_rows[row * kIndexDim + dim] = __float2bfloat16(0.0F);
    return;
  }
  if (end % 4 != 3) {
    compressed_rows[row * kIndexDim + dim] = __float2bfloat16(0.0F);
    return;
  }
  float pooled = 0.0F;
  for (int offset = 0; offset < 4; ++offset) {
    const std::int64_t position = end - 3 + offset;
    const __nv_bfloat16* source = nullptr;
    int source_candidate = -1;
    for (int candidate = 0; candidate <= row; ++candidate) {
      if (token_to_request[candidate] == request &&
          logical_positions[candidate] == position) {
        source = raw_key + candidate * kIndexDim;
        source_candidate = candidate;
      }
    }
    if (!source) {
      source = reinterpret_cast<const __nv_bfloat16*>(
          active_raw +
          (static_cast<std::size_t>(request) * 8 + position % 8) * 280);
    }
    pooled += __bfloat162float(source[dim]);
    if (offset == 0 && dim < 3) {
      if (source_candidate >= 0) {
        first_positions[dim] = positions[dim * rows + source_candidate];
      } else {
        first_positions[dim] =
            reinterpret_cast<const std::int64_t*>(source + kIndexDim)[dim];
      }
    }
  }
  pooled = __bfloat162float(__float2bfloat16(pooled * 0.25F));
  const float sum = block_sum<kIndexDim>(pooled * pooled, scratch);
  float normalized = pooled * rsqrtf(sum / kIndexDim + kEpsilon) *
                     (1.0F + __bfloat162float(k_norm[dim]));
  normalized = __bfloat162float(__float2bfloat16(normalized));
  if (dim < kRotary) {
    const int frequency = dim % 32;
    const int axis = rope_axis(frequency);
    const float angle = static_cast<float>(first_positions[axis]) *
                        powf(kTheta, -2.0F * frequency / kRotary);
    const float cosine = __bfloat162float(__float2bfloat16(cosf(angle)));
    const float sine = __bfloat162float(__float2bfloat16(sinf(angle)));
    const int pair = dim < 32 ? dim + 32 : dim - 32;
    // Recompute the paired pooled value. The fixed four-row group keeps this
    // bounded and avoids a second shared 128-float buffer.
    float paired = 0.0F;
    for (int offset = 0; offset < 4; ++offset) {
      const std::int64_t position = end - 3 + offset;
      const __nv_bfloat16* source = nullptr;
      for (int candidate = 0; candidate <= row; ++candidate)
        if (token_to_request[candidate] == request &&
            logical_positions[candidate] == position)
          source = raw_key + candidate * kIndexDim;
      if (!source)
        source = reinterpret_cast<const __nv_bfloat16*>(
            active_raw +
            (static_cast<std::size_t>(request) * 8 + position % 8) * 280);
      paired += __bfloat162float(source[pair]);
    }
    paired = __bfloat162float(__float2bfloat16(paired * 0.25F));
    paired *= rsqrtf(sum / kIndexDim + kEpsilon) *
              (1.0F + __bfloat162float(k_norm[pair]));
    paired = __bfloat162float(__float2bfloat16(paired));
    normalized = dim < 32 ? normalized * cosine - paired * sine
                          : normalized * cosine + paired * sine;
  }
  compressed_rows[row * kIndexDim + dim] = __float2bfloat16(normalized);
}

__global__ void apply_gate(__nv_bfloat16* attention,
                           const __nv_bfloat16* gate, int rows) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * kHeads * kHeadDim;
  if (index < elements) {
    const float x = __bfloat162float(gate[index]);
    const float sigmoid = 1.0F / (1.0F + expf(-x));
    attention[index] = __float2bfloat16(
        __bfloat162float(attention[index]) * sigmoid);
  }
}

}  // namespace

extern "C" int qwen38_qsa_preprocess(
    const __nv_bfloat16* hidden, const __nv_bfloat16* qkv,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    const __nv_bfloat16* index_qk_first,
    const __nv_bfloat16* index_qk_second,
    const __nv_bfloat16* index_q_norm,
    const __nv_bfloat16* index_k_norm,
    const std::int64_t* positions, int rows,
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    __nv_bfloat16* gate, __nv_bfloat16* index_query,
    __nv_bfloat16* index_raw_key, __nv_bfloat16* index_projected_scratch,
    cudaStream_t stream) {
  error.clear();
  if (!hidden || !qkv || !q_norm || !k_norm || !index_qk_first ||
      !index_qk_second || !index_q_norm || !index_k_norm || !positions ||
      !query || !key || !value || !gate || !index_query || !index_raw_key ||
      !index_projected_scratch || !stream || rows < 1 || rows > 128) {
    error = "invalid exact QSA preprocessing arguments";
    return 1;
  }
  postprocess_qkv<<<dim3(rows, kHeads + 2), 256, 0, stream>>>(
      qkv, q_norm, k_norm, positions, rows, query, key, value, gate);
  if (!ok(cudaGetLastError(), "postprocess QKV")) return 1;
  index_projection<<<dim3(rows, kIndexWidth), 256, 0, stream>>>(
      hidden, index_qk_first, index_qk_second, rows, index_projected_scratch);
  if (!ok(cudaGetLastError(), "project replicated index QK")) return 1;
  postprocess_index<<<dim3(rows, kIndexHeads + 1), 128, 0, stream>>>(
      index_projected_scratch, index_q_norm, index_k_norm, positions, rows, index_query,
      index_raw_key);
  return ok(cudaGetLastError(), "postprocess index QK") ? 0 : 1;
}

extern "C" int qwen38_qsa_format_state_rows(
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    const __nv_bfloat16* index_raw_key,
    const __nv_bfloat16* index_k_norm,
    const std::int64_t* positions, const std::int64_t* logical_positions,
    const std::int32_t* token_to_request, const void* active_raw_state,
    int rows, void* main_rows_fp8, void* raw_rows_bf16,
    __nv_bfloat16* compressed_rows, cudaStream_t stream) {
  error.clear();
  if (!key || !value || !index_raw_key || !index_k_norm || !positions ||
      !logical_positions || !token_to_request || !active_raw_state ||
      !main_rows_fp8 || !raw_rows_bf16 || !compressed_rows || !stream ||
      rows < 1 || rows > 128) {
    error = "invalid exact QSA state formatting arguments";
    return 1;
  }
  format_main_rows<<<rows, 512, 0, stream>>>(
      key, value, rows, static_cast<__nv_fp8_e4m3*>(main_rows_fp8));
  if (!ok(cudaGetLastError(), "format FP8 main-cache rows")) return 1;
  format_side_rows<<<rows, 128, 0, stream>>>(
      index_raw_key, index_k_norm, positions, logical_positions,
      token_to_request, static_cast<const std::uint8_t*>(active_raw_state),
      rows, static_cast<std::uint8_t*>(raw_rows_bf16), compressed_rows);
  return ok(cudaGetLastError(), "format QSA side-cache rows") ? 0 : 1;
}

extern "C" int qwen38_qsa_apply_output_gate(__nv_bfloat16* attention,
                                               const __nv_bfloat16* gate,
                                               int rows,
                                               cudaStream_t stream) {
  error.clear();
  if (!attention || !gate || !stream || rows < 1 || rows > 128) {
    error = "invalid exact QSA output gate arguments";
    return 1;
  }
  const int elements = rows * kHeads * kHeadDim;
  apply_gate<<<(elements + 255) / 256, 256, 0, stream>>>(attention, gate, rows);
  return ok(cudaGetLastError(), "apply sigmoid output gate") ? 0 : 1;
}

extern "C" const char* qwen38_qsa_preprocess_last_error() {
  return error.c_str();
}

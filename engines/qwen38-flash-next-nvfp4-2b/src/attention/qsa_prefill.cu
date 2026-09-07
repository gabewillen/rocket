// SPDX-License-Identifier: Apache-2.0
// Query-pair union tiling follows FlashInfer 91bda04c
// flashinfer/msa_ops/cute_dsl/sparse_prefill_sm12x.py (Apache-2.0).
// The scalar control follows vLLM 8e685d198
// models/qwen3_8_flash_next/nvidia/ops/qsa.py (Apache-2.0).
#include "attention/qsa_prefill.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <array>
#include <cmath>
#include <mutex>
#include <string>

namespace {
constexpr int kHeads = 12;
constexpr int kDim = 256;
constexpr int kTopk = 2051;
constexpr int kTile = 16;
constexpr int kThreads = 512;
constexpr int kQueryRows = 32;
constexpr std::size_t kQueryElements = kQueryRows * kDim;
constexpr std::size_t kKvElements = 2 * kTile * kDim;
constexpr std::size_t kAccumulatorElements = kQueryRows * kDim;
constexpr std::size_t kScoreElements = kQueryRows * kTile;
constexpr std::size_t kSharedBytes =
    (kQueryElements + 2 * kKvElements + kScoreElements) *
        sizeof(__nv_bfloat16) +
    (kAccumulatorElements + kScoreElements) * sizeof(float) +
    512;
thread_local std::string error;
std::mutex prepare_mutex;
std::array<bool, 32> prepared{};

bool ok(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  error = std::string(operation) + ": " + cudaGetErrorString(status);
  return false;
}

struct Shared {
  __nv_bfloat16* query;
  __nv_bfloat16* key;
  __nv_bfloat16* value;
  __nv_bfloat16* probabilities;
  float* accumulator;
  float* scores;
  float* maxima;
  float* normalizers;
  float* alpha;
  std::int32_t* logical;
  std::uint8_t* mask;
  std::int32_t* merge;
};

__device__ Shared bind_shared(unsigned char* base) {
  Shared s{};
  auto* bf16 = reinterpret_cast<__nv_bfloat16*>(base);
  s.query = bf16;
  s.key = s.query + kQueryElements;
  s.value = s.key + kKvElements;
  s.probabilities = s.value + kKvElements;
  s.accumulator = reinterpret_cast<float*>(s.probabilities + kScoreElements);
  s.scores = s.accumulator + kAccumulatorElements;
  s.maxima = s.scores + kScoreElements;
  s.normalizers = s.maxima + kQueryRows;
  s.alpha = s.normalizers + kQueryRows;
  s.logical = reinterpret_cast<std::int32_t*>(s.alpha + kQueryRows);
  s.mask = reinterpret_cast<std::uint8_t*>(s.logical + kTile);
  s.merge = reinterpret_cast<std::int32_t*>(s.mask + kTile);
  return s;
}

__global__ void union2_prefill(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const std::int32_t* __restrict__ indices, int query_tokens,
    int context_tokens,
    __nv_bfloat16* __restrict__ output,
    Qwen38QsaPrefillCounters* __restrict__ counters) {
  using namespace nvcuda;
  extern __shared__ __align__(16) unsigned char storage[];
  Shared s = bind_shared(storage);
  const int sequence = blockIdx.y;
  const int first = blockIdx.x * 2;
  const int row0 = sequence * query_tokens + first;
  const int row1 = row0 + 1;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const bool second = first + 1 < query_tokens;
  for (int index = tid; index < kQueryElements; index += blockDim.x) {
    const int local_row = index / kDim;
    const int dim = index % kDim;
    const int row = local_row < 16 ? row0 : row1;
    const int head = local_row & 15;
    s.query[index] = head < kHeads && (local_row < 16 || second)
                         ? query[(row * kHeads + head) * kDim + dim]
                         : __float2bfloat16(0.0F);
    s.accumulator[index] = 0.0F;
  }
  if (tid < kQueryRows) {
    s.maxima[tid] = -INFINITY;
    s.normalizers[tid] = 0.0F;
  }
  if (tid == 0) {
    for (int item = 0; item < 5; ++item) s.merge[item] = 0;
  }
  __syncthreads();

  while (true) {
    if (tid == 0) {
      for (int slot = 0; slot < kTile; ++slot) {
        const bool load_a = s.merge[0] < kTopk;
        const bool load_b = second && s.merge[1] < kTopk;
        const int a =
            load_a ? indices[row0 * kTopk + s.merge[0]] : -1;
        const int b = load_b
                          ? indices[row1 * kTopk + s.merge[1]]
                          : -1;
        s.merge[4] += load_a + load_b;
        if (a < 0 && b < 0) {
          s.logical[slot] = -1;
          s.mask[slot] = 0;
          s.merge[0] = kTopk;
          s.merge[1] = kTopk;
          continue;
        }
        int logical = -1;
        std::uint8_t mask = 0;
        if (b < 0 || (a >= 0 && a < b)) {
          logical = a;
          mask = 1;
          ++s.merge[0];
        } else if (a < 0 || b < a) {
          logical = b;
          mask = 2;
          ++s.merge[1];
        } else {
          logical = a;
          mask = 3;
          ++s.merge[0];
          ++s.merge[1];
        }
        s.logical[slot] = logical;
        s.mask[slot] = mask;
        if (logical >= 0) {
          ++s.merge[2];
          s.merge[3] += (mask & 1U) + ((mask >> 1U) & 1U);
        }
      }
    }
    __syncthreads();
    if (s.mask[0] == 0) break;

    for (int index = tid; index < kTile * kDim; index += blockDim.x) {
      const int token = index / kDim;
      const int dim = index % kDim;
      const int logical = s.logical[token];
      const std::size_t base =
          (static_cast<std::size_t>(sequence) * context_tokens +
           max(logical, 0)) *
          kDim;
      const __nv_bfloat16 kval =
          logical >= 0 ? key[base + dim] : __float2bfloat16(0.0F);
      const __nv_bfloat16 vval =
          logical >= 0 ? value[base + dim] : __float2bfloat16(0.0F);
      s.key[index] = kval;
      s.value[index] = vval;
      s.key[kTile * kDim + index] = kval;
      s.value[kTile * kDim + index] = vval;
    }
    __syncthreads();

    if (warp < 2) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> product;
      wmma::fill_fragment(product, 0.0F);
      for (int k = 0; k < kDim; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major> q_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                       wmma::col_major> k_fragment;
        wmma::load_matrix_sync(q_fragment,
                               s.query + warp * 16 * kDim + k, kDim);
        wmma::load_matrix_sync(k_fragment,
                               s.key + warp * kTile * kDim + k, kDim);
        wmma::mma_sync(product, q_fragment, k_fragment, product);
      }
      wmma::store_matrix_sync(s.scores + warp * 16 * kTile, product,
                              kTile, wmma::mem_row_major);
    }
    __syncthreads();

    if (tid < kQueryRows) {
      const int pair_row = tid >> 4;
      const int head = tid & 15;
      float tile_maximum = -INFINITY;
      for (int token = 0; token < kTile; ++token) {
        const bool valid = head < kHeads &&
                           (s.mask[token] & (1U << pair_row));
        if (valid)
          tile_maximum = fmaxf(
              tile_maximum, s.scores[tid * kTile + token] * 0.0625F);
      }
      const float next = fmaxf(s.maxima[tid], tile_maximum);
      const float rescale =
          isfinite(s.maxima[tid]) ? expf(s.maxima[tid] - next) : 0.0F;
      float tile_normalizer = 0.0F;
      for (int token = 0; token < kTile; ++token) {
        const bool valid = head < kHeads &&
                           (s.mask[token] & (1U << pair_row));
        const float probability =
            valid ? expf(s.scores[tid * kTile + token] * 0.0625F - next)
                  : 0.0F;
        s.probabilities[tid * kTile + token] =
            __float2bfloat16(probability);
        tile_normalizer += probability;
      }
      s.alpha[tid] = rescale;
      s.normalizers[tid] =
          s.normalizers[tid] * rescale + tile_normalizer;
      s.maxima[tid] = next;
    }
    __syncthreads();
    for (int index = tid; index < kAccumulatorElements;
         index += blockDim.x)
      s.accumulator[index] *= s.alpha[index / kDim];
    __syncthreads();

    for (int pair_row = 0; pair_row < 2; ++pair_row) {
      const int dimension_tile = warp;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major> p_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major> v_fragment;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_fragment;
      const int row_offset = pair_row * 16 * kDim;
      wmma::load_matrix_sync(p_fragment,
                             s.probabilities + pair_row * 16 * kTile, kTile);
      wmma::load_matrix_sync(v_fragment,
                             s.value + pair_row * kTile * kDim +
                                 dimension_tile * 16,
                             kDim);
      wmma::load_matrix_sync(out_fragment,
                             s.accumulator + row_offset + dimension_tile * 16,
                             kDim, wmma::mem_row_major);
      wmma::mma_sync(out_fragment, p_fragment, v_fragment, out_fragment);
      wmma::store_matrix_sync(
          s.accumulator + row_offset + dimension_tile * 16, out_fragment,
          kDim, wmma::mem_row_major);
    }
    __syncthreads();
  }

  for (int index = tid; index < kQueryElements; index += blockDim.x) {
    const int local_row = index / kDim;
    const int pair_row = local_row >> 4;
    const int head = local_row & 15;
    if (head < kHeads && (pair_row == 0 || second)) {
      const int row = pair_row == 0 ? row0 : row1;
      const float norm = s.normalizers[local_row];
      output[(row * kHeads + head) * kDim + index % kDim] =
          __float2bfloat16(norm > 0.0F ? s.accumulator[index] / norm : 0.0F);
    }
  }
  if (tid == 0 && counters) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&counters->union_tokens),
              static_cast<unsigned long long>(s.merge[2]));
    atomicAdd(
        reinterpret_cast<unsigned long long*>(&counters->selected_tokens),
        static_cast<unsigned long long>(s.merge[3]));
    atomicAdd(reinterpret_cast<unsigned long long*>(&counters->index_loads),
              static_cast<unsigned long long>(s.merge[4]));
  }
}

__global__ void scalar_prefill(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const std::int32_t* __restrict__ indices, int query_tokens,
    int context_tokens,
    __nv_bfloat16* __restrict__ output) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  const int sequence = row / query_tokens;
  float query_registers[8];
  float accumulator[8]{};
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    const int dim = lane + item * 32;
    query_registers[item] =
        __bfloat162float(query[(row * kHeads + head) * kDim + dim]);
  }
  float maximum = -INFINITY;
  float normalizer = 0.0F;
  for (int selected = 0; selected < kTopk; ++selected) {
    const int logical = indices[row * kTopk + selected];
    const bool valid = logical >= 0;
    const std::size_t base =
        (static_cast<std::size_t>(sequence) * context_tokens +
         max(logical, 0)) *
        kDim;
    float dot = 0.0F;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const int dim = lane + item * 32;
      if (valid)
        dot += query_registers[item] * __bfloat162float(key[base + dim]);
    }
    for (int delta = 16; delta; delta >>= 1)
      dot += __shfl_down_sync(0xffffffffU, dot, delta);
    const float score = __shfl_sync(0xffffffffU,
                                    valid ? dot * 0.0625F : -INFINITY, 0);
    const float previous = maximum;
    const float next = fmaxf(previous, score);
    const float alpha = isfinite(previous) ? expf(previous - next) : 0.0F;
    const float probability = valid ? expf(score - next) : 0.0F;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const int dim = lane + item * 32;
      accumulator[item] =
          accumulator[item] * alpha +
          probability *
              (valid ? __bfloat162float(value[base + dim]) : 0.0F);
    }
    normalizer = normalizer * alpha + probability;
    maximum = next;
  }
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    const int dim = lane + item * 32;
    output[(row * kHeads + head) * kDim + dim] = __float2bfloat16(
        normalizer > 0.0F ? accumulator[item] / normalizer : 0.0F);
  }
}

bool shape_ok(int sequences, int query_tokens, int context_tokens) {
  const bool bucket = sequences == 1 || sequences == 2 || sequences == 4 ||
                      sequences == 8 || sequences == 16;
  return bucket &&
         ((query_tokens == 300 && context_tokens == 8492) ||
          (query_tokens == 8192 && context_tokens == 8192));
}
}  // namespace

extern "C" int qwen38_qsa_prefill_prepare(int device) {
  error.clear();
  if (device < 0 || device >= static_cast<int>(prepared.size())) {
    error = "invalid QSA prefill device";
    return 1;
  }
  std::lock_guard<std::mutex> lock(prepare_mutex);
  if (prepared[device]) return 0;
  if (!ok(cudaSetDevice(device), "set QSA prefill device") ||
      !ok(cudaFuncSetAttribute(union2_prefill,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(kSharedBytes)),
          "opt in QSA prefill shared memory"))
    return 1;
  prepared[device] = true;
  return 0;
}

extern "C" int qwen38_qsa_prefill_union2(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const std::int32_t* indices, int sequences,
    int query_tokens, int context_tokens, __nv_bfloat16* output,
    Qwen38QsaPrefillCounters* counters,
    cudaStream_t stream) {
  error.clear();
  int device = -1;
  if (!query || !key || !value || !indices || !output ||
      !stream || !shape_ok(sequences, query_tokens, context_tokens) ||
      cudaGetDevice(&device) != cudaSuccess || device < 0 ||
      device >= static_cast<int>(prepared.size()) || !prepared[device]) {
    error = "QSA prefill union launch contract changed";
    return 1;
  }
  union2_prefill<<<dim3((query_tokens + 1) / 2, sequences), kThreads,
                   kSharedBytes,
                   stream>>>(query, key, value, indices, query_tokens,
                             context_tokens, output, counters);
  return ok(cudaGetLastError(), "launch QSA union prefill") ? 0 : 1;
}

extern "C" int qwen38_qsa_prefill_control(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const std::int32_t* indices, int sequences,
    int query_tokens, int context_tokens, __nv_bfloat16* output,
    cudaStream_t stream) {
  error.clear();
  if (!query || !key || !value || !indices || !output || !stream ||
      !shape_ok(sequences, query_tokens, context_tokens)) {
    error = "QSA prefill control launch contract changed";
    return 1;
  }
  scalar_prefill<<<dim3(sequences * query_tokens, kHeads), 32, 0, stream>>>(
      query, key, value, indices, query_tokens, context_tokens, output);
  return ok(cudaGetLastError(), "launch scalar QSA prefill") ? 0 : 1;
}

extern "C" const char* qwen38_qsa_prefill_last_error() {
  return error.c_str();
}

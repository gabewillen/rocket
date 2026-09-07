// SPDX-License-Identifier: Apache-2.0
// Fixed-shape adaptation of vLLM 8e685d198
// vllm/third_party/flash_linear_attention/ops/fused_recurrent.py and
// FlashInfer's Apache-2.0 gdn_decode_pretranspose.py and gdn_decode_mtp.py.
// General sequence, layout, dtype, and head-shape dispatch are removed for
// Qwen3.8 TP2.
#include "linear_attention/gdn_core.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <new>
#include <stdexcept>
#include <string>

namespace rocket::qwen38::linear_attention {
namespace {

thread_local std::string last_error;
constexpr int kThreads = 128;
constexpr int kRowsPerBlock = 16;
constexpr int kValueTiles = kHeadDim / kRowsPerBlock;
constexpr float kEpsilon = 1.0e-6F;

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

__device__ __forceinline__ float block_sum(float value, float* warp_sums) {
  for (int delta = 16; delta > 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  float total = threadIdx.x < 4 ? warp_sums[lane] : 0.0F;
  if (warp == 0) {
    for (int delta = 16; delta > 0; delta >>= 1) {
      total += __shfl_down_sync(0xffffffffU, total, delta);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) warp_sums[0] = total;
  __syncthreads();
  const float result = warp_sums[0];
  // No warp may reuse warp_sums for the next back-to-back reduction until
  // every thread has consumed this result.
  __syncthreads();
  return result;
}

__global__ void causal_conv_update(
    const __nv_bfloat16* qkvz, const __nv_bfloat16* weight,
    __nv_bfloat16* state, std::size_t slot_stride,
    const std::int32_t* state_indices, __nv_bfloat16* mixed_qkv, int m) {
  const int row = blockIdx.y;
  const int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || dim >= kQkvWidth) return;
  const int slot = state_indices[row];
  if (slot <= 0) {
    mixed_qkv[row * kQkvWidth + dim] = __float2bfloat16(0.0F);
    return;
  }
  __nv_bfloat16* history = state + static_cast<std::size_t>(slot) * slot_stride;
  const float x0 = __bfloat162float(history[dim]);
  const float x1 = __bfloat162float(history[kQkvWidth + dim]);
  const float x2 = __bfloat162float(history[2 * kQkvWidth + dim]);
  const float x3 = __bfloat162float(qkvz[row * (kQkvWidth + kGateWidth) + dim]);
  history[dim] = __float2bfloat16(x1);
  history[kQkvWidth + dim] = __float2bfloat16(x2);
  history[2 * kQkvWidth + dim] = __float2bfloat16(x3);
  const std::size_t w = static_cast<std::size_t>(dim) * kConvKernel;
  float value = x0 * __bfloat162float(weight[w]) +
                x1 * __bfloat162float(weight[w + 1]) +
                x2 * __bfloat162float(weight[w + 2]) +
                x3 * __bfloat162float(weight[w + 3]);
  value *= 1.0F / (1.0F + __expf(-value));
  mixed_qkv[row * kQkvWidth + dim] = __float2bfloat16(value);
}

__global__ void causal_conv_verify(
    const __nv_bfloat16* qkvz, const __nv_bfloat16* weight,
    const __nv_bfloat16* accepted_state,
    const std::int32_t* accepted_indices, __nv_bfloat16* mixed_qkv,
    int sequences, int verify_width) {
  const int sequence = blockIdx.y;
  const int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (sequence >= sequences || dim >= kQkvWidth) return;
  const int slot = accepted_indices[sequence];
  if (slot <= 0) {
    for (int prefix = 0; prefix < verify_width; ++prefix) {
      const int row = prefix * sequences + sequence;
      mixed_qkv[static_cast<std::size_t>(row) * kQkvWidth + dim] =
          __float2bfloat16(0.0F);
    }
    return;
  }
  const __nv_bfloat16* history = accepted_state +
      static_cast<std::size_t>(slot) * kConvStateRows * kQkvWidth;
  float x0 = __bfloat162float(history[dim]);
  float x1 = __bfloat162float(history[kQkvWidth + dim]);
  float x2 = __bfloat162float(history[2 * kQkvWidth + dim]);
  for (int prefix = 0; prefix < verify_width; ++prefix) {
    const int row = prefix * sequences + sequence;
    const float x3 = __bfloat162float(
        qkvz[static_cast<std::size_t>(row) * (kQkvWidth + kGateWidth) + dim]);
    const std::size_t w = static_cast<std::size_t>(dim) * kConvKernel;
    float value = x0 * __bfloat162float(weight[w]) +
                  x1 * __bfloat162float(weight[w + 1]) +
                  x2 * __bfloat162float(weight[w + 2]) +
                  x3 * __bfloat162float(weight[w + 3]);
    value *= 1.0F / (1.0F + __expf(-value));
    mixed_qkv[static_cast<std::size_t>(row) * kQkvWidth + dim] =
        __float2bfloat16(value);
    x0 = x1;
    x1 = x2;
    x2 = x3;
  }
}

__global__ void causal_conv_accept(
    const __nv_bfloat16* qkvz, __nv_bfloat16* accepted_state,
    const std::int32_t* accepted_indices,
    const std::int32_t* accepted_prefixes, int sequences) {
  const int sequence = blockIdx.y;
  const int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (sequence >= sequences || dim >= kQkvWidth) return;
  const int prefix_count = accepted_prefixes[sequence];
  const int slot = accepted_indices[sequence];
  if (prefix_count <= 0 || slot <= 0) return;
  __nv_bfloat16* history = accepted_state +
      static_cast<std::size_t>(slot) * kConvStateRows * kQkvWidth;
  float x0 = __bfloat162float(history[dim]);
  float x1 = __bfloat162float(history[kQkvWidth + dim]);
  float x2 = __bfloat162float(history[2 * kQkvWidth + dim]);
  for (int prefix = 0; prefix < prefix_count; ++prefix) {
    const int row = prefix * sequences + sequence;
    const float x3 = __bfloat162float(
        qkvz[static_cast<std::size_t>(row) * (kQkvWidth + kGateWidth) + dim]);
    x0 = x1;
    x1 = x2;
    x2 = x3;
  }
  history[dim] = __float2bfloat16(x0);
  history[kQkvWidth + dim] = __float2bfloat16(x1);
  history[2 * kQkvWidth + dim] = __float2bfloat16(x2);
}

__global__ __launch_bounds__(kThreads) void recurrent_gdn(
    const __nv_bfloat16* mixed_qkv, const __nv_bfloat16* ba,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    float* state, std::size_t slot_stride,
    const std::int32_t* state_indices, __nv_bfloat16* output, int m) {
  const int tile = blockIdx.x;
  const int hv = blockIdx.y;
  const int row = blockIdx.z;
  const int k_dim = threadIdx.x;
  if (row >= m) return;
  const int slot = state_indices[row];
  const int v0 = tile * kRowsPerBlock;
  if (slot <= 0) {
    for (int v = v0 + k_dim; v < v0 + kRowsPerBlock; v += kThreads) {
      output[(row * kValueHeads + hv) * kHeadDim + v] =
          __float2bfloat16(0.0F);
    }
    return;
  }

  __shared__ float q[kHeadDim];
  __shared__ float key[kHeadDim];
  __shared__ float warp_sums[4];
  const int h = hv / (kValueHeads / kKeyHeads);
  const std::size_t mixed_base = static_cast<std::size_t>(row) * kQkvWidth;
  q[k_dim] = __bfloat162float(mixed_qkv[mixed_base + h * kHeadDim + k_dim]);
  key[k_dim] = __bfloat162float(
      mixed_qkv[mixed_base + kKeyHeads * kHeadDim + h * kHeadDim + k_dim]);
  __syncthreads();
  const float q_norm = rsqrtf(block_sum(q[k_dim] * q[k_dim], warp_sums) + kEpsilon) *
                       0.08838834764831845F;
  const float k_norm = rsqrtf(block_sum(key[k_dim] * key[k_dim], warp_sums) +
                              kEpsilon);
  q[k_dim] *= q_norm;
  key[k_dim] *= k_norm;
  __syncthreads();

  const float a = __bfloat162float(ba[row * (2 * kValueHeads) +
                                      kValueHeads + hv]);
  const float beta = 1.0F /
      (1.0F + __expf(-__bfloat162float(ba[row * (2 * kValueHeads) + hv])));
  const float x = a + __bfloat162float(dt_bias[hv]);
  const float softplus = x <= 20.0F ? log1pf(__expf(x)) : x;
  // The authenticated serving slab preserves the checkpoint BF16 source.
  // Promotion happens before the FP32 decay arithmetic used by the FLA path.
  const float decay =
      __expf(-__expf(__bfloat162float(a_log[hv])) * softplus);
  float* head_state = state + static_cast<std::size_t>(slot) * slot_stride +
                      static_cast<std::size_t>(hv) * kHeadDim * kHeadDim;

  for (int v = v0; v < v0 + kRowsPerBlock; ++v) {
    float current = head_state[v * kHeadDim + k_dim] * decay;
    const float prediction = block_sum(current * key[k_dim], warp_sums);
    const float v_value = __bfloat162float(
        mixed_qkv[mixed_base + 2 * kKeyHeads * kHeadDim + hv * kHeadDim + v]);
    const float delta = beta * (v_value - prediction);
    current += delta * key[k_dim];
    head_state[v * kHeadDim + k_dim] = current;
    const float projected = block_sum(current * q[k_dim], warp_sums);
    if (k_dim == 0) {
      output[(row * kValueHeads + hv) * kHeadDim + v] =
          __float2bfloat16(projected);
    }
  }
}

__global__ __launch_bounds__(kThreads) void recurrent_gdn_verify(
    const __nv_bfloat16* mixed_qkv, const __nv_bfloat16* ba,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    const float* accepted_state, const std::int32_t* accepted_indices,
    __nv_bfloat16* output, int sequences, int verify_width) {
  const int tile = blockIdx.x;
  const int hv = blockIdx.y;
  const int sequence = blockIdx.z;
  const int k_dim = threadIdx.x;
  if (sequence >= sequences) return;
  const int slot = accepted_indices[sequence];
  const int v0 = tile * kRowsPerBlock;
  if (slot <= 0) {
    for (int prefix = 0; prefix < verify_width; ++prefix) {
      const int row = prefix * sequences + sequence;
      for (int v = v0 + k_dim; v < v0 + kRowsPerBlock; v += kThreads) {
        output[(static_cast<std::size_t>(row) * kValueHeads + hv) *
                   kHeadDim + v] = __float2bfloat16(0.0F);
      }
    }
    return;
  }
  __shared__ float q[kMaxVerifyWidth][kHeadDim];
  __shared__ float key[kMaxVerifyWidth][kHeadDim];
  __shared__ float decay[kMaxVerifyWidth];
  __shared__ float beta[kMaxVerifyWidth];
  __shared__ float warp_sums[4];
  const int h = hv / (kValueHeads / kKeyHeads);
  const float* head_state = accepted_state +
      static_cast<std::size_t>(slot) * kValueHeads * kHeadDim * kHeadDim +
      static_cast<std::size_t>(hv) * kHeadDim * kHeadDim;

  // FlashInfer's MTP loop order: normalize each time step once, then keep
  // four independent V rows resident in registers across every prefix.
  for (int prefix = 0; prefix < verify_width; ++prefix) {
    const int row = prefix * sequences + sequence;
    const std::size_t mixed_base = static_cast<std::size_t>(row) * kQkvWidth;
    float q_value = __bfloat162float(
        mixed_qkv[mixed_base + h * kHeadDim + k_dim]);
    float key_value = __bfloat162float(
        mixed_qkv[mixed_base + kKeyHeads * kHeadDim + h * kHeadDim + k_dim]);
    q_value *= rsqrtf(block_sum(q_value * q_value, warp_sums) + kEpsilon) *
               0.08838834764831845F;
    key_value *= rsqrtf(block_sum(key_value * key_value, warp_sums) +
                        kEpsilon);
    q[prefix][k_dim] = q_value;
    key[prefix][k_dim] = key_value;
    const float a = __bfloat162float(
        ba[row * (2 * kValueHeads) + kValueHeads + hv]);
    beta[prefix] = 1.0F /
        (1.0F + __expf(-__bfloat162float(ba[row * (2 * kValueHeads) + hv])));
    const float x = a + __bfloat162float(dt_bias[hv]);
    const float softplus = x <= 20.0F ? log1pf(__expf(x)) : x;
    decay[prefix] =
        __expf(-__expf(__bfloat162float(a_log[hv])) * softplus);
  }
  __syncthreads();
  #pragma unroll
  for (int v_offset = 0; v_offset < kRowsPerBlock; v_offset += 4) {
    float current[4];
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      current[lane] = head_state[(v0 + v_offset + lane) * kHeadDim + k_dim];
    }
    for (int prefix = 0; prefix < verify_width; ++prefix) {
      const int row = prefix * sequences + sequence;
      const std::size_t mixed_base = static_cast<std::size_t>(row) * kQkvWidth;
      float prediction[4];
      #pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        current[lane] *= decay[prefix];
        prediction[lane] =
            block_sum(current[lane] * key[prefix][k_dim], warp_sums);
      }
      #pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        const int v = v0 + v_offset + lane;
        const float v_value = __bfloat162float(
            mixed_qkv[mixed_base + 2 * kKeyHeads * kHeadDim +
                      hv * kHeadDim + v]);
        current[lane] += beta[prefix] * (v_value - prediction[lane]) *
                         key[prefix][k_dim];
      }
      float projected[4];
      #pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        projected[lane] =
            block_sum(current[lane] * q[prefix][k_dim], warp_sums);
      }
      if (k_dim == 0) {
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
          const int v = v0 + v_offset + lane;
          output[(static_cast<std::size_t>(row) * kValueHeads + hv) *
                     kHeadDim + v] = __float2bfloat16(projected[lane]);
        }
      }
    }
  }
}

__global__ __launch_bounds__(kThreads) void recurrent_gdn_accept(
    const __nv_bfloat16* mixed_qkv, const __nv_bfloat16* ba,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias, float* state,
    const std::int32_t* accepted_indices,
    const std::int32_t* accepted_prefixes, int sequences) {
  const int tile = blockIdx.x;
  const int hv = blockIdx.y;
  const int sequence = blockIdx.z;
  const int k_dim = threadIdx.x;
  if (sequence >= sequences) return;
  const int prefix_count = accepted_prefixes[sequence];
  const int slot = accepted_indices[sequence];
  if (prefix_count <= 0 || slot <= 0) return;
  const int v0 = tile * kRowsPerBlock;
  __shared__ float key[kMaxVerifyWidth][kHeadDim];
  __shared__ float decay[kMaxVerifyWidth];
  __shared__ float beta[kMaxVerifyWidth];
  __shared__ float warp_sums[4];
  const int h = hv / (kValueHeads / kKeyHeads);
  for (int prefix = 0; prefix < prefix_count; ++prefix) {
    const int row = prefix * sequences + sequence;
    const std::size_t base = static_cast<std::size_t>(row) * kQkvWidth;
    float key_value = __bfloat162float(
        mixed_qkv[base + kKeyHeads * kHeadDim + h * kHeadDim + k_dim]);
    key_value *= rsqrtf(block_sum(key_value * key_value, warp_sums) +
                        kEpsilon);
    key[prefix][k_dim] = key_value;
    const float a = __bfloat162float(
        ba[row * (2 * kValueHeads) + kValueHeads + hv]);
    beta[prefix] = 1.0F /
        (1.0F + __expf(-__bfloat162float(ba[row * (2 * kValueHeads) + hv])));
    const float x = a + __bfloat162float(dt_bias[hv]);
    const float softplus = x <= 20.0F ? log1pf(__expf(x)) : x;
    decay[prefix] =
        __expf(-__expf(__bfloat162float(a_log[hv])) * softplus);
  }
  __syncthreads();
  float* head_state = state +
      static_cast<std::size_t>(slot) * kValueHeads * kHeadDim * kHeadDim +
      static_cast<std::size_t>(hv) * kHeadDim * kHeadDim;
  #pragma unroll
  for (int v_offset = 0; v_offset < kRowsPerBlock; v_offset += 4) {
    float current[4];
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      current[lane] = head_state[(v0 + v_offset + lane) * kHeadDim + k_dim];
    }
    for (int prefix = 0; prefix < prefix_count; ++prefix) {
      const int row = prefix * sequences + sequence;
      const std::size_t base = static_cast<std::size_t>(row) * kQkvWidth;
      float prediction[4];
      #pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        current[lane] *= decay[prefix];
        prediction[lane] =
            block_sum(current[lane] * key[prefix][k_dim], warp_sums);
      }
      #pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        const int v = v0 + v_offset + lane;
        const float v_value = __bfloat162float(
            mixed_qkv[base + 2 * kKeyHeads * kHeadDim + hv * kHeadDim + v]);
        current[lane] += beta[prefix] * (v_value - prediction[lane]) *
                         key[prefix][k_dim];
      }
    }
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      head_state[(v0 + v_offset + lane) * kHeadDim + k_dim] = current[lane];
    }
  }
}

__global__ __launch_bounds__(kThreads) void gated_rmsnorm_verify(
    const __nv_bfloat16* core, const __nv_bfloat16* qkvz,
    const __nv_bfloat16* norm_weight, __nv_bfloat16* output, int rows) {
  const int hv = blockIdx.x;
  const int row = blockIdx.y;
  const int dim = threadIdx.x;
  if (row >= rows) return;
  const std::size_t offset =
      (static_cast<std::size_t>(row) * kValueHeads + hv) * kHeadDim + dim;
  __shared__ float warp_sums[4];
  const float value = __bfloat162float(core[offset]);
  const float inv_rms = rsqrtf(block_sum(value * value, warp_sums) /
                               static_cast<float>(kHeadDim) + kEpsilon);
  const std::size_t z_offset = static_cast<std::size_t>(row) *
                                   (kQkvWidth + kGateWidth) +
                               kQkvWidth + hv * kHeadDim + dim;
  const float z = __bfloat162float(qkvz[z_offset]);
  output[offset] = __float2bfloat16(
      value * inv_rms * __bfloat162float(norm_weight[dim]) *
      (z / (1.0F + __expf(-z))));
}

__global__ __launch_bounds__(kThreads) void gated_rmsnorm(
    const __nv_bfloat16* core, const __nv_bfloat16* qkvz,
    const __nv_bfloat16* norm_weight, const std::int32_t* state_indices,
    __nv_bfloat16* output, int m) {
  const int hv = blockIdx.x;
  const int row = blockIdx.y;
  const int dim = threadIdx.x;
  if (row >= m) return;
  const std::size_t offset =
      (static_cast<std::size_t>(row) * kValueHeads + hv) * kHeadDim + dim;
  if (state_indices[row] <= 0) {
    output[offset] = __float2bfloat16(0.0F);
    return;
  }
  __shared__ float warp_sums[4];
  const float value = __bfloat162float(core[offset]);
  const float inv_rms = rsqrtf(block_sum(value * value, warp_sums) /
                               static_cast<float>(kHeadDim) + kEpsilon);
  const std::size_t z_offset = static_cast<std::size_t>(row) *
                                   (kQkvWidth + kGateWidth) +
                               kQkvWidth + hv * kHeadDim + dim;
  const float z = __bfloat162float(qkvz[z_offset]);
  const float gate = z / (1.0F + __expf(-z));
  output[offset] = __float2bfloat16(
      value * inv_rms * __bfloat162float(norm_weight[dim]) * gate);
}

}  // namespace

struct CorePlan::Impl {
  int device;
  const __nv_bfloat16* conv_weight;
  const __nv_bfloat16* a_log;
  const __nv_bfloat16* dt_bias;
  const __nv_bfloat16* norm_weight;
  __nv_bfloat16* mixed_qkv = nullptr;
  __nv_bfloat16* core = nullptr;
  __nv_bfloat16* output = nullptr;
};

CorePlan::CorePlan(int device, const __nv_bfloat16* conv_weight,
                   const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
                   const __nv_bfloat16* norm_weight)
    : impl_(new Impl{device, conv_weight, a_log, dt_bias, norm_weight}) {
  if (device < 0 || !conv_weight || !a_log || !dt_bias || !norm_weight) {
    delete impl_;
    impl_ = nullptr;
    throw std::invalid_argument("exact Qwen GDN weights and device are required");
  }
  try {
    cuda_check(cudaSetDevice(device), "cudaSetDevice");
    cuda_check(cudaMalloc(&impl_->mixed_qkv,
                          kMaxVerifierRows * kQkvWidth * sizeof(__nv_bfloat16)),
               "cudaMalloc mixed_qkv");
    cuda_check(cudaMalloc(&impl_->core,
                          kMaxVerifierRows * kGateWidth * sizeof(__nv_bfloat16)),
               "cudaMalloc core");
    cuda_check(cudaMalloc(&impl_->output,
                          kMaxVerifierRows * kGateWidth * sizeof(__nv_bfloat16)),
               "cudaMalloc output");
  } catch (...) {
    if (impl_->output) cudaFree(impl_->output);
    if (impl_->core) cudaFree(impl_->core);
    if (impl_->mixed_qkv) cudaFree(impl_->mixed_qkv);
    delete impl_;
    impl_ = nullptr;
    throw;
  }
}

void CorePlan::launch_verifier(
    const __nv_bfloat16* qkvz, const __nv_bfloat16* ba,
    const __nv_bfloat16* accepted_conv_state,
    const float* accepted_recurrent_state,
    const std::int32_t* accepted_state_indices, int sequences,
    int verify_width, cudaStream_t stream) {
  if (!qkvz || !ba || !accepted_conv_state || !accepted_recurrent_state ||
      !accepted_state_indices || !allowed_m(sequences) ||
      verify_width < 1 || verify_width > 8 ||
      sequences * verify_width > kMaxVerifierRows || !stream) {
    throw std::invalid_argument("exact Qwen GDN verifier core contract changed");
  }
  const int rows = sequences * verify_width;
  if (rows < kMaxVerifierRows) {
    cuda_check(cudaMemsetAsync(
                   impl_->output + static_cast<std::size_t>(rows) * kGateWidth,
                   0,
                   static_cast<std::size_t>(kMaxVerifierRows - rows) *
                       kGateWidth * sizeof(__nv_bfloat16),
                   stream),
               "clear inactive GDN verifier rows");
  }
  causal_conv_verify<<<dim3((kQkvWidth + 255) / 256, sequences), 256, 0,
                       stream>>>(qkvz, impl_->conv_weight,
                                 accepted_conv_state, accepted_state_indices,
                                 impl_->mixed_qkv, sequences, verify_width);
  recurrent_gdn_verify<<<dim3(kValueTiles, kValueHeads, sequences), kThreads,
                         0, stream>>>(
      impl_->mixed_qkv, ba, impl_->a_log, impl_->dt_bias,
      accepted_recurrent_state, accepted_state_indices, impl_->core,
      sequences, verify_width);
  gated_rmsnorm_verify<<<dim3(kValueHeads, rows), kThreads, 0, stream>>>(
      impl_->core, qkvz, impl_->norm_weight, impl_->output, rows);
  cuda_check(cudaPeekAtLastError(), "Qwen GDN verifier core launch");
}

void CorePlan::accept_verifier(
    const __nv_bfloat16* qkvz, const __nv_bfloat16* ba,
    __nv_bfloat16* accepted_conv_state, float* accepted_recurrent_state,
    const std::int32_t* accepted_state_indices,
    const std::int32_t* accepted_prefixes, int sequences, int verify_width,
    cudaStream_t stream) {
  if (!qkvz || !ba || !accepted_conv_state || !accepted_recurrent_state ||
      !accepted_state_indices || !accepted_prefixes || !allowed_m(sequences) ||
      verify_width < 1 || verify_width > kMaxVerifyWidth || !stream) {
    throw std::invalid_argument(
        "exact Qwen GDN verifier acceptance contract changed");
  }
  causal_conv_accept<<<dim3((kQkvWidth + 255) / 256, sequences), 256, 0,
                       stream>>>(qkvz, accepted_conv_state,
                                 accepted_state_indices, accepted_prefixes,
                                 sequences);
  recurrent_gdn_accept<<<dim3(kValueTiles, kValueHeads, sequences), kThreads,
                         0, stream>>>(
      impl_->mixed_qkv, ba, impl_->a_log, impl_->dt_bias,
      accepted_recurrent_state, accepted_state_indices, accepted_prefixes,
      sequences);
  cuda_check(cudaPeekAtLastError(), "Qwen GDN verifier accepted replay");
}

CorePlan::~CorePlan() {
  if (!impl_) return;
  cudaSetDevice(impl_->device);
  cudaFree(impl_->output);
  cudaFree(impl_->core);
  cudaFree(impl_->mixed_qkv);
  delete impl_;
}

void CorePlan::launch(
    const __nv_bfloat16* qkvz, const __nv_bfloat16* ba,
    __nv_bfloat16* conv_state, std::size_t conv_slot_stride,
    float* recurrent_state, std::size_t recurrent_slot_stride,
    const std::int32_t* state_indices, int m, cudaStream_t stream) {
  if (!qkvz || !ba || !conv_state || !recurrent_state || !state_indices ||
      !allowed_m(m) || !stream ||
      conv_slot_stride < kConvStateRows * kQkvWidth ||
      recurrent_slot_stride <
          static_cast<std::size_t>(kValueHeads) * kHeadDim * kHeadDim) {
    throw std::invalid_argument("exact Qwen GDN launch contract changed");
  }
  cuda_check(cudaSetDevice(impl_->device), "cudaSetDevice");
  // The fixed output GEMM consumes 16 rows. Clear the inactive tail on the
  // launch stream so a smaller bucket cannot observe a prior larger request.
  if (m < kMaxRows) {
    cuda_check(cudaMemsetAsync(
                   impl_->output + static_cast<std::size_t>(m) * kGateWidth, 0,
                   static_cast<std::size_t>(kMaxRows - m) * kGateWidth *
                       sizeof(__nv_bfloat16),
                   stream),
               "clear inactive GDN rows");
  }
  causal_conv_update<<<dim3((kQkvWidth + 255) / 256, m), 256, 0, stream>>>(
      qkvz, impl_->conv_weight, conv_state, conv_slot_stride, state_indices,
      impl_->mixed_qkv, m);
  recurrent_gdn<<<dim3(kValueTiles, kValueHeads, m), kThreads, 0, stream>>>(
      impl_->mixed_qkv, ba, impl_->a_log, impl_->dt_bias, recurrent_state,
      recurrent_slot_stride, state_indices, impl_->core, m);
  gated_rmsnorm<<<dim3(kValueHeads, m), kThreads, 0, stream>>>(
      impl_->core, qkvz, impl_->norm_weight, state_indices, impl_->output, m);
  cuda_check(cudaPeekAtLastError(), "Qwen GDN core launch");
}

const __nv_bfloat16* CorePlan::output() const noexcept {
  return impl_ ? impl_->output : nullptr;
}

}  // namespace rocket::qwen38::linear_attention

namespace {
template <typename F>
int wrap(F&& fn) noexcept {
  try {
    fn();
    return 0;
  } catch (const std::exception& error) {
    rocket::qwen38::linear_attention::last_error = error.what();
    return 1;
  } catch (...) {
    rocket::qwen38::linear_attention::last_error = "unknown failure";
    return 1;
  }
}
}  // namespace

extern "C" int qwen38_gdn_core_create(
    int device, const __nv_bfloat16* conv_weight, const __nv_bfloat16* a_log,
    const __nv_bfloat16* dt_bias, const __nv_bfloat16* norm_weight,
    void** plan) {
  return wrap([&] {
    if (!plan) throw std::invalid_argument("plan output is required");
    *plan = nullptr;
    *plan = new rocket::qwen38::linear_attention::CorePlan(
        device, conv_weight, a_log, dt_bias, norm_weight);
  });
}

extern "C" int qwen38_gdn_core_launch(
    void* plan, const __nv_bfloat16* qkvz, const __nv_bfloat16* ba,
    __nv_bfloat16* conv_state, std::size_t conv_slot_stride,
    float* recurrent_state, std::size_t recurrent_slot_stride,
    const std::int32_t* state_indices, int m, cudaStream_t stream) {
  return wrap([&] {
    if (!plan) throw std::invalid_argument("GDN plan is required");
    static_cast<rocket::qwen38::linear_attention::CorePlan*>(plan)->launch(
        qkvz, ba, conv_state, conv_slot_stride, recurrent_state,
        recurrent_slot_stride, state_indices, m, stream);
  });
}

extern "C" int qwen38_gdn_core_output(void* plan, void** output_bf16,
                                        std::size_t* elements) {
  return wrap([&] {
    if (!plan || !output_bf16 || !elements) {
      throw std::invalid_argument("GDN output arguments are required");
    }
    *output_bf16 = const_cast<__nv_bfloat16*>(
        static_cast<rocket::qwen38::linear_attention::CorePlan*>(plan)->output());
    *elements = rocket::qwen38::linear_attention::kMaxRows *
                rocket::qwen38::linear_attention::kGateWidth;
  });
}

extern "C" int qwen38_gdn_core_destroy(void* plan) {
  return wrap([&] {
    delete static_cast<rocket::qwen38::linear_attention::CorePlan*>(plan);
  });
}

extern "C" const char* qwen38_gdn_core_last_error() {
  return rocket::qwen38::linear_attention::last_error.c_str();
}

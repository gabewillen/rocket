// SPDX-License-Identifier: Apache-2.0
// Shape-specialized fusion of the two pinned vLLM prefill stages:
// causal_conv1d_fn and fused_gdn_prefill_post_conv.fused_post_conv_prep.
#include "linear_attention/gdn_prefill_fused.h"
#include "linear_attention/gdn_prefill_width4.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace rocket::qwen38::linear_attention::prefill {
namespace {
constexpr int kThreads = 128;
constexpr int kTokenTile = 16;

__device__ float warp_sum(float value) {
  for (int delta = 16; delta != 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  return __shfl_sync(0xffffffffU, value, 0);
}
__device__ float raw(const __nv_bfloat16* input,
                     const __nv_bfloat16* initial_state, int token,
                     int channel, int lag) {
  const int source = token - lag;
  if (source >= 0) {
    return __bfloat162float(
        input[static_cast<std::size_t>(source) * kQkvWidth + channel]);
  }
  return __bfloat162float(
      initial_state[channel * kConvStateWidth + initial_state_slot(source)]);
}

__device__ float conv(const __nv_bfloat16* input,
                      const __nv_bfloat16* initial_state,
                      const __nv_bfloat16* weight, int token, int channel) {
  float output = 0.0F;
#pragma unroll
  for (int tap = 0; tap < kConvWidth; ++tap) {
    const int lag = kConvWidth - 1 - tap;
    output = fmaf(raw(input, initial_state, token, channel, lag),
                  __bfloat162float(weight[channel * kConvWidth + tap]), output);
  }
  return output / (1.0F + expf(-output));
}

__global__ void fused_qk(const __nv_bfloat16* input,
                         const __nv_bfloat16* initial_state,
                         const __nv_bfloat16* weight, __nv_bfloat16* q,
                         __nv_bfloat16* k, int tokens) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int head = blockIdx.y;
  for (int local_token = warp; local_token < kTokenTile; local_token += 4) {
    const int token = blockIdx.x * kTokenTile + local_token;
    if (token >= tokens) {
      continue;
    }
    float q_values[4];
    float k_values[4];
    float q_square_sum = 0.0F;
    float k_square_sum = 0.0F;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int dimension = lane + part * 32;
      q_values[part] = __bfloat162float(__float2bfloat16(
          conv(input, initial_state, weight, token,
               head * kHeadDim + dimension)));
      k_values[part] = __bfloat162float(__float2bfloat16(conv(
          input, initial_state, weight, token,
          kKeyHeads * kHeadDim + head * kHeadDim + dimension)));
      q_square_sum = fmaf(q_values[part], q_values[part], q_square_sum);
      k_square_sum = fmaf(k_values[part], k_values[part], k_square_sum);
    }
    const float q_inverse = rsqrtf(warp_sum(q_square_sum) + 1.0e-6F);
    const float k_inverse = rsqrtf(warp_sum(k_square_sum) + 1.0e-6F);
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int dimension = lane + part * 32;
      const auto index =
          (static_cast<std::size_t>(token) * kKeyHeads + head) * kHeadDim +
          dimension;
      q[index] = __float2bfloat16(q_values[part] * q_inverse);
      k[index] = __float2bfloat16(k_values[part] * k_inverse);
    }
  }
}
__global__ void fused_v_gate(
    const __nv_bfloat16* input, const __nv_bfloat16* ba,
    const __nv_bfloat16* initial_state, const __nv_bfloat16* weight,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    __nv_bfloat16* v, float* g, float* beta, int tokens) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int head = blockIdx.y;
  for (int local_token = warp; local_token < kTokenTile; local_token += 4) {
    const int token = blockIdx.x * kTokenTile + local_token;
    if (token >= tokens) {
      continue;
    }
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int dimension = lane + part * 32;
      const int channel =
          2 * kKeyHeads * kHeadDim + head * kHeadDim + dimension;
      const auto index =
          (static_cast<std::size_t>(token) * kValueHeads + head) * kHeadDim +
          dimension;
      v[index] = __float2bfloat16(
          conv(input, initial_state, weight, token, channel));
    }
    if (lane == 0) {
      const float a =
          __bfloat162float(ba[token * kGateWidth + kValueHeads + head]) +
          __bfloat162float(dt_bias[head]);
      const float softplus = a > 20.0F
                                 ? a
                                 : (a > 0.0F ? a + log1pf(expf(-a))
                                             : log1pf(expf(a)));
      g[token * kValueHeads + head] =
          -expf(__bfloat162float(a_log[head])) * softplus;
      const float b = __bfloat162float(ba[token * kGateWidth + head]);
      beta[token * kValueHeads + head] = 1.0F / (1.0F + expf(-b));
    }
  }
}
__global__ void materialize(const __nv_bfloat16* input,
                            const __nv_bfloat16* initial_state,
                            const __nv_bfloat16* weight,
                            __nv_bfloat16* output,
                            int tokens) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < static_cast<std::size_t>(tokens) * kQkvWidth) {
    const int token = index / kQkvWidth;
    const int channel = index % kQkvWidth;
    output[index] = __float2bfloat16(
        conv(input, initial_state, weight, token, channel));
  }
}

__global__ void prep_qk(const __nv_bfloat16* input, __nv_bfloat16* q,
                        __nv_bfloat16* k, int tokens) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int head = blockIdx.y;
  for (int local = warp; local < kTokenTile; local += 4) {
    const int token = blockIdx.x * kTokenTile + local;
    if (token >= tokens) {
      continue;
    }
    float q_values[4];
    float k_values[4];
    float q_square_sum = 0.0F;
    float k_square_sum = 0.0F;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int dimension = lane + part * 32;
      q_values[part] = __bfloat162float(
          input[static_cast<std::size_t>(token) * kQkvWidth +
                head * kHeadDim + dimension]);
      k_values[part] = __bfloat162float(
          input[static_cast<std::size_t>(token) * kQkvWidth +
                kKeyHeads * kHeadDim + head * kHeadDim + dimension]);
      q_square_sum =
          fmaf(q_values[part], q_values[part], q_square_sum);
      k_square_sum =
          fmaf(k_values[part], k_values[part], k_square_sum);
    }
    const float q_inverse = rsqrtf(warp_sum(q_square_sum) + 1.0e-6F);
    const float k_inverse = rsqrtf(warp_sum(k_square_sum) + 1.0e-6F);
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int dimension = lane + part * 32;
      const auto index =
          (static_cast<std::size_t>(token) * kKeyHeads + head) * kHeadDim +
          dimension;
      q[index] = __float2bfloat16(q_values[part] * q_inverse);
      k[index] = __float2bfloat16(k_values[part] * k_inverse);
    }
  }
}

__global__ void prep_v_gate(const __nv_bfloat16* input,
                            const __nv_bfloat16* ba,
                            const __nv_bfloat16* a_log,
                            const __nv_bfloat16* dt_bias, __nv_bfloat16* v,
                            float* g, float* beta, int tokens) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int head = blockIdx.y;
  for (int local = warp; local < kTokenTile; local += 4) {
    const int token = blockIdx.x * kTokenTile + local;
    if (token >= tokens) {
      continue;
    }
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int dimension = lane + part * 32;
      const int channel =
          2 * kKeyHeads * kHeadDim + head * kHeadDim + dimension;
      v[(static_cast<std::size_t>(token) * kValueHeads + head) * kHeadDim +
        dimension] = input[static_cast<std::size_t>(token) * kQkvWidth +
                           channel];
    }
    if (lane == 0) {
      const float a =
          __bfloat162float(ba[token * kGateWidth + kValueHeads + head]) +
          __bfloat162float(dt_bias[head]);
      const float softplus = a > 20.0F
                                 ? a
                                 : (a > 0.0F ? a + log1pf(expf(-a))
                                             : log1pf(expf(a)));
      g[token * kValueHeads + head] =
          -expf(__bfloat162float(a_log[head])) * softplus;
      const float b = __bfloat162float(ba[token * kGateWidth + head]);
      beta[token * kValueHeads + head] = 1.0F / (1.0F + expf(-b));
    }
  }
}
__global__ void publish_conv(const __nv_bfloat16* input,
                             const __nv_bfloat16* initial_state,
                             __nv_bfloat16* state, int tokens) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= kQkvWidth) {
    return;
  }
  for (int slot = 0; slot < kConvStateWidth; ++slot) {
    const int token = published_source_token(tokens, slot);
    state[channel * kConvStateWidth + slot] =
        token >= 0
            ? input[static_cast<std::size_t>(token) * kQkvWidth + channel]
            : initial_state[channel * kConvStateWidth +
                            initial_state_slot(token)];
  }
}

bool valid(const __nv_bfloat16* input, const __nv_bfloat16* ba,
           const __nv_bfloat16* weight,
           const __nv_bfloat16* initial_state, const __nv_bfloat16* a_log,
           const __nv_bfloat16* dt_bias, Outputs outputs, int tokens) {
  return input != nullptr && ba != nullptr && weight != nullptr &&
         initial_state != nullptr && a_log != nullptr && dt_bias != nullptr &&
         outputs.q != nullptr && outputs.k != nullptr && outputs.v != nullptr &&
         outputs.g != nullptr && outputs.beta != nullptr &&
         outputs.final_conv_state != nullptr && tokens > 0 &&
         tokens <= kMaxTokens;
}

void prep(const __nv_bfloat16* input, const __nv_bfloat16* ba,
          const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
          Outputs outputs, int tokens, cudaStream_t stream) {
  const int token_blocks = (tokens + kTokenTile - 1) / kTokenTile;
  prep_qk<<<dim3(token_blocks, kKeyHeads), kThreads, 0, stream>>>(
      input, outputs.q, outputs.k, tokens);
  prep_v_gate<<<dim3(token_blocks, kValueHeads), kThreads, 0, stream>>>(
      input, ba, a_log, dt_bias, outputs.v, outputs.g, outputs.beta, tokens);
}
}  // namespace

cudaError_t launch_fused(
    const __nv_bfloat16* input, const __nv_bfloat16* ba,
    const __nv_bfloat16* weight, const __nv_bfloat16* initial_state,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    Outputs outputs, int tokens, cudaStream_t stream) noexcept {
  if (!valid(input, ba, weight, initial_state, a_log, dt_bias, outputs,
             tokens)) {
    return cudaErrorInvalidValue;
  }
  const int token_blocks = (tokens + kTokenTile - 1) / kTokenTile;
  fused_qk<<<dim3(token_blocks, kKeyHeads), kThreads, 0, stream>>>(
      input, initial_state, weight, outputs.q, outputs.k, tokens);
  fused_v_gate<<<dim3(token_blocks, kValueHeads), kThreads, 0, stream>>>(
      input, ba, initial_state, weight, a_log, dt_bias, outputs.v, outputs.g,
      outputs.beta, tokens);
  publish_conv<<<(kQkvWidth + 255) / 256, 256, 0, stream>>>(
      input, initial_state, outputs.final_conv_state, tokens);
  return cudaPeekAtLastError();
}

cudaError_t launch_materialized_reference(
    const __nv_bfloat16* input, const __nv_bfloat16* ba,
    const __nv_bfloat16* weight, const __nv_bfloat16* initial_state,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    __nv_bfloat16* conv_output, Outputs outputs, int tokens,
    cudaStream_t stream) noexcept {
  if (conv_output == nullptr ||
      !valid(input, ba, weight, initial_state, a_log, dt_bias, outputs,
             tokens)) {
    return cudaErrorInvalidValue;
  }
  const auto elements = static_cast<std::size_t>(tokens) * kQkvWidth;
  materialize<<<(elements + 255) / 256, 256, 0, stream>>>(
      input, initial_state, weight, conv_output, tokens);
  prep(conv_output, ba, a_log, dt_bias, outputs, tokens, stream);
  publish_conv<<<(kQkvWidth + 255) / 256, 256, 0, stream>>>(
      input, initial_state, outputs.final_conv_state, tokens);
  return cudaPeekAtLastError();
}
}  // namespace rocket::qwen38::linear_attention::prefill

// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace rocket::qwen38::linear_attention::prefill {

inline constexpr int kConvWidth = 4;
inline constexpr int kConvStateWidth = kConvWidth - 1;

#if defined(__CUDACC__)
#define ROCKET_QWEN38_HOST_DEVICE __host__ __device__
#else
#define ROCKET_QWEN38_HOST_DEVICE
#endif

// vLLM causal_conv1d stores weights oldest to current. Negative token sources
// address the three-token state, also oldest to newest.
ROCKET_QWEN38_HOST_DEVICE constexpr int causal_source_token(
    int token, int weight_tap) noexcept {
  return token - (kConvWidth - 1 - weight_tap);
}

ROCKET_QWEN38_HOST_DEVICE constexpr int initial_state_slot(
    int source_token) noexcept {
  return kConvStateWidth + source_token;
}

ROCKET_QWEN38_HOST_DEVICE constexpr int published_source_token(
    int tokens, int state_slot) noexcept {
  return tokens - kConvStateWidth + state_slot;
}

#undef ROCKET_QWEN38_HOST_DEVICE

}  // namespace rocket::qwen38::linear_attention::prefill

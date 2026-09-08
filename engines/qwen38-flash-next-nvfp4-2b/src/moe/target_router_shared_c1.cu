// SPDX-License-Identifier: Apache-2.0
#include "moe/target_router_shared_c1.h"

#include "moe/target_moe_b12x_aot.h"
#include "target_moe_artifact_key.h"

#include <math_constants.h>

#include <algorithm>
#include <cmath>

namespace rocket::qwen38::moe {
namespace {

TargetDenseFailure diagnose_identity(const TargetDenseIdentity& identity) noexcept {
  std::array<std::uint8_t, 32> artifact{};
  if (!parse_target_moe_artifact_key(kRocketQwen38TargetMoeArtifactKey,
                                     &artifact) ||
      identity.artifact_sha256 != artifact)
    return TargetDenseFailure::kArtifact;
  if (!std::any_of(identity.layout_sha256.begin(), identity.layout_sha256.end(),
                   [](std::uint8_t byte) { return byte != 0; }))
    return TargetDenseFailure::kLayout;
  if (identity.rank != 0 && identity.rank != 1)
    return TargetDenseFailure::kRank;
  if (identity.layer != kTargetCompositionLayer)
    return TargetDenseFailure::kLayer;
  return TargetDenseFailure::kNone;
}

__host__ __device__ __forceinline__ float e2m1(std::uint8_t code) {
  constexpr float magnitude[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                                  2.0F, 3.0F, 4.0F, 6.0F};
  const float value = magnitude[code & 7];
  return code & 8 ? -value : value;
}

__host__ __device__ __forceinline__ float packed_e2m1(std::uint8_t pair,
                                                       int column) {
  return e2m1(column & 1 ? static_cast<std::uint8_t>(pair >> 4)
                         : static_cast<std::uint8_t>(pair & 15));
}

__host__ __device__ __forceinline__ float e4m3(std::uint8_t code) {
  const int sign = code >> 7;
  const int exponent = (code >> 3) & 15;
  const int mantissa = code & 7;
  float value = 0.0F;
  if (exponent == 0) {
    value = ldexpf(static_cast<float>(mantissa), -9);
  } else {
    value = ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F,
                   exponent - 7);
  }
  return sign ? -value : value;
}

__host__ __device__ __forceinline__ int sfb_offset(int row, int group) {
  constexpr int kGroups = kTargetRouterHidden / 16;
  constexpr int kTiles = (kGroups + 3) / 4;
  const int row_tile = row / 128;
  const int row_in_tile = row % 128;
  const int group_tile = group / 4;
  const int group_in_tile = group % 4;
  return (row_tile * kTiles + group_tile) * 512 +
         (row_in_tile % 32) * 16 + (row_in_tile / 32) * 4 + group_in_tile;
}

__global__ void router_logits_kernel(const __nv_bfloat16* hidden,
                                     const std::uint8_t* packed,
                                     const std::uint8_t* scales,
                                     const float* alpha, float* logits) {
  const int expert = threadIdx.x;
  if (expert >= kTargetRouterExperts) return;
  float sum = 0.0F;
  for (int column = 0; column < kTargetRouterHidden; ++column) {
    const std::uint8_t pair =
        packed[expert * (kTargetRouterHidden / 2) + column / 2];
    const float scale = e4m3(scales[sfb_offset(expert, column / 16)]);
    sum += __bfloat162float(hidden[column]) * packed_e2m1(pair, column) * scale;
  }
  logits[expert] = sum * alpha[0];
}

__global__ void router_top10_kernel(const float* logits, std::int32_t* ids,
                                    float* weights,
                                    std::uint64_t* source_generation,
                                    const std::uint64_t* requested_generation) {
  if (threadIdx.x != 0) return;
  float selected_logits[kTargetRouterTopK];
  bool used[kTargetRouterExperts]{};
  for (int slot = 0; slot < kTargetRouterTopK; ++slot) {
    int selected = -1;
    float best = -CUDART_INF_F;
    for (int expert = 0; expert < kTargetRouterExperts; ++expert) {
      const float value = logits[expert];
      if (!used[expert] &&
          (value > best || (value == best && expert < selected))) {
        best = value;
        selected = expert;
      }
    }
    used[selected] = true;
    ids[slot] = selected;
    selected_logits[slot] = best;
  }
  float denominator = 0.0F;
  for (int slot = 0; slot < kTargetRouterTopK; ++slot) {
    weights[slot] = expf(selected_logits[slot] - selected_logits[0]);
    denominator += weights[slot];
  }
  for (int slot = 0; slot < kTargetRouterTopK; ++slot)
    weights[slot] /= denominator;
  *source_generation = *requested_generation;
}

__global__ void shared_up_gate_kernel(
    const __nv_bfloat16* hidden, const __nv_bfloat16* gate,
    const __nv_bfloat16* up, int begin, float* gate_scratch,
    float* up_scratch) {
  const int intermediate = threadIdx.x;
  if (intermediate >= kTargetSharedRankIntermediate) return;
  const int global = begin + intermediate;
  float gate_value = 0.0F;
  float up_value = 0.0F;
  for (int column = 0; column < kTargetRouterHidden; ++column) {
    const float input = __bfloat162float(hidden[column]);
    gate_value += input * __bfloat162float(
                              gate[global * kTargetRouterHidden + column]);
    up_value += input *
                __bfloat162float(up[global * kTargetRouterHidden + column]);
  }
  gate_scratch[intermediate] =
      gate_value / (1.0F + expf(-gate_value));
  up_scratch[intermediate] = up_value;
}

__global__ void shared_gate_kernel(const __nv_bfloat16* hidden,
                                   const __nv_bfloat16* shared_gate,
                                   float* output) {
  if (threadIdx.x != 0) return;
  float value = 0.0F;
  for (int column = 0; column < kTargetRouterHidden; ++column)
    value += __bfloat162float(hidden[column]) *
             __bfloat162float(shared_gate[column]);
  output[0] = 1.0F / (1.0F + expf(-value));
}

__global__ void shared_down_add_kernel(
    const float* gate_scratch, const float* up_scratch,
    const __nv_bfloat16* down, int begin, const float* shared_gate,
    __nv_bfloat16* output) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= kTargetRouterHidden) return;
  float partial = 0.0F;
  for (int intermediate = 0; intermediate < kTargetSharedRankIntermediate;
       ++intermediate) {
    partial += gate_scratch[intermediate] * up_scratch[intermediate] *
               __bfloat162float(
                   down[column * kTargetSharedIntermediate + begin + intermediate]);
  }
  output[column] = __float2bfloat16(
      __bfloat162float(output[column]) + partial * shared_gate[0]);
}

}  // namespace

TargetDenseFailure diagnose_target_router_c1(
    const TargetDenseIdentity& identity,
    const TargetRouterNvfp4Weights& weights,
    const TargetRouterC1Launch& launch) noexcept {
  const auto identity_failure = diagnose_identity(identity);
  if (identity_failure != TargetDenseFailure::kNone) return identity_failure;
  if (!weights.packed_e2m1 || !weights.cutlass_sfb_e4m3 ||
      !weights.weight_scale_2)
    return TargetDenseFailure::kWeight;
  if (!launch.hidden_bf16) return TargetDenseFailure::kActivation;
  if (!launch.logits_f32 || !launch.global_ids_i32 ||
      !launch.routing_weights_f32)
    return TargetDenseFailure::kOutput;
  if (!launch.source_generation || !launch.requested_generation)
    return TargetDenseFailure::kGeneration;
  if (!launch.stream) return TargetDenseFailure::kStream;
  return TargetDenseFailure::kNone;
}

TargetDenseFailure diagnose_target_shared_c1(
    const TargetDenseIdentity& identity,
    const TargetSharedBf16Weights& weights,
    const TargetSharedC1Launch& launch) noexcept {
  const auto identity_failure = diagnose_identity(identity);
  if (identity_failure != TargetDenseFailure::kNone) return identity_failure;
  if (!weights.gate || !weights.up || !weights.down || !weights.shared_gate)
    return TargetDenseFailure::kWeight;
  if (!launch.hidden_bf16) return TargetDenseFailure::kActivation;
  if (!launch.routed_plus_shared_bf16) return TargetDenseFailure::kOutput;
  if (!launch.gate_scratch_f32 || !launch.up_scratch_f32 ||
      !launch.shared_gate_scratch_f32)
    return TargetDenseFailure::kWorkspace;
  if (!launch.stream) return TargetDenseFailure::kStream;
  return TargetDenseFailure::kNone;
}

TargetDenseOutcome enqueue_target_router_c1(
    const TargetDenseIdentity& identity,
    const TargetRouterNvfp4Weights& weights,
    const TargetRouterC1Launch& launch) noexcept {
  if (diagnose_target_router_c1(identity, weights, launch) !=
      TargetDenseFailure::kNone)
    return TargetDenseOutcome::kContractError;
  router_logits_kernel<<<1, kTargetRouterExperts, 0, launch.stream>>>(
      launch.hidden_bf16, weights.packed_e2m1, weights.cutlass_sfb_e4m3,
      weights.weight_scale_2, launch.logits_f32);
  router_top10_kernel<<<1, 1, 0, launch.stream>>>(
      launch.logits_f32, launch.global_ids_i32, launch.routing_weights_f32,
      launch.source_generation, launch.requested_generation);
  return cudaPeekAtLastError() == cudaSuccess ? TargetDenseOutcome::kOk
                                               : TargetDenseOutcome::kCudaError;
}

TargetDenseOutcome enqueue_target_shared_c1(
    const TargetDenseIdentity& identity,
    const TargetSharedBf16Weights& weights,
    const TargetSharedC1Launch& launch) noexcept {
  if (diagnose_target_shared_c1(identity, weights, launch) !=
      TargetDenseFailure::kNone)
    return TargetDenseOutcome::kContractError;
  const int begin = identity.rank * kTargetSharedRankIntermediate;
  shared_up_gate_kernel<<<1, kTargetSharedRankIntermediate, 0, launch.stream>>>(
      launch.hidden_bf16, weights.gate, weights.up, begin,
      launch.gate_scratch_f32, launch.up_scratch_f32);
  shared_gate_kernel<<<1, 1, 0, launch.stream>>>(
      launch.hidden_bf16, weights.shared_gate, launch.shared_gate_scratch_f32);
  shared_down_add_kernel<<<(kTargetRouterHidden + 255) / 256, 256, 0,
                           launch.stream>>>(
      launch.gate_scratch_f32, launch.up_scratch_f32, weights.down, begin,
      launch.shared_gate_scratch_f32, launch.routed_plus_shared_bf16);
  return cudaPeekAtLastError() == cudaSuccess ? TargetDenseOutcome::kOk
                                               : TargetDenseOutcome::kCudaError;
}

}  // namespace rocket::qwen38::moe

extern "C" int rocket_qwen38_target_router_c1_enqueue(
    const rocket::qwen38::moe::TargetDenseIdentity* identity,
    const rocket::qwen38::moe::TargetRouterNvfp4Weights* weights,
    const rocket::qwen38::moe::TargetRouterC1Launch* launch) noexcept {
  using namespace rocket::qwen38::moe;
  if (!identity || !weights || !launch)
    return static_cast<int>(TargetDenseOutcome::kContractError);
  return static_cast<int>(enqueue_target_router_c1(*identity, *weights, *launch));
}

extern "C" int rocket_qwen38_target_shared_c1_enqueue(
    const rocket::qwen38::moe::TargetDenseIdentity* identity,
    const rocket::qwen38::moe::TargetSharedBf16Weights* weights,
    const rocket::qwen38::moe::TargetSharedC1Launch* launch) noexcept {
  using namespace rocket::qwen38::moe;
  if (!identity || !weights || !launch)
    return static_cast<int>(TargetDenseOutcome::kContractError);
  return static_cast<int>(enqueue_target_shared_c1(*identity, *weights, *launch));
}

extern "C" float rocket_qwen38_target_nvfp4_e2m1_host(
    std::uint8_t code) noexcept {
  return rocket::qwen38::moe::e2m1(code);
}

extern "C" float rocket_qwen38_target_nvfp4_packed_host(
    std::uint8_t pair, int column) noexcept {
  return rocket::qwen38::moe::packed_e2m1(pair, column);
}

extern "C" float rocket_qwen38_target_nvfp4_e4m3_host(
    std::uint8_t code) noexcept {
  return rocket::qwen38::moe::e4m3(code);
}

extern "C" int rocket_qwen38_target_router_sfb_offset_host(
    int row, int group) noexcept {
  if (row < 0 || row >= rocket::qwen38::moe::kTargetRouterExperts || group < 0 ||
      group >= rocket::qwen38::moe::kTargetRouterHidden / 16)
    return -1;
  return rocket::qwen38::moe::sfb_offset(row, group);
}

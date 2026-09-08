// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>

namespace rocket::qwen38::moe {

inline constexpr int kTargetRouterHidden = 2'560;
inline constexpr int kTargetRouterExperts = 512;
inline constexpr int kTargetRouterTopK = 10;
inline constexpr int kTargetSharedIntermediate = 320;
inline constexpr int kTargetSharedRankIntermediate = 160;
inline constexpr int kTargetCompositionLayer = 3;

enum class TargetDenseOutcome : std::uint8_t {
  kOk,
  kContractError,
  kCudaError,
};

// Fixed-cardinality first-invalid-field values for post-fence telemetry.
enum class TargetDenseFailure : std::uint8_t {
  kNone,
  kArtifact,
  kLayout,
  kRank,
  kLayer,
  kWeight,
  kActivation,
  kOutput,
  kWorkspace,
  kGeneration,
  kStream,
};

struct TargetDenseIdentity {
  std::array<std::uint8_t, 32> artifact_sha256;
  std::array<std::uint8_t, 32> layout_sha256;
  int rank;
  int layer;
};

struct TargetRouterNvfp4Weights {
  const std::uint8_t* packed_e2m1;
  const std::uint8_t* cutlass_sfb_e4m3;
  const float* weight_scale_2;
};

struct TargetRouterC1Launch {
  const __nv_bfloat16* hidden_bf16;
  float* logits_f32;
  std::int32_t* global_ids_i32;
  float* routing_weights_f32;
  std::uint64_t* source_generation;
  const std::uint64_t* requested_generation;
  cudaStream_t stream;
};

struct TargetSharedBf16Weights {
  const __nv_bfloat16* gate;
  const __nv_bfloat16* up;
  const __nv_bfloat16* down;
  const __nv_bfloat16* shared_gate;
};

struct TargetSharedC1Launch {
  const __nv_bfloat16* hidden_bf16;
  __nv_bfloat16* routed_plus_shared_bf16;
  float* gate_scratch_f32;
  float* up_scratch_f32;
  float* shared_gate_scratch_f32;
  cudaStream_t stream;
};

[[nodiscard]] TargetDenseFailure diagnose_target_router_c1(
    const TargetDenseIdentity& identity,
    const TargetRouterNvfp4Weights& weights,
    const TargetRouterC1Launch& launch) noexcept;
[[nodiscard]] TargetDenseFailure diagnose_target_shared_c1(
    const TargetDenseIdentity& identity,
    const TargetSharedBf16Weights& weights,
    const TargetSharedC1Launch& launch) noexcept;

// Fixed-c1 launches. All pointers and the stream are borrowed. Enqueue does no
// allocation, host transfer, synchronization, Torch, or Python work.
[[nodiscard]] TargetDenseOutcome enqueue_target_router_c1(
    const TargetDenseIdentity& identity,
    const TargetRouterNvfp4Weights& weights,
    const TargetRouterC1Launch& launch) noexcept;
[[nodiscard]] TargetDenseOutcome enqueue_target_shared_c1(
    const TargetDenseIdentity& identity,
    const TargetSharedBf16Weights& weights,
    const TargetSharedC1Launch& launch) noexcept;

}  // namespace rocket::qwen38::moe

extern "C" {
int rocket_qwen38_target_router_c1_enqueue(
    const rocket::qwen38::moe::TargetDenseIdentity* identity,
    const rocket::qwen38::moe::TargetRouterNvfp4Weights* weights,
    const rocket::qwen38::moe::TargetRouterC1Launch* launch) noexcept;
int rocket_qwen38_target_shared_c1_enqueue(
    const rocket::qwen38::moe::TargetDenseIdentity* identity,
    const rocket::qwen38::moe::TargetSharedBf16Weights* weights,
    const rocket::qwen38::moe::TargetSharedC1Launch* launch) noexcept;
float rocket_qwen38_target_nvfp4_e2m1_host(std::uint8_t code) noexcept;
float rocket_qwen38_target_nvfp4_packed_host(std::uint8_t pair,
                                             int column) noexcept;
float rocket_qwen38_target_nvfp4_e4m3_host(std::uint8_t code) noexcept;
int rocket_qwen38_target_router_sfb_offset_host(int row, int group) noexcept;
}

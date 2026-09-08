// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "moe/target_moe_c1.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>

namespace rocket::qwen38::moe {

inline constexpr int kTargetMoeHidden = 2'560;
inline constexpr int kTargetMoeLogicalIntermediate = 640;
inline constexpr int kTargetMoePhysicalIntermediate = 768;
inline constexpr int kTargetMoeStateExperts = 257;
inline constexpr int kTargetMoeMaxRows = 10;

struct TargetMoeB12xIdentity {
  std::array<std::uint8_t, 32> artifact_sha256;
  std::array<std::uint8_t, 32> layout_sha256;
  int rank;
  int layer;
};

struct TargetMoeB12xWeights {
  // Init-padded physical N768 ModelOpt NVFP4 tensors. Logical source N640 is
  // authenticated by TargetMoeB12xIdentity before module construction.
  const std::uint8_t* w13_packed;
  const std::uint8_t* w13_scale;
  const std::uint8_t* down_packed;
  const std::uint8_t* down_scale;
  const float* input_global_scale;
  const float* w1_alpha;
  const float* w2_alpha;
  const float* down_input_scale;
};

struct TargetMoeB12xWorkspace {
  std::uint8_t* packed_a;
  std::uint8_t* packed_a_scale;
  void* route_output_scratch_bf16;
  std::int32_t* barrier_count;
  std::int32_t* barrier_epoch;
  std::int32_t* row_counts;
  std::int32_t* active_expert_count;
  std::int32_t* weight_expert_ids;
  std::int32_t* global_to_local_expert;
  std::int32_t* virtual_route_scratch;
  std::int32_t* token_map;
  float* token_weights;
};

struct TargetMoeB12xLaunch {
  const void* hidden_bf16;
  const std::int32_t* local_expert_ids;
  const float* local_routing_weights;
  void* output_bf16;
  TargetMoeB12xWorkspace workspace;
  cudaStream_t stream;
};

// Owns the pinned fixed-c1 CuTe module only. All activations, dense routes,
// weights, output, workspace, and stream remain caller-owned. Construction is
// outside graph capture; launch performs no allocation, D2H, synchronization,
// Python, or Torch work.
class TargetMoeB12xAot final {
 public:
  TargetMoeB12xAot(int device, TargetMoeB12xIdentity identity,
                   TargetMoeB12xWeights weights);
  ~TargetMoeB12xAot();
  TargetMoeB12xAot(const TargetMoeB12xAot&) = delete;
  TargetMoeB12xAot& operator=(const TargetMoeB12xAot&) = delete;

  [[nodiscard]] TargetMoeOutcome enqueue(
      const TargetMoeB12xLaunch& launch) const noexcept;
  [[nodiscard]] const TargetMoeB12xIdentity& identity() const noexcept;

 private:
  struct Impl;
  Impl* impl_;
};

[[nodiscard]] bool target_moe_b12x_aot_compiled() noexcept;

}  // namespace rocket::qwen38::moe

extern "C" {

// Stable validation/embedding ABI. Creation and destruction are outside the
// hot path. Enqueue has the same no-allocation and borrowed-stream contract as
// TargetMoeB12xAot::enqueue.
int rocket_qwen38_target_moe_b12x_create(
    int device, const rocket::qwen38::moe::TargetMoeB12xIdentity* identity,
    const rocket::qwen38::moe::TargetMoeB12xWeights* weights,
    void** handle) noexcept;
int rocket_qwen38_target_moe_b12x_enqueue(
    void* handle,
    const rocket::qwen38::moe::TargetMoeB12xLaunch* launch) noexcept;
void rocket_qwen38_target_moe_b12x_destroy(void* handle) noexcept;

}

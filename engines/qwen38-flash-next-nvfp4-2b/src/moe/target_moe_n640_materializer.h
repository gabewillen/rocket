// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "moe/target_moe_b12x_aot.h"

namespace rocket::qwen38::moe {

// Host-side logical planes in the same expert-major order authenticated by
// OwnerLocalMoeSlab. These are source N640 bytes, never serving N768 pointers.
struct TargetMoeLogicalN640 {
  std::span<const std::uint8_t> w13_packed;
  std::span<const std::uint8_t> w13_scale;
  std::span<const std::uint8_t> down_packed;
  std::span<const std::uint8_t> down_scale;
  std::span<const float> input_global_scale;
  std::span<const float> w1_alpha;
  std::span<const float> w2_alpha;
  std::span<const float> down_input_scale;
};

struct TargetMoeTransformedIdentity {
  std::array<std::uint8_t, 32> source_layout_sha256{};
  std::array<std::uint8_t, 32> source_planes_sha256{};
  std::array<std::uint8_t, 32> physical_planes_sha256{};
  int logical_intermediate = kTargetMoeLogicalIntermediate;
  int physical_intermediate = kTargetMoePhysicalIntermediate;
  int experts = 256;
  int hidden = kTargetMoeHidden;
  int rank = -1;
  int layer = 3;
};

TargetMoeTransformedIdentity target_moe_layer3_transformed_identity(int rank);

struct TargetMoePhysicalN768Host {
  std::vector<std::uint8_t> w13_packed;
  std::vector<std::uint8_t> w13_scale;
  std::vector<std::uint8_t> down_packed;
  std::vector<std::uint8_t> down_scale;
  std::vector<float> input_global_scale;
  std::vector<float> folded_w1_alpha;
  std::vector<float> w2_alpha;
  std::vector<float> down_input_scale;
};

// Exact host transform derived under Apache-2.0 from FlashInfer commit
// 91bda04c66f7cb851e1ab3b78b9fecea644b9844, files
// flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py and
// flashinfer/cute_dsl/utils.py. `_pad_intermediate_to_tile` pads gate/up
// halves independently,
// down columns are padded, and E4M3 block scales are remapped through the
// same logical scale coordinates. Zero is the neutral packed FP4/scale code.
TargetMoePhysicalN768Host materialize_target_moe_n640_host(
    const TargetMoeLogicalN640& source);
std::array<std::uint8_t, 32> target_moe_n640_sha256(
    const TargetMoeLogicalN640& source);
std::array<std::uint8_t, 32> target_moe_n768_sha256(
    const TargetMoePhysicalN768Host& physical);
std::array<std::array<std::uint8_t, 32>, 8> target_moe_n768_plane_sha256(
    const TargetMoePhysicalN768Host& physical);

// One-time device owner. Construction transforms and copies on its private
// stream, records an event, then publishes pointers only after the event has
// completed. No member method allocates or copies during captured enqueue.
class TargetMoeN768DeviceOwner final {
 public:
  TargetMoeN768DeviceOwner(int device, const TargetMoeLogicalN640& source,
                           TargetMoeTransformedIdentity identity);
  ~TargetMoeN768DeviceOwner();
  TargetMoeN768DeviceOwner(const TargetMoeN768DeviceOwner&) = delete;
  TargetMoeN768DeviceOwner& operator=(const TargetMoeN768DeviceOwner&) = delete;

  const TargetMoeB12xWeights& weights() const noexcept { return weights_; }
  const TargetMoeTransformedIdentity& identity() const noexcept {
    return identity_;
  }

 private:
  int device_ = -1;
  std::vector<void*> allocations_;
  TargetMoeB12xWeights weights_{};
  TargetMoeTransformedIdentity identity_{};
};

}  // namespace rocket::qwen38::moe

extern "C" {
struct RocketQwen38TargetMoeLogicalN640 {
  const std::uint8_t* w13_packed;
  const std::uint8_t* w13_scale;
  const std::uint8_t* down_packed;
  const std::uint8_t* down_scale;
  const float* input_global_scale;
  const float* w1_alpha;
  const float* w2_alpha;
  const float* down_input_scale;
};

int rocket_qwen38_target_moe_n640_hash(
    const RocketQwen38TargetMoeLogicalN640* source,
    std::uint8_t source_sha256[32],
    std::uint8_t physical_sha256[32]) noexcept;
int rocket_qwen38_target_moe_n640_plane_hashes(
    const RocketQwen38TargetMoeLogicalN640* source,
    std::uint8_t physical_plane_sha256[8][32]) noexcept;
const char* rocket_qwen38_target_moe_n640_last_error() noexcept;
}

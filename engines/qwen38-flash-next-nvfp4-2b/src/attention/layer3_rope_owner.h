// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace rocket::qwen38::attention {

inline constexpr int kLayer3RopeLayer = 3;
inline constexpr int kLayer3RopeRows = 35;
inline constexpr int kLayer3RopePairs = 32;
inline constexpr int kLayer3RopeColumns = 64;
inline constexpr float kLayer3RopeTheta = 10'000'000.0f;
inline constexpr std::string_view kLayer3RopeRocketBaseRevision =
    "58f4d1ed59074a181c9db6e4c5a950db14e7ad0b";
inline constexpr std::string_view kLayer3RopeCheckpoint =
    "nvidia/Qwen3.8-Flash-Next-NVFP4";
inline constexpr std::string_view kLayer3RopeCheckpointRevision =
    "fc694b54fb0174e0913e6adf86691ef85a4ead47";
inline constexpr std::string_view kLayer3RopeConfigSha256 =
    "deef67a61f3311faf051b23dc4192f442c7fee4f9cd2f38cbcbe4da55c763a80";
inline constexpr std::string_view kLayer3RopeVllmRevision =
    "8e685d198";
inline constexpr std::string_view kLayer3RopeVllmImageDigest =
    "sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8";
inline constexpr std::string_view kLayer3RopePayloadSha256 =
    "f22ad8a42a36ec8078d7f43a89bdb98cbb05ef7139c34b09a7cd5af62b98a516";

struct Layer3RopeIdentity {
  std::string_view checkpoint_revision;
  std::string_view config_sha256;
  std::string_view vllm_revision;
  int rank;
  int layer;
  int first_position;
  int rows;
  int rotary_dim;
  float rope_theta;
  bool uses_mrope;
};

struct Layer3RopeView {
  const __nv_bfloat16* cos_sin = nullptr;
  cudaEvent_t ready = nullptr;
  std::string_view payload_sha256;
  int rows = 0;
  int columns = 0;
  int row_stride = 0;
};

Layer3RopeIdentity layer3_rope_identity(int rank);
// The same position-only RoPE table is shared by all 12 target QSA layers.
// The returned identity remains layer-bound so a layer owner cannot accept a
// descriptor for another attention position in the schedule.
Layer3RopeIdentity target_qsa_rope_identity(int rank, int layer);

// Exact pinned-vLLM BF16 payload consumed by NativeQsaFullAttentionGraph,
// row-major [35,64] with stride [64,1]. Columns [0,32) are cosine and
// [32,64) are sine. This host view exists for independent CPU contract tests;
// the operational path publishes only device storage.
std::span<const std::uint16_t> layer3_rope_host_bits() noexcept;

class Layer3RopeDeviceOwner final {
 public:
  Layer3RopeDeviceOwner(int device, Layer3RopeIdentity identity);
  ~Layer3RopeDeviceOwner();
  Layer3RopeDeviceOwner(const Layer3RopeDeviceOwner&) = delete;
  Layer3RopeDeviceOwner& operator=(const Layer3RopeDeviceOwner&) = delete;

  const Layer3RopeIdentity& identity() const noexcept { return identity_; }
  // Consumers must enqueue wait() before reading view().cos_sin. The event is
  // the publication boundary; construction does not synchronize the device.
  Layer3RopeView view() const noexcept;
  void wait(cudaStream_t consumer_stream) const;

 private:
  int device_ = -1;
  Layer3RopeIdentity identity_{};
  __nv_bfloat16* cos_sin_ = nullptr;
  cudaStream_t initialization_stream_ = nullptr;
  cudaEvent_t ready_ = nullptr;
};

}  // namespace rocket::qwen38::attention

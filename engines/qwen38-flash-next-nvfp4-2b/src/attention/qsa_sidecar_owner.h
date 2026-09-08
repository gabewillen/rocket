// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace rocket::qwen38::attention {

inline constexpr std::string_view kQsaSidecarArtifactKey =
    "bdbebd4f45c398f090a41ab98cd3881b969d958d8ae0bc42f3411844d3262edd";
inline constexpr std::size_t kQsaSidecarBytes = 39'321'600;
inline constexpr std::size_t kQsaIndexerLayer3Offset = 0;
inline constexpr std::size_t kQsaIndexerLayer3Bytes = 3'276'800;

struct QsaSidecarIdentity {
  std::string_view artifact_key;
  std::array<std::uint8_t, 32> payload_sha256;
  std::array<std::uint8_t, 32> layer3_sha256;
  int rank;
  int layer;
};

QsaSidecarIdentity layer3_qsa_sidecar_identity(int rank);
std::vector<std::uint8_t> authenticate_qsa_sidecar_host(
    const std::filesystem::path& payload, const QsaSidecarIdentity& identity);

// Target-only sidecar owner. It authenticates the complete content-addressed
// payload and the bound layer-3 component before allocation. A private H2D
// stream/event fence completes before either device pointer is published.
class QsaSidecarDeviceOwner final {
 public:
  QsaSidecarDeviceOwner(int device, const std::filesystem::path& payload,
                        QsaSidecarIdentity identity);
  ~QsaSidecarDeviceOwner();
  QsaSidecarDeviceOwner(const QsaSidecarDeviceOwner&) = delete;
  QsaSidecarDeviceOwner& operator=(const QsaSidecarDeviceOwner&) = delete;

  const std::uint8_t* payload() const noexcept { return payload_; }
  const __nv_bfloat16* index_qk_proj() const noexcept {
    return reinterpret_cast<const __nv_bfloat16*>(
        payload_ + kQsaIndexerLayer3Offset);
  }
  const QsaSidecarIdentity& identity() const noexcept { return identity_; }

 private:
  int device_ = -1;
  std::uint8_t* payload_ = nullptr;
  QsaSidecarIdentity identity_{};
};

}  // namespace rocket::qwen38::attention

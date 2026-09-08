// SPDX-License-Identifier: Apache-2.0
#include "attention/layer3_rope_owner.h"
#include "attention/qsa_sidecar_owner.h"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace attention = rocket::qwen38::attention;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
  std::array<std::array<std::uint8_t, 32>, attention::kQsaSidecarLayers>
      component_digests{};
  for (int index = 0; index < attention::kQsaSidecarLayers; ++index) {
    const int layer = 4 * index + 3;
    const auto rank0 = attention::target_qsa_sidecar_identity(0, layer);
    const auto rank1 = attention::target_qsa_sidecar_identity(1, layer);
    const auto rope = attention::target_qsa_rope_identity(0, layer);
    require(rank0.rank == 0 && rank1.rank == 1 && rank0.layer == layer &&
                rank1.layer == layer && rank0.artifact_key == rank1.artifact_key &&
                rank0.payload_sha256 == rank1.payload_sha256 &&
                rank0.layer3_sha256 == rank1.layer3_sha256,
            "QSA shared-asset rank binding changed");
    require(attention::target_qsa_sidecar_offset(layer) ==
                static_cast<std::size_t>(index) *
                    attention::kQsaIndexerLayer3Bytes &&
                rope.layer == layer,
            "QSA shared-asset schedule changed");
    component_digests[static_cast<std::size_t>(index)] = rank0.layer3_sha256;
  }
  for (std::size_t left = 0; left < component_digests.size(); ++left)
    for (std::size_t right = left + 1; right < component_digests.size(); ++right)
      require(component_digests[left] != component_digests[right],
              "QSA sidecar component identity is not layer-specific");
  for (const int layer : {-1, 0, 4, 48}) {
    try {
      (void)attention::target_qsa_sidecar_identity(0, layer);
      require(false, "non-QSA sidecar layer accepted");
    } catch (const std::invalid_argument&) {
    }
  }
}

// SPDX-License-Identifier: Apache-2.0
#include "attention/layer3_rope_owner.h"

#include <array>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace attention = rocket::qwen38::attention;

namespace {
void require(bool condition) {
  if (!condition) throw std::runtime_error("layer3 RoPE CPU golden mismatch");
}
}  // namespace

int main() {
  const auto rank0 = attention::layer3_rope_identity(0);
  const auto rank1 = attention::layer3_rope_identity(1);
  for (int layer = 3; layer < 48; layer += 4) {
    const auto identity = attention::target_qsa_rope_identity(0, layer);
    require(identity.layer == layer && identity.rank == 0);
  }
  try {
    (void)attention::target_qsa_rope_identity(0, 4);
    require(false);
  } catch (const std::invalid_argument&) {
  }
  require(rank0.rank == 0 && rank1.rank == 1 && rank0.layer == 3 &&
          rank0.first_position == 0 && rank0.rows == 35 &&
          rank0.rotary_dim == 64 && !rank0.uses_mrope);
  const auto bits = attention::layer3_rope_host_bits();
  require(bits.size() == 35 * 64);
  for (int column = 0; column < 32; ++column) {
    require(bits[column] == 0x3f80);
    require(bits[32 + column] == 0x0000);
  }
  struct Golden { int row; int column; std::uint16_t bits; };
  constexpr std::array goldens{
      Golden{1, 0, 0x3f0a}, Golden{1, 31, 0x3f80},
      Golden{1, 32, 0x3f57}, Golden{1, 63, 0x3432},
      Golden{17, 0, 0xbe8d}, Golden{17, 31, 0x3f80},
      Golden{17, 32, 0xbf76}, Golden{17, 63, 0x363d},
      Golden{34, 0, 0xbf59}, Golden{34, 31, 0x3f80},
      Golden{34, 32, 0x3f07}, Golden{34, 63, 0x36bd}};
  for (const auto& golden : goldens)
    require(bits[golden.row * 64 + golden.column] == golden.bits);

  const auto c1_identity = attention::target_qsa_c1_rope_identity(1, 47);
  require(c1_identity.rank == 1 && c1_identity.layer == 47 &&
          c1_identity.first_position == 0 && c1_identity.rows == 36);
  const auto c1_bits = attention::layer3_rope_c1_host_bits();
  require(c1_bits.size() == 36 * 64);
  for (std::size_t index = 0; index < bits.size(); ++index)
    require(c1_bits[index] == bits[index]);
  require(c1_bits[35 * 64] == 0xbf67);
  require(c1_bits[35 * 64 + 31] == 0x3f80);
  require(c1_bits[35 * 64 + 32] == 0xbedb);
  require(c1_bits[35 * 64 + 63] == 0x36c2);

  bool rejected = false;
  try {
    auto drift = rank0;
    drift.rows = 34;
    // A CUDA-free identity rejection must happen before device selection.
    attention::Layer3RopeDeviceOwner owner(-1, drift);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected);
  if (std::getenv("ROCKET_QWEN38_LAYER3_ROPE_C1_DEVICE_PROOF")) {
    attention::Layer3RopeC1DeviceOwner owner(
        0, attention::target_qsa_c1_rope_identity(0, 3));
    const auto view = owner.view();
    require(view.cos_sin != nullptr && view.ready != nullptr &&
            view.payload_sha256 == attention::kLayer3RopeC1PayloadSha256 &&
            view.rows == 36 && view.columns == 64 && view.row_stride == 64);
    std::array<std::uint16_t, 64> position35{};
    require(cudaEventSynchronize(view.ready) == cudaSuccess);
    require(cudaMemcpy(position35.data(), view.cos_sin + 35 * 64,
                       position35.size() * sizeof(position35[0]),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (std::size_t index = 0; index < position35.size(); ++index)
      require(position35[index] == c1_bits[35 * 64 + index]);
  }
  rejected = false;
  try {
    auto drift = c1_identity;
    drift.rows = 35;
    attention::Layer3RopeC1DeviceOwner owner(-1, drift);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected);
  std::cout << "layer3_rope valid=1 complete=1 phase=cpu_golden rows=35 columns=64 sha256="
            << attention::kLayer3RopePayloadSha256 << '\n';
}

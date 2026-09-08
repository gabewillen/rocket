// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_n640_materializer.h"

#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace moe = rocket::qwen38::moe;

int main() {
  constexpr std::size_t e = 256, h = 2560, n = 640, np = 768;
  std::vector<std::uint8_t> w13(e * 2 * n * h / 2, 0x31);
  std::vector<std::uint8_t> s13(e * 2 * n * (h / 16), 0x42);
  std::vector<std::uint8_t> down(e * h * n / 2, 0x53);
  std::vector<std::uint8_t> sd(e * h * (n / 16), 0x64);
  std::vector<float> input(e, 2.0F), a1(e, 3.0F), a2(e, 4.0F), ds(e, 5.0F);
  const auto output = moe::materialize_target_moe_n640_host(
      {w13, s13, down, sd, input, a1, a2, ds});
  if (output.w13_packed.size() != e * 2 * np * h / 2 ||
      output.w13_scale.size() != e * 2 * np * (h / 16) ||
      output.down_packed.size() != e * h * np / 2 ||
      output.down_scale.size() != e * h * (np / 16) ||
      output.folded_w1_alpha.front() != 6.0F)
    return 1;
  // Packed gap between up and gate and down's padded tail are neutral zero.
  if (output.w13_packed[n * h / 2] != 0 ||
      output.w13_packed[np * h / 2] != 0x31 ||
      output.down_packed[n / 2] != 0 ||
      output.down_packed[np / 2] != 0x53)
    return 2;
  if (std::count(output.w13_scale.begin(), output.w13_scale.end(), 0x42) !=
          static_cast<std::ptrdiff_t>(s13.size()) ||
      std::count(output.down_scale.begin(), output.down_scale.end(), 0x64) !=
          static_cast<std::ptrdiff_t>(sd.size()))
    return 3;
  const auto source_digest = moe::target_moe_n640_sha256(
      {w13, s13, down, sd, input, a1, a2, ds});
  const auto physical_digest = moe::target_moe_n768_sha256(output);
  if (source_digest == physical_digest || source_digest[0] == 0 ||
      physical_digest[0] == 0)
    return 4;
  const auto rank0 = moe::target_moe_layer3_transformed_identity(0);
  const auto rank1 = moe::target_moe_layer3_transformed_identity(1);
  if (rank0.rank != 0 || rank1.rank != 1 ||
      rank0.source_planes_sha256 == rank1.source_planes_sha256 ||
      rank0.physical_planes_sha256 == rank1.physical_planes_sha256 ||
      rank0.source_layout_sha256 == rank1.source_layout_sha256)
    return 5;
  try {
    (void)moe::target_moe_layer3_transformed_identity(2);
    return 6;
  } catch (const std::invalid_argument&) {
  }
  return 0;
}

// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer0_boundary_comparator.h"

#include <bit>
#include <cstring>
#include <filesystem>
#include <vector>

int main(int argc, char** argv) {
  using namespace rocket::qwen38::decode;
  if (argc != 2) return 2;
  const auto references = authenticate_target_layer0_boundary_references(
      std::filesystem::path(argv[1]));
  for (const auto& reference : references) {
    const auto exact = compare_target_layer0_boundary_bytes(
        reference, reference.bytes.data(), reference.bytes.size());
    if (!exact.exact || exact.mismatch_count != 0) return 3;
    auto changed = reference.bytes;
    changed[changed.size() / 2] ^= 1;
    const auto mismatch = compare_target_layer0_boundary_bytes(
        reference, changed.data(), changed.size());
    if (mismatch.exact || mismatch.mismatch_count != 1 ||
        mismatch.first_mismatch != changed.size() / 2) return 4;
  }
  std::vector<float> attention(2'560);
  for (std::size_t i = 0; i < attention.size(); ++i) {
    std::uint16_t expected = 0;
    std::memcpy(&expected, references[0].bytes.data() + i * sizeof(expected),
                sizeof(expected));
    attention[i] = std::bit_cast<float>(
        static_cast<std::uint32_t>(expected) << 16);
  }
  const auto rounded = round_target_layer0_attention_to_bf16(
      attention.data(), attention.size());
  const auto attention_exact = compare_target_layer0_boundary_bytes(
      references[0], rounded.data(), rounded.size());
  if (!attention_exact.exact) return 5;
  bool rejected = false;
  try {
    (void)round_target_layer0_attention_to_bf16(attention.data(), 2'559);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) return 6;
}

// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer0_boundary_comparator.h"

#include <filesystem>

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
}

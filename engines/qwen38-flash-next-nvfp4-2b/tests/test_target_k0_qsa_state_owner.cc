// SPDX-License-Identifier: Apache-2.0
#include "attention/target_k0_qsa_state_owner.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

int main() {
  using namespace rocket::qwen38::attention;
  std::vector<std::uint8_t> storage(target_k0_qsa_state_bytes() + 256);
  auto binding = bind_target_k0_qsa_state_storage(
      storage.data(), target_k0_qsa_state_bytes(), 1, 35);
  if (binding.used_bytes != target_k0_qsa_state_bytes() ||
      target_k0_qsa_state_layer_bytes() * 12 != binding.used_bytes)
    return 1;
  for (int index = 0; index < 12; ++index) {
    const auto& view = binding.views[index];
    if (view.rank != 1 || view.layer != 4 * index + 3 || view.rows != 1 ||
        view.main_blocks != 1 || view.compressed_blocks != 1 ||
        view.uses_mrope) return 2;
  }
  bool rejected = false;
  try {
    (void)bind_target_k0_qsa_state_storage(
        storage.data(), target_k0_qsa_state_bytes() - 1, 1, 35);
  } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) return 3;
  rejected = false;
  try {
    (void)bind_target_k0_qsa_state_storage(
        storage.data(), target_k0_qsa_state_bytes(), 0, 87);
  } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) return 4;
  std::printf("qwen38 K0 QSA state: layers=12 bytes=%zu stride=%zu\n",
              binding.used_bytes, target_k0_qsa_state_layer_bytes());
}

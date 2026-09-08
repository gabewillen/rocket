// SPDX-License-Identifier: Apache-2.0
#include "attention/qsa_m35_state_seed.h"

#include <array>
#include <cstdint>

int main() {
  using rocket::qwen38::attention::validate_target_qsa_m35_seed_slots;
  std::array<std::int64_t, 35> main{};
  std::array<std::int64_t, 4> raw{};
  std::array<std::int64_t, 8> compressed{};
  for (std::size_t i = 0; i < main.size(); ++i) main[i] = 1600 + i;
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = 8 + i;
  }
  for (std::size_t i = 0; i < compressed.size(); ++i)
    compressed[i] = 400 + i;
  if (!validate_target_qsa_m35_seed_slots(main.data(), raw.data(),
                                           compressed.data()))
    return 1;
  main[34] = 1635;
  if (validate_target_qsa_m35_seed_slots(main.data(), raw.data(),
                                          compressed.data()))
    return 2;
  main[34] = 1634;
  raw[3] = 9;
  if (validate_target_qsa_m35_seed_slots(main.data(), raw.data(),
                                          compressed.data()))
    return 3;
  raw[3] = 11;
  compressed[0] = 401;
  if (validate_target_qsa_m35_seed_slots(main.data(), raw.data(),
                                          compressed.data()))
    return 4;
}

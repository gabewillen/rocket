// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_b12x_aot.h"

// A shared-library anchor keeps the stable C ABI linkable for validation and
// embedding while its implementation remains in the AOT static target.
extern "C" bool rocket_qwen38_target_moe_b12x_available() noexcept {
  return rocket::qwen38::moe::target_moe_b12x_aot_compiled();
}

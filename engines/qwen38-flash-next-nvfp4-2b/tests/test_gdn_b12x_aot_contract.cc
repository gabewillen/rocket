// SPDX-License-Identifier: Apache-2.0
#include "linear_attention/gdn_b12x_aot.h"

#include <cstdlib>
#include <stdexcept>

namespace linear = rocket::qwen38::linear_attention;

int main() {
  static_assert(linear::kGdnB12xInputWidth == 2'560);
  static_assert(linear::kGdnB12xQkvzWidth == 8'192);
  static_assert(linear::kGdnB12xBaWidth == 48);
  if (linear::gdn_b12x_aot_compiled()) return 0;
  try {
    linear::GdnB12xAot unavailable(0);
    std::abort();
  } catch (const std::runtime_error&) {
  }
  return 0;
}

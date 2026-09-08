// SPDX-License-Identifier: Apache-2.0
#include "linear_attention/gdn_cutlass.h"

#include <cstdlib>
#include <vector>

namespace linear = rocket::qwen38::linear_attention;

namespace {

void check(bool condition) {
  if (!condition) std::abort();
}

void check_sfa_bijection(int tokens, int width) {
  const int padded_tokens = ((tokens + 127) / 128) * 128;
  const int columns = width / 16;
  std::vector<bool> touched(linear::prefill_sfa_bytes(tokens, width));
  for (int row = 0; row < padded_tokens; ++row) {
    for (int column = 0; column < columns; ++column) {
      const auto offset = linear::prefill_sfa_offset(row, column, columns);
      check(offset < touched.size());
      check(!touched[offset]);
      touched[offset] = true;
    }
  }
  for (bool value : touched) check(value);
}

}  // namespace

int main() {
  check(linear::allowed_prefill_tokens(300));
  check(linear::allowed_prefill_tokens(8'192));
  check(!linear::allowed_prefill_tokens(299));
  check(!linear::allowed_prefill_tokens(8'193));
  check(linear::prefill_sfa_bytes(300, 2'560) == 61'440);
  check(linear::prefill_sfa_bytes(300, 3'072) == 73'728);
  check(linear::prefill_sfa_bytes(8'192, 2'560) == 1'310'720);
  check(linear::prefill_sfa_bytes(8'192, 3'072) == 1'572'864);
  check(linear::prefill_sfa_bytes(128, 2'560) == 0);
  check(linear::prefill_sfa_bytes(300, 2'559) == 0);
  check_sfa_bijection(300, 2'560);
  check_sfa_bijection(8'192, 3'072);
  check(linear::CutlassGdnPrefillProjection::input_quantizations_per_launch() ==
        1);
  check(linear::CutlassGdnPrefillProjection::
            reference_input_quantizations_per_launch() == 2);
  check(linear::GdnPrefillInputBackend::kB12x !=
        linear::GdnPrefillInputBackend::kCutlassControl);
  check(std::string_view(linear::kPrefillB12xQuantSourceRevision) ==
        "8e685d198");
  return 0;
}

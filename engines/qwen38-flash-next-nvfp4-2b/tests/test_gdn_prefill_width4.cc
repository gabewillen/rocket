// SPDX-License-Identifier: Apache-2.0
#include "linear_attention/gdn_prefill_width4.h"

#include <array>
#include <cstdlib>

namespace prefill = rocket::qwen38::linear_attention::prefill;

namespace {

void check(bool condition) {
  if (!condition) {
    std::abort();
  }
}

float source_value(const std::array<float, 3>& initial,
                   const std::array<float, 5>& input, int source) {
  return source < 0 ? initial[prefill::initial_state_slot(source)]
                    : input[source];
}

float convolve(const std::array<float, 3>& initial,
               const std::array<float, 5>& input,
               const std::array<float, 4>& weight, int token) {
  float value = 0.0F;
  for (int tap = 0; tap < prefill::kConvWidth; ++tap) {
    const int source = prefill::causal_source_token(token, tap);
    value += source_value(initial, input, source) * weight[tap];
  }
  return value;
}

std::array<float, 3> publish(const std::array<float, 3>& initial,
                             const std::array<float, 5>& input, int tokens) {
  std::array<float, 3> result{};
  for (int slot = 0; slot < prefill::kConvStateWidth; ++slot) {
    result[slot] = source_value(
        initial, input, prefill::published_source_token(tokens, slot));
  }
  return result;
}

}  // namespace

int main() {
  constexpr std::array<float, 3> initial{10.0F, 20.0F, 30.0F};
  constexpr std::array<float, 5> input{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
  constexpr std::array<float, 4> weight{1000.0F, 100.0F, 10.0F, 1.0F};

  check(convolve(initial, input, weight, 0) == 12'301.0F);
  check(convolve(initial, input, weight, 1) == 23'012.0F);
  check(convolve(initial, input, weight, 3) == 1'234.0F);

  check((publish(initial, input, 1) ==
         std::array<float, 3>{20.0F, 30.0F, 1.0F}));
  check((publish(initial, input, 2) ==
         std::array<float, 3>{30.0F, 1.0F, 2.0F}));
  check((publish(initial, input, 3) ==
         std::array<float, 3>{1.0F, 2.0F, 3.0F}));
  check((publish(initial, input, 5) ==
         std::array<float, 3>{3.0F, 4.0F, 5.0F}));
  return 0;
}

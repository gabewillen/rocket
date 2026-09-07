// SPDX-License-Identifier: Apache-2.0
#include "linear_attention/gdn_verifier.h"

#include <iostream>

int main() {
  using rocket::qwen38::linear_attention::VerifierShape;
  using rocket::qwen38::linear_attention::allowed_verifier_shape;
  int failures = 0;
  const int sequences[] = {1, 2, 4, 8, 16};
  for (const int count : sequences) {
    for (int width = 1; width <= 8; ++width) {
      if (!allowed_verifier_shape({count, width}) ||
          VerifierShape{count, width}.token_rows() != count * width) {
        ++failures;
      }
    }
  }
  const VerifierShape rejected[] = {
      {0, 1}, {3, 1}, {16, 0}, {16, 9}, {32, 4},
  };
  for (const auto shape : rejected) {
    if (allowed_verifier_shape(shape)) ++failures;
  }
  if (failures != 0) {
    std::cerr << "GDN verifier shape contract failures=" << failures << '\n';
    return 1;
  }
  std::cout << "gdn_verifier_shapes=40 max_rows=128 rejected_shapes=5\n";
  return 0;
}

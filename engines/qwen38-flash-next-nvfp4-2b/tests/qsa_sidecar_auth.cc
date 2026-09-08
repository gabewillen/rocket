// SPDX-License-Identifier: Apache-2.0
#include "attention/qsa_sidecar_owner.h"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace attention = rocket::qwen38::attention;

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    const std::string_view rank_argument(argv[1]);
    if (rank_argument != "0" && rank_argument != "1")
      throw std::invalid_argument("QSA sidecar rank argument changed");
    const int rank = rank_argument == "0" ? 0 : 1;
    const auto payload = attention::authenticate_qsa_sidecar_host(
        argv[2], attention::layer3_qsa_sidecar_identity(rank));
    std::cout << "qsa_sidecar_auth rank=" << rank << " layer=3 bytes="
              << payload.size() << " valid=1 complete=1\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "qsa_sidecar_auth valid=0 complete=0 phase=authenticate reason="
              << error.what() << '\n';
    return 1;
  }
}

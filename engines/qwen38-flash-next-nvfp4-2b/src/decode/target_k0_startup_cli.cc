// SPDX-License-Identifier: Apache-2.0
#include "output/native_token_io.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string_view>

int main(int argc, char** argv) {
  try {
    if (argc != 5 || std::string_view(argv[1]) != "--tokenizer" ||
        std::string_view(argv[3]) != "--oracle")
      throw std::invalid_argument("expected authenticated roots");
    const auto roots =
        rocket::qwen38::output::authenticate_token_io_artifact_roots(
            std::filesystem::path(argv[2]), std::filesystem::path(argv[4]));
    if (roots.tokenizer_identity_sha256.empty() ||
        roots.oracle_manifest_sha256.empty())
      throw std::logic_error("K0 startup root authentication failed");
    std::puts("{\"phase\":\"composition\",\"outcome\":\"incomplete\","
              "\"missing_dependency\":\"all48_concrete_layer_ports\","
              "\"cuda_launches\":0}");
    return 2;
  } catch (const std::exception&) {
    std::puts("{\"phase\":\"startup\",\"outcome\":\"failure\","
              "\"cuda_launches\":0}");
    return 1;
  }
}

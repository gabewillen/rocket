// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_startup.h"

#include <cstdio>
#include <stdexcept>

namespace decode = rocket::qwen38::decode;
namespace output = rocket::qwen38::output;

int main(int argc, char** argv) {
  try {
    bool rejected = false;
    try {
      output::TokenIoArtifactRoots forged{
          "/forged", "/forged", std::string(64, '0'),
          std::string(decode::kTargetK0OracleManifestSha256)};
      (void)decode::detokenize_target_k0_token(forged, 248'046);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 1;
    if (argc == 1) {
      std::puts("qwen38 K0 startup: forged tokenizer rejected");
      return 0;
    }
    if (argc != 3) return 2;
    auto roots = output::authenticate_token_io_artifact_roots(argv[1], argv[2]);
    if (!decode::detokenize_target_k0_token(roots, 248'046).empty()) return 3;
    rejected = false;
    try { (void)decode::detokenize_target_k0_token(roots, 13); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 4;
    std::puts("qwen38 K0 startup: authenticated EOS detokenization passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 5;
  }
}

#include "hyperconnection/native_adapter.h"

#include <cstdio>
#include <stdexcept>
#include <type_traits>

int main() {
  try {
    if (!std::is_final_v<
            rocket::qwen38::hyperconnection::NativeFullAttentionHyperConnection>)
      throw std::runtime_error("native HC adapter must remain final");
    std::puts("qwen38 native HC adapter type contract passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

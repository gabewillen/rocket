#include "attention/native_qsa_graph.h"

#include <cstdio>
#include <stdexcept>
#include <string_view>

int main() {
  try {
    void* handle = nullptr;
    if (rocket_qwen38_target_qsa_create_c1(
            0, 0, 3, nullptr, nullptr, nullptr, nullptr, &handle) == 0 ||
        handle != nullptr)
      throw std::runtime_error("incomplete QSA factory input was accepted");
    const __nv_bfloat16* output = nullptr;
    if (rocket_qwen38_target_qsa_projected_output_c1(nullptr, &output) == 0)
      throw std::runtime_error("missing QSA graph published an output");
    if (rocket_qwen38_target_qsa_launch_c1(nullptr, nullptr, nullptr, 1, 1,
                                           nullptr) == 0)
      throw std::runtime_error("missing QSA graph launched");
    rocket_qwen38_target_qsa_destroy_c1(nullptr);
    if (!rocket_qwen38_target_qsa_last_error())
      throw std::runtime_error("QSA failure telemetry is unavailable");
    std::puts("qwen38 native QSA C ABI rejection contract passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

// SPDX-License-Identifier: Apache-2.0
#include <stdexcept>

extern "C" {
int qwen38_layer3_rope_c1_create(int, int, int, void**) noexcept;
int qwen38_layer3_rope_c1_view(void*, const void**, void**, int*, int*) noexcept;
int qwen38_layer3_rope_c1_wait(void*, void*) noexcept;
int qwen38_layer3_rope_c1_destroy(void*) noexcept;
}

int main() {
  void* owner = nullptr;
  if (qwen38_layer3_rope_c1_create(-1, 0, 4, &owner) == 0 || owner)
    throw std::runtime_error("invalid c1 RoPE identity was accepted");
  const void* pointer = nullptr;
  void* event = nullptr;
  int rows = 0;
  int columns = 0;
  if (qwen38_layer3_rope_c1_view(nullptr, &pointer, &event, &rows, &columns) == 0)
    throw std::runtime_error("missing c1 RoPE owner published a view");
  if (qwen38_layer3_rope_c1_wait(nullptr, nullptr) == 0)
    throw std::runtime_error("missing c1 RoPE owner accepted a wait");
  return qwen38_layer3_rope_c1_destroy(nullptr);
}

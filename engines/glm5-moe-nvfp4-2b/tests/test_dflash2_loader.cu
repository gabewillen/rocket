// Real-checkpoint DFlash2 loader smoke. Skips when the 2.2 GiB draft shard is absent.
#include "dflash2.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

int main() {
  const char* env = std::getenv("ROCKET_DFLASH2_DIR");
  const std::filesystem::path dir =
      env ? std::filesystem::path(env)
          : std::filesystem::path(std::getenv("HOME")) /
                ".cache/rocket-glm53-exl3-host/models/glm-5.3-flash-dflash2";
  const auto file = dir / "model.safetensors";
  if (!std::filesystem::exists(file)) {
    std::printf("SKIP: %s absent\n", file.c_str());
    return 77;
  }
  rocket::engine::DFlash2Weights w;
  w.load(file);
  const struct {
    const char* name;
    std::size_t bytes;
  } required[] = {
      {"fc.weight", 4096ull * 20480 * 2},
      {"layers.0.self_attn.q_proj.weight", 4096ull * 4096 * 2},
      {"layers.4.mlp.down_proj.weight", 4096ull * 12288 * 2},
      {"candidate_selector.predecessor_codebook", 154880ull * 256 * 2},
      {"candidate_selector.successor_codebook", 154880ull * 256 * 2},
  };
  for (const auto& r : required) {
    if (!w.tensor_data(r.name) || w.tensor_bytes(r.name) != r.bytes) {
      std::fprintf(stderr, "FAIL: %s got %zu bytes expected %zu\n", r.name,
                   w.tensor_bytes(r.name), r.bytes);
      return 1;
    }
  }
  std::printf("PASS: DFlash2 loaded 81 BF16 tensors and required shapes\n");
  return 0;
}

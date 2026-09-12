// DFlash2 draft model for the CUDA engine.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace rocket::engine {

struct DFlash2Config {
  int num_layers = 5;
  int hidden_size = 4096;
  int num_kv_heads = 8;
  int head_dim = 128;
  int intermediate_size = 12288;
  int sliding_window = 2048;
  int mask_token_id = 154856;
  int selector_top_k = 16;
  int selector_rank = 256;
  int target_layer_ids[5] = {5, 14, 24, 33, 42};
  float input_embedding_scale = 1.0f;
};

struct DFlash2Weights {
  DFlash2Config cfg;
  std::vector<char> blob_;     // file contents
  const char* data_ = nullptr; // blob start of tensor data

  void load(const std::filesystem::path& ckpt_file);
  const void* tensor_data(const char* name) const;
  std::size_t tensor_bytes(const char* name) const;
};

}  // namespace rocket::engine

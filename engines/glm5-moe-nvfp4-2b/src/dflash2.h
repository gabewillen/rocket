// DFlash2 draft model weights and raw CUDA forward interface.
#pragma once

#include <cstddef>
#include <filesystem>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kv/nvme_prefix_store.h"

namespace rocket::fabric { class ExpertParallel; }

namespace rocket::engine {

struct DFlash2Config {
  int num_layers = 5;
  int hidden_size = 4096;
  int num_heads = 32;
  int num_kv_heads = 8;
  int head_dim = 128;
  int intermediate_size = 12288;
  int sliding_window = 2048;
  int mask_token_id = 154856;
  int vocab_size = 154880;
  int conv_kernel_size = 2;
  int conv_group_size = 16;
  int selector_top_k = 16;
  int selector_rank = 256;
  int target_layer_ids[5] = {5, 14, 24, 33, 42};
  float input_embedding_scale = 1.0f;
  float rms_norm_eps = 1.0e-5f;
};

class DFlash2Weights {
 public:
  DFlash2Weights() = default;
  ~DFlash2Weights();
  DFlash2Weights(const DFlash2Weights&) = delete;
  DFlash2Weights& operator=(const DFlash2Weights&) = delete;

  void load(const std::filesystem::path& ckpt_file);
  const void* tensor_data(const char* name) const;
  std::size_t tensor_bytes(const char* name) const;
  const std::vector<std::int64_t>& tensor_shape(const char* name) const;

  DFlash2Config cfg;

 private:
  void* device_slab_ = nullptr;
  std::map<std::string, void*, std::less<>> tensors_dev_;
  std::map<std::string, std::size_t, std::less<>> bytes_;
  std::map<std::string, std::vector<std::int64_t>, std::less<>> shapes_;
};

// Exact qwen3_dflash2.py grouped dynamic convolution. `coeff` is workspace
// [rows,2,taps,hidden/group_size] produced by kernel_projection.
void dflash2_grouped_conv_prepare(__nv_bfloat16* out, __nv_bfloat16* coeff,
                                  const __nv_bfloat16* hidden,
                                  const __nv_bfloat16* kernel_projection,
                                  const __nv_bfloat16* base_kernel, int rows, int hidden_size,
                                  int block_size, int group_size, int taps, cudaStream_t stream);
void dflash2_grouped_conv_finish(__nv_bfloat16* out, const __nv_bfloat16* hidden,
                                 const __nv_bfloat16* coeff,
                                 const __nv_bfloat16* base_kernel, int rows, int hidden_size,
                                 int block_size, int group_size, int taps, cudaStream_t stream);

class DFlash2DraftEngine {
 public:
  DFlash2DraftEngine(const std::filesystem::path& dir, const __nv_bfloat16* target_embed,
                     const __nv_bfloat16* target_lm_head, int max_batch, int max_tokens,
                     int max_draft_tokens, rocket::fabric::ExpertParallel* ep = nullptr);
  ~DFlash2DraftEngine();
  DFlash2DraftEngine(const DFlash2DraftEngine&) = delete;
  DFlash2DraftEngine& operator=(const DFlash2DraftEngine&) = delete;

  // Aux layout is [5][aux_stride_rows][4096]. Accepted target rows are
  // position-major [position][batch]; base_pos is each stream's first row.
  void append_context(const __nv_bfloat16* aux, int aux_stride_rows, int positions, int batch,
                      const std::vector<int>& base_pos, const std::vector<int>& accepted,
                      cudaStream_t stream);
  // Produces `draft_tokens` tokens per stream in position-major order.
  void propose(const std::vector<int>& anchor, const std::vector<int>& position, int batch,
               int draft_tokens, std::vector<int>& out, cudaStream_t stream);
  // The proposal model only attends to its configured trailing window. These
  // methods persist and restore exactly that rank-local K/V state.
  std::size_t prefix_state_bytes(int position) const;
  std::uint64_t prefix_state_digest(int slot, int position) const;
  bool prefix_state_equal(int a, int b, int position) const;
  void save_prefix_state(kv::NvmePrefixStore& store, const kv::PrefixRecordKey& key,
                         int slot, int position, cudaStream_t stream);
  bool load_prefix_state(kv::NvmePrefixStore& store, const kv::PrefixRecordKey& key,
                         int slot, int position, cudaStream_t stream);
  void copy_prefix_state(int dst_slot, int src_slot, int position, cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace rocket::engine

// Weight residency for one booster.
//
// The NVFP4 checkpoint is 182 GiB and a booster is 123.73 GiB
// (blog/posts/hardware/2026-09-05-spark-cluster-baseline/), so the whole fuel
// cannot be resident on one node. It does not have to be: only the routed
// experts are quantized, and only 8 of 288 of them fire per token per layer.
//
// So residency splits in two:
//   resident   every BF16 tensor: embeddings, all 45 layers of attention,
//              the 3 dense MLPs, the shared experts, the routers, the
//              hyper-connection parameters, the final norm and lm_head.
//   streamed   the routed experts, fetched from the mmap'd checkpoint into a
//              bounded device cache keyed by (layer, expert) and evicted LRU.
//
// The split is what makes a single-booster decode step possible at all. The
// two-booster expert-parallel arrangement replaces the streaming half.
#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "model_config.h"
#include "safetensors.h"

namespace rocket::engine {

using bf16 = __nv_bfloat16;

struct HyperConnW {
  const bf16* fn = nullptr;     // [hc_mix, hc*hidden]
  const float* base = nullptr;  // [hc_mix]
  const float* scale = nullptr; // [3]
};

struct KdaW {
  const bf16* qkv = nullptr;      // [3*qkv_dim, hidden], q then k then v
  const bf16* conv = nullptr;     // [3*qkv_dim, kernel]
  const bf16* f_a = nullptr;      // [head_dim, hidden]
  const bf16* f_b = nullptr;      // [qkv_dim, head_dim]
  const bf16* g_a = nullptr;
  const bf16* g_b = nullptr;
  const bf16* b_proj = nullptr;   // [heads, hidden]
  const float* a_log = nullptr;   // [heads]
  const float* dt_bias = nullptr; // [qkv_dim]
  const bf16* o_norm = nullptr;   // [head_dim]
  const bf16* o_proj = nullptr;   // [hidden, qkv_dim]
};

struct MlaW {
  const bf16* q_a = nullptr;        // [q_lora, hidden]
  const bf16* q_a_norm = nullptr;   // [q_lora]
  const bf16* q_b = nullptr;        // [heads*qk, q_lora]
  const bf16* kv_a = nullptr;       // [kv_lora, hidden]
  const bf16* kv_a_norm = nullptr;  // [kv_lora]
  const bf16* kv_b = nullptr;       // [heads*(nope+v), kv_lora]
  const bf16* o_proj = nullptr;     // [hidden, heads*v]
  const bf16* idx_wk = nullptr;     // [index_head_dim, hidden]
  const bf16* idx_wq_b = nullptr;   // [index_heads*index_head_dim, q_lora]
  const bf16* idx_k_norm_w = nullptr;
  const bf16* idx_k_norm_b = nullptr;
  const bf16* idx_weights = nullptr;  // [index_heads, hidden]
  const bf16* idx_gate = nullptr;     // [index_head_dim, hidden]
  const bf16* idx_ape = nullptr;      // [kpool, index_head_dim]
};

struct DenseMlpW {
  const bf16* gate = nullptr;  // [inter, hidden]
  const bf16* up = nullptr;
  const bf16* down = nullptr;  // [hidden, inter]
};

struct MoeW {
  const bf16* router = nullptr;       // [n_experts, hidden]
  const float* router_bias = nullptr; // [n_experts]
  DenseMlpW shared;                   // moe_intermediate * n_shared wide
};

struct LayerW {
  HyperConnW attn_hc, ffn_hc;
  const bf16* input_norm = nullptr;
  const bf16* post_attn_norm = nullptr;
  KdaW kda;
  MlaW mla;
  DenseMlpW dense;
  MoeW moe;
};

// One routed expert, resident in the device cache.
struct ExpertDev {
  const std::uint8_t* gate_packed = nullptr;
  const std::uint8_t* gate_scale = nullptr;
  float gate_global = 0.0f;
  const std::uint8_t* up_packed = nullptr;
  const std::uint8_t* up_scale = nullptr;
  float up_global = 0.0f;
  const std::uint8_t* down_packed = nullptr;
  const std::uint8_t* down_scale = nullptr;
  float down_global = 0.0f;
};

class WeightStore {
 public:
  // expert_cache_bytes bounds the streamed half. Throws std::runtime_error if
  // the resident half does not fit or a tensor is missing.
  WeightStore(const fuel::ModelConfig& cfg, const std::filesystem::path& snapshot_dir,
              std::size_t expert_cache_bytes);
  ~WeightStore();

  const LayerW& layer(int i) const { return layers_[i]; }
  const bf16* embed() const { return embed_; }
  const bf16* final_norm() const { return final_norm_; }
  const bf16* lm_head() const { return lm_head_; }

  // Returns the expert, fetching it from the checkpoint if it is not resident.
  const ExpertDev& expert(int layer, int expert_id, cudaStream_t s);

  std::size_t resident_bytes() const { return resident_bytes_; }
  std::size_t expert_slot_bytes() const { return slot_bytes_; }
  std::size_t expert_slots() const { return slots_.size(); }
  std::uint64_t expert_hits() const { return hits_; }
  std::uint64_t expert_misses() const { return misses_; }
  std::size_t expert_bytes_streamed() const { return streamed_bytes_; }

 private:
  const bf16* upload_bf16(std::string_view name, std::int64_t expect_numel);
  const float* upload_f32(std::string_view name, std::int64_t expect_numel);
  // Concatenates several tensors of equal trailing shape into one device buffer.
  const bf16* upload_concat(const std::vector<std::string>& names, std::int64_t expect_numel);
  void* device_alloc(std::size_t bytes);
  void copy_in(void* dst, const void* src, std::size_t bytes);

  fuel::ModelConfig cfg_;
  fuel::Checkpoint ckpt_;
  std::vector<LayerW> layers_;
  const bf16* embed_ = nullptr;
  const bf16* final_norm_ = nullptr;
  const bf16* lm_head_ = nullptr;

  std::vector<void*> owned_;
  std::size_t resident_bytes_ = 0;

  // --- streamed experts ---
  struct Slot {
    std::uint8_t* base = nullptr;
    long long key = -1;  // layer * n_experts + expert
  };
  std::vector<Slot> slots_;
  std::list<int> lru_;                                  // front is most recent
  std::unordered_map<long long, int> resident_expert_;  // key -> slot
  std::vector<std::list<int>::iterator> lru_pos_;
  std::vector<ExpertDev> slot_view_;
  std::size_t slot_bytes_ = 0;
  std::uint8_t* pinned_ = nullptr;
  std::size_t pinned_bytes_ = 0;
  std::uint64_t hits_ = 0, misses_ = 0;
  std::size_t streamed_bytes_ = 0;
};

}  // namespace rocket::engine

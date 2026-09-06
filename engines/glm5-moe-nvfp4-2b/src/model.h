// One decode step of GLM-5.3-Flash on one booster, layers 0..44.
//
// Batch 1, one token per call. Prefill runs the same path once per prompt
// token, because the KDA recurrence is sequential and a chunked prefill would
// be a second implementation of it.
//
// Layer 45 (the MTP draft layer) is loaded by nothing here and speculation is
// off; the vision tower is ignored.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model_config.h"
#include "weights.h"

namespace rocket::engine {

// Wall time attributed to each stage of one step, in milliseconds. Collected
// with an explicit sync per stage, so the sum is larger than an uninstrumented
// step; `total_uninstrumented_ms` is measured separately.
struct StageMs {
  double embed = 0, hyper_connection = 0, norms = 0, kda = 0, mla = 0, indexer = 0;
  double dense_mlp = 0, moe_router = 0, moe_experts = 0, moe_shared = 0, expert_stream = 0;
  double lm_head = 0;
  double sum() const {
    return embed + hyper_connection + norms + kda + mla + indexer + dense_mlp + moe_router +
           moe_experts + moe_shared + expert_stream + lm_head;
  }
};

class DecodeEngine {
 public:
  DecodeEngine(const fuel::ModelConfig& cfg, const std::filesystem::path& snapshot_dir,
               std::size_t expert_cache_bytes, int max_tokens);
  ~DecodeEngine();

  // Runs one step for `token` at the current position and returns the greedy
  // next token. Advances the position and every cache.
  int step(int token, bool collect_stages);

  void reset();

  const StageMs& stages() const { return stages_; }
  // RMS of the hyper-connection stream mean after each layer, from the last step.
  const std::vector<float>& layer_rms() const { return layer_rms_; }
  // Shannon entropy of the router distribution, averaged over the MoE layers
  // of the last step, in nats.
  double router_entropy() const { return router_entropy_; }
  int position() const { return pos_; }
  WeightStore& weights() { return w_; }

 private:
  void run_kda(int layer, int slot);
  void run_mla(int layer, int slot);
  void run_dense_mlp(int layer);
  void run_moe(int layer);
  float sync_rms(const bf16* x, int n);

  fuel::ModelConfig cfg_;
  WeightStore w_;
  cudaStream_t stream_ = nullptr;
  int max_tokens_ = 0;
  int pos_ = 0;

  std::vector<int> kda_slot_, mla_slot_;

  // caches
  float* kda_state_ = nullptr;
  bf16* kda_conv_ = nullptr;
  bf16* mla_latent_ = nullptr;
  bf16* idx_key_ = nullptr;
  bf16* idx_gate_ = nullptr;

  // activations
  bf16 *streams_ = nullptr, *residual_ = nullptr, *collapsed_ = nullptr, *normed_ = nullptr;
  bf16 *sublayer_out_ = nullptr, *hmean_ = nullptr;
  float *mix_ = nullptr, *post_ = nullptr, *comb_ = nullptr;
  bf16 *qkv_ = nullptr, *qkv_conv_ = nullptr, *lr_a_ = nullptr, *lr_b_ = nullptr, *gate_ = nullptr;
  bf16 *beta_raw_ = nullptr, *beta_ = nullptr, *kda_o_ = nullptr, *kda_on_ = nullptr;
  bf16 *q_resid_raw_ = nullptr, *q_resid_ = nullptr, *q_ = nullptr, *ckv_ = nullptr;
  bf16 *q_idx_ = nullptr, *idx_k_raw_ = nullptr, *pool_keys_ = nullptr, *v_out_ = nullptr;
  float *q_abs_ = nullptr, *scores_ = nullptr, *ctx_ = nullptr, *pool_scores_ = nullptr;
  float* head_w_ = nullptr;
  int *sel_pools_ = nullptr, *sel_tokens_ = nullptr, *n_sel_ = nullptr, *n_tok_ = nullptr;
  float *router_logits_ = nullptr, *topk_w_ = nullptr;
  int* topk_idx_ = nullptr;
  bf16 *mlp_gate_ = nullptr, *mlp_up_ = nullptr, *mlp_h_ = nullptr, *mlp_out_ = nullptr;
  bf16* acc_ = nullptr;
  float *logits_ = nullptr, *scratch_f_ = nullptr;
  int* argmax_i_ = nullptr;

  std::vector<void*> owned_;
  StageMs stages_;
  std::vector<float> layer_rms_;
  double router_entropy_ = 0.0;
};

}  // namespace rocket::engine

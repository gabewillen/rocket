// The GLM-5.3-Flash decode step on one booster, layers 0..44, M concurrent
// streams.
//
// Every call to step() advances every active stream by exactly one position.
// Streams occupy slots [0, batch) of every per-stream buffer; batch may vary
// call to call, but a caller that mixes batch sizes across calls on the same
// engine is mixing which physical slots are live and must reset() first.
// Prefill is the same call with `collect_stages` off, one token per stream
// per call, run once per prompt token; the KDA recurrence is sequential, so a
// chunked prefill would be a second implementation of it.
//
// Only the batched decode-step glue is stage 1 (embeddings, norms,
// hyper-connections, router, dense MLP, shared expert, lm_head, and the
// per-stream paged MLA/indexer KV). The routed-expert path stays the
// dequant-GEMV looped over (stream, selected expert) pairs; stage 2 replaces
// that loop with a CUTLASS grouped GEMM (see kernels.h: nvfp4_quantize_rows,
// swiglu_grouped, moe_gather_rows, moe_scatter_add, all declared and
// unimplemented on purpose -- that is stage 2's seam).
//
// Layer 45 (the MTP draft layer) is loaded by nothing here and speculation is
// off; the vision tower is ignored.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernels.h"
#include "model_config.h"
#include "weights.h"

namespace rocket::engine {

// Wall time attributed to each stage of one step, in milliseconds, summed
// over every layer of that kind. Collected with an explicit sync per stage,
// so the sum is larger than an uninstrumented step; `total_uninstrumented_ms`
// is measured separately by the caller.
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
  // max_batch bounds every per-stream buffer and the trivial one-page-per-
  // stream KV table (see model.cu: stage 1 gives every stream slot its own
  // physical page rather than a real evicting page pool; the KvPages plumbing
  // is real, only the allocator behind it is the simplification).
  DecodeEngine(const fuel::ModelConfig& cfg, const std::filesystem::path& snapshot_dir,
               std::size_t expert_cache_bytes, int max_tokens, int max_batch);
  ~DecodeEngine();

  // Runs one batched step. tokens[i] is the input token for stream slot i,
  // for i in [0, tokens.size()). Every active slot's KDA state, paged KV, and
  // position advance by one. out_tokens is resized to tokens.size() and
  // filled with each slot's greedy next token.
  void step(const std::vector<int>& tokens, std::vector<int>& out_tokens, bool collect_stages);

  // Resets every stream slot's KDA state, hyper-connection streams, and
  // position to 0. The paged MLA/indexer KV is not cleared: positions restart
  // at 0, so nothing downstream reads the stale entries above the new
  // position, exactly as the single-stream engine already relied on before
  // batching (its KV was never cleared here either).
  void reset();

  const StageMs& stages() const { return stages_; }
  // RMS of stream slot 0's hyper-connection stream mean after each layer,
  // from the last instrumented step. Multi-stream RMS is not collected: this
  // is a debugging signal, not a per-stream correctness contract.
  const std::vector<float>& layer_rms() const { return layer_rms_; }
  // Shannon entropy of the router distribution, averaged over slot 0's MoE
  // layers of the last instrumented step, in nats.
  double router_entropy() const { return router_entropy_; }
  int position(int stream) const { return pos_[stream]; }
  int max_batch() const { return max_batch_; }
  WeightStore& weights() { return w_; }

  // Telemetry: per-projection-site activation absmax, calibration capture
  // stage 1 (fuels/glm-5.3-flash/fuel.yaml, serving_regime.quantization_plan).
  // Every linear projection at a hyper-connection site shares that site's
  // `normed_` input, so one absmax per (layer, site) covers every projection
  // fed by it. Off by default; costs one extra reduction pass per site per
  // step when on.
  void set_telemetry(bool on) { telemetry_ = on; }
  bool telemetry() const { return telemetry_; }
  const std::unordered_map<std::string, float>& telemetry_absmax() const {
    return telemetry_absmax_;
  }

 private:
  void run_kda(int layer, int slot, int batch);
  void run_mla(int layer, int slot, int batch, const int* n_tokens_dev, int n_pools_max);
  void run_dense_mlp(int layer, int batch);
  void run_moe(int layer, int batch);
  float sync_rms_slot0(const bf16* x, int n);
  void record_absmax(const std::string& name, const bf16* x, int batch, int n);

  fuel::ModelConfig cfg_;
  WeightStore w_;
  cudaStream_t stream_ = nullptr;
  int max_tokens_ = 0;
  int max_batch_ = 0;
  std::vector<int> pos_;  // per stream slot, host

  int kda_layers_ = 0, mla_layers_ = 0;
  std::vector<int> kda_slot_, mla_slot_;

  // --- KDA state, per (layer slot, stream slot); layer-slot-major so a
  // per-layer call sees a plain [batch, ...] contiguous block. ---
  bf16 *q_conv_state_ = nullptr, *k_conv_state_ = nullptr, *v_conv_state_ = nullptr;  // [kda_layers][max_batch][qkv][taps]
  float* kda_state_ = nullptr;  // [kda_layers][max_batch][heads][hd][hd]

  // --- paged MLA latent + indexer key/gate, one physical page per stream
  // slot (max_pages = 1, page_tokens = max_tokens). ---
  KvPages mla_kv_;
  int* kv_table_ = nullptr;  // [max_batch], table[m] = m

  // per-step device scratch
  int* tokens_dev_ = nullptr;
  int* pos_dev_ = nullptr;
  int* n_tokens_dev_ = nullptr;
  int* n_pools_dev_ = nullptr;

  // activations, all [max_batch, width] row-major unless noted
  bf16 *streams_ = nullptr, *residual_ = nullptr, *collapsed_ = nullptr, *normed_ = nullptr;
  bf16 *sublayer_out_ = nullptr, *hmean_ = nullptr;
  float *mix_ = nullptr, *post_ = nullptr, *comb_ = nullptr;
  bf16 *q_raw_ = nullptr, *k_raw_ = nullptr, *v_raw_ = nullptr;
  bf16 *q_conv_ = nullptr, *k_conv_ = nullptr, *v_conv_ = nullptr;
  bf16 *lr_a_ = nullptr, *lr_b_ = nullptr, *gate_ = nullptr;
  bf16 *beta_raw_ = nullptr, *beta_ = nullptr, *kda_o_ = nullptr, *kda_on_ = nullptr;
  bf16 *q_resid_raw_ = nullptr, *q_resid_ = nullptr, *q_ = nullptr, *ckv_ = nullptr;
  bf16 *latent_stage_ = nullptr, *v_out_ = nullptr;
  bf16 *q_idx_ = nullptr, *idx_k_raw_ = nullptr, *idx_k_stage_ = nullptr, *idx_g_stage_ = nullptr;
  bf16 *pool_keys_ = nullptr;
  float *q_abs_ = nullptr, *scores_ = nullptr, *ctx_ = nullptr, *pool_scores_ = nullptr;
  float* head_w_ = nullptr;
  int *sel_pools_ = nullptr, *sel_tokens_ = nullptr, *n_sel_ = nullptr, *n_tok_ = nullptr;
  float *router_logits_ = nullptr, *topk_w_ = nullptr;
  int* topk_idx_ = nullptr;
  // single-row scratch for the unbatched routed-expert GEMV loop
  bf16 *exp_gate_ = nullptr, *exp_up_ = nullptr, *exp_h_ = nullptr, *exp_out_ = nullptr;
  bf16 *mlp_gate_ = nullptr, *mlp_up_ = nullptr, *mlp_h_ = nullptr, *mlp_out_ = nullptr;
  bf16* acc_ = nullptr;
  float *logits_ = nullptr, *scratch_f_ = nullptr;
  int* argmax_i_ = nullptr;

  int sel_stride_ = 0;       // == max_sel_tokens, also n_sel_max for mla_scores/context
  int pool_stride_ = 0;      // == n_pools_max bound (max_tokens / kpool + 1)
  int sel_pool_stride_ = 0;  // == index_select_pools()

  std::vector<void*> owned_;
  StageMs stages_;
  std::vector<float> layer_rms_;
  double router_entropy_ = 0.0;

  bool telemetry_ = false;
  std::unordered_map<std::string, float> telemetry_absmax_;
  float* telemetry_scratch_ = nullptr;  // [max_batch]
};

}  // namespace rocket::engine

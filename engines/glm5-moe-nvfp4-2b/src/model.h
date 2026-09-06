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
// Stage 1 batched the decode-step glue (embeddings, norms, hyper-connections,
// router, dense MLP, shared expert, lm_head, and the per-stream paged
// MLA/indexer KV). The routed-expert path stayed a dequant-GEMV loop over
// (stream, selected expert) pairs, run once per pair regardless of batch.
//
// Stage 2 (blog/posts/runtime/2026-09-07-*) replaces that loop above a
// measured crossover batch (kGroupedMoeMinBatch, model.cu) with a CUTLASS
// grouped GEMM: gather every (stream, expert) row across the whole batch,
// quantize to NVFP4, one grouped GEMM per stage (w13 fused gate+up, w2
// down), swiglu across the fused w13 halves, scatter-add into the batch
// accumulator. Below the crossover the GEMV loop stays the runtime path: at
// M=1 there is only one group per expert and no batching to win back the
// fixed cost of the grouped path's host-side bookkeeping. MoePath lets a test
// force either path at any batch to compare them directly.
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

namespace rocket::fabric {
class ExpertParallel;
}

namespace rocket::engine {

// Wall time attributed to each stage of one step, in milliseconds, summed
// over every layer of that kind. Collected with an explicit sync per stage,
// so the sum is larger than an uninstrumented step; `total_uninstrumented_ms`
// is measured separately by the caller.
struct StageMs {
  double embed = 0, hyper_connection = 0, norms = 0, kda = 0, mla = 0, indexer = 0;
  double dense_mlp = 0, moe_router = 0, moe_experts = 0, moe_shared = 0, expert_stream = 0;
  double lm_head = 0;
  // Host-only wall time (std::chrono, no GPU sync) spent building the
  // std::map<expert_id, rows> grouping in run_moe_grouped, summed over every
  // MoE layer of the step. Not part of sum(): it overlaps moe_experts, it is
  // a diagnostic for the "Next" item in blog/posts/runtime/2026-09-07-
  // grouped-gemm-beats-gemv-at-every-m/ (replace with a flat counting pass
  // if it shows up), not a stage the decode step is attributed to.
  double moe_group_host_ms = 0;
  // Wall time inside the two-booster routed-expert row exchange
  // (src/fabric/expert_parallel.h), summed over the step's MoE layers. Zero
  // on a single booster. Counted in sum() because it is serial in the step:
  // the layer cannot scatter-add until the peer's rows have landed. It
  // includes waiting for the peer, so it is the split's exposed cost and an
  // upper bound on the transport's own.
  double fabric = 0;
  double sum() const {
    return embed + hyper_connection + norms + kda + mla + indexer + dense_mlp + moe_router +
           moe_experts + moe_shared + expert_stream + lm_head + fabric;
  }
};

enum class MoePath { kAuto, kForceGemv, kForceGrouped };

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

  // Test-only override of the GEMV/grouped-GEMM crossover (model.cu). kAuto
  // (default) picks by batch size against the measured kGroupedMoeMinBatch.
  void set_moe_path(MoePath p) { moe_path_ = p; }
  MoePath moe_path() const { return moe_path_; }

  // Mounts this engine as one rank of the two-booster expert split. Must be
  // set before the first step(), and the matching WeightStore::set_expert_range
  // must already be in place, since run_moe_grouped will only ever fetch the
  // experts this rank owns. Null (the default) is the single-booster engine,
  // unchanged.
  void set_expert_parallel(fabric::ExpertParallel* ep) { ep_ = ep; }
  fabric::ExpertParallel* expert_parallel() const { return ep_; }

  // Test-only: CUDA graph capture of the decode step's non-routing work
  // (blog/posts/runtime/2026-09-07-grouped-gemm-beats-gemv-at-every-m/'s
  // Next item 2). Off by default. Only takes effect when collect_stages is
  // false and telemetry() is off, since both read back per-stage state with
  // a host sync that cannot be captured; step() falls back to the direct
  // per-layer path otherwise. See model.cu::ensure_graphs_built for what is
  // and is not captured, and why.
  void set_use_cuda_graph(bool on) { use_cuda_graph_ = on; }
  bool use_cuda_graph() const { return use_cuda_graph_; }

  // Test-only: the raw lm_head logits row for one stream slot from the last
  // step() call, copied to host. Used to compare the grouped and GEMV MoE
  // paths directly (tests/test_moe_grouped.cu) rather than only through the
  // greedy token they produce.
  std::vector<float> last_logits(int stream) const;
  const std::unordered_map<std::string, float>& telemetry_absmax() const {
    return telemetry_absmax_;
  }

 private:
  void run_kda(int layer, int slot, int batch);
  void run_mla(int layer, int slot, int batch, const int* n_tokens_dev, int n_pools_max);
  void run_dense_mlp(int layer, int batch);
  void run_moe(int layer, int batch);
  void run_moe_gemv(int layer, int batch, const std::vector<int>& idx);
  bool run_moe_grouped(int layer, int batch, const std::vector<int>& idx,
                       const std::vector<float>& wts);
  float sync_rms_slot0(const bf16* x, int n);
  void record_absmax(const std::string& name, const bf16* x, int batch, int n);

  // ---- CUDA graph capture path (kept deliberately separate from run_moe
  // and the direct per-layer loop in step(), not a refactor of them: the
  // direct path stays byte-for-byte what it always was, so batch-parity and
  // moe-grouped exercise it unchanged regardless of use_cuda_graph_). A
  // layer's attention site and a dense FFN site have no host dependency and
  // are captured once per batch size, replayed every step. A MoE FFN site
  // splits into a captured router half (ends at an async device->host copy
  // of the routing decision, no sync), an uncaptured dispatch half
  // (WeightStore::expert() is host-driven LRU/mmap bookkeeping -- reading a
  // checkpoint, updating a std::list -- that graph capture cannot express,
  // and it must see the routing decision before it can pick pointers for
  // the grouped GEMM, so this is a structural reason, not a missing
  // optimization), and a captured shared-expert-FFN-plus-combine tail,
  // deferred to the *start* of the next graph segment since it depends on
  // acc_ written by the uncaptured dispatch. See model.cu::ensure_graphs_built.
  void run_attn_site(int layer, int batch, int n_pools_launch);
  void run_ffn_site_dense(int layer, int batch);
  void run_moe_router_stage(int layer, int batch);
  void run_moe_dispatch_stage(int layer, int batch);
  void run_moe_post_stage(int layer, int batch);
  void run_step_layers_graph(int batch);
  void ensure_graphs_built(int batch);
  void destroy_graphs();

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

  // --- grouped-GEMM (stage 2) routed-expert scratch, sized to the worst
  // case of one group per gathered row: max_batch * num_experts_per_tok. ---
  int moe_max_rows_ = 0;
  bf16* moe_x_ = nullptr;              // [rows, H] gathered stream residuals
  std::uint8_t* moe_a1_packed_ = nullptr;  // [rows, H/2] NVFP4 activations, w13
  std::uint8_t* moe_a1_sf_ = nullptr;      // swizzled SFA slabs, w13
  bf16* moe_gu_ = nullptr;             // [rows, 2*MI] fused w13 raw output
  bf16* moe_h_ = nullptr;              // [rows, MI] post-swiglu
  std::uint8_t* moe_a2_packed_ = nullptr;  // [rows, MI/2] NVFP4 activations, w2
  std::uint8_t* moe_a2_sf_ = nullptr;      // swizzled SFA slabs, w2
  bf16* moe_out_ = nullptr;            // [rows, H] w2 raw output
  int* moe_row_of_ = nullptr;          // [rows] source/dest stream slot
  int* moe_row_in_group_ = nullptr;    // [rows] row index within its group
  int* moe_group_of_row_ = nullptr;    // [rows] which group (active expert)
  long long* moe_sf1_base_ = nullptr;  // [rows] per-group SFA byte offset, w13
  long long* moe_sf2_base_ = nullptr;  // [rows] per-group SFA byte offset, w2
  float* moe_gate_global_ = nullptr;   // [rows] per-group gate weight_scale_2
  float* moe_up_global_ = nullptr;     // [rows] per-group up weight_scale_2
  float* moe_scatter_w_ = nullptr;     // [rows] topk_w * down weight_scale_2
  MoePath moe_path_ = MoePath::kAuto;
  fabric::ExpertParallel* ep_ = nullptr;

  // ---- CUDA graph capture (model.cu::ensure_graphs_built) ----
  std::vector<int> moe_layer_ids_;   // text-layer indices with sparse MLP, ascending
  bool use_cuda_graph_ = false;
  int graph_batch_ = -1;             // batch the cached graphs below were built for, -1 = none
  std::vector<cudaGraph_t> graphs_;
  std::vector<cudaGraphExec_t> graph_execs_;
  // Async device->host target for one MoE layer's routing decision inside a
  // captured graph segment (run_moe_router_stage); pinned so the copy is a
  // real DMA and a legal graph memcpy node. Reused every MoE layer of every
  // step: the dispatch that reads it always runs, synchronously, before the
  // next graph segment's router stage can overwrite it.
  int* moe_idx_pinned_ = nullptr;    // [moe_max_rows_]
  float* moe_wts_pinned_ = nullptr;  // [moe_max_rows_]
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

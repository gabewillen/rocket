#include "model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "kernels.h"

namespace rocket::engine {
namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("rocket::engine::model: " + what);
}
void cuda_check(cudaError_t e, const std::string& what) {
  if (e != cudaSuccess) fail(what + ": " + cudaGetErrorString(e));
}

using Clock = std::chrono::steady_clock;

class StageTimer {
 public:
  StageTimer(cudaStream_t s, double* sink, bool on) : s_(s), sink_(sink), on_(on) {
    if (on_) t0_ = Clock::now();
  }
  ~StageTimer() {
    if (!on_) return;
    cudaStreamSynchronize(s_);
    *sink_ += std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
  }

 private:
  cudaStream_t s_;
  double* sink_;
  bool on_;
  Clock::time_point t0_;
};

}  // namespace

DecodeEngine::DecodeEngine(const fuel::ModelConfig& cfg, const std::filesystem::path& snapshot_dir,
                           std::size_t expert_cache_bytes, int max_tokens, int max_batch)
    : cfg_(cfg),
      w_(cfg, snapshot_dir, expert_cache_bytes),
      max_tokens_(max_tokens),
      max_batch_(max_batch) {
  if (max_batch_ <= 0) fail("max_batch must be positive");
  cuda_check(cudaStreamCreate(&stream_), "stream");

  const int H = cfg_.hidden_size;
  const int hc = cfg_.hc_mult;
  const int qkv = cfg_.kda_qkv_dim();
  const int hd = cfg_.kda_head_dim;
  const int taps = cfg_.conv_state_taps();
  const int heads_kda = cfg_.kda_heads;
  const int heads_mla = cfg_.mla_heads;
  const int kvl = cfg_.kv_lora_rank;
  const int ih = cfg_.index_n_heads, ihd = cfg_.index_head_dim;
  sel_stride_ = cfg_.index_topk + cfg_.index_kpool - 1;
  pool_stride_ = max_tokens / cfg_.index_kpool + 1;
  sel_pool_stride_ = cfg_.index_select_pools();
  const int big_inter =
      std::max(cfg_.intermediate_size, cfg_.moe_intermediate_size * cfg_.n_shared_experts);
  const int MB = max_batch_;

  auto alloc = [&](std::size_t bytes) {
    void* p = nullptr;
    cuda_check(cudaMalloc(&p, bytes), "cudaMalloc activations");
    cudaMemset(p, 0, bytes);
    owned_.push_back(p);
    return p;
  };
  auto A = [&](std::size_t n) { return static_cast<bf16*>(alloc(n * sizeof(bf16))); };
  auto F = [&](std::size_t n) { return static_cast<float*>(alloc(n * sizeof(float))); };
  auto I = [&](std::size_t n) { return static_cast<int*>(alloc(n * sizeof(int))); };

  kda_slot_.assign(cfg_.text_layers, -1);
  mla_slot_.assign(cfg_.text_layers, -1);
  int nk = 0, nm = 0;
  for (int l = 0; l < cfg_.text_layers; ++l) {
    if (cfg_.layers[l].attn == fuel::AttnKind::kKda) kda_slot_[l] = nk++;
    else mla_slot_[l] = nm++;
  }
  kda_layers_ = nk;
  mla_layers_ = nm;

  // --- KDA state, layer-slot-major so a per-layer call is a plain
  // [batch, ...] contiguous block. ---
  q_conv_state_ = A(static_cast<std::size_t>(nk) * MB * qkv * taps);
  k_conv_state_ = A(static_cast<std::size_t>(nk) * MB * qkv * taps);
  v_conv_state_ = A(static_cast<std::size_t>(nk) * MB * qkv * taps);
  kda_state_ = F(static_cast<std::size_t>(nk) * MB * heads_kda * hd * hd);

  // --- paged MLA latent + indexer key/gate: one physical page per stream
  // slot, sized to the whole context window. ---
  mla_kv_.latent = A(static_cast<std::size_t>(MB) * nm * max_tokens * kvl);
  mla_kv_.key = A(static_cast<std::size_t>(MB) * nm * max_tokens * ihd);
  mla_kv_.gate = A(static_cast<std::size_t>(MB) * nm * max_tokens * ihd);
  mla_kv_.max_pages = 1;
  mla_kv_.page_tokens = max_tokens;
  mla_kv_.layers = nm;
  mla_kv_.kv_lora = kvl;
  mla_kv_.index_head_dim = ihd;
  kv_table_ = I(MB);
  {
    std::vector<int> table(MB);
    std::iota(table.begin(), table.end(), 0);
    cuda_check(cudaMemcpy(kv_table_, table.data(), MB * sizeof(int), cudaMemcpyHostToDevice),
              "kv table upload");
  }
  mla_kv_.table = kv_table_;

  tokens_dev_ = I(MB);
  pos_dev_ = I(MB);
  n_tokens_dev_ = I(MB);
  n_pools_dev_ = I(MB);

  streams_ = A(static_cast<std::size_t>(MB) * hc * H);
  residual_ = A(static_cast<std::size_t>(MB) * hc * H);
  collapsed_ = A(static_cast<std::size_t>(MB) * H);
  normed_ = A(static_cast<std::size_t>(MB) * H);
  sublayer_out_ = A(static_cast<std::size_t>(MB) * H);
  hmean_ = A(static_cast<std::size_t>(MB) * H);
  mix_ = F(static_cast<std::size_t>(MB) * cfg_.hc_mix());
  post_ = F(static_cast<std::size_t>(MB) * hc);
  comb_ = F(static_cast<std::size_t>(MB) * hc * hc);

  q_raw_ = A(static_cast<std::size_t>(MB) * qkv);
  k_raw_ = A(static_cast<std::size_t>(MB) * qkv);
  v_raw_ = A(static_cast<std::size_t>(MB) * qkv);
  q_conv_ = A(static_cast<std::size_t>(MB) * qkv);
  k_conv_ = A(static_cast<std::size_t>(MB) * qkv);
  v_conv_ = A(static_cast<std::size_t>(MB) * qkv);
  lr_a_ = A(static_cast<std::size_t>(MB) * hd);
  lr_b_ = A(static_cast<std::size_t>(MB) * qkv);
  gate_ = A(static_cast<std::size_t>(MB) * qkv);
  beta_raw_ = A(static_cast<std::size_t>(MB) * heads_kda);
  beta_ = A(static_cast<std::size_t>(MB) * heads_kda);
  kda_o_ = A(static_cast<std::size_t>(MB) * qkv);
  kda_on_ = A(static_cast<std::size_t>(MB) * qkv);

  q_resid_raw_ = A(static_cast<std::size_t>(MB) * cfg_.q_lora_rank);
  q_resid_ = A(static_cast<std::size_t>(MB) * cfg_.q_lora_rank);
  q_ = A(static_cast<std::size_t>(MB) * heads_mla * cfg_.qk_head_dim());
  ckv_ = A(static_cast<std::size_t>(MB) * kvl);
  latent_stage_ = A(static_cast<std::size_t>(MB) * kvl);
  v_out_ = A(static_cast<std::size_t>(MB) * heads_mla * cfg_.v_head_dim);
  q_abs_ = F(static_cast<std::size_t>(MB) * heads_mla * kvl);
  ctx_ = F(static_cast<std::size_t>(MB) * heads_mla * kvl);
  scores_ = F(static_cast<std::size_t>(MB) * heads_mla * sel_stride_);

  q_idx_ = A(static_cast<std::size_t>(MB) * ih * ihd);
  idx_k_raw_ = A(static_cast<std::size_t>(MB) * ihd);
  idx_k_stage_ = A(static_cast<std::size_t>(MB) * ihd);
  idx_g_stage_ = A(static_cast<std::size_t>(MB) * ihd);
  pool_keys_ = A(static_cast<std::size_t>(MB) * pool_stride_ * ihd);
  pool_scores_ = F(static_cast<std::size_t>(MB) * pool_stride_);
  head_w_ = F(static_cast<std::size_t>(MB) * ih);
  sel_pools_ = I(static_cast<std::size_t>(MB) * sel_pool_stride_);
  sel_tokens_ = I(static_cast<std::size_t>(MB) * sel_stride_);
  n_sel_ = I(MB);
  n_tok_ = I(MB);

  router_logits_ = F(static_cast<std::size_t>(MB) * cfg_.n_routed_experts);
  topk_w_ = F(static_cast<std::size_t>(MB) * cfg_.num_experts_per_tok);
  topk_idx_ = I(static_cast<std::size_t>(MB) * cfg_.num_experts_per_tok);
  exp_gate_ = A(cfg_.moe_intermediate_size);
  exp_up_ = A(cfg_.moe_intermediate_size);
  exp_h_ = A(cfg_.moe_intermediate_size);
  exp_out_ = A(H);
  mlp_gate_ = A(static_cast<std::size_t>(MB) * big_inter);
  mlp_up_ = A(static_cast<std::size_t>(MB) * big_inter);
  mlp_h_ = A(static_cast<std::size_t>(MB) * big_inter);
  mlp_out_ = A(static_cast<std::size_t>(MB) * H);
  acc_ = A(static_cast<std::size_t>(MB) * H);
  logits_ = F(static_cast<std::size_t>(MB) * cfg_.vocab_size);
  scratch_f_ = F(MB);
  argmax_i_ = I(MB);
  telemetry_scratch_ = F(MB);

  pos_.assign(MB, 0);
  layer_rms_.assign(cfg_.text_layers, 0.0f);
}

DecodeEngine::~DecodeEngine() {
  for (void* p : owned_) cudaFree(p);
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

void DecodeEngine::reset() {
  std::fill(pos_.begin(), pos_.end(), 0);
  const int H = cfg_.hidden_size;
  const int qkv = cfg_.kda_qkv_dim();
  const int taps = cfg_.conv_state_taps();
  const int MB = max_batch_;
  const std::size_t conv_bytes =
      static_cast<std::size_t>(kda_layers_) * MB * qkv * taps * sizeof(bf16);
  const std::size_t state_bytes = static_cast<std::size_t>(kda_layers_) * MB * cfg_.kda_heads *
                                  cfg_.kda_head_dim * cfg_.kda_head_dim * sizeof(float);
  cudaMemsetAsync(q_conv_state_, 0, conv_bytes, stream_);
  cudaMemsetAsync(k_conv_state_, 0, conv_bytes, stream_);
  cudaMemsetAsync(v_conv_state_, 0, conv_bytes, stream_);
  cudaMemsetAsync(kda_state_, 0, state_bytes, stream_);
  cudaMemsetAsync(streams_, 0, static_cast<std::size_t>(MB) * cfg_.hc_mult * H * sizeof(bf16),
                  stream_);
  cudaStreamSynchronize(stream_);
}

float DecodeEngine::sync_rms_slot0(const bf16* x, int n) {
  sumsq_bf16(scratch_f_, x, 1, n, stream_);
  float ss = 0.0f;
  cudaMemcpyAsync(&ss, scratch_f_, sizeof(float), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  return std::sqrt(ss / static_cast<float>(n));
}

void DecodeEngine::record_absmax(const std::string& name, const bf16* x, int batch, int n) {
  absmax_bf16(telemetry_scratch_, x, batch, n, stream_);
  std::vector<float> h(batch);
  cudaMemcpyAsync(h.data(), telemetry_scratch_, batch * sizeof(float), cudaMemcpyDeviceToHost,
                  stream_);
  cudaStreamSynchronize(stream_);
  float mx = 0.0f;
  for (const float v : h) mx = std::max(mx, v);
  auto it = telemetry_absmax_.find(name);
  if (it == telemetry_absmax_.end())
    telemetry_absmax_.emplace(name, mx);
  else
    it->second = std::max(it->second, mx);
}

void DecodeEngine::run_kda(int layer, int slot, int batch) {
  const KdaW& k = w_.layer(layer).kda;
  const int H = cfg_.hidden_size;
  const int qkv = cfg_.kda_qkv_dim();
  const int hd = cfg_.kda_head_dim;
  const int taps = cfg_.conv_state_taps();
  const int heads = cfg_.kda_heads;
  const int MB = max_batch_;

  gemm_bf16(q_raw_, k.qkv, normed_, batch, qkv, H, stream_);
  gemm_bf16(k_raw_, k.qkv + static_cast<std::size_t>(qkv) * H, normed_, batch, qkv, H, stream_);
  gemm_bf16(v_raw_, k.qkv + static_cast<std::size_t>(2) * qkv * H, normed_, batch, qkv, H, stream_);

  bf16* qstate = q_conv_state_ + static_cast<std::size_t>(slot) * MB * qkv * taps;
  bf16* kstate = k_conv_state_ + static_cast<std::size_t>(slot) * MB * qkv * taps;
  bf16* vstate = v_conv_state_ + static_cast<std::size_t>(slot) * MB * qkv * taps;
  kda_conv_update(q_conv_, qstate, q_raw_, k.conv, batch, qkv, cfg_.kda_conv_kernel, stream_);
  kda_conv_update(k_conv_, kstate, k_raw_, k.conv + static_cast<std::size_t>(qkv) * cfg_.kda_conv_kernel,
                  batch, qkv, cfg_.kda_conv_kernel, stream_);
  kda_conv_update(v_conv_, vstate, v_raw_,
                  k.conv + static_cast<std::size_t>(2) * qkv * cfg_.kda_conv_kernel, batch, qkv,
                  cfg_.kda_conv_kernel, stream_);
  kda_norm_qk(q_conv_, k_conv_, batch, heads, hd, /*row_stride=*/qkv, stream_);

  gemm_bf16(lr_a_, k.f_a, normed_, batch, hd, H, stream_);
  gemm_bf16(lr_b_, k.f_b, lr_a_, batch, qkv, hd, stream_);
  kda_forget_gate(gate_, lr_b_, k.dt_bias, k.a_log, batch, heads, hd, cfg_.kda_gate_lower_bound,
                  stream_);

  gemm_bf16(beta_raw_, k.b_proj, normed_, batch, heads, H, stream_);
  kda_sigmoid(beta_, beta_raw_, batch * heads, stream_);

  float* state = kda_state_ + static_cast<std::size_t>(slot) * MB * heads * hd * hd;
  kda_recurrent_step(state, kda_o_, q_conv_, k_conv_, v_conv_, gate_, beta_, batch, heads, hd,
                     /*row_stride=*/qkv, stream_);

  gemm_bf16(lr_a_, k.g_a, normed_, batch, hd, H, stream_);
  gemm_bf16(lr_b_, k.g_b, lr_a_, batch, qkv, hd, stream_);
  kda_gated_norm(kda_on_, kda_o_, lr_b_, k.o_norm, batch, heads, hd, cfg_.rms_norm_eps, stream_);
  gemm_bf16(sublayer_out_, k.o_proj, kda_on_, batch, H, qkv, stream_);

  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".attn.normed", normed_, batch, H);
}

void DecodeEngine::run_mla(int layer, int slot, int batch, const int* n_tokens_dev,
                           int n_pools_max) {
  const MlaW& m = w_.layer(layer).mla;
  const int H = cfg_.hidden_size;
  const int heads = cfg_.mla_heads;
  const int kvl = cfg_.kv_lora_rank;
  const int ihd = cfg_.index_head_dim;
  const int ih = cfg_.index_n_heads;
  const int kpool = cfg_.index_kpool;

  gemm_bf16(q_resid_raw_, m.q_a, normed_, batch, cfg_.q_lora_rank, H, stream_);
  rmsnorm(q_resid_, q_resid_raw_, m.q_a_norm, batch, cfg_.q_lora_rank, cfg_.rms_norm_eps, stream_);
  gemm_bf16(q_, m.q_b, q_resid_, batch, heads * cfg_.qk_head_dim(), cfg_.q_lora_rank, stream_);

  gemm_bf16(ckv_, m.kv_a, normed_, batch, kvl, H, stream_);
  rmsnorm(latent_stage_, ckv_, m.kv_a_norm, batch, kvl, cfg_.rms_norm_eps, stream_);

  gemm_bf16(q_idx_, m.idx_wq_b, q_resid_, batch, ih * ihd, cfg_.q_lora_rank, stream_);
  gemm_bf16(idx_k_raw_, m.idx_wk, normed_, batch, ihd, H, stream_);
  layernorm(idx_k_stage_, idx_k_raw_, m.idx_k_norm_w, m.idx_k_norm_b, batch, ihd, 1e-6f, stream_);
  gemm_bf16(idx_g_stage_, m.idx_gate, normed_, batch, ihd, H, stream_);
  // The reference scales these by index_n_heads^-0.5 before the weighted sum.
  // A positive constant multiplies every pool score equally, so it cannot
  // move the top-k, and the scores are used for nothing else.
  gemm_bf16_f32(head_w_, m.idx_weights, normed_, batch, ih, H, stream_);

  kv_write_latent(mla_kv_, latent_stage_, pos_dev_, batch, slot, stream_);
  kv_write_index(mla_kv_, idx_k_stage_, idx_g_stage_, pos_dev_, batch, slot, stream_);

  if (n_pools_max > 0) {
    indexer_pool_keys(pool_keys_, mla_kv_, m.idx_ape, n_pools_dev_, n_pools_max, pool_stride_,
                      batch, slot, kpool, ihd, stream_);
    indexer_scores(pool_scores_, q_idx_, pool_keys_, head_w_, n_pools_dev_, n_pools_max,
                   pool_stride_, batch, ih, ihd, stream_);
    indexer_select(sel_pools_, n_sel_, pool_scores_, n_pools_dev_, n_pools_max, pool_stride_,
                   sel_pool_stride_, batch, cfg_.index_select_pools(), stream_);
  } else {
    cudaMemsetAsync(n_sel_, 0, static_cast<std::size_t>(batch) * sizeof(int), stream_);
  }
  indexer_expand(sel_tokens_, n_tok_, sel_pools_, n_sel_, n_tokens_dev, sel_pool_stride_,
                sel_stride_, batch, kpool, stream_);

  const float scaling = 1.0f / std::sqrt(static_cast<float>(cfg_.qk_head_dim()));
  mla_absorb_q(q_abs_, m.kv_b, q_, batch, heads, cfg_.qk_nope_head_dim, cfg_.v_head_dim, kvl,
              stream_);
  mla_scores(scores_, q_abs_, mla_kv_, sel_tokens_, n_tok_, sel_stride_, sel_stride_, batch, slot,
            heads, kvl, scaling, stream_);
  mla_softmax(scores_, n_tok_, sel_stride_, batch, heads, stream_);
  mla_context(ctx_, scores_, mla_kv_, sel_tokens_, n_tok_, sel_stride_, sel_stride_, batch, slot,
             heads, kvl, stream_);
  mla_expand_v(v_out_, m.kv_b, ctx_, batch, heads, cfg_.qk_nope_head_dim, cfg_.v_head_dim, kvl,
              stream_);
  gemm_bf16(sublayer_out_, m.o_proj, v_out_, batch, H, heads * cfg_.v_head_dim, stream_);

  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".attn.normed", normed_, batch, H);
}

void DecodeEngine::run_dense_mlp(int layer, int batch) {
  const DenseMlpW& d = w_.layer(layer).dense;
  const int H = cfg_.hidden_size;
  const int I = cfg_.intermediate_size;
  gemm_bf16(mlp_gate_, d.gate, normed_, batch, I, H, stream_);
  gemm_bf16(mlp_up_, d.up, normed_, batch, I, H, stream_);
  swiglu_clamped(mlp_h_, mlp_gate_, mlp_up_, batch * I, cfg_.swiglu_limit, stream_);
  gemm_bf16(sublayer_out_, d.down, mlp_h_, batch, H, I, stream_);
  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".ffn.normed", normed_, batch, H);
}

void DecodeEngine::run_moe(int layer, int batch) {
  const MoeW& mo = w_.layer(layer).moe;
  const int H = cfg_.hidden_size;
  const int MI = cfg_.moe_intermediate_size;
  const int SI = MI * cfg_.n_shared_experts;
  const int K = cfg_.num_experts_per_tok;

  gemm_bf16_f32(router_logits_, mo.router, normed_, batch, cfg_.n_routed_experts, H, stream_);
  moe_router(topk_idx_, topk_w_, router_logits_, mo.router_bias, batch, cfg_.n_routed_experts, K,
            cfg_.norm_topk_prob, cfg_.routed_scaling_factor, stream_);

  std::vector<int> idx(static_cast<std::size_t>(batch) * K);
  std::vector<float> wts(static_cast<std::size_t>(batch) * K);
  cudaMemcpyAsync(idx.data(), topk_idx_, idx.size() * sizeof(int), cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(wts.data(), topk_w_, wts.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  double ent = 0.0;
  double norm = 0.0;
  for (int t = 0; t < K; ++t) norm += wts[t];  // slot 0 only, see model.h::router_entropy
  for (int t = 0; t < K; ++t) {
    const double p = wts[t] / (norm > 0.0 ? norm : 1.0);
    if (p > 0.0) ent -= p * std::log(p);
  }
  router_entropy_ += ent;

  cudaMemsetAsync(acc_, 0, static_cast<std::size_t>(batch) * H * sizeof(bf16), stream_);
  for (int m = 0; m < batch; ++m) {
    const bf16* x_row = normed_ + static_cast<std::size_t>(m) * H;
    for (int t = 0; t < K; ++t) {
      const int expert_id = idx[static_cast<std::size_t>(m) * K + t];
      const auto t_stream = Clock::now();
      const ExpertDev& e = w_.expert(layer, expert_id, stream_);
      stages_.expert_stream +=
          std::chrono::duration<double, std::milli>(Clock::now() - t_stream).count();
      gemv_nvfp4(exp_gate_, e.gate_packed, e.gate_scale, e.gate_global, x_row, MI, H, stream_);
      gemv_nvfp4(exp_up_, e.up_packed, e.up_scale, e.up_global, x_row, MI, H, stream_);
      swiglu_clamped(exp_h_, exp_gate_, exp_up_, MI, cfg_.swiglu_limit, stream_);
      gemv_nvfp4(exp_out_, e.down_packed, e.down_scale, e.down_global, exp_h_, H, MI, stream_);
      axpy_bf16(acc_ + static_cast<std::size_t>(m) * H, exp_out_, topk_w_ + static_cast<std::size_t>(m) * K,
               t, K, /*batch=*/1, H, stream_);
    }
  }

  gemm_bf16(mlp_gate_, mo.shared.gate, normed_, batch, SI, H, stream_);
  gemm_bf16(mlp_up_, mo.shared.up, normed_, batch, SI, H, stream_);
  swiglu_clamped(mlp_h_, mlp_gate_, mlp_up_, batch * SI, cfg_.swiglu_limit, stream_);
  gemm_bf16(mlp_out_, mo.shared.down, mlp_h_, batch, H, SI, stream_);
  add_bf16(acc_, mlp_out_, batch * H, stream_);
  cudaMemcpyAsync(sublayer_out_, acc_, static_cast<std::size_t>(batch) * H * sizeof(bf16),
                  cudaMemcpyDeviceToDevice, stream_);

  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".ffn.normed", normed_, batch, H);
}

void DecodeEngine::step(const std::vector<int>& tokens, std::vector<int>& out_tokens,
                       bool collect_stages) {
  const int batch = static_cast<int>(tokens.size());
  if (batch <= 0 || batch > max_batch_) fail("batch out of [1, max_batch] range");
  for (const int p : pos_)
    if (p > max_tokens_) fail("position exceeds the compiled max_tokens");

  const int H = cfg_.hidden_size;
  const int hc = cfg_.hc_mult;
  stages_ = StageMs{};
  router_entropy_ = 0.0;

  std::vector<int> pos_h(batch), ntok_h(batch);
  int n_pools_max = 0;
  for (int m = 0; m < batch; ++m) {
    pos_h[m] = pos_[m];
    ntok_h[m] = pos_[m] + 1;
    n_pools_max = std::max(n_pools_max, ntok_h[m] / cfg_.index_kpool);
  }
  cudaMemcpyAsync(tokens_dev_, tokens.data(), batch * sizeof(int), cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(pos_dev_, pos_h.data(), batch * sizeof(int), cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(n_tokens_dev_, ntok_h.data(), batch * sizeof(int), cudaMemcpyHostToDevice,
                  stream_);

  {
    StageTimer t(stream_, &stages_.embed, collect_stages);
    embed_streams(streams_, w_.embed(), tokens_dev_, batch, hc, H, stream_);
  }

  for (int l = 0; l < cfg_.text_layers; ++l) {
    const LayerW& lw = w_.layer(l);

    // ---- attention site ----
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages);
      cudaMemcpyAsync(residual_, streams_, static_cast<std::size_t>(batch) * hc * H * sizeof(bf16),
                      cudaMemcpyDeviceToDevice, stream_);
      hc_mix_gemv(mix_, lw.attn_hc.fn, streams_, batch, cfg_.hc_mix(), hc, H, cfg_.rms_norm_eps,
                 stream_);
      hc_split(post_, comb_, collapsed_, mix_, lw.attn_hc.base, lw.attn_hc.scale, streams_, batch,
              hc, H, cfg_.hc_eps, cfg_.hc_sinkhorn_iters, stream_);
    }
    {
      StageTimer t(stream_, &stages_.norms, collect_stages);
      rmsnorm(normed_, collapsed_, lw.input_norm, batch, H, cfg_.rms_norm_eps, stream_);
    }
    if (cfg_.layers[l].attn == fuel::AttnKind::kKda) {
      StageTimer t(stream_, &stages_.kda, collect_stages);
      run_kda(l, kda_slot_[l], batch);
    } else {
      // The indexer runs inside the MLA layer; its cost is folded in here
      // rather than isolated by a sync, since a batched step has no
      // per-layer host readback of the selection count anymore.
      StageTimer t(stream_, &stages_.mla, collect_stages);
      run_mla(l, mla_slot_[l], batch, n_tokens_dev_, n_pools_max);
    }
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages);
      hc_combine(streams_, post_, sublayer_out_, comb_, residual_, batch, hc, H, stream_);
    }

    // ---- feed-forward site ----
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages);
      cudaMemcpyAsync(residual_, streams_, static_cast<std::size_t>(batch) * hc * H * sizeof(bf16),
                      cudaMemcpyDeviceToDevice, stream_);
      hc_mix_gemv(mix_, lw.ffn_hc.fn, streams_, batch, cfg_.hc_mix(), hc, H, cfg_.rms_norm_eps,
                 stream_);
      hc_split(post_, comb_, collapsed_, mix_, lw.ffn_hc.base, lw.ffn_hc.scale, streams_, batch, hc,
              H, cfg_.hc_eps, cfg_.hc_sinkhorn_iters, stream_);
    }
    {
      StageTimer t(stream_, &stages_.norms, collect_stages);
      rmsnorm(normed_, collapsed_, lw.post_attn_norm, batch, H, cfg_.rms_norm_eps, stream_);
    }
    if (cfg_.layers[l].mlp == fuel::MlpKind::kDense) {
      StageTimer t(stream_, &stages_.dense_mlp, collect_stages);
      run_dense_mlp(l, batch);
    } else {
      StageTimer t(stream_, &stages_.moe_experts, collect_stages);
      run_moe(l, batch);
    }
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages);
      hc_combine(streams_, post_, sublayer_out_, comb_, residual_, batch, hc, H, stream_);
    }

    if (collect_stages) {
      hc_head_mean(hmean_, streams_, batch, hc, H, stream_);
      layer_rms_[l] = sync_rms_slot0(hmean_, H);
    }
  }

  std::vector<int> next(batch, 0);
  {
    StageTimer t(stream_, &stages_.lm_head, collect_stages);
    hc_head_mean(hmean_, streams_, batch, hc, H, stream_);
    rmsnorm(normed_, hmean_, w_.final_norm(), batch, H, cfg_.rms_norm_eps, stream_);
    gemm_bf16_f32(logits_, w_.lm_head(), normed_, batch, cfg_.vocab_size, H, stream_);
    argmax_f32(argmax_i_, scratch_f_, logits_, batch, cfg_.vocab_size, stream_);
    cudaMemcpyAsync(next.data(), argmax_i_, batch * sizeof(int), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }

  int moe_layers = 0;
  for (const fuel::LayerSpec& s : cfg_.layers)
    if (s.mlp == fuel::MlpKind::kSparse) ++moe_layers;
  router_entropy_ /= (moe_layers > 0 ? moe_layers : 1);

  for (int m = 0; m < batch; ++m) ++pos_[m];
  cuda_check(cudaGetLastError(), "decode step");
  out_tokens = std::move(next);
}

}  // namespace rocket::engine

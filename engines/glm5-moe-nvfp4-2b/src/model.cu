#include "model.h"

#include <chrono>
#include <cmath>
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
                           std::size_t expert_cache_bytes, int max_tokens)
    : cfg_(cfg), w_(cfg, snapshot_dir, expert_cache_bytes), max_tokens_(max_tokens) {
  cuda_check(cudaStreamCreate(&stream_), "stream");

  const int H = cfg_.hidden_size;
  const int hc = cfg_.hc_mult;
  const int qkv = cfg_.kda_qkv_dim();
  const int hd = cfg_.kda_head_dim;
  const int heads = cfg_.mla_heads;
  const int kvl = cfg_.kv_lora_rank;
  const int ih = cfg_.index_n_heads, ihd = cfg_.index_head_dim;
  const int max_sel_tokens = cfg_.index_topk + cfg_.index_kpool - 1;
  const int max_pools = max_tokens / cfg_.index_kpool + 1;
  const int big_inter = std::max(cfg_.intermediate_size,
                                 cfg_.moe_intermediate_size * cfg_.n_shared_experts);

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

  kda_state_ = F(static_cast<std::size_t>(nk) * cfg_.kda_heads * hd * hd);
  kda_conv_ = A(static_cast<std::size_t>(nk) * 3 * qkv * cfg_.conv_state_taps());
  mla_latent_ = A(static_cast<std::size_t>(nm) * max_tokens * kvl);
  idx_key_ = A(static_cast<std::size_t>(nm) * max_tokens * ihd);
  idx_gate_ = A(static_cast<std::size_t>(nm) * max_tokens * ihd);

  streams_ = A(static_cast<std::size_t>(hc) * H);
  residual_ = A(static_cast<std::size_t>(hc) * H);
  collapsed_ = A(H);
  normed_ = A(H);
  sublayer_out_ = A(H);
  hmean_ = A(H);
  mix_ = F(cfg_.hc_mix());
  post_ = F(hc);
  comb_ = F(static_cast<std::size_t>(hc) * hc);

  qkv_ = A(3 * static_cast<std::size_t>(qkv));
  qkv_conv_ = A(3 * static_cast<std::size_t>(qkv));
  lr_a_ = A(hd);
  lr_b_ = A(qkv);
  gate_ = A(qkv);
  beta_raw_ = A(cfg_.kda_heads);
  beta_ = A(cfg_.kda_heads);
  kda_o_ = A(qkv);
  kda_on_ = A(qkv);

  q_resid_raw_ = A(cfg_.q_lora_rank);
  q_resid_ = A(cfg_.q_lora_rank);
  q_ = A(static_cast<std::size_t>(heads) * cfg_.qk_head_dim());
  ckv_ = A(kvl);
  v_out_ = A(static_cast<std::size_t>(heads) * cfg_.v_head_dim);
  q_abs_ = F(static_cast<std::size_t>(heads) * kvl);
  ctx_ = F(static_cast<std::size_t>(heads) * kvl);
  scores_ = F(static_cast<std::size_t>(heads) * max_sel_tokens);

  q_idx_ = A(static_cast<std::size_t>(ih) * ihd);
  idx_k_raw_ = A(ihd);
  pool_keys_ = A(static_cast<std::size_t>(max_pools) * ihd);
  pool_scores_ = F(max_pools);
  head_w_ = F(ih);
  sel_pools_ = I(cfg_.index_select_pools());
  sel_tokens_ = I(max_sel_tokens);
  n_sel_ = I(1);
  n_tok_ = I(1);

  router_logits_ = F(cfg_.n_routed_experts);
  topk_w_ = F(cfg_.num_experts_per_tok);
  topk_idx_ = I(cfg_.num_experts_per_tok);
  mlp_gate_ = A(big_inter);
  mlp_up_ = A(big_inter);
  mlp_h_ = A(big_inter);
  mlp_out_ = A(H);
  acc_ = A(H);
  logits_ = F(cfg_.vocab_size);
  scratch_f_ = F(4);
  argmax_i_ = I(1);

  layer_rms_.assign(cfg_.text_layers, 0.0f);
}

DecodeEngine::~DecodeEngine() {
  for (void* p : owned_) cudaFree(p);
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

void DecodeEngine::reset() {
  pos_ = 0;
  const int H = cfg_.hidden_size;
  const int qkv = cfg_.kda_qkv_dim();
  int nk = 0, nm = 0;
  for (int l = 0; l < cfg_.text_layers; ++l)
    (cfg_.layers[l].attn == fuel::AttnKind::kKda ? nk : nm)++;
  cudaMemsetAsync(kda_state_, 0,
                  static_cast<std::size_t>(nk) * cfg_.kda_heads * cfg_.kda_head_dim *
                      cfg_.kda_head_dim * sizeof(float),
                  stream_);
  cudaMemsetAsync(kda_conv_, 0,
                  static_cast<std::size_t>(nk) * 3 * qkv * cfg_.conv_state_taps() * sizeof(bf16),
                  stream_);
  cudaMemsetAsync(streams_, 0, static_cast<std::size_t>(cfg_.hc_mult) * H * sizeof(bf16), stream_);
  cudaStreamSynchronize(stream_);
}

float DecodeEngine::sync_rms(const bf16* x, int n) {
  sumsq_bf16(scratch_f_, x, n, stream_);
  float ss = 0.0f;
  cudaMemcpyAsync(&ss, scratch_f_, sizeof(float), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  return std::sqrt(ss / static_cast<float>(n));
}

void DecodeEngine::run_kda(int layer, int slot) {
  const KdaW& k = w_.layer(layer).kda;
  const int H = cfg_.hidden_size;
  const int qkv = cfg_.kda_qkv_dim();
  const int hd = cfg_.kda_head_dim;
  const int heads = cfg_.kda_heads;

  gemv_bf16(qkv_, k.qkv, normed_, 3 * qkv, H, stream_);
  kda_conv_update(qkv_conv_, kda_conv_ + static_cast<std::size_t>(slot) * 3 * qkv * cfg_.conv_state_taps(),
                  qkv_, k.conv, 3 * qkv, cfg_.kda_conv_kernel, stream_);
  bf16* q = qkv_conv_;
  bf16* kk = qkv_conv_ + qkv;
  bf16* v = qkv_conv_ + 2 * qkv;
  kda_norm_qk(q, kk, heads, hd, stream_);

  gemv_bf16(lr_a_, k.f_a, normed_, hd, H, stream_);
  gemv_bf16(lr_b_, k.f_b, lr_a_, qkv, hd, stream_);
  kda_forget_gate(gate_, lr_b_, k.dt_bias, k.a_log, heads, hd, cfg_.kda_gate_lower_bound, stream_);

  gemv_bf16(beta_raw_, k.b_proj, normed_, heads, H, stream_);
  kda_sigmoid(beta_, beta_raw_, heads, stream_);

  kda_recurrent_step(kda_state_ + static_cast<std::size_t>(slot) * heads * hd * hd, kda_o_, q, kk, v,
                     gate_, beta_, heads, hd, stream_);

  gemv_bf16(lr_a_, k.g_a, normed_, hd, H, stream_);
  gemv_bf16(lr_b_, k.g_b, lr_a_, qkv, hd, stream_);
  kda_gated_norm(kda_on_, kda_o_, lr_b_, k.o_norm, heads, hd, cfg_.rms_norm_eps, stream_);
  gemv_bf16(sublayer_out_, k.o_proj, kda_on_, H, qkv, stream_);
}

void DecodeEngine::run_mla(int layer, int slot) {
  const MlaW& m = w_.layer(layer).mla;
  const int H = cfg_.hidden_size;
  const int heads = cfg_.mla_heads;
  const int kvl = cfg_.kv_lora_rank;
  const int ihd = cfg_.index_head_dim;
  const int ih = cfg_.index_n_heads;
  const int kpool = cfg_.index_kpool;
  const int n_tokens = pos_ + 1;

  gemv_bf16(q_resid_raw_, m.q_a, normed_, cfg_.q_lora_rank, H, stream_);
  rmsnorm(q_resid_, q_resid_raw_, m.q_a_norm, cfg_.q_lora_rank, cfg_.rms_norm_eps, stream_);
  gemv_bf16(q_, m.q_b, q_resid_, heads * cfg_.qk_head_dim(), cfg_.q_lora_rank, stream_);

  bf16* latent_base = mla_latent_ + static_cast<std::size_t>(slot) * max_tokens_ * kvl;
  gemv_bf16(ckv_, m.kv_a, normed_, kvl, H, stream_);
  rmsnorm(latent_base + static_cast<std::size_t>(pos_) * kvl, ckv_, m.kv_a_norm, kvl,
          cfg_.rms_norm_eps, stream_);

  // --- indexer ------------------------------------------------------------
  bf16* key_base = idx_key_ + static_cast<std::size_t>(slot) * max_tokens_ * ihd;
  bf16* gate_base = idx_gate_ + static_cast<std::size_t>(slot) * max_tokens_ * ihd;
  gemv_bf16(q_idx_, m.idx_wq_b, q_resid_, ih * ihd, cfg_.q_lora_rank, stream_);
  gemv_bf16(idx_k_raw_, m.idx_wk, normed_, ihd, H, stream_);
  layernorm(key_base + static_cast<std::size_t>(pos_) * ihd, idx_k_raw_, m.idx_k_norm_w,
            m.idx_k_norm_b, ihd, 1e-6f, stream_);
  gemv_bf16(gate_base + static_cast<std::size_t>(pos_) * ihd, m.idx_gate, normed_, ihd, H, stream_);
  // The reference scales these by index_n_heads^-0.5 before the weighted sum.
  // A positive constant multiplies every pool score equally, so it cannot move
  // the top-k, and the scores are used for nothing else.
  gemv_bf16_f32(head_w_, m.idx_weights, normed_, ih, H, stream_);

  const int n_pools = n_tokens / kpool;
  if (n_pools > 0) {
    indexer_pool_keys(pool_keys_, key_base, gate_base, m.idx_ape, n_pools, kpool, ihd, stream_);
    indexer_scores(pool_scores_, q_idx_, pool_keys_, head_w_, n_pools, ih, ihd, stream_);
    indexer_select(sel_pools_, n_sel_, pool_scores_, n_pools, cfg_.index_select_pools(), stream_);
  } else {
    cudaMemsetAsync(n_sel_, 0, sizeof(int), stream_);
  }
  indexer_expand(sel_tokens_, n_tok_, sel_pools_, n_sel_, kpool, n_tokens, stream_);

  int n_sel_tokens = 0;
  cudaMemcpyAsync(&n_sel_tokens, n_tok_, sizeof(int), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  if (n_sel_tokens <= 0) fail("indexer selected no tokens");

  // --- absorbed attention over the selected set ---------------------------
  const float scaling = 1.0f / std::sqrt(static_cast<float>(cfg_.qk_head_dim()));
  mla_absorb_q(q_abs_, m.kv_b, q_, heads, cfg_.qk_nope_head_dim, cfg_.v_head_dim, kvl, stream_);
  mla_scores(scores_, q_abs_, latent_base, sel_tokens_, n_sel_tokens, heads, kvl, scaling, stream_);
  mla_softmax(scores_, heads, n_sel_tokens, stream_);
  mla_context(ctx_, scores_, latent_base, sel_tokens_, n_sel_tokens, heads, kvl, stream_);
  mla_expand_v(v_out_, m.kv_b, ctx_, heads, cfg_.qk_nope_head_dim, cfg_.v_head_dim, kvl, stream_);
  gemv_bf16(sublayer_out_, m.o_proj, v_out_, H, heads * cfg_.v_head_dim, stream_);
}

void DecodeEngine::run_dense_mlp(int layer) {
  const DenseMlpW& d = w_.layer(layer).dense;
  const int H = cfg_.hidden_size;
  const int I = cfg_.intermediate_size;
  gemv_bf16(mlp_gate_, d.gate, normed_, I, H, stream_);
  gemv_bf16(mlp_up_, d.up, normed_, I, H, stream_);
  swiglu_clamped(mlp_h_, mlp_gate_, mlp_up_, I, cfg_.swiglu_limit, stream_);
  gemv_bf16(sublayer_out_, d.down, mlp_h_, H, I, stream_);
}

void DecodeEngine::run_moe(int layer) {
  const MoeW& mo = w_.layer(layer).moe;
  const int H = cfg_.hidden_size;
  const int MI = cfg_.moe_intermediate_size;
  const int SI = MI * cfg_.n_shared_experts;
  const int K = cfg_.num_experts_per_tok;

  gemv_bf16_f32(router_logits_, mo.router, normed_, cfg_.n_routed_experts, H, stream_);
  moe_router(topk_idx_, topk_w_, router_logits_, mo.router_bias, cfg_.n_routed_experts, K,
             cfg_.norm_topk_prob, cfg_.routed_scaling_factor, stream_);

  std::vector<int> idx(K);
  std::vector<float> wts(K);
  cudaMemcpyAsync(idx.data(), topk_idx_, K * sizeof(int), cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(wts.data(), topk_w_, K * sizeof(float), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  double ent = 0.0;
  double norm = 0.0;
  for (const float v : wts) norm += v;
  for (const float v : wts) {
    const double p = v / (norm > 0.0 ? norm : 1.0);
    if (p > 0.0) ent -= p * std::log(p);
  }
  router_entropy_ += ent;

  cudaMemsetAsync(acc_, 0, static_cast<std::size_t>(H) * sizeof(bf16), stream_);
  for (int t = 0; t < K; ++t) {
    const auto t_stream = Clock::now();
    const ExpertDev& e = w_.expert(layer, idx[t], stream_);
    stages_.expert_stream +=
        std::chrono::duration<double, std::milli>(Clock::now() - t_stream).count();
    gemv_nvfp4(mlp_gate_, e.gate_packed, e.gate_scale, e.gate_global, normed_, MI, H, stream_);
    gemv_nvfp4(mlp_up_, e.up_packed, e.up_scale, e.up_global, normed_, MI, H, stream_);
    swiglu_clamped(mlp_h_, mlp_gate_, mlp_up_, MI, cfg_.swiglu_limit, stream_);
    gemv_nvfp4(mlp_out_, e.down_packed, e.down_scale, e.down_global, mlp_h_, H, MI, stream_);
    axpy_bf16(acc_, mlp_out_, topk_w_, t, H, stream_);
  }

  gemv_bf16(mlp_gate_, mo.shared.gate, normed_, SI, H, stream_);
  gemv_bf16(mlp_up_, mo.shared.up, normed_, SI, H, stream_);
  swiglu_clamped(mlp_h_, mlp_gate_, mlp_up_, SI, cfg_.swiglu_limit, stream_);
  gemv_bf16(mlp_out_, mo.shared.down, mlp_h_, H, SI, stream_);
  add_bf16(acc_, mlp_out_, H, stream_);
  cudaMemcpyAsync(sublayer_out_, acc_, static_cast<std::size_t>(H) * sizeof(bf16),
                  cudaMemcpyDeviceToDevice, stream_);
}

int DecodeEngine::step(int token, bool collect_stages) {
  if (pos_ >= max_tokens_) fail("position exceeds the compiled max_tokens");
  const int H = cfg_.hidden_size;
  const int hc = cfg_.hc_mult;
  stages_ = StageMs{};
  router_entropy_ = 0.0;

  {
    StageTimer t(stream_, &stages_.embed, collect_stages);
    // Every hyper-connection stream starts as a copy of the embedding.
    for (int i = 0; i < hc; ++i) {
      cudaMemcpyAsync(streams_ + static_cast<std::size_t>(i) * H,
                      w_.embed() + static_cast<std::size_t>(token) * H,
                      static_cast<std::size_t>(H) * sizeof(bf16), cudaMemcpyDeviceToDevice, stream_);
    }
  }

  for (int l = 0; l < cfg_.text_layers; ++l) {
    const LayerW& lw = w_.layer(l);

    // ---- attention site ----
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages);
      cudaMemcpyAsync(residual_, streams_, static_cast<std::size_t>(hc) * H * sizeof(bf16),
                      cudaMemcpyDeviceToDevice, stream_);
      hc_mix_gemv(mix_, lw.attn_hc.fn, streams_, cfg_.hc_mix(), hc, H, cfg_.rms_norm_eps, stream_);
      hc_split(post_, comb_, collapsed_, mix_, lw.attn_hc.base, lw.attn_hc.scale, streams_, hc, H,
               cfg_.hc_eps, cfg_.hc_sinkhorn_iters, stream_);
    }
    {
      StageTimer t(stream_, &stages_.norms, collect_stages);
      rmsnorm(normed_, collapsed_, lw.input_norm, H, cfg_.rms_norm_eps, stream_);
    }
    if (cfg_.layers[l].attn == fuel::AttnKind::kKda) {
      StageTimer t(stream_, &stages_.kda, collect_stages);
      run_kda(l, kda_slot_[l]);
    } else {
      // The indexer runs inside the MLA layer; its cost is separated by the
      // sync the top-k read-back already forces.
      StageTimer t(stream_, &stages_.mla, collect_stages);
      run_mla(l, mla_slot_[l]);
    }
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages);
      hc_combine(streams_, post_, sublayer_out_, comb_, residual_, hc, H, stream_);
    }

    // ---- feed-forward site ----
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages);
      cudaMemcpyAsync(residual_, streams_, static_cast<std::size_t>(hc) * H * sizeof(bf16),
                      cudaMemcpyDeviceToDevice, stream_);
      hc_mix_gemv(mix_, lw.ffn_hc.fn, streams_, cfg_.hc_mix(), hc, H, cfg_.rms_norm_eps, stream_);
      hc_split(post_, comb_, collapsed_, mix_, lw.ffn_hc.base, lw.ffn_hc.scale, streams_, hc, H,
               cfg_.hc_eps, cfg_.hc_sinkhorn_iters, stream_);
    }
    {
      StageTimer t(stream_, &stages_.norms, collect_stages);
      rmsnorm(normed_, collapsed_, lw.post_attn_norm, H, cfg_.rms_norm_eps, stream_);
    }
    if (cfg_.layers[l].mlp == fuel::MlpKind::kDense) {
      StageTimer t(stream_, &stages_.dense_mlp, collect_stages);
      run_dense_mlp(l);
    } else {
      StageTimer t(stream_, &stages_.moe_experts, collect_stages);
      run_moe(l);
    }
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages);
      hc_combine(streams_, post_, sublayer_out_, comb_, residual_, hc, H, stream_);
    }

    if (collect_stages) {
      hc_head_mean(hmean_, streams_, hc, H, stream_);
      layer_rms_[l] = sync_rms(hmean_, H);
    }
  }

  int next = 0;
  {
    StageTimer t(stream_, &stages_.lm_head, collect_stages);
    hc_head_mean(hmean_, streams_, hc, H, stream_);
    rmsnorm(normed_, hmean_, w_.final_norm(), H, cfg_.rms_norm_eps, stream_);
    gemv_bf16_f32(logits_, w_.lm_head(), normed_, cfg_.vocab_size, H, stream_);
    argmax_f32(argmax_i_, scratch_f_, logits_, cfg_.vocab_size, stream_);
    cudaMemcpyAsync(&next, argmax_i_, sizeof(int), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }

  int moe_layers = 0;
  for (const fuel::LayerSpec& s : cfg_.layers)
    if (s.mlp == fuel::MlpKind::kSparse) ++moe_layers;
  router_entropy_ /= (moe_layers > 0 ? moe_layers : 1);

  ++pos_;
  cuda_check(cudaGetLastError(), "decode step");
  return next;
}

}  // namespace rocket::engine

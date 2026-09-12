#include "model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

#include "kernels.h"
#include "fabric/expert_parallel.h"
#include "moe_grouped.h"
#include "exl3_moe_bridge.h"
#include "exl3_bridge.h"

namespace rocket::engine {
namespace {

inline std::uint16_t fp16_bits(float v) {
  __half h = __float2half(v);
  std::uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("rocket::engine::model: " + what);
}
void cuda_check(cudaError_t e, const std::string& what) {
  if (e != cudaSuccess) fail(what + ": " + cudaGetErrorString(e));
}

using Clock = std::chrono::steady_clock;

// Measured crossover between the GEMV loop and the grouped GEMM
// (blog/posts/runtime/2026-09-07-*, tests/test_moe_grouped.cu M-sweep on this
// booster): the grouped path wins at every M tested, M=1 included, because
// the GEMV kernel's scalar BF16 loads were already the bottleneck the
// per-token baseline identified (blog/posts/runtime/2026-09-07-one-booster-
// decodes-end-to-end/), not the batching this stage adds. Grouping 8 experts
// of a single stream into 8 one-row CUTLASS groups still beats 8 GEMV
// kernels. kGroupedMoeMinBatch is 1 rather than "always" so the GEMV path
// stays reachable through MoePath::kForceGemv (tests, debugging) and as the
// runtime fallback when the grouped launcher reports it cannot implement a
// shape (run_moe_grouped's return value).
constexpr int kGroupedMoeMinBatch = 1;

// Event-based stage timer: records two events per stage and accumulates the
// GPU elapsed time at the end of the step (finish_stage_events). Unlike the
// old sync-and-wallclock version this never stalls the pipeline, so the
// per-stage numbers are the true GPU cost even when stages overlap.
class StageTimer {
 public:
  StageTimer(cudaStream_t s, double* sink, bool on,
             std::vector<cudaEvent_t>* starts, std::vector<cudaEvent_t>* stops,
             std::vector<double*>* sinks)
      : s_(s), stops_(stops), on_(on) {
    if (!on_) return;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, s);
    starts->push_back(a);
    stops->push_back(b);
    sinks->push_back(sink);
  }
  ~StageTimer() {
    if (!on_) return;
    cudaEventRecord(stops_->back(), s_);
  }

 private:
  cudaStream_t s_;
  std::vector<cudaEvent_t>* stops_;
  bool on_;
};

}  // namespace

DecodeEngine::DecodeEngine(const fuel::ModelConfig& cfg, const std::filesystem::path& snapshot_dir,
                           std::size_t expert_cache_bytes, int max_tokens, int max_batch,
                           int kv_pool_pages, int kv_page_tokens)
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
  auto U8 = [&](std::size_t n) { return static_cast<std::uint8_t*>(alloc(n)); };
  auto I64 = [&](std::size_t n) { return static_cast<long long*>(alloc(n * sizeof(long long))); };

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

  // Exact-byte guard against fuels/glm-5.3-flash/attention.yaml's own
  // bytes.kda_state_per_stream accounting: recurrent_per_layer is
  // heads * head_dim * head_dim * 2 B (2097152 for this fuel's 64/128/128),
  // conv_per_layer is 3 tensors * qkv_width * conv_state_taps * 2 B
  // (147456). A missing factor here (e.g. one head_dim term dropped) would
  // silently underallocate kda_state_ and kda_stage_ (model.cu::kda_pack /
  // kda_unpack, used by kv_fork/kv_detach/kv_resume) by exactly that
  // dropped factor, so this is checked once at construction rather than
  // trusted from the arithmetic alone.
  {
    const std::size_t recurrent_per_layer =
        static_cast<std::size_t>(heads_kda) * hd * hd * sizeof(bf16);
    const std::size_t conv_per_layer =
        3ull * cfg_.kda_qkv_dim() * cfg_.conv_state_taps() * sizeof(bf16);
    if (recurrent_per_layer != 2097152ull || conv_per_layer != 147456ull)
      fail("KDA state byte accounting drifted from fuels/glm-5.3-flash/attention.yaml "
          "bytes.kda_state_per_stream (recurrent_per_layer=" +
          std::to_string(recurrent_per_layer) + ", conv_per_layer=" +
          std::to_string(conv_per_layer) + ", expected 2097152 / 147456)");
  }

  // --- MLA latent + indexer key/gate: the radix-tree paged pool
  // (src/kv/page_pool.h), the only backend the kernels' block table reads
  // through kv_locate(). kv_pool_pages <= 0 auto-sizes to one full private
  // context's worth of pages per stream slot (see model.h's constructor
  // comment), matching the deleted stage-1 allocator's worst-case memory.
  {
    const kv::KvGeometry geom{kv_page_tokens, nm, kvl, ihd, cfg_.index_kpool};
    const std::string bad = geom.why_invalid();
    if (!bad.empty()) fail("kv pool geometry: " + bad);
    if (max_tokens % kv_page_tokens != 0)
      fail("max_tokens must be a whole number of KV pages");
    const int pages_per_stream = max_tokens / kv_page_tokens;
    const int pool_pages = kv_pool_pages > 0 ? kv_pool_pages : MB * pages_per_stream;
    if (pool_pages < pages_per_stream)
      fail("kv_pool_pages cannot back even one full-context stream");
    kv_arena_ = std::make_unique<kv::KvArena>(geom, pool_pages, MB, pages_per_stream, stream_);
    kv_pool_ = std::make_unique<kv::PagePool>(pool_pages, kv_page_tokens);
    kv_tree_ = std::make_unique<kv::PrefixTree>();
    kv_cache_ = std::make_unique<kv::KvCache>(geom, kv_pool_.get(), kv_tree_.get(),
                                              kv_arena_.get());
    kda_store_ = std::make_unique<kv::HostKdaStateStore>(stream_);
    mla_kv_ = kv_arena_->pages();
    kv_seq_of_slot_.assign(MB, -1);
    for (int m = 0; m < MB; ++m) kv_seq_of_slot_[m] = kv_cache_->open(m);
    kda_stage_ = alloc(kda_bytes_per_stream());
  }

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

  // --- grouped-GEMM (stage 2) routed-expert scratch, worst case one group
  // per gathered row: max_batch * num_experts_per_tok rows/groups. ---
  const int MI = cfg_.moe_intermediate_size;
  moe_max_rows_ = MB * cfg_.num_experts_per_tok;
  const std::size_t MR = static_cast<std::size_t>(moe_max_rows_);
  moe_x_ = A(MR * H);
  moe_a1_packed_ = U8(MR * H / 2);
  moe_a1_sf_ = U8(MR * 64 * 512);  // k_tiles(H=4096)=64 atoms of 512 B, worst case
  moe_gu_ = A(MR * 2 * MI);
  moe_h_ = A(MR * MI);
  moe_a2_packed_ = U8(MR * MI / 2);
  moe_a2_sf_ = U8(MR * 32 * 512);  // k_tiles(MI=2048)=32 atoms of 512 B, worst case

  const int DI = cfg_.intermediate_size;
  // --- dense-MLP fp4 grouped-path scratch: one group per projection, so the
  // row space is the batch itself. ---
  dense_a1_packed_ = U8(static_cast<std::size_t>(MB) * H / 2);
  dense_a1_sf_ = U8(64 * 512);  // one 128-row tile, k_tiles(H)=64
  dense_gu_ = A(static_cast<std::size_t>(MB) * 2 * DI);
  dense_a2_packed_ = U8(static_cast<std::size_t>(MB) * DI / 2);
  dense_a2_sf_ = U8(192 * 512);  // k_tiles(I=12288)=192
  dense_down_raw_ = A(static_cast<std::size_t>(MB) * H);
  shared_gu_ = A(static_cast<std::size_t>(MB) * 2 * MI * cfg_.n_shared_experts);

  // --- EXL3-fuel routed-expert scratch ---
  if (w_.exl3_fuel()) {
    exl3_hidden_fp16_ = nullptr;
    cuda_check(cudaMalloc(&exl3_hidden_fp16_,
                          static_cast<std::size_t>(MB) * H * 2), "exl3 hidden fp16");
    cuda_check(cudaMalloc(&exl3_out_fp32_,
                          static_cast<std::size_t>(MB) * H * 4), "exl3 out fp32");
    cuda_check(cudaMalloc(&exl3_tok_sorted_,
                          static_cast<std::size_t>(MR) * 8), "exl3 tok sorted");
    cuda_check(cudaMalloc(&exl3_w_sorted_,
                          static_cast<std::size_t>(MR) * 2), "exl3 w sorted");
    cuda_check(cudaMalloc(&exl3_expert_count_,
                          static_cast<std::size_t>(cfg_.n_routed_experts) * 8),
               "exl3 expert count");
    cuda_check(cudaMalloc(&exl3_ptrs_dev_,
                          static_cast<std::size_t>(cfg_.n_routed_experts) * 9 * 8),
               "exl3 ptrs dev");
    cuda_check(cudaHostAlloc(&exl3_ptrs_stage_,
                             static_cast<std::size_t>(cfg_.n_routed_experts) * 9 * 8,
                             cudaHostAllocDefault), "exl3 ptrs stage");
    cuda_check(cudaHostAlloc(&exl3_counts_stage_,
                             static_cast<std::size_t>(cfg_.n_routed_experts) * 8,
                             cudaHostAllocDefault), "exl3 counts stage");
    cuda_check(cudaHostAlloc(&exl3_tok_stage_,
                             static_cast<std::size_t>(MR) * 10,
                             cudaHostAllocDefault), "exl3 tok stage");
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    exl3_conc_ = num_sms / 12;
    if (exl3_conc_ < 1) exl3_conc_ = 1;
    exl3_max_tpe_ = MR;  // worst case: all slots on one expert
    cuda_check(cudaMalloc(&exl3_temp_g_,
                          static_cast<std::size_t>(exl3_conc_) * exl3_max_tpe_ * H * 2),
               "exl3 temp g");
    cuda_check(cudaMalloc(&exl3_temp_u_,
                          static_cast<std::size_t>(exl3_conc_) * exl3_max_tpe_ * H * 2),
               "exl3 temp u");
    cuda_check(cudaMalloc(&exl3_temp_ig_,
                          static_cast<std::size_t>(exl3_conc_) * exl3_max_tpe_ * MI * 2),
               "exl3 temp ig");
    cuda_check(cudaMalloc(&exl3_temp_iu_,
                          static_cast<std::size_t>(exl3_conc_) * exl3_max_tpe_ * MI * 2),
               "exl3 temp iu");
    cuda_check(cudaMalloc(&exl3_a_had_,
                          static_cast<std::size_t>(MB) * H * 2), "exl3 a_had");
  }
  dense_row_in_group_ = I(MB);
  dense_group_of_row_ = I(MB);
  dense_sf_base_ = I64(1);
  dense_gate_global_ = F(1);
  dense_up_global_ = F(1);
  dense_down_global_ = F(1);
  {
    std::vector<int> seq(static_cast<std::size_t>(MB));
    for (int m = 0; m < MB; ++m) seq[m] = m;
    cudaMemcpyAsync(dense_row_in_group_, seq.data(), MB * sizeof(int), cudaMemcpyHostToDevice,
                    stream_);
    cudaStreamSynchronize(stream_);
  }
  moe_out_ = A(MR * H);
  moe_row_of_ = I(MR);
  moe_row_in_group_ = I(MR);
  moe_group_of_row_ = I(MR);
  moe_crow_of_ = I(MR);
  moe_send_rows_ = I(MR);
  moe_recv_rows_ = I(MR);
  moe_sf1_base_ = I64(MR);
  moe_sf2_base_ = I64(MR);
  moe_gate_global_ = F(MR);
  moe_up_global_ = F(MR);
  moe_scatter_w_ = F(MR);
  cuda_check(cudaHostAlloc(&moe_idx_pinned_, MR * sizeof(int), cudaHostAllocDefault),
            "pinned moe idx");
  cuda_check(cudaHostAlloc(&moe_wts_pinned_, MR * sizeof(float), cudaHostAllocDefault),
            "pinned moe wts");
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

  for (int l = 0; l < cfg_.text_layers; ++l)
    if (cfg_.layers[l].mlp == fuel::MlpKind::kSparse) moe_layer_ids_.push_back(l);
  expert_fire_.assign(static_cast<std::size_t>(cfg_.n_routed_experts), 0ull);
}

std::vector<float> DecodeEngine::last_logits(int stream) const {
  std::vector<float> out(static_cast<std::size_t>(cfg_.vocab_size));
  cudaMemcpyAsync(out.data(), logits_ + static_cast<std::size_t>(stream) * cfg_.vocab_size,
                  out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  return out;
}

DecodeEngine::~DecodeEngine() {
  destroy_graphs();
  if (moe_idx_pinned_ != nullptr) cudaFreeHost(moe_idx_pinned_);
  if (moe_wts_pinned_ != nullptr) cudaFreeHost(moe_wts_pinned_);
  for (void* p : owned_) cudaFree(p);
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

void DecodeEngine::destroy_graphs() {
  for (cudaGraphExec_t e : graph_execs_)
    if (e != nullptr) cudaGraphExecDestroy(e);
  for (cudaGraph_t g : graphs_)
    if (g != nullptr) cudaGraphDestroy(g);
  graph_execs_.clear();
  graphs_.clear();
  graph_batch_ = -1;
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

  // The paged KV pool cannot leave stale pages behind the way a cleared
  // stage-1 buffer could: they are refcounted and would never come back
  // without an explicit destroy. Every slot gets a fresh empty sequence
  // instead.
  for (int m = 0; m < MB; ++m) {
    if (kv_seq_of_slot_[m] >= 0) kv_cache_->destroy(kv_seq_of_slot_[m]);
    kv_seq_of_slot_[m] = kv_cache_->open(m);
  }
}

// ------------------------------------------------------------- paged KV

std::size_t DecodeEngine::kda_bytes_per_stream() const {
  const std::size_t conv = 3ull * cfg_.kda_qkv_dim() * cfg_.conv_state_taps() * sizeof(bf16);
  const std::size_t state = static_cast<std::size_t>(cfg_.kda_heads) * cfg_.kda_head_dim *
                            cfg_.kda_head_dim * sizeof(float);
  return static_cast<std::size_t>(kda_layers_) * (conv + state);
}

int DecodeEngine::kv_pinned_pages() const { return kv_pool_->pinned_pages(); }

int DecodeEngine::kv_session_in_slot(int slot) const {
  if (slot < 0 || slot >= max_batch_) return -1;
  return kv_seq_of_slot_[slot];
}

// One stream's KDA state is strided across kda_layers_ (the buffers are
// layer-slot-major so a per-layer call sees a contiguous [batch, ...] block),
// so packing it is one copy per tensor per layer.
void DecodeEngine::kda_pack(int slot, void* dst) {
  const int MB = max_batch_;
  const std::size_t conv_n = static_cast<std::size_t>(cfg_.kda_qkv_dim()) * cfg_.conv_state_taps();
  const std::size_t state_n = static_cast<std::size_t>(cfg_.kda_heads) * cfg_.kda_head_dim *
                              cfg_.kda_head_dim;
  auto* out = static_cast<std::uint8_t*>(dst);
  for (int li = 0; li < kda_layers_; ++li) {
    const std::size_t coff = (static_cast<std::size_t>(li) * MB + slot) * conv_n;
    for (const bf16* src : {q_conv_state_, k_conv_state_, v_conv_state_}) {
      cuda_check(cudaMemcpyAsync(out, src + coff, conv_n * sizeof(bf16),
                                 cudaMemcpyDeviceToDevice, stream_), "kda pack conv");
      out += conv_n * sizeof(bf16);
    }
    const std::size_t soff = (static_cast<std::size_t>(li) * MB + slot) * state_n;
    cuda_check(cudaMemcpyAsync(out, kda_state_ + soff, state_n * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_), "kda pack state");
    out += state_n * sizeof(float);
  }
  cuda_check(cudaStreamSynchronize(stream_), "kda pack sync");
}

void DecodeEngine::kda_unpack(int slot, const void* src) {
  const int MB = max_batch_;
  const std::size_t conv_n = static_cast<std::size_t>(cfg_.kda_qkv_dim()) * cfg_.conv_state_taps();
  const std::size_t state_n = static_cast<std::size_t>(cfg_.kda_heads) * cfg_.kda_head_dim *
                              cfg_.kda_head_dim;
  const auto* in = static_cast<const std::uint8_t*>(src);
  for (int li = 0; li < kda_layers_; ++li) {
    const std::size_t coff = (static_cast<std::size_t>(li) * MB + slot) * conv_n;
    for (bf16* dst : {q_conv_state_, k_conv_state_, v_conv_state_}) {
      cuda_check(cudaMemcpyAsync(dst + coff, in, conv_n * sizeof(bf16),
                                 cudaMemcpyDeviceToDevice, stream_), "kda unpack conv");
      in += conv_n * sizeof(bf16);
    }
    const std::size_t soff = (static_cast<std::size_t>(li) * MB + slot) * state_n;
    cuda_check(cudaMemcpyAsync(kda_state_ + soff, in, state_n * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_), "kda unpack state");
    in += state_n * sizeof(float);
  }
  cuda_check(cudaStreamSynchronize(stream_), "kda unpack sync");
}

void DecodeEngine::kda_copy_slot(int dst_slot, int src_slot) {
  kda_pack(src_slot, kda_stage_);
  kda_unpack(dst_slot, kda_stage_);
}

int DecodeEngine::kv_fork(int parent_slot, int fork_pos, int child_slot) {
  if (parent_slot < 0 || parent_slot >= max_batch_) fail("kv_fork: parent slot out of range");
  if (child_slot < 0 || child_slot >= max_batch_) fail("kv_fork: child slot out of range");
  if (kv_seq_of_slot_[parent_slot] < 0) fail("kv_fork: parent slot holds no sequence");
  if (kv_seq_of_slot_[child_slot] >= 0) kv_cache_->destroy(kv_seq_of_slot_[child_slot]);
  const int child = kv_cache_->fork(kv_seq_of_slot_[parent_slot], fork_pos, child_slot);
  kv_seq_of_slot_[child_slot] = child;
  kv_arena_->upload_table(child_slot, kv_cache_->page_table(child));
  // The recurrent state is not shareable, so the child pays a full copy.
  kda_copy_slot(child_slot, parent_slot);
  pos_[child_slot] = fork_pos;
  return child;
}

int DecodeEngine::kv_detach(int slot) {
  if (slot < 0 || slot >= max_batch_ || kv_seq_of_slot_[slot] < 0)
    fail("kv_detach: slot holds no sequence");
  const int session = kv_seq_of_slot_[slot];
  kda_pack(slot, kda_stage_);
  kda_store_->save(session, kda_stage_, kda_bytes_per_stream());
  kv_cache_->detach(session);
  kv_seq_of_slot_[slot] = -1;
  return session;
}

void DecodeEngine::kv_resume(int session, int slot) {
  if (slot < 0 || slot >= max_batch_) fail("kv_resume: slot out of range");
  if (kv_seq_of_slot_[slot] >= 0) fail("kv_resume: slot is occupied");
  kv_cache_->attach(session, slot);
  kv_seq_of_slot_[slot] = session;
  kv_arena_->upload_table(slot, kv_cache_->page_table(session));
  kda_store_->load(session, kda_stage_, kda_bytes_per_stream());
  kda_unpack(slot, kda_stage_);
  pos_[slot] = kv_cache_->info(session).length;
}

void DecodeEngine::kv_destroy(int session) {
  const int slot = kv_cache_->slot_of(session);
  if (slot >= 0) kv_seq_of_slot_[slot] = -1;
  kv_cache_->destroy(session);
  kda_store_->drop(session);
}

// Reserves this step's token in every active slot and reuploads the block
// table of any slot whose table changed, which is one slot in every
// page_tokens steps plus whatever copy on extend privatised.
void DecodeEngine::kv_advance(const std::vector<int>& tokens, int batch) {
  for (int m = 0; m < batch; ++m) {
    const int seq = kv_seq_of_slot_[m];
    if (seq < 0) fail("step: stream slot holds no KV sequence");
    const kv::AppendSite site = kv_cache_->append_token(seq, tokens[m]);
    if (site.page < 0) fail("step: KV pool exhausted");
    if (site.grew_table || site.copied_on_extend)
      kv_arena_->upload_table(m, kv_cache_->page_table(seq));
  }
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
  const bool kdbg = std::getenv("ROCKET_DEBUG_KDA") != nullptr && layer == 0;
  const auto k_t0 = Clock::now();
  auto k_mark = [&](const char* what) {
    if (!kdbg) return;
    cudaStreamSynchronize(stream_);
    std::printf("[kda-prof] %-14s %8.3f ms\n", what,
                std::chrono::duration<double, std::milli>(Clock::now() - k_t0).count());
  };
  const KdaW& k = w_.layer(layer).kda;
  const int H = cfg_.hidden_size;
  const int qkv = cfg_.kda_qkv_dim();
  const int hd = cfg_.kda_head_dim;
  const int taps = cfg_.conv_state_taps();
  const int heads = cfg_.kda_heads;
  const int MB = max_batch_;

  if (k.qkv_fp8.packed != nullptr) {
    // FP8-per-row q/k/v (same row order as the bf16 concat): half the weight
    // bytes of the largest per-token family. Batch-invariant kernel keeps
    // M=1 and M=B bit-identical.
    gemm_fp8_row(q_raw_, k.qkv_fp8.packed, k.qkv_fp8.scales, normed_, batch, qkv, H, 0, stream_);
    gemm_fp8_row(k_raw_, k.qkv_fp8.packed, k.qkv_fp8.scales, normed_, batch, qkv, H, qkv, stream_);
    gemm_fp8_row(v_raw_, k.qkv_fp8.packed, k.qkv_fp8.scales, normed_, batch, qkv, H,
                 2 * qkv, stream_);
  } else if (k.q_overlay.packed) {
    for (int m = 0; m < batch; ++m)
      gemv_nvfp4(q_raw_ + static_cast<std::size_t>(m) * qkv, k.q_overlay.packed,
                 k.q_overlay.scale, k.q_overlay.global,
                 normed_ + static_cast<std::size_t>(m) * H, qkv, H, stream_);
  } else {
    gemm_bf16(q_raw_, k.qkv, normed_, batch, qkv, H, stream_);
  }
  gemm_bf16(k_raw_, k.qkv + static_cast<std::size_t>(qkv) * H, normed_, batch, qkv, H, stream_);
  gemm_bf16(v_raw_, k.qkv + static_cast<std::size_t>(2) * qkv * H, normed_, batch, qkv, H, stream_);
  k_mark("qkv_gemm");
  if (telemetry_) {
    const std::string p = "layer" + std::to_string(layer) + ".";
    record_absmax(p + "kda_q.output", q_raw_, batch, qkv);
    record_absmax(p + "kda_k.output", k_raw_, batch, qkv);
    record_absmax(p + "kda_v.output", v_raw_, batch, qkv);
  }

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
  k_mark("conv3+normqk");
  if (std::getenv("ROCKET_DEBUG_LAYER0") != nullptr && layer == 0) {
    auto dump_bf16 = [&](const char* what, const void* p, std::size_t n) {
      std::vector<std::uint16_t> h(n);
      cudaMemcpy(h.data(), p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
      double ss = 0; float mx = 0;
      for (auto u : h) { std::uint32_t bits = std::uint32_t(u) << 16; float v; memcpy(&v, &bits, 4);
        ss += double(v) * v; mx = std::max(mx, std::fabs(v)); }
      std::printf("[dbg-kda] %-18s rms=%.6f amax=%.6f\n", what, std::sqrt(ss / n), mx);
    };
    cudaStreamSynchronize(stream_);
    dump_bf16("q_raw", q_raw_, static_cast<std::size_t>(batch) * qkv);
    dump_bf16("k_raw", k_raw_, static_cast<std::size_t>(batch) * qkv);
    dump_bf16("v_raw", v_raw_, static_cast<std::size_t>(batch) * qkv);
    dump_bf16("q_conv", q_conv_, static_cast<std::size_t>(batch) * qkv);
    dump_bf16("k_conv", k_conv_, static_cast<std::size_t>(batch) * qkv);
    dump_bf16("v_conv", v_conv_, static_cast<std::size_t>(batch) * qkv);
    // conv weight regions as loaded on the device
    const int ck = cfg_.kda_conv_kernel;
    {
      std::vector<std::uint16_t> h(16);
      cudaMemcpy(h.data(), k.conv, 32, cudaMemcpyDeviceToHost);
      std::printf("[dbg-kda] wconv_q first16 hex:");
      for (auto u : h) std::printf(" %04x", u);
      std::printf("\n");
      cudaMemcpy(h.data(), k.conv + static_cast<std::size_t>(2) * qkv * ck, 32,
                 cudaMemcpyDeviceToHost);
      std::printf("[dbg-kda] wconv_v first16 hex:");
      for (auto u : h) std::printf(" %04x", u);
      std::printf("\n");
      std::vector<std::uint16_t> st(8);
      cudaMemcpy(st.data(), vstate, 16, cudaMemcpyDeviceToHost);
      std::printf("[dbg-kda] vstate[0..8] hex:");
      for (auto u : st) std::printf(" %04x", u);
      std::printf("\n");
    }
    dump_bf16("wconv_q", k.conv, static_cast<std::size_t>(qkv) * ck);
    dump_bf16("wconv_k", k.conv + static_cast<std::size_t>(qkv) * ck,
              static_cast<std::size_t>(qkv) * ck);
    dump_bf16("wconv_v", k.conv + static_cast<std::size_t>(2) * qkv * ck,
              static_cast<std::size_t>(qkv) * ck);
  }

  gemm_bf16(lr_a_, k.f_a, normed_, batch, hd, H, stream_);
  gemm_bf16(lr_b_, k.f_b, lr_a_, batch, qkv, hd, stream_);
  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".kda_gates.decay_output", lr_b_, batch,
                  qkv);
  kda_forget_gate(gate_, lr_b_, k.dt_bias, k.a_log, batch, heads, hd, cfg_.kda_gate_lower_bound,
                  stream_);
  k_mark("gates");

  gemm_bf16(beta_raw_, k.b_proj, normed_, batch, heads, H, stream_);
  kda_sigmoid(beta_, beta_raw_, batch * heads, stream_);
  k_mark("beta");

  float* state = kda_state_ + static_cast<std::size_t>(slot) * MB * heads * hd * hd;
  kda_recurrent_step(state, kda_o_, q_conv_, k_conv_, v_conv_, gate_, beta_, batch, heads, hd,
                     /*row_stride=*/qkv, stream_);
  k_mark("recurrent");
  if (std::getenv("ROCKET_DEBUG_LAYER0") != nullptr && layer == 0) {
    auto dump_bf16 = [&](const char* what, const void* p, std::size_t n) {
      std::vector<std::uint16_t> h(n);
      cudaMemcpy(h.data(), p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
      double ss = 0; float mx = 0;
      for (auto u : h) { std::uint32_t bits = std::uint32_t(u) << 16; float v; memcpy(&v, &bits, 4);
        ss += double(v) * v; mx = std::max(mx, std::fabs(v)); }
      std::printf("[dbg-kda] %-18s rms=%.6f amax=%.6f\n", what, std::sqrt(ss / n), mx);
    };
    auto dump_f32 = [&](const char* what, const void* p, std::size_t n) {
      std::vector<float> h(n);
      cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost);
      double ss = 0; float mx = 0;
      for (float v : h) { ss += double(v) * v; mx = std::max(mx, std::fabs(v)); }
      std::printf("[dbg-kda] %-18s rms=%.6f amax=%.6f\n", what, std::sqrt(ss / n), mx);
    };
    cudaStreamSynchronize(stream_);
    dump_bf16("gate", gate_, static_cast<std::size_t>(batch) * qkv);
    dump_bf16("beta", beta_, static_cast<std::size_t>(batch) * heads);
    dump_f32("state_after", state, static_cast<std::size_t>(batch) * heads * hd * hd);
    dump_bf16("kda_o", kda_o_, static_cast<std::size_t>(batch) * qkv);
  }

  gemm_bf16(lr_a_, k.g_a, normed_, batch, hd, H, stream_);
  gemm_bf16(lr_b_, k.g_b, lr_a_, batch, qkv, hd, stream_);
  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".kda_gates.output_gate", lr_b_, batch,
                  qkv);
  kda_gated_norm(kda_on_, kda_o_, lr_b_, k.o_norm, batch, heads, hd, cfg_.rms_norm_eps, stream_);
  k_mark("o_norm");
  if (k.o_proj_fp8.packed != nullptr)
    gemm_fp8_row(sublayer_out_, k.o_proj_fp8.packed, k.o_proj_fp8.scales, kda_on_, batch, H, qkv,
                 0, stream_);
  else
    gemm_bf16(sublayer_out_, k.o_proj, kda_on_, batch, H, qkv, stream_);
  k_mark("o_proj");
  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".kda_o.output", sublayer_out_, batch, H);

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

  if (m.q_a_fp8.packed != nullptr)
    gemm_fp8_row(q_resid_raw_, m.q_a_fp8.packed, m.q_a_fp8.scales, normed_, batch,
                 cfg_.q_lora_rank, H, 0, stream_);
  else
    gemm_bf16(q_resid_raw_, m.q_a, normed_, batch, cfg_.q_lora_rank, H, stream_);
  rmsnorm(q_resid_, q_resid_raw_, m.q_a_norm, batch, cfg_.q_lora_rank, cfg_.rms_norm_eps, stream_);
  if (m.q_b_fp8.packed != nullptr)
    gemm_fp8_row(q_, m.q_b_fp8.packed, m.q_b_fp8.scales, q_resid_, batch,
                 heads * cfg_.qk_head_dim(), cfg_.q_lora_rank, 0, stream_);
  else
    gemm_bf16(q_, m.q_b, q_resid_, batch, heads * cfg_.qk_head_dim(), cfg_.q_lora_rank, stream_);
  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".mla_q.output", q_, batch,
                  heads * cfg_.qk_head_dim());

  if (m.kv_a_fp8.packed != nullptr)
    gemm_fp8_row(ckv_, m.kv_a_fp8.packed, m.kv_a_fp8.scales, normed_, batch, kvl, H, 0, stream_);
  else
    gemm_bf16(ckv_, m.kv_a, normed_, batch, kvl, H, stream_);
  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".mla_kv.output", ckv_, batch, kvl);
  rmsnorm(latent_stage_, ckv_, m.kv_a_norm, batch, kvl, cfg_.rms_norm_eps, stream_);

  if (m.idx_wq_b_fp8.packed != nullptr)
    gemm_fp8_row(q_idx_, m.idx_wq_b_fp8.packed, m.idx_wq_b_fp8.scales, q_resid_, batch, ih * ihd,
                 cfg_.q_lora_rank, 0, stream_);
  else
    gemm_bf16(q_idx_, m.idx_wq_b, q_resid_, batch, ih * ihd, cfg_.q_lora_rank, stream_);
  gemm_bf16(idx_k_raw_, m.idx_wk, normed_, batch, ihd, H, stream_);
  layernorm(idx_k_stage_, idx_k_raw_, m.idx_k_norm_w, m.idx_k_norm_b, batch, ihd, 1e-6f, stream_);
  gemm_bf16(idx_g_stage_, m.idx_gate, normed_, batch, ihd, H, stream_);
  if (telemetry_) {
    const std::string p = "layer" + std::to_string(layer) + ".mla_indexer.";
    record_absmax(p + "key_output", idx_k_raw_, batch, ihd);
    record_absmax(p + "gate_output", idx_g_stage_, batch, ihd);
  }
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
  if (m.o_proj_fp8.packed != nullptr)
    gemm_fp8_row(sublayer_out_, m.o_proj_fp8.packed, m.o_proj_fp8.scales, v_out_, batch, H,
                 heads * cfg_.v_head_dim, 0, stream_);
  else
    gemm_bf16(sublayer_out_, m.o_proj, v_out_, batch, H, heads * cfg_.v_head_dim, stream_);
  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".mla_o.output", sublayer_out_, batch, H);

  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".attn.normed", normed_, batch, H);
}

void DecodeEngine::run_dense_mlp(int layer, int batch) {
  const DenseMlpW& d = w_.layer(layer).dense;
  const int H = cfg_.hidden_size;
  const int I = cfg_.intermediate_size;
  if (std::getenv("ROCKET_DENSE_MLP_OFF") != nullptr) {
    // Diagnostic: zero the dense MLP contribution (bypasses the fp4 path).
    cudaMemsetAsync(sublayer_out_, 0, static_cast<std::size_t>(batch) * H * sizeof(bf16), stream_);
    return;
  }
  if (d.gate == nullptr && d.fp4_gate.packed != nullptr) {
    // NVFP4 dense MLP through the grouped path, one group per projection.
    // Single code path at every batch keeps M=1 and M=B bit-identical, and
    // the weight matrices are read once per step instead of once per token.
    // gate|up live back to back in one device slab, so the first GEMM is the
    // fused w13 shape exactly like the MoE experts.
    cudaMemcpyAsync(dense_gate_global_, &d.fp4_gate.global, sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(dense_up_global_, &d.fp4_up.global, sizeof(float), cudaMemcpyHostToDevice,
                    stream_);
    cudaMemcpyAsync(dense_down_global_, &d.fp4_down.global, sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    nvfp4_quantize_rows(dense_a1_packed_, dense_a1_sf_, normed_, dense_row_in_group_,
                        dense_group_of_row_, dense_sf_base_, batch, H, stream_);
    GroupedGemmGroup g1;
    g1.m = batch;
    g1.a_packed = dense_a1_packed_;
    g1.a_scale = dense_a1_sf_;
    g1.b_packed = d.fp4_gate.packed;  // gate|up fused slab, n = 2*I
    g1.b_scale = d.fp4_gate_sw;
    g1.d_out = dense_gu_;
    if (!grouped_gemm_nvfp4_sticky({g1}, 2 * I, H, stream_))
      fail("dense fp4 grouped GEMM1 failed");
    swiglu_grouped(mlp_h_, dense_gu_, dense_gate_global_, dense_up_global_, dense_group_of_row_,
                   batch, I, cfg_.swiglu_limit, stream_);
    nvfp4_quantize_rows(dense_a2_packed_, dense_a2_sf_, mlp_h_, dense_row_in_group_,
                        dense_group_of_row_, dense_sf_base_, batch, I, stream_);
    GroupedGemmGroup g2;
    g2.m = batch;
    g2.a_packed = dense_a2_packed_;
    g2.a_scale = dense_a2_sf_;
    g2.b_packed = d.fp4_down.packed;
    g2.b_scale = d.fp4_down_sw;
    g2.d_out = dense_down_raw_;
    if (!grouped_gemm_nvfp4_sticky({g2}, H, I, stream_))
      fail("dense fp4 grouped GEMM2 failed");
    mul_scalar_bf16(sublayer_out_, dense_down_raw_, d.fp4_down.global,
                    static_cast<long long>(batch) * H, stream_);
  } else {
    gemm_bf16(mlp_gate_, d.gate, normed_, batch, I, H, stream_);
    gemm_bf16(mlp_up_, d.up, normed_, batch, I, H, stream_);
    swiglu_clamped(mlp_h_, mlp_gate_, mlp_up_, batch * I, cfg_.swiglu_limit, stream_);
    gemm_bf16(sublayer_out_, d.down, mlp_h_, batch, H, I, stream_);
  }
  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".ffn.normed", normed_, batch, H);
}

// Stage 1's per-(stream, expert) GEMV loop, unchanged. Runtime fallback
// below kGroupedMoeMinBatch (model.cu top of file) and when the grouped
// launcher reports it cannot implement a shape.
void DecodeEngine::run_moe_gemv(int layer, int batch, const std::vector<int>& idx) {
  const int H = cfg_.hidden_size;
  const int MI = cfg_.moe_intermediate_size;
  const int K = cfg_.num_experts_per_tok;
  for (int m = 0; m < batch; ++m) {
    const bf16* x_row = normed_ + static_cast<std::size_t>(m) * H;
    for (int t = 0; t < K; ++t) {
      const int expert_id = idx[static_cast<std::size_t>(m) * K + t];
      const auto t_stream = Clock::now();
      const ExpertDev& e = w_.expert(layer, expert_id, stream_);
      stages_.expert_stream +=
          std::chrono::duration<double, std::milli>(Clock::now() - t_stream).count();
      gemv_nvfp4(exp_gate_, e.gate_packed, e.gate_scale, e.gate_global, x_row,
                 MI, H, stream_);
      gemv_nvfp4(exp_up_, e.up_packed, e.up_scale, e.up_global, x_row, MI, H,
                 stream_);
      swiglu_clamped(exp_h_, exp_gate_, exp_up_, MI, cfg_.swiglu_limit, stream_);
      gemv_nvfp4(exp_out_, e.down_packed, e.down_scale, e.down_global, exp_h_,
                 H, MI, stream_);
      axpy_bf16(acc_ + static_cast<std::size_t>(m) * H, exp_out_, topk_w_ + static_cast<std::size_t>(m) * K,
               t, K, /*batch=*/1, H, stream_);
    }
  }
}

// Stage 2: CUTLASS grouped GEMM over every (stream, expert) pair the batch
// routed to this layer, grouped by expert id. See kernels.h::nvfp4_quantize_rows,
// swiglu_grouped, moe_gather_rows, moe_scatter_add and moe_grouped.h.
// Returns false if the grouped launcher could not implement a shape (should
// not happen at these fixed dims, but the caller falls back to the GEMV loop
// rather than trust that it never does).
bool DecodeEngine::run_moe_grouped(int layer, int batch, const std::vector<int>& idx,
                                   const std::vector<float>& wts) {
  const int H = cfg_.hidden_size;
  const int MI = cfg_.moe_intermediate_size;
  const int K = cfg_.num_experts_per_tok;
  const int rows = batch * K;

  // Bucket every (stream, k-th choice) slot by expert id. std::map (not
  // unordered_map) so group order -- and therefore every row/group index
  // downstream -- is a pure function of which experts fired, not of hash
  // iteration order, which keeps the grouped path as reproducible as the
  // GEMV loop it replaces.
  const auto t_group0 = Clock::now();
  std::map<int, std::vector<int>> by_expert;
  for (int slot = 0; slot < rows; ++slot) by_expert[idx[static_cast<std::size_t>(slot)]].push_back(slot);
  stages_.moe_group_host_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - t_group0).count();

  // Two row spaces, because a rank's expert set is no longer required to be a
  // contiguous id range (src/fabric/expert_balance.h):
  //
  //   true rows     [0, rows), expert-ascending, the order moe_scatter_add
  //                 accumulates in. Both ranks build the identical list, and
  //                 the merged accumulator stays bit-identical to the
  //                 single-booster one because that order never changes.
  //   compact rows  [0, own_n), this rank's owned rows only, in true-row
  //                 order. Everything the grouped GEMM consumes is indexed
  //                 here, so the owned work is one contiguous run even when
  //                 the owned experts are scattered through the id space.
  //
  // The second GEMM still writes its output at the *true* row offset, so
  // moe_out_ is laid out identically on both ranks and the scatter-add is the
  // same call the single-booster engine makes.
  std::vector<int> row_of(rows);
  std::vector<float> scatter_w(rows);
  std::vector<int> crow_of, row_in_group, group_of_row, send_rows, recv_rows;
  crow_of.reserve(static_cast<std::size_t>(rows));
  send_rows.reserve(static_cast<std::size_t>(rows));
  recv_rows.reserve(static_cast<std::size_t>(rows));
  std::vector<long long> sf1_base, sf2_base;
  std::vector<float> gate_global, up_global;
  std::vector<int> group_expert, group_crow_start, group_row_start;
  std::vector<GroupedGemmGroup> groups1, groups2;
  groups1.reserve(by_expert.size());
  groups2.reserve(by_expert.size());

  int row_start = 0;    // next true row
  int crow = 0;         // next compact row
  long long sf1_off = 0, sf2_off = 0;
  int g = 0;
  int rank0_rows = 0;   // rows rank 0 computes; both ranks compute the same number
  for (const auto& [expert_id, slots] : by_expert) {
    const int Mg = static_cast<int>(slots.size());
    const bool mine = ep_ == nullptr || ep_->owns(expert_id);
    if (ep_ != nullptr && ep_->owner_of(expert_id) == 0) rank0_rows += Mg;
    float down_global = 0.0f;

    if (mine) {
      const auto t_stream = Clock::now();
      const ExpertDev& e = w_.expert(layer, expert_id, stream_);
      stages_.expert_stream +=
          std::chrono::duration<double, std::milli>(Clock::now() - t_stream).count();
      down_global = e.down_global;

      // Parallel indexes for the GEMV fallback (see the grouped launch below):
      // per group, which expert's packed weights to use, where its compact rows
      // start, and where its true rows start. All three are pure functions of
      // the ordered walk this loop already performs.
      group_expert.push_back(expert_id);
      group_crow_start.push_back(crow);
      group_row_start.push_back(row_start);

      const long long mn_tiles = (Mg + 127) / 128;
      sf1_base.push_back(sf1_off);
      sf2_base.push_back(sf2_off);
      gate_global.push_back(e.gate_global);
      up_global.push_back(e.up_global);

      GroupedGemmGroup gr1;
      gr1.m = Mg;
      gr1.a_packed = moe_a1_packed_ + static_cast<std::size_t>(crow) * (H / 2);
      gr1.a_scale = moe_a1_sf_ + sf1_off;
      gr1.b_packed = e.gate_packed;  // fused with up_packed, see weights.h
      gr1.b_scale = e.w13_scale;
      gr1.d_out = moe_gu_ + static_cast<std::size_t>(crow) * (2 * MI);
      groups1.push_back(gr1);

      GroupedGemmGroup gr2;
      gr2.m = Mg;
      gr2.a_packed = moe_a2_packed_ + static_cast<std::size_t>(crow) * (MI / 2);
      gr2.a_scale = moe_a2_sf_ + sf2_off;
      gr2.b_packed = e.down_packed;
      gr2.b_scale = e.down_scale_swizzled;
      gr2.d_out = moe_out_ + static_cast<std::size_t>(row_start) * H;  // true row
      groups2.push_back(gr2);

      sf1_off += mn_tiles * 64 * 512;  // k_tiles(H=4096)=64
      sf2_off += mn_tiles * 32 * 512;  // k_tiles(MI=2048)=32
      ++g;
    } else {
      // The peer computes these rows and writes them into the staging region
      // over the fabric. This rank still needs their scatter weight, so it
      // reads the foreign expert's down weight_scale_2 out of the loader's
      // table (weights.h::expert_down_global) rather than the expert cache it
      // is not allowed to fetch into.
      down_global = w_.expert_down_global(layer, expert_id);
    }

    for (int li = 0; li < Mg; ++li) {
      const int row = row_start + li;
      const int slot = slots[static_cast<std::size_t>(li)];
      row_of[row] = slot / K;  // stream slot this row's activation comes from / goes to
      // The down-projection's own weight_scale_2 is folded into the scatter
      // weight here rather than the GEMM epilogue (moe_grouped.h) or a
      // per-row kernel pass; the router weight and this global both apply
      // once per row with no elementwise interaction, so one multiply on
      // the host, once per (stream, expert) pair, covers both.
      scatter_w[row] = wts[static_cast<std::size_t>(slot)] * down_global;
      if (mine) {
        crow_of.push_back(slot / K);
        row_in_group.push_back(li);
        group_of_row.push_back(g - 1);
        send_rows.push_back(row);
      } else {
        recv_rows.push_back(row);
      }
    }

    if (mine) crow += Mg;
    row_start += Mg;
  }
  const int own_n = crow;

  cudaMemsetAsync(moe_a1_sf_, 0, static_cast<std::size_t>(sf1_off), stream_);
  cudaMemsetAsync(moe_a2_sf_, 0, static_cast<std::size_t>(sf2_off), stream_);
  cudaMemcpyAsync(moe_row_of_, row_of.data(), rows * sizeof(int), cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(moe_scatter_w_, scatter_w.data(), rows * sizeof(float), cudaMemcpyHostToDevice,
                  stream_);
  if (own_n > 0) {
    cudaMemcpyAsync(moe_crow_of_, crow_of.data(), own_n * sizeof(int), cudaMemcpyHostToDevice,
                    stream_);
    cudaMemcpyAsync(moe_row_in_group_, row_in_group.data(), own_n * sizeof(int),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(moe_group_of_row_, group_of_row.data(), own_n * sizeof(int),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(moe_send_rows_, send_rows.data(), own_n * sizeof(int), cudaMemcpyHostToDevice,
                    stream_);
    cudaMemcpyAsync(moe_sf1_base_, sf1_base.data(), sf1_base.size() * sizeof(long long),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(moe_sf2_base_, sf2_base.data(), sf2_base.size() * sizeof(long long),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(moe_gate_global_, gate_global.data(), gate_global.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(moe_up_global_, up_global.data(), up_global.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
  }
  if (!recv_rows.empty())
    cudaMemcpyAsync(moe_recv_rows_, recv_rows.data(), recv_rows.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream_);

  // Every operand below is compact-indexed. On one booster own_n is rows and
  // the compact space is the true space, so these are the same calls as before.
  //
  // Pair-mode fallback: if the grouped launcher cannot run a shape, do not
  // unwind. Both ranks must advance the fabric's shared doorbell sequence
  // exactly once per layer, and the missing half must come out of the exchange,
  // so the GEMV loop below produces the same owned rows into the same buffers
  // (moe_gu_ / moe_out_, raw GEMM outputs, globals left to the shared tail)
  // and control reaches exchange_begin on every path. The peer's sequence
  // arithmetic and the static-fire bit parity are unchanged because the swap
  // is local to this rank (the gather/scatter maps and row order are shared).
  static const bool force_grouped_fail =
      std::getenv("ROCKET_MOE_FORCE_FALLBACK") != nullptr;
  bool gemv_fallback = false;
  if (own_n > 0) {
    moe_gather_rows(moe_x_, normed_, moe_crow_of_, own_n, H, stream_);
    nvfp4_quantize_rows(moe_a1_packed_, moe_a1_sf_, moe_x_, moe_row_in_group_, moe_group_of_row_,
                        moe_sf1_base_, own_n, H, stream_);
    gemv_fallback =
        force_grouped_fail || !grouped_gemm_nvfp4(groups1, 2 * MI, H, stream_);
    if (gemv_fallback) {
      for (std::size_t g_fb = 0; g_fb < group_expert.size(); ++g_fb) {
        const ExpertDev& e = w_.expert(layer, group_expert[g_fb], stream_);
        for (int li = 0; li < groups1[g_fb].m; ++li) {
          const int crow = group_crow_start[g_fb] + li;
          gemv_nvfp4(moe_gu_ + static_cast<std::size_t>(crow) * (2 * MI), e.gate_packed,
                     e.gate_scale, 1.0f, moe_x_ + static_cast<std::size_t>(crow) * H, MI, H,
                     stream_);
          gemv_nvfp4(moe_gu_ + static_cast<std::size_t>(crow) * (2 * MI) + MI, e.up_packed,
                     e.up_scale, 1.0f, moe_x_ + static_cast<std::size_t>(crow) * H, MI, H,
                     stream_);
        }
      }
    }
  }
  if (own_n > 0) {
    // Shared tail in both paths: moe_gu_ is complete by now (grouped GEMM or
    // the stage-1 GEMV fallback above), so the activation variant of the
    // swiglu and the second quantize run unconditionally.
    swiglu_grouped(moe_h_, moe_gu_, moe_gate_global_, moe_up_global_, moe_group_of_row_, own_n,
                   MI, cfg_.swiglu_limit, stream_);
    nvfp4_quantize_rows(moe_a2_packed_, moe_a2_sf_, moe_h_, moe_row_in_group_, moe_group_of_row_,
                        moe_sf2_base_, own_n, MI, stream_);
    gemv_fallback = gemv_fallback || !grouped_gemm_nvfp4(groups2, H, MI, stream_);
  }
  if (gemv_fallback) {
    for (std::size_t g_fb = 0; g_fb < group_expert.size(); ++g_fb) {
      const ExpertDev& e = w_.expert(layer, group_expert[g_fb], stream_);
      for (int li = 0; li < groups2[g_fb].m; ++li) {
        const int crow = group_crow_start[g_fb] + li;
        gemv_nvfp4(moe_out_ + static_cast<std::size_t>(group_row_start[g_fb] + li) * H,
                   e.down_packed, e.down_scale, 1.0f,
                   moe_h_ + static_cast<std::size_t>(crow) * MI, H, MI, stream_);
      }
    }
  }

  if (ep_ == nullptr) {
    moe_scatter_add(acc_, moe_out_, moe_row_of_, moe_scatter_w_, rows, batch, H, stream_);
    return true;
  }

  // Staging layout, agreed by both ranks without a message: rank 0's computed
  // rows first, then rank 1's, each in true-row-ascending order. Both ranks
  // know every expert's owner and every expert's row count, so both compute
  // the same rank0_rows and write into disjoint halves of the same region.
  const int send_off = (ep_->rank() == 0) ? 0 : rank0_rows;
  const int recv_off = (ep_->rank() == 0) ? rank0_rows : 0;
  auto* stage = static_cast<bf16*>(ep_->stage());
  if (own_n > 0)
    moe_gather_rows(stage + static_cast<std::size_t>(send_off) * H, moe_out_, moe_send_rows_, own_n,
                    H, stream_);
  // The NIC reads this memory next, so the GPU's writes have to be complete
  // and not merely enqueued.
  cuda_check(cudaStreamSynchronize(stream_), "sync before RDMA write");
  ep_->exchange_begin(send_off, own_n);
  moe_pending_ = true;
  moe_recv_off_rows_ = recv_off;
  moe_recv_n_ = rows - own_n;
  moe_rows_ = rows;
  moe_batch_ = batch;
  return true;
}

void DecodeEngine::finish_moe_exchange() {
  if (!moe_pending_) return;
  const int H = cfg_.hidden_size;
  ep_->exchange_end(&stages_.fabric);
  moe_pending_ = false;
  auto* stage = static_cast<bf16*>(ep_->stage());
  if (moe_recv_n_ > 0)
    moe_scatter_rows(moe_out_, stage + static_cast<std::size_t>(moe_recv_off_rows_) * H,
                     moe_recv_rows_, moe_recv_n_, H, stream_);
  moe_scatter_add(acc_, moe_out_, moe_row_of_, moe_scatter_w_, moe_rows_, moe_batch_, H, stream_);
}

// Host-side per-expert firing counter (model.h::expert_fire_counts). Reads the
// routing decision run_moe already copied back, so it adds no sync and no
// device work; it is the input to the balanced expert partition
// (src/fabric/expert_balance.h).
void DecodeEngine::record_expert_fire(const std::vector<int>& idx) {
  for (const int e : idx)
    if (e >= 0 && e < static_cast<int>(expert_fire_.size()))
      ++expert_fire_[static_cast<std::size_t>(e)];
}

// EXL3-fuel routed experts through the ported batched trellis kernel: one
// launch per layer (gather, gate|up, silu+clamp, down, scatter-accumulate).
// The host walk emits the expert-sorted arrays the kernel consumes and the
// per-expert device pointer arrays from the streaming slots.
void DecodeEngine::run_moe_exl3(int layer, int batch, const std::vector<int>& idx,
                                const std::vector<float>& wts) {
  const int H = cfg_.hidden_size;
  const int MI = cfg_.moe_intermediate_size;
  const int K = cfg_.num_experts_per_tok;
  const int rows = batch * K;
  const int NE = cfg_.n_routed_experts;

  // Per-expert exl3_gemm_raw path: 3 launches per fired expert (fused
  // gate|up trellis slab, then down), with fp16 conversion bridges. This is
  // the correctness baseline; the batched MoE kernel replaces it for speed
  // once the outputs match the vLLM oracle.
  cuda_check(cudaMemsetAsync(acc_, 0, static_cast<std::size_t>(batch) * H * sizeof(bf16),
                             stream_), "exl3 acc zero");

  // fp16 activation scratch
  bf16_to_fp16_rows(exl3_hidden_fp16_, normed_, static_cast<std::size_t>(batch) * H, stream_);

  // Per-expert: gather rows, gemm gate, gemm up, swiglu, gemm down,
  // scatter-add with the router weight into acc_.
  std::vector<std::int64_t> off(NE + 1, 0);
  for (int slot = 0; slot < rows; ++slot) ++off[idx[static_cast<std::size_t>(slot)] + 1];
  for (int e = 0; e < NE; ++e) off[e + 1] += off[e];

  for (int e = 0; e < NE; ++e) {
    const int cnt = static_cast<int>(off[e + 1] - off[e]);
    if (cnt == 0) continue;
    const Exl3ExpertView& v = w_.exl3_expert(layer, e, stream_);

    // Gather the input rows for this expert into contiguous fp16 scratch.
    for (int si = 0; si < cnt; ++si) {
      const int slot = idx[static_cast<std::size_t>(off[e] + si)];
      const int tok = slot / K;
      bf16_to_fp16_rows(static_cast<char*>(exl3_hidden_fp16_) + static_cast<std::size_t>(si) * H * 2,
                        normed_ + static_cast<std::size_t>(tok) * H, H, stream_);
    }

    // gate: [cnt, 4096] @ [2048, 4096]^T -> [cnt, 2048] fp16
    // (gate and up trellis blobs are separate; call per projection)
    exl3_gemm_raw(exl3_hidden_fp16_, v.gate_trellis, exl3_temp_ig_,
                  v.gate_suh, exl3_a_had_, v.gate_svh,
                  cnt, H, MI, 4, true, false, stream_);
    if (std::getenv("ROCKET_TRACE_COPY") && e == idx[0]) {
      // Check the gate output is non-zero
      static bool checked = false;
      if (!checked) {
        checked = true;
        cuda_check(cudaStreamSynchronize(stream_), "gate sync");
        float sum = 0;
        // copy a few values to host
        std::vector<__half> h(16);
        cudaMemcpy(h.data(), exl3_temp_ig_, 16 * sizeof(__half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 16; ++i) sum += __half2float(h[i]);
        std::fprintf(stderr, "[exl3-dequant] gate output sum (first 16) = %f\n", sum);
        std::fprintf(stderr, "\n");
      }
    }
    exl3_gemm_raw(exl3_hidden_fp16_, v.up_trellis, exl3_temp_iu_,
                  v.up_suh, exl3_a_had_, v.up_svh,
                  cnt, H, MI, 4, true, false, stream_);

    // swiglu on fp16: silu(gate) * clamp(up, ±limit)
    swiglu_fp16(exl3_temp_ig_, exl3_temp_ig_, exl3_temp_iu_,
                static_cast<long long>(cnt) * MI, cfg_.swiglu_limit, stream_);

    // down: [cnt, 2048] @ [4096, 2048]^T -> [cnt, 4096] fp16
    // (reuse temp_state_u as the output scratch)
    exl3_gemm_raw(exl3_temp_ig_, v.down_trellis, exl3_temp_iu_,
                  v.down_suh, exl3_a_had_mi_, v.down_svh,
                  cnt, MI, H, 4, true, false, stream_);

    // scatter-add: acc_[tok] += wts[slot] * fp32(down_out[si])
    for (int si = 0; si < cnt; ++si) {
      const int slot = idx[static_cast<std::size_t>(off[e] + si)];
      const int tok = slot / K;
      const float w = wts[static_cast<std::size_t>(slot)];
      // axpy: acc_[tok] += w * fp16_out[si]
      const __half* src = reinterpret_cast<const __half*>(exl3_temp_iu_) +
                          static_cast<std::size_t>(si) * H;
      bf16* dst = acc_ + static_cast<std::size_t>(tok) * H;
      axpy_bf16_scalar(dst, src, w, H, stream_);
    }
  }
}


void DecodeEngine::run_moe(int layer, int batch) {
  const bool mdbg = std::getenv("ROCKET_DEBUG_MOE") != nullptr && layer == 4;
  const auto m_t0 = Clock::now();
  auto m_mark = [&](const char* what) {
    if (!mdbg) return;
    cudaStreamSynchronize(stream_);
    std::printf("[moe-prof] %-14s %8.3f ms\n", what,
                std::chrono::duration<double, std::milli>(Clock::now() - m_t0).count());
  };
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
  m_mark("router+readback");

  double ent = 0.0;
  double norm = 0.0;
  for (int t = 0; t < K; ++t) norm += wts[t];  // slot 0 only, see model.h::router_entropy
  for (int t = 0; t < K; ++t) {
    const double p = wts[t] / (norm > 0.0 ? norm : 1.0);
    if (p > 0.0) ent -= p * std::log(p);
  }
  router_entropy_ += ent;
  record_expert_fire(idx);

  if (w_.exl3_fuel()) {
    // DEBUG: no-op MoE to isolate the sticky error source
    cudaMemsetAsync(acc_, 0, static_cast<std::size_t>(batch) * H * sizeof(bf16), stream_);
    return;
  }

  cudaMemsetAsync(acc_, 0, static_cast<std::size_t>(batch) * H * sizeof(bf16), stream_);
  const bool want_grouped =
      moe_path_ == MoePath::kForceGrouped ||
      (moe_path_ == MoePath::kAuto && batch >= kGroupedMoeMinBatch);
  if (moe_path_ != MoePath::kForceGemv && want_grouped) {
    m_mark("pre-grouped");
  if (!run_moe_grouped(layer, batch, idx, wts)) {
      cudaMemsetAsync(acc_, 0, static_cast<std::size_t>(batch) * H * sizeof(bf16), stream_);
      run_moe_gemv(layer, batch, idx);
    }
  } else {
    run_moe_gemv(layer, batch, idx);
  }
  m_mark("grouped_gemm_total");

  // Without overlap the step blocks here, exactly where the serialized
  // exchange always sat. With it, the shared expert's dense GEMMs are enqueued
  // first and run on the device while the host waits on the peer's doorbell.
  // The scatter-add is still enqueued before add_bf16 either way, so acc_ is
  // accumulated in the same order and the merged result is unchanged.
  if (!ep_overlap_) finish_moe_exchange();

  run_shared_mlp(layer, batch);
  if (ep_overlap_) finish_moe_exchange();
  add_bf16(acc_, mlp_out_, batch * H, stream_);
  cudaMemcpyAsync(sublayer_out_, acc_, static_cast<std::size_t>(batch) * H * sizeof(bf16),
                  cudaMemcpyDeviceToDevice, stream_);

  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".ffn.normed", normed_, batch, H);
}

// ---------------------------------------------------------------------------
// CUDA graph capture path (blog/posts/runtime/2026-09-07-grouped-gemm-beats-
// gemv-at-every-m/'s Next item 2). Deliberately not a refactor of run_kda,
// run_mla, run_dense_mlp, or run_moe above: those stay the reference path,
// exercised unchanged by every existing test regardless of use_cuda_graph_.
// The functions below duplicate their call sequence for capture; kept in
// sync by hand, same tradeoff this file already made for MoePath::kForceGemv
// vs kForceGrouped.
//
// A layer's attention site never touches WeightStore::expert() or reads
// back to the host, so it captures exactly like the direct path executes it
// -- with one exception: run_mla's indexer calls take a host int
// (n_pools_max) as a *launch bound*, not a data value (kernels.h says so
// explicitly for indexer_pool_keys/indexer_scores: "it only sizes the
// launch, so no kernel reads past a stream's own n_pools", gated by the
// device array n_pools_dev_ inside the kernel). n_pools_max grows with
// position, so a graph captured at one position and replayed at a later one
// would under-launch if it baked in the live value. Passing the worst case
// (max_tokens_ / index_kpool, safely inside pool_stride_'s buffer bound)
// instead makes every MLA attention-site graph replay-correct at any
// position for the engine's whole compiled context, at the cost of some
// idle blocks at low position -- exactly the tradeoff the doc comment
// promises is safe.
void DecodeEngine::run_attn_site(int layer, int batch, int n_pools_launch) {
  const LayerW& lw = w_.layer(layer);
  const int H = cfg_.hidden_size;
  const int hc = cfg_.hc_mult;
  const bool dbg = std::getenv("ROCKET_DEBUG_LAYER0") != nullptr && layer == 0;
  cudaMemcpyAsync(residual_, streams_, static_cast<std::size_t>(batch) * hc * H * sizeof(bf16),
                  cudaMemcpyDeviceToDevice, stream_);
  hc_mix_gemv(mix_, lw.attn_hc.fn, streams_, batch, cfg_.hc_mix(), hc, H, cfg_.rms_norm_eps,
             stream_);
  hc_split(post_, comb_, collapsed_, mix_, lw.attn_hc.base, lw.attn_hc.scale, streams_, batch, hc,
          H, cfg_.hc_eps, cfg_.hc_sinkhorn_iters, stream_);
  rmsnorm(normed_, collapsed_, lw.input_norm, batch, H, cfg_.rms_norm_eps, stream_);
  if (dbg) {
    auto dump = [&](const char* what, const void* p, bool is_float) {
      std::vector<float> h(static_cast<std::size_t>(batch) * hc * H);
      cudaMemcpy(h.data(), p, h.size() * sizeof(float), is_float ? cudaMemcpyDeviceToHost
                                                                 : cudaMemcpyDeviceToHost);
      (void)is_float;
      double ss = 0;
      float mx = 0;
      for (float v : h) { ss += double(v) * v; mx = std::max(mx, std::fabs(v)); }
      std::printf("[dbg-l0] %-18s rms=%.6f amax=%.6f\n", what, std::sqrt(ss / h.size()), mx);
    };
    // bf16 buffers need conversion; read raw and treat pairs via host bf16 table
    auto dump_bf16 = [&](const char* what, const void* p, std::size_t n) {
      std::vector<std::uint16_t> h(n);
      cudaMemcpy(h.data(), p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
      double ss = 0;
      float mx = 0;
      for (auto u : h) {
        std::uint32_t bits = std::uint32_t(u) << 16;
        float v;
        memcpy(&v, &bits, 4);
        ss += double(v) * v;
        mx = std::max(mx, std::fabs(v));
      }
      std::printf("[dbg-l0] %-18s rms=%.6f amax=%.6f\n", what, std::sqrt(ss / n), mx);
    };
    cudaStreamSynchronize(stream_);
    dump_bf16("streams_in", streams_, static_cast<std::size_t>(batch) * hc * H);
    dump("mix", mix_, true);
    dump_bf16("collapsed", collapsed_, static_cast<std::size_t>(batch) * H);
    dump_bf16("normed", normed_, static_cast<std::size_t>(batch) * H);
  }
  if (cfg_.layers[layer].attn == fuel::AttnKind::kKda) {
    run_kda(layer, kda_slot_[layer], batch);
  } else {
    run_mla(layer, mla_slot_[layer], batch, n_tokens_dev_, n_pools_launch);
  }
  if (dbg) {
    auto dump_bf16 = [&](const char* what, const void* p, std::size_t n) {
      std::vector<std::uint16_t> h(n);
      cudaMemcpy(h.data(), p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
      double ss = 0;
      float mx = 0;
      for (auto u : h) {
        std::uint32_t bits = std::uint32_t(u) << 16;
        float v;
        memcpy(&v, &bits, 4);
        ss += double(v) * v;
        mx = std::max(mx, std::fabs(v));
      }
      std::printf("[dbg-l0] %-18s rms=%.6f amax=%.6f\n", what, std::sqrt(ss / n), mx);
    };
    cudaStreamSynchronize(stream_);
    dump_bf16("attn_out", sublayer_out_, static_cast<std::size_t>(batch) * H);
    dump_bf16("streams_post_attn", streams_, static_cast<std::size_t>(batch) * hc * H);
  }
  hc_combine(streams_, post_, sublayer_out_, comb_, residual_, batch, hc, H, stream_);
  if (dbg) {
    auto dump_bf16 = [&](const char* what, const void* p, std::size_t n) {
      std::vector<std::uint16_t> h(n);
      cudaMemcpy(h.data(), p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
      double ss = 0;
      float mx = 0;
      for (auto u : h) {
        std::uint32_t bits = std::uint32_t(u) << 16;
        float v;
        memcpy(&v, &bits, 4);
        ss += double(v) * v;
        mx = std::max(mx, std::fabs(v));
      }
      std::printf("[dbg-l0] %-18s rms=%.6f amax=%.6f\n", what, std::sqrt(ss / n), mx);
    };
    cudaStreamSynchronize(stream_);
    dump_bf16("streams_post_combine", streams_, static_cast<std::size_t>(batch) * hc * H);
  }
}

// FFN site of a dense-MLP layer. Never called for a sparse (MoE) layer;
// ensure_graphs_built's segment loop only reaches this for layers strictly
// between two moe_layer_ids_ entries (or before the first / after the last),
// which are dense by construction of that list.
void DecodeEngine::run_ffn_site_dense(int layer, int batch) {
  const LayerW& lw = w_.layer(layer);
  const int H = cfg_.hidden_size;
  const int hc = cfg_.hc_mult;
  cudaMemcpyAsync(residual_, streams_, static_cast<std::size_t>(batch) * hc * H * sizeof(bf16),
                  cudaMemcpyDeviceToDevice, stream_);
  hc_mix_gemv(mix_, lw.ffn_hc.fn, streams_, batch, cfg_.hc_mix(), hc, H, cfg_.rms_norm_eps,
             stream_);
  hc_split(post_, comb_, collapsed_, mix_, lw.ffn_hc.base, lw.ffn_hc.scale, streams_, batch, hc, H,
          cfg_.hc_eps, cfg_.hc_sinkhorn_iters, stream_);
  rmsnorm(normed_, collapsed_, lw.post_attn_norm, batch, H, cfg_.rms_norm_eps, stream_);
  run_dense_mlp(layer, batch);
  hc_combine(streams_, post_, sublayer_out_, comb_, residual_, batch, hc, H, stream_);
}

// Captured half of a MoE FFN site: hc-mix/split, norm, router GEMM, top-k
// (device-only, kernels.cu::moe_router_kernel -- no host sync forces this
// one), and the async device->host copy of the routing decision. Ends
// without a sync: the caller (run_step_layers_graph) syncs once after
// launching the graph this sits in, which is the one host sync per MoE
// layer that stays, same as the direct path's run_moe.
void DecodeEngine::run_moe_router_stage(int layer, int batch) {
  const MoeW& mo = w_.layer(layer).moe;
  const LayerW& lw = w_.layer(layer);
  const int H = cfg_.hidden_size;
  const int hc = cfg_.hc_mult;
  const int K = cfg_.num_experts_per_tok;
  cudaMemcpyAsync(residual_, streams_, static_cast<std::size_t>(batch) * hc * H * sizeof(bf16),
                  cudaMemcpyDeviceToDevice, stream_);
  hc_mix_gemv(mix_, lw.ffn_hc.fn, streams_, batch, cfg_.hc_mix(), hc, H, cfg_.rms_norm_eps,
             stream_);
  hc_split(post_, comb_, collapsed_, mix_, lw.ffn_hc.base, lw.ffn_hc.scale, streams_, batch, hc, H,
          cfg_.hc_eps, cfg_.hc_sinkhorn_iters, stream_);
  rmsnorm(normed_, collapsed_, lw.post_attn_norm, batch, H, cfg_.rms_norm_eps, stream_);

  gemm_bf16_f32(router_logits_, mo.router, normed_, batch, cfg_.n_routed_experts, H, stream_);
  moe_router(topk_idx_, topk_w_, router_logits_, mo.router_bias, batch, cfg_.n_routed_experts, K,
            cfg_.norm_topk_prob, cfg_.routed_scaling_factor, stream_);
  const int rows = batch * K;
  cudaMemcpyAsync(moe_idx_pinned_, topk_idx_, static_cast<std::size_t>(rows) * sizeof(int),
                  cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(moe_wts_pinned_, topk_w_, static_cast<std::size_t>(rows) * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);
  cudaMemsetAsync(acc_, 0, static_cast<std::size_t>(batch) * H * sizeof(bf16), stream_);
}

// Uncaptured half: the routing decision is on the host now (the caller
// synced after this layer's router-stage graph), so this reads
// moe_idx_pinned_/moe_wts_pinned_, does the router-entropy accumulation
// run_moe does inline, then dispatches to the grouped GEMM or the GEMV
// fallback exactly as run_moe does. Never appears inside a captured graph:
// w_.expert()'s LRU/mmap bookkeeping is host code with no CUDA-graph
// analogue, and it must run after this layer's routing is known.
void DecodeEngine::run_moe_dispatch_stage(int layer, int batch) {
  const int H = cfg_.hidden_size;
  const int K = cfg_.num_experts_per_tok;
  const int rows = batch * K;
  std::vector<int> idx(moe_idx_pinned_, moe_idx_pinned_ + rows);
  std::vector<float> wts(moe_wts_pinned_, moe_wts_pinned_ + rows);

  double ent = 0.0;
  double norm = 0.0;
  for (int t = 0; t < K; ++t) norm += wts[static_cast<std::size_t>(t)];
  for (int t = 0; t < K; ++t) {
    const double pr = wts[static_cast<std::size_t>(t)] / (norm > 0.0 ? norm : 1.0);
    if (pr > 0.0) ent -= pr * std::log(pr);
  }
  router_entropy_ += ent;
  record_expert_fire(idx);

  const bool want_grouped =
      moe_path_ == MoePath::kForceGrouped ||
      (moe_path_ == MoePath::kAuto && batch >= kGroupedMoeMinBatch);
  if (moe_path_ != MoePath::kForceGemv && want_grouped) {
    if (!run_moe_grouped(layer, batch, idx, wts)) {
      cudaMemsetAsync(acc_, 0, static_cast<std::size_t>(batch) * H * sizeof(bf16), stream_);
      run_moe_gemv(layer, batch, idx);
    }
  } else {
    run_moe_gemv(layer, batch, idx);
  }
  // No overlap on the graph path: run_moe_post_stage (the shared expert) is
  // inside the next captured segment, which cannot be launched before acc_
  // holds the routed half.
  finish_moe_exchange();
}

// Captured tail of a MoE FFN site: shared-expert dense FFN, added into acc_
// (written by the uncaptured dispatch immediately before this runs), then
// hc-combine. Deferred to the *start* of the next graph segment
// (ensure_graphs_built) because it must run after dispatch, which is never
// itself inside a graph.

// Shared-expert FFN into mlp_out_: bf16 GEMMs, or the NVFP4 grouped path
// (one group per projection, fused gate|up slab) when the fp4 overlays are
// loaded. The sticky variant keeps the graph-segment calls capture-safe and
// the batch-invariant kernels keep M=1 and M=B bit-identical.
void DecodeEngine::run_shared_mlp(int layer, int batch) {
  const MoeW& mo = w_.layer(layer).moe;
  const int H = cfg_.hidden_size;
  const int MI = cfg_.moe_intermediate_size;
  const int SI = MI * cfg_.n_shared_experts;
  const DenseMlpW& sh = mo.shared;
  if (sh.fp4_gate.packed != nullptr) {
    cudaMemcpyAsync(dense_gate_global_, &sh.fp4_gate.global, sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(dense_up_global_, &sh.fp4_up.global, sizeof(float), cudaMemcpyHostToDevice,
                    stream_);
    cudaMemcpyAsync(dense_down_global_, &sh.fp4_down.global, sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    nvfp4_quantize_rows(dense_a1_packed_, dense_a1_sf_, normed_, dense_row_in_group_,
                        dense_group_of_row_, dense_sf_base_, batch, H, stream_);
    GroupedGemmGroup g1;
    g1.m = batch;
    g1.a_packed = dense_a1_packed_;
    g1.a_scale = dense_a1_sf_;
    g1.b_packed = sh.fp4_gate.packed;  // fused gate|up, n = 2*SI
    g1.b_scale = sh.fp4_gate_sw;
    g1.d_out = shared_gu_;
    if (!grouped_gemm_nvfp4_sticky({g1}, 2 * SI, H, stream_))
      fail("shared fp4 grouped GEMM1 failed");
    swiglu_grouped(mlp_h_, shared_gu_, dense_gate_global_, dense_up_global_, dense_group_of_row_,
                   batch, SI, cfg_.swiglu_limit, stream_);
    nvfp4_quantize_rows(dense_a2_packed_, dense_a2_sf_, mlp_h_, dense_row_in_group_,
                        dense_group_of_row_, dense_sf_base_, batch, SI, stream_);
    GroupedGemmGroup g2;
    g2.m = batch;
    g2.a_packed = dense_a2_packed_;
    g2.a_scale = dense_a2_sf_;
    g2.b_packed = sh.fp4_down.packed;
    g2.b_scale = sh.fp4_down_sw;
    g2.d_out = dense_down_raw_;
    if (!grouped_gemm_nvfp4_sticky({g2}, H, SI, stream_)) fail("shared fp4 grouped GEMM2 failed");
    mul_scalar_bf16(mlp_out_, dense_down_raw_, sh.fp4_down.global,
                    static_cast<long long>(batch) * H, stream_);
    return;
  }
  gemm_bf16(mlp_gate_, sh.gate, normed_, batch, SI, H, stream_);
  gemm_bf16(mlp_up_, sh.up, normed_, batch, SI, H, stream_);
  swiglu_clamped(mlp_h_, mlp_gate_, mlp_up_, batch * SI, cfg_.swiglu_limit, stream_);
  gemm_bf16(mlp_out_, sh.down, mlp_h_, batch, H, SI, stream_);
}

void DecodeEngine::run_moe_post_stage(int layer, int batch) {
  const MoeW& mo = w_.layer(layer).moe;
  const int H = cfg_.hidden_size;
  const int MI = cfg_.moe_intermediate_size;
  const int SI = MI * cfg_.n_shared_experts;
  const int hc = cfg_.hc_mult;
  run_shared_mlp(layer, batch);
  add_bf16(acc_, mlp_out_, batch * H, stream_);
  cudaMemcpyAsync(sublayer_out_, acc_, static_cast<std::size_t>(batch) * H * sizeof(bf16),
                  cudaMemcpyDeviceToDevice, stream_);
  hc_combine(streams_, post_, sublayer_out_, comb_, residual_, batch, hc, H, stream_);
}

// Builds (or rebuilds, if batch changed) one CUDA graph per segment: segment
// i covers everything from just after moe_layer_ids_[i-1]'s dispatch (its
// deferred post-stage) through every following dense/KDA/MLA layer up to
// and including moe_layer_ids_[i]'s router stage; the last segment runs
// through the end of the layer stack. num_moe_layers segments have a router
// stage at their end and are followed by an uncaptured dispatch;
// num_moe_layers + 1 segments exist in total. Rebuilding tears down any
// prior graphs first (destroy_graphs) since a graph captured for one batch
// size bakes in that batch's kernel launch dimensions.
void DecodeEngine::ensure_graphs_built(int batch) {
  // Warm the grouped-GEMM workspace outside capture: the dense fp4 path runs
  // inside captured segments, and a first-call workspace cudaMalloc cannot
  // happen during stream capture. The warm-up writes only scratch buffers
  // that every real layer overwrites.
  if (w_.layer(0).dense.fp4_gate.packed != nullptr) {
    for (int l = 0; l < cfg_.text_layers; ++l)
      if (cfg_.layers[l].mlp == fuel::MlpKind::kDense) run_dense_mlp(l, batch);
  }
  if (graph_batch_ == batch && !graph_execs_.empty()) return;
  destroy_graphs();
  graph_batch_ = batch;
  const int num_moe = static_cast<int>(moe_layer_ids_.size());
  const int n_pools_worst = max_tokens_ / cfg_.index_kpool;
  graphs_.assign(static_cast<std::size_t>(num_moe) + 1, nullptr);
  graph_execs_.assign(static_cast<std::size_t>(num_moe) + 1, nullptr);

  int layer = 0;
  for (int seg = 0; seg <= num_moe; ++seg) {
    cuda_check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
              "graph capture begin");
    if (seg > 0) run_moe_post_stage(moe_layer_ids_[static_cast<std::size_t>(seg) - 1], batch);
    const int stop_at = (seg < num_moe) ? moe_layer_ids_[static_cast<std::size_t>(seg)]
                                        : cfg_.text_layers;
    for (; layer < stop_at; ++layer) {
      run_attn_site(layer, batch, n_pools_worst);
      run_ffn_site_dense(layer, batch);
    }
    if (seg < num_moe) {
      run_attn_site(layer, batch, n_pools_worst);
      run_moe_router_stage(layer, batch);
      ++layer;  // this layer's post-stage is deferred to segment seg + 1
    }
    cudaGraph_t g = nullptr;
    cuda_check(cudaStreamEndCapture(stream_, &g), "graph capture end");
    graphs_[static_cast<std::size_t>(seg)] = g;
    cuda_check(cudaGraphInstantiate(&graph_execs_[static_cast<std::size_t>(seg)], g, 0),
              "graph instantiate");
  }
}

// Replays the graphs built by ensure_graphs_built, doing the one unavoidable
// per-MoE-layer host sync and uncaptured dispatch between segments i and
// i + 1. This is the entire routing-forced-sync count for the step: 42 on
// this chemistry (cfg_.layers with mlp == kSparse), not 45 and not more --
// the indexer's own top-k (kernels.cu::indexer_select) needs none, since it
// is device-only and gated by a device array, not a host readback.
void DecodeEngine::run_step_layers_graph(int batch) {
  const int num_moe = static_cast<int>(moe_layer_ids_.size());
  for (int seg = 0; seg <= num_moe; ++seg) {
    cuda_check(cudaGraphLaunch(graph_execs_[static_cast<std::size_t>(seg)], stream_),
              "graph launch");
    if (seg < num_moe) {
      cuda_check(cudaStreamSynchronize(stream_), "graph segment sync");
      run_moe_dispatch_stage(moe_layer_ids_[static_cast<std::size_t>(seg)], batch);
    }
  }
}

void DecodeEngine::step(const std::vector<int>& tokens, std::vector<int>& out_tokens,
                       bool collect_stages) {
  const int batch = static_cast<int>(tokens.size());
  if (batch <= 0 || batch > max_batch_) fail("batch out of [1, max_batch] range");
  for (const int p : pos_)
    if (p > max_tokens_) fail("position exceeds the compiled max_tokens");

  // Must run before any KV write: it is what decides which physical page
  // this step's position resolves to.
  kv_advance(tokens, batch);

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
    StageTimer t(stream_, &stages_.embed, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
    embed_streams(streams_, w_.embed(), tokens_dev_, batch, hc, H, stream_);
  }

  // Graphs need collect_stages off (StageTimer's destructor syncs to
  // attribute wall time to a stage, which cannot happen inside a capture)
  // and telemetry off (record_absmax syncs too); step() falls back to the
  // direct per-layer loop below for either, exactly as if use_cuda_graph_
  // were never set.
  const bool use_graph = use_cuda_graph_ && !collect_stages && !telemetry_;
  if (use_graph) {
    static bool said = false;
    if (!said) { std::printf("[graph] capturing/replaying graph path\n"); said = true; }
    ensure_graphs_built(batch);
    run_step_layers_graph(batch);
  } else {
  for (int l = 0; l < cfg_.text_layers; ++l) {
    const LayerW& lw = w_.layer(l);
    const bool dbg_l0 = std::getenv("ROCKET_DEBUG_LAYER0") != nullptr && l == 0;
    const bool first_site = dbg_l0;

    // ---- attention site ----
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      cudaMemcpyAsync(residual_, streams_, static_cast<std::size_t>(batch) * hc * H * sizeof(bf16),
                      cudaMemcpyDeviceToDevice, stream_);
      hc_mix_gemv(mix_, lw.attn_hc.fn, streams_, batch, cfg_.hc_mix(), hc, H, cfg_.rms_norm_eps,
                 stream_);
      hc_split(post_, comb_, collapsed_, mix_, lw.attn_hc.base, lw.attn_hc.scale, streams_, batch,
              hc, H, cfg_.hc_eps, cfg_.hc_sinkhorn_iters, stream_);
    }
    {
      StageTimer t(stream_, &stages_.norms, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      rmsnorm(normed_, collapsed_, lw.input_norm, batch, H, cfg_.rms_norm_eps, stream_);
    }
    if (std::getenv("ROCKET_TRACE_COPY")) {
      const cudaError_t le = cudaGetLastError();
      if (le != cudaSuccess)
        std::fprintf(stderr, "[sticky] layer %d entry: %s\n", l, cudaGetErrorString(le));
    }
    if (cfg_.layers[l].attn == fuel::AttnKind::kKda) {
      StageTimer t(stream_, &stages_.kda, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      run_kda(l, kda_slot_[l], batch);
    } else {
      // The indexer runs inside the MLA layer; its cost is folded in here
      // rather than isolated by a sync, since a batched step has no
      // per-layer host readback of the selection count anymore.
      StageTimer t(stream_, &stages_.mla, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      run_mla(l, mla_slot_[l], batch, n_tokens_dev_, n_pools_max);
    }
    if (dbg_l0) {
      auto dump_bf16 = [&](const char* what, const void* p, std::size_t n) {
        std::vector<std::uint16_t> h(n);
        cudaMemcpy(h.data(), p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        double ss = 0;
        float mx = 0;
        for (auto u : h) {
          std::uint32_t bits = std::uint32_t(u) << 16;
          float v;
          memcpy(&v, &bits, 4);
          ss += double(v) * v;
          mx = std::max(mx, std::fabs(v));
        }
        std::printf("[dbg-l0] %-20s rms=%.6f amax=%.6f\n", what, std::sqrt(ss / n), mx);
      };
      auto dump_f32 = [&](const char* what, const void* p, std::size_t n) {
        std::vector<float> h(n);
        cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost);
        double ss = 0;
        float mx = 0;
        for (float v : h) { ss += double(v) * v; mx = std::max(mx, std::fabs(v)); }
        std::printf("[dbg-l0] %-20s rms=%.6f amax=%.6f\n", what, std::sqrt(ss / n), mx);
      };
      cudaStreamSynchronize(stream_);
      if (first_site) {
        dump_bf16("streams_in", streams_, static_cast<std::size_t>(batch) * hc * H);
        dump_f32("mix", mix_, static_cast<std::size_t>(batch) * cfg_.hc_mix());
        dump_bf16("collapsed", collapsed_, static_cast<std::size_t>(batch) * H);
        dump_bf16("normed", normed_, static_cast<std::size_t>(batch) * H);
      }
      dump_bf16("attn_out", sublayer_out_, static_cast<std::size_t>(batch) * H);
      dump_bf16("streams_post_attn_combine", streams_, static_cast<std::size_t>(batch) * hc * H);
    }
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      hc_combine(streams_, post_, sublayer_out_, comb_, residual_, batch, hc, H, stream_);
    }

    // ---- feed-forward site ----
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      cudaMemcpyAsync(residual_, streams_, static_cast<std::size_t>(batch) * hc * H * sizeof(bf16),
                      cudaMemcpyDeviceToDevice, stream_);
      hc_mix_gemv(mix_, lw.ffn_hc.fn, streams_, batch, cfg_.hc_mix(), hc, H, cfg_.rms_norm_eps,
                 stream_);
      hc_split(post_, comb_, collapsed_, mix_, lw.ffn_hc.base, lw.ffn_hc.scale, streams_, batch, hc,
              H, cfg_.hc_eps, cfg_.hc_sinkhorn_iters, stream_);
    }
    {
      StageTimer t(stream_, &stages_.norms, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      rmsnorm(normed_, collapsed_, lw.post_attn_norm, batch, H, cfg_.rms_norm_eps, stream_);
    }
    if (cfg_.layers[l].mlp == fuel::MlpKind::kDense) {
      StageTimer t(stream_, &stages_.dense_mlp, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      run_dense_mlp(l, batch);
    } else {
      StageTimer t(stream_, &stages_.moe_experts, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      run_moe(l, batch);
    }
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      hc_combine(streams_, post_, sublayer_out_, comb_, residual_, batch, hc, H, stream_);
    }

    if (collect_stages) {
      hc_head_mean(hmean_, streams_, batch, hc, H, stream_);
      layer_rms_[l] = sync_rms_slot0(hmean_, H);
    }
  }
  }  // else (!use_graph)

  std::vector<int> next(batch, 0);
  {
    StageTimer t(stream_, &stages_.lm_head, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
    hc_head_mean(hmean_, streams_, batch, hc, H, stream_);
    rmsnorm(normed_, hmean_, w_.final_norm(), batch, H, cfg_.rms_norm_eps, stream_);
    gemm_bf16_f32(logits_, w_.lm_head(), normed_, batch, cfg_.vocab_size, H, stream_);
    argmax_f32(argmax_i_, scratch_f_, logits_, batch, cfg_.vocab_size, stream_);
    cudaMemcpyAsync(next.data(), argmax_i_, batch * sizeof(int), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }

  

  finish_stage_events();

  int moe_layers = 0;
  for (const fuel::LayerSpec& s : cfg_.layers)
    if (s.mlp == fuel::MlpKind::kSparse) ++moe_layers;
  router_entropy_ /= (moe_layers > 0 ? moe_layers : 1);

  for (int m = 0; m < batch; ++m) ++pos_[m];
  cuda_check(cudaGetLastError(), "decode step");
  out_tokens = std::move(next);
}

void DecodeEngine::finish_stage_events() {
  if (prof_starts_.empty()) return;
  for (std::size_t i = 0; i < prof_starts_.size(); ++i) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, prof_starts_[i], prof_stops_[i]);
    *prof_sinks_[i] += ms;
    cudaEventDestroy(prof_starts_[i]);
    cudaEventDestroy(prof_stops_[i]);
  }
  prof_starts_.clear();
  prof_stops_.clear();
  prof_sinks_.clear();
}

}  // namespace rocket::engine

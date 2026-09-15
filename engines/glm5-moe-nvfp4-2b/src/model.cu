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

namespace rocket::engine {
namespace {

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
    index_ = stops->size();
    starts->push_back(a);
    stops->push_back(b);
    sinks->push_back(sink);
  }
  ~StageTimer() {
    if (!on_) return;
    cudaEventRecord((*stops_)[index_], s_);
  }

 private:
  cudaStream_t s_;
  std::vector<cudaEvent_t>* stops_;
  std::size_t index_ = 0;
  bool on_;
};

}  // namespace

DecodeEngine::DecodeEngine(const fuel::ModelConfig& cfg, const std::filesystem::path& snapshot_dir,
                           std::size_t expert_cache_bytes, int max_tokens, int max_batch,
                           int kv_pool_pages, int kv_page_tokens, int max_prefill_chunk)
    : cfg_(cfg),
      w_(cfg, snapshot_dir, expert_cache_bytes),
      max_tokens_(max_tokens),
      max_batch_(max_batch),
      max_work_k_(std::max(kSpecMax, max_prefill_chunk)) {
  if (max_batch_ <= 0) fail("max_batch must be positive");
  if (max_prefill_chunk < 1 || max_prefill_chunk > 64)
    fail("max_prefill_chunk must be in [1,64]");
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
  // Work-row space may be wider for prefill, while DFlash2 verification and
  // its transactional replay rails remain fixed at kSpecMax=8.
  const int SB = MB * max_work_k_;
  const int RB = MB * DecodeEngine::kSpecMax;

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
  kda_state_draft_ = F(static_cast<std::size_t>(nk) * MB * heads_kda * hd * hd);
  q_conv_state_draft_ = A(static_cast<std::size_t>(nk) * MB * qkv * taps);
  k_conv_state_draft_ = A(static_cast<std::size_t>(nk) * MB * qkv * taps);
  v_conv_state_draft_ = A(static_cast<std::size_t>(nk) * MB * qkv * taps);

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
    restored_kda_chain_.assign(MB, 0);
    restored_next_token_.assign(MB, 0);
    for (int m = 0; m < MB; ++m) kv_seq_of_slot_[m] = kv_cache_->open(m);
    kda_stage_ = alloc(kda_bytes_per_stream());
  }

  tokens_dev_ = I(MB);
  prefix_token_dev_ = I(1);
  pos_dev_ = I(MB);
  n_tokens_dev_ = I(MB);
  n_pools_dev_ = I(MB);

  streams_ = A(static_cast<std::size_t>(SB) * hc * H);
  residual_ = A(static_cast<std::size_t>(SB) * hc * H);
  collapsed_ = A(static_cast<std::size_t>(SB) * H);
  normed_ = A(static_cast<std::size_t>(SB) * H);
  sublayer_out_ = A(static_cast<std::size_t>(SB) * H);
  hmean_ = A(static_cast<std::size_t>(SB) * H);
  dflash_aux_hidden_ = A(static_cast<std::size_t>(5) * SB * H);
  mix_ = F(static_cast<std::size_t>(SB) * cfg_.hc_mix());
  post_ = F(static_cast<std::size_t>(SB) * hc);
  comb_ = F(static_cast<std::size_t>(SB) * hc * hc);

  q_raw_ = A(static_cast<std::size_t>(SB) * qkv);
  k_raw_ = A(static_cast<std::size_t>(SB) * qkv);
  v_raw_ = A(static_cast<std::size_t>(SB) * qkv);
  q_conv_ = A(static_cast<std::size_t>(SB) * qkv);
  k_conv_ = A(static_cast<std::size_t>(SB) * qkv);
  v_conv_ = A(static_cast<std::size_t>(SB) * qkv);
  lr_a_ = A(static_cast<std::size_t>(SB) * hd);
  lr_b_ = A(static_cast<std::size_t>(SB) * qkv);
  gate_ = A(static_cast<std::size_t>(SB) * qkv);
  beta_raw_ = A(static_cast<std::size_t>(SB) * heads_kda);
  beta_ = A(static_cast<std::size_t>(SB) * heads_kda);
  kda_o_ = A(static_cast<std::size_t>(SB) * qkv);
  kda_on_ = A(static_cast<std::size_t>(SB) * qkv);

  q_resid_raw_ = A(static_cast<std::size_t>(SB) * cfg_.q_lora_rank);
  q_resid_ = A(static_cast<std::size_t>(SB) * cfg_.q_lora_rank);
  q_ = A(static_cast<std::size_t>(SB) * heads_mla * cfg_.qk_head_dim());
  ckv_ = A(static_cast<std::size_t>(SB) * kvl);
  latent_stage_ = A(static_cast<std::size_t>(SB) * kvl);
  v_out_ = A(static_cast<std::size_t>(SB) * heads_mla * cfg_.v_head_dim);
  q_abs_ = F(static_cast<std::size_t>(SB) * heads_mla * kvl);
  ctx_ = F(static_cast<std::size_t>(SB) * heads_mla * kvl);
  scores_ = F(static_cast<std::size_t>(SB) * heads_mla * sel_stride_);

  q_idx_ = A(static_cast<std::size_t>(SB) * ih * ihd);
  idx_k_raw_ = A(static_cast<std::size_t>(SB) * ihd);
  idx_k_stage_ = A(static_cast<std::size_t>(SB) * ihd);
  idx_g_stage_ = A(static_cast<std::size_t>(SB) * ihd);
  pool_keys_ = A(static_cast<std::size_t>(SB) * pool_stride_ * ihd);
  pool_scores_ = F(static_cast<std::size_t>(SB) * pool_stride_);
  head_w_ = F(static_cast<std::size_t>(SB) * ih);
  sel_pools_ = I(static_cast<std::size_t>(SB) * sel_pool_stride_);
  sel_tokens_ = I(static_cast<std::size_t>(SB) * sel_stride_);
  n_sel_ = I(SB);
  n_tok_ = I(SB);

  router_logits_ = F(static_cast<std::size_t>(SB) * cfg_.n_routed_experts);
  topk_w_ = F(static_cast<std::size_t>(SB) * cfg_.num_experts_per_tok);
  topk_idx_ = I(static_cast<std::size_t>(SB) * cfg_.num_experts_per_tok);
  exp_gate_ = A(cfg_.moe_intermediate_size);
  exp_up_ = A(cfg_.moe_intermediate_size);
  exp_h_ = A(cfg_.moe_intermediate_size);
  exp_out_ = A(H);

  // --- grouped-GEMM (stage 2) routed-expert scratch, worst case one group
  // per gathered row: max_batch * num_experts_per_tok rows/groups. ---
  const int MI = cfg_.moe_intermediate_size;
  moe_max_rows_ = SB * cfg_.num_experts_per_tok;
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
  dense_a1_packed_ = U8(static_cast<std::size_t>(SB) * H / 2);
  const std::size_t dense_m_tiles = (static_cast<std::size_t>(SB) + 127) / 128;
  dense_a1_sf_ = U8(dense_m_tiles * 64 * 512);  // mn_tiles * k_tiles(H) * atom
  dense_gu_ = A(static_cast<std::size_t>(SB) * 2 * DI);
  dense_a2_packed_ = U8(static_cast<std::size_t>(SB) * DI / 2);
  dense_a2_sf_ = U8(dense_m_tiles * 192 * 512);  // mn_tiles * k_tiles(I) * atom
  dense_down_raw_ = A(static_cast<std::size_t>(SB) * H);
  shared_gu_ = A(static_cast<std::size_t>(SB) * 2 * MI * cfg_.n_shared_experts);

  // Spec verify batches kSpecMax rows per stream through the dense/shared MLPs
  // (run_dense_mlp(l, total)), so these row maps must cover the SB row space,
  // not the plain batch MB.
  dense_row_in_group_ = I(SB);
  dense_group_of_row_ = I(SB);
  dense_sf_base_ = I64(1);
  dense_gate_global_ = F(1);
  dense_up_global_ = F(1);
  dense_down_global_ = F(1);
  {
    std::vector<int> seq(static_cast<std::size_t>(SB));
    for (int m = 0; m < SB; ++m) seq[m] = m;
    cudaMemcpyAsync(dense_row_in_group_, seq.data(), SB * sizeof(int), cudaMemcpyHostToDevice,
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
  mlp_gate_ = A(static_cast<std::size_t>(SB) * big_inter);
  mlp_up_ = A(static_cast<std::size_t>(SB) * big_inter);
  mlp_h_ = A(static_cast<std::size_t>(SB) * big_inter);
  mlp_out_ = A(static_cast<std::size_t>(SB) * H);
  acc_ = A(static_cast<std::size_t>(SB) * H);
  logits_ = F(static_cast<std::size_t>(SB) * cfg_.vocab_size);
  scratch_f_ = F(SB);
  argmax_i_ = I(SB);
  tokens_spec_dev_ = I(SB);
  kda_spec_normed_rail_ = A(static_cast<std::size_t>(nk) * RB * H);
  kda_spec_qkv_rail_ = A(static_cast<std::size_t>(nk) * 3 * RB * qkv);
  kda_spec_gate_rail_ = A(static_cast<std::size_t>(nk) * RB * qkv);
  kda_spec_beta_rail_ = A(static_cast<std::size_t>(nk) * RB * heads_kda);
  kda_spec_on_rail_ = A(static_cast<std::size_t>(nk) * RB * qkv);
  kda_spec_cut_dev_ = I(MB);
  pos_spec_dev_ = I(SB);
  ntok_spec_dev_ = I(SB);
  npool_spec_dev_ = I(SB);
  telemetry_scratch_ = F(MB);

  pos_.assign(MB, 0);
  layer_rms_.assign(cfg_.text_layers, 0.0f);

  for (int l = 0; l < cfg_.text_layers; ++l)
    if (cfg_.layers[l].mlp == fuel::MlpKind::kSparse) moe_layer_ids_.push_back(l);
  expert_fire_.assign(static_cast<std::size_t>(cfg_.n_routed_experts), 0ull);
}

std::vector<float> DecodeEngine::last_logits(int stream) const {
  return last_logits_row(stream);
}

std::vector<float> DecodeEngine::last_logits_row(int row) const {
  if (row < 0 || row >= max_batch_ * kSpecMax) fail("logit row is outside the work buffer");
  std::vector<float> out(static_cast<std::size_t>(cfg_.vocab_size));
  cudaMemcpyAsync(out.data(), logits_ + static_cast<std::size_t>(row) * cfg_.vocab_size,
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

// KV-divergence probe (ROCKET_KV_CHECKSUM=1): fp32 sums of the paged MLA
// latent/key/gate values for logical slots [from, to) of slot-0's sequence,
// over every MLA layer. Reveals whether verify-written KV equals the plain
// path's writes.
void DecodeEngine::mla_dump_ints(const char* tag, int layer, const int* sel, const int* n_tok,
                                 int stride, int batch) {
  static bool on = std::getenv("ROCKET_LHC_CHECKSUM") != nullptr;
  if (!on) return;
  std::vector<int> hsel(static_cast<std::size_t>(batch) * stride, 0);
  std::vector<int> hn(std::size_t(batch), 0);
  cudaMemcpy(hsel.data(), sel, hsel.size() * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(hn.data(), n_tok, hn.size() * sizeof(int), cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(stream_);
  for (int m = 0; m < batch; ++m) {
    const int n = hn[m];
    long long s = 0;
    for (int i = 0; i < n && i < stride; ++i) s += hsel[static_cast<std::size_t>(m) * stride + i];
    std::string heads;
    for (int i = 0; i < 6 && i < n && i < stride; ++i)
      heads += (i ? "," : "") +
               std::to_string(hsel[static_cast<std::size_t>(m) * stride + i]);
    fprintf(stderr, "[lhc] %-11s lay%d row%d n=%d sum=%lld first=[%s]\n", tag, layer, m, n, s,
            heads.c_str());
  }
}

void DecodeEngine::mla_dump_score(const char* tag, int layer, const float* ctx, int batch,
                                  int heads) {
  static bool on = std::getenv("ROCKET_LHC_CHECKSUM") != nullptr;
  if (!on) return;
  std::vector<float> h(static_cast<std::size_t>(batch) * heads, 0.f);
  cudaMemcpy(h.data(), ctx, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(stream_);
  for (int m = 0; m < batch; ++m) {
    double s = 0.0, amax = 0.0;
    for (int hd = 0; hd < heads; ++hd) {
      double v = h[static_cast<std::size_t>(m) * heads + hd];
      s += v; amax = std::max(amax, std::fabs(v));
    }
    fprintf(stderr, "[lhc] %-11s lay%d row%d nsum=%.8g amax=%.8g\n", tag, layer, m, s, amax);
  }
}

void DecodeEngine::print_streams_checksum(int layer, const char* tag, int rows, std::size_t n,
                                          const bf16* buf, int row_stride) {
  static bool on = std::getenv("ROCKET_LHC_CHECKSUM") != nullptr;
  if (!on) return;
  static std::vector<bf16> h;
  if (!buf) buf = streams_;
  if (row_stride <= 0) row_stride = static_cast<int>(n / static_cast<std::size_t>(rows));
  h.resize(n);
  cudaStreamSynchronize(stream_);
  cudaMemcpy(h.data(), buf, n * sizeof(bf16), cudaMemcpyDeviceToHost);
  for (int r = 0; r < rows; ++r) {
    double s = 0.0, amax = 0.0;
    for (int i = 0; i < row_stride; ++i) {
      double v = __bfloat162float(h[static_cast<std::size_t>(r) * row_stride + i]);
      s += v; amax = std::max(amax, std::fabs(v));
    }
    fprintf(stderr, "[lhc] %-11s lay%d row%d buf=%.8g amax=%.8g\n", tag, layer, r, s, amax);
  }
}

void DecodeEngine::print_kv_checksum(const char* tag, int from, int to) {
  static bool kvs = std::getenv("ROCKET_KV_CHECKSUM") != nullptr;
  if (!kvs || !mla_kv_.latent) return;
  static const int at = std::getenv("ROCKET_KV_AT") ? std::atoi(std::getenv("ROCKET_KV_AT")) : -1;
  if (at >= 0 && to != at) return;
  if (at >= 0 && from < at - 1) from = at - 1;
  const int pt = mla_kv_.page_tokens;
  const int kvl = mla_kv_.kv_lora;
  const int ihd = mla_kv_.index_head_dim;
  static std::vector<bf16> hlat, hkey, hgate;
  hlat.resize(static_cast<std::size_t>(pt) * kvl);
  hkey.resize(static_cast<std::size_t>(pt) * ihd);
  hgate.resize(hkey.size());
  for (int lay = 0; lay < mla_kv_.layers; ++lay) {
    const bf16* dlat = mla_kv_.latent + static_cast<std::size_t>(lay) * pt * kvl;
    const bf16* dkey = mla_kv_.key + static_cast<std::size_t>(lay) * pt * ihd;
    const bf16* dgate = mla_kv_.gate + static_cast<std::size_t>(lay) * pt * ihd;
    cudaMemcpy(hlat.data(), dlat, hlat.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
    cudaMemcpy(hkey.data(), dkey, hkey.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
    cudaMemcpy(hgate.data(), dgate, hgate.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
    double sl = 0.0, sk = 0.0, sg = 0.0;
    for (int pos = from; pos < to; ++pos) {
      double s1 = 0.0, k1 = 0.0, g1 = 0.0;
      for (int c = 0; c < kvl; ++c) s1 += static_cast<double>(__bfloat162float(hlat[pos * kvl + c]));
      for (int c = 0; c < ihd; ++c) {
        k1 += static_cast<double>(__bfloat162float(hkey[pos * ihd + c]));
        g1 += static_cast<double>(__bfloat162float(hgate[pos * ihd + c]));
      }
      sl += s1; sk += k1; sg += g1;
      if (to - from <= 4)
        fprintf(stderr, "[kvslot] %-10s lay%d pos%d latent=%.8g key=%.8g gate=%.8g\n", tag, lay,
                pos, s1, k1, g1);
    }
    fprintf(stderr, "[kv] %-10s lay%d latent=%.8g key=%.8g gate=%.8g\n", tag, lay, sl, sk, sg);
  }
}

// State-divergence probe (ROCKET_SPEC_CHECKSUM=1): layer-0 bf16 conv +
// fp32 recurrence state sums for slot 0.
void DecodeEngine::print_kda_checksum(int layer, const char* tag) {
  static bool kcs = std::getenv("ROCKET_SPEC_CHECKSUM") != nullptr;
  if (!kcs) return;
  const int qkv = cfg_.kda_qkv_dim();
  const int taps = cfg_.conv_state_taps();
  const int hd2 = cfg_.kda_heads * cfg_.kda_head_dim * cfg_.kda_head_dim;
  const int slot = 0;
  {
    static std::vector<float> hs;
    hs.resize(hd2);
    cudaMemcpyAsync(hs.data(), kda_state_ + static_cast<std::size_t>(slot) * max_batch_ * hd2,
                    hd2 * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    double sum = 0.0;
    float mn = 3.4e38f, mx = -3.4e38f;
    for (float v : hs) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
    fprintf(stderr, "[cksum] %-12s L%d rec sum=%.8g min%.4g max%.4g\n", tag, layer, sum, mn, mx);
  }
  for (const char* nm : { "q", "k", "v" }) {
    static std::vector<bf16> hb;
    hb.resize(qkv * taps);
    const bf16* base = (nm[0] == 'q' ? q_conv_state_
                          : nm[0] == 'k' ? k_conv_state_ : v_conv_state_) +
                       static_cast<std::size_t>(slot) * max_batch_ * qkv * taps;
    cudaMemcpyAsync(hb.data(), base, hb.size() * sizeof(bf16), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    double sum = 0.0;
    for (bf16 b : hb) sum += static_cast<double>(__bfloat162float(b));
    fprintf(stderr, "[cksum] %-12s L%d %sconv sum=%.8g\n", tag, layer, nm, sum);
  }
}

void DecodeEngine::commit_positions(const std::vector<int>& accepted) {
  const int k = spec_k_;
  if (static_cast<int>(accepted.size()) != spec_batch_)
    fail("commit_positions: accepted size must equal the speculative batch");
  bool any_reject = false;
  for (int n : accepted) {
    if (n < 0 || n > k) fail("commit_positions: accepted count out of range");
    if (n < k) any_reject = true;
  }
  if (any_reject && k > kSpecMax)
    fail("wide prefill transactions must commit every position");
  for (int m = 0; m < spec_batch_; ++m) {
    pos_[m] += accepted[m];
    const int seq = kv_seq_of_slot_[m];
    if (seq < 0) fail("commit_positions: stream slot holds no KV sequence");
    kv_cache_->truncate(seq, pos_[m]);
  }
  print_kv_checksum("commit-kv", 0, pos_[0]);
  if (!any_reject) {
    // Verification wrote every active stream's final state out of place. A
    // whole-arena swap is valid only when every slot participated; otherwise
    // it would promote stale draft state for inactive slots.
    if (spec_batch_ == max_batch_) {
      std::swap(q_conv_state_, q_conv_state_draft_);
      std::swap(k_conv_state_, k_conv_state_draft_);
      std::swap(v_conv_state_, v_conv_state_draft_);
      std::swap(kda_state_, kda_state_draft_);
    } else {
      const std::size_t conv_row = static_cast<std::size_t>(cfg_.kda_qkv_dim()) *
                                   cfg_.conv_state_taps() * sizeof(bf16);
      const std::size_t rec_row = static_cast<std::size_t>(cfg_.kda_heads) * cfg_.kda_head_dim *
                                  cfg_.kda_head_dim * sizeof(float);
      for (int li = 0; li < kda_layers_; ++li) {
        const std::size_t conv_off = static_cast<std::size_t>(li) * max_batch_ * conv_row;
        const std::size_t rec_off = static_cast<std::size_t>(li) * max_batch_ * rec_row;
        for (auto [dst, src] : {std::pair<void*, const void*>(q_conv_state_, q_conv_state_draft_),
                                std::pair<void*, const void*>(k_conv_state_, k_conv_state_draft_),
                                std::pair<void*, const void*>(v_conv_state_, v_conv_state_draft_)})
          cudaMemcpyAsync(static_cast<char*>(dst) + conv_off,
                          static_cast<const char*>(src) + conv_off,
                          static_cast<std::size_t>(spec_batch_) * conv_row,
                          cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(reinterpret_cast<char*>(kda_state_) + rec_off,
                        reinterpret_cast<const char*>(kda_state_draft_) + rec_off,
                        static_cast<std::size_t>(spec_batch_) * rec_row,
                        cudaMemcpyDeviceToDevice, stream_);
      }
    }
    return;
  }
  // Verification left committed state untouched. Replay only the accepted
  // prefix in place; rejected positions never enter persistent state.
  std::vector<int> cut(max_batch_, 0);
  for (int m = 0; m < static_cast<int>(cut.size()) && m < static_cast<int>(accepted.size()); ++m)
    cut[m] = accepted[m];
  cudaMemcpyAsync(kda_spec_cut_dev_, cut.data(), cut.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream_);
  for (int l = 0; l < cfg_.text_layers; ++l) {
    if (cfg_.layers[l].attn != fuel::AttnKind::kKda) continue;
    const KdaW& kda = w_.layer(l).kda;
    const int slot = kda_slot_[l];
    const int batch = spec_batch_;
    const int rows = spec_k_ * batch;
    const int qkv = cfg_.kda_qkv_dim();
    const int heads = cfg_.kda_heads;
    const int hd = cfg_.kda_head_dim;
    const int taps = cfg_.conv_state_taps();
    const std::size_t rail_rows = static_cast<std::size_t>(kSpecMax) * max_batch_;
    const std::size_t rail_qkv = static_cast<std::size_t>(slot) * 3 * rail_rows * qkv;
    const std::size_t rail_gate = static_cast<std::size_t>(slot) * rail_rows * qkv;
    const std::size_t rail_beta = static_cast<std::size_t>(slot) * rail_rows * heads;

    const bf16* q_rail = kda_spec_qkv_rail_ + rail_qkv;
    const bf16* k_rail = kda_spec_qkv_rail_ + rail_qkv + rail_rows * qkv;
    const bf16* v_rail = kda_spec_qkv_rail_ + rail_qkv + 2 * rail_rows * qkv;
    const bf16* gate_rail = kda_spec_gate_rail_ + rail_gate;
    const bf16* beta_rail = kda_spec_beta_rail_ + rail_beta;

    bf16* qstate = q_conv_state_ + static_cast<std::size_t>(slot) * max_batch_ * qkv * taps;
    bf16* kstate = k_conv_state_ + static_cast<std::size_t>(slot) * max_batch_ * qkv * taps;
    bf16* vstate = v_conv_state_ + static_cast<std::size_t>(slot) * max_batch_ * qkv * taps;
    kda_conv_update_cut(q_conv_, qstate, q_rail, kda.conv, kda_spec_cut_dev_, batch, qkv,
                        cfg_.kda_conv_kernel, spec_k_, stream_);
    kda_conv_update_cut(k_conv_, kstate, k_rail,
                        kda.conv + static_cast<std::size_t>(qkv) * cfg_.kda_conv_kernel,
                        kda_spec_cut_dev_, batch, qkv, cfg_.kda_conv_kernel, spec_k_, stream_);
    kda_conv_update_cut(v_conv_, vstate, v_rail,
                        kda.conv + static_cast<std::size_t>(2) * qkv * cfg_.kda_conv_kernel,
                        kda_spec_cut_dev_, batch, qkv, cfg_.kda_conv_kernel, spec_k_, stream_);
    kda_norm_qk(q_conv_, k_conv_, rows, heads, hd, qkv, stream_);
    float* state = kda_state_ + static_cast<std::size_t>(slot) * max_batch_ * heads * hd * hd;
    kda_recurrent_cut(state, kda_o_, q_conv_, k_conv_, v_conv_, gate_rail, beta_rail,
                      kda_spec_cut_dev_, batch, heads, hd, qkv, spec_k_, stream_);
  }
}

void DecodeEngine::reset_slot(int slot) {
  if (slot < 0 || slot >= max_batch_) fail("reset_slot: slot out of range");
  pos_[slot] = 0;
  restored_kda_chain_[slot] = 0;
  restored_next_token_[slot] = 0;
  const int H = cfg_.hidden_size;
  const int qkv = cfg_.kda_qkv_dim();
  const int taps = cfg_.conv_state_taps();
  const int MB = max_batch_;
  for (int li = 0; li < kda_layers_; ++li) {
    const std::size_t conv_off = (static_cast<std::size_t>(li) * MB + slot) * qkv * taps;
    const std::size_t conv_bytes = static_cast<std::size_t>(qkv) * taps * sizeof(bf16);
    cuda_check(cudaMemsetAsync(q_conv_state_ + conv_off, 0, conv_bytes, stream_), "reset slot q conv");
    cuda_check(cudaMemsetAsync(k_conv_state_ + conv_off, 0, conv_bytes, stream_), "reset slot k conv");
    cuda_check(cudaMemsetAsync(v_conv_state_ + conv_off, 0, conv_bytes, stream_), "reset slot v conv");
    const std::size_t state_off = (static_cast<std::size_t>(li) * MB + slot) *
                                  cfg_.kda_heads * cfg_.kda_head_dim * cfg_.kda_head_dim;
    const std::size_t state_bytes = static_cast<std::size_t>(cfg_.kda_heads) * cfg_.kda_head_dim *
                                    cfg_.kda_head_dim * sizeof(float);
    cuda_check(cudaMemsetAsync(kda_state_ + state_off, 0, state_bytes, stream_), "reset slot kda state");
  }
  cuda_check(cudaMemsetAsync(streams_ + static_cast<std::size_t>(slot) * cfg_.hc_mult * H, 0,
                             static_cast<std::size_t>(cfg_.hc_mult) * H * sizeof(bf16), stream_),
             "reset slot streams");
  cuda_check(cudaStreamSynchronize(stream_), "reset slot sync");
  if (kv_seq_of_slot_[slot] >= 0) kv_cache_->destroy(kv_seq_of_slot_[slot]);
  const std::uint64_t seed = kv_nvme_backing_ ? kv_hash_seed(slot) : 0;
  kv_seq_of_slot_[slot] = kv_cache_->open(slot, seed);
}

void DecodeEngine::reset() {
  std::fill(pos_.begin(), pos_.end(), 0);
  std::fill(restored_kda_chain_.begin(), restored_kda_chain_.end(), 0);
  std::fill(restored_next_token_.begin(), restored_next_token_.end(), 0);
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
    const std::uint64_t seed = kv_nvme_backing_ ? kv_hash_seed(m) : 0;
    kv_seq_of_slot_[m] = kv_cache_->open(m, seed);
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

PrefixStateDigest DecodeEngine::kv_state_digest(int slot) {
  if (slot < 0 || slot >= max_batch_ || kv_seq_of_slot_[slot] < 0)
    fail("kv_state_digest: invalid slot");
  auto hash_bytes = [](std::uint64_t h, const std::uint8_t* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
  };
  PrefixStateDigest out{0xcbf29ce484222325ull, 0xcbf29ce484222325ull};
  for (int page : kv_cache_->page_table(kv_seq_of_slot_[slot])) {
    const auto bytes = kv_arena_->read_page(page);
    out.target = hash_bytes(out.target, bytes.data(), bytes.size());
  }
  kda_pack(slot, kda_stage_);
  std::vector<std::uint8_t> host(kda_bytes_per_stream());
  cuda_check(cudaMemcpy(host.data(), kda_stage_, host.size(), cudaMemcpyDeviceToHost),
             "kv_state_digest KDA");
  out.kda = hash_bytes(out.kda, host.data(), host.size());
  return out;
}

bool DecodeEngine::kv_state_equal(int a, int b) {
  if (a < 0 || b < 0 || a >= max_batch_ || b >= max_batch_ ||
      kv_seq_of_slot_[a] < 0 || kv_seq_of_slot_[b] < 0)
    fail("kv_state_equal: invalid slot");
  const auto& ap = kv_cache_->page_table(kv_seq_of_slot_[a]);
  const auto& bp = kv_cache_->page_table(kv_seq_of_slot_[b]);
  if (ap.size() != bp.size()) return false;
  for (std::size_t i = 0; i < ap.size(); ++i)
    if (kv_arena_->read_page(ap[i]) != kv_arena_->read_page(bp[i])) return false;
  kda_pack(a, kda_stage_);
  std::vector<std::uint8_t> first(kda_bytes_per_stream());
  cuda_check(cudaMemcpy(first.data(), kda_stage_, first.size(), cudaMemcpyDeviceToHost),
             "kv_state_equal first KDA");
  kda_pack(b, kda_stage_);
  std::vector<std::uint8_t> second(kda_bytes_per_stream());
  cuda_check(cudaMemcpy(second.data(), kda_stage_, second.size(), cudaMemcpyDeviceToHost),
             "kv_state_equal second KDA");
  return first == second;
}

void DecodeEngine::enable_nvme_prefix_cache(kv::NvmePrefixOptions options,
                                             std::uint64_t namespace_hash) {
  if (kv_nvme_backing_) fail("NVMe prefix cache is already enabled");
  kv_namespace_hash_ = namespace_hash;
  const kv::KvGeometry geom = kv_cache_->geometry();
  kv_nvme_backing_ = std::make_unique<kv::NvmeArenaPageBacking>(
      kv_arena_.get(), geom, std::move(options), namespace_hash, stream_);
  kv_cache_->set_backing(kv_nvme_backing_.get());
  kda_store_ = std::make_unique<kv::NvmeKdaStateStore>(
      &kv_nvme_backing_->store(), namespace_hash);
}

const kv::NvmePrefixStats* DecodeEngine::kv_nvme_stats() const {
  return kv_nvme_backing_ ? &kv_nvme_backing_->store().stats() : nullptr;
}

kv::NvmePrefixStore* DecodeEngine::kv_nvme_store() {
  return kv_nvme_backing_ ? &kv_nvme_backing_->store() : nullptr;
}

std::uint64_t DecodeEngine::kv_hash_seed(int slot) const {
  if (slot < 0 || slot >= max_batch_) fail("kv hash seed: slot out of range");
  // Grouped routed-expert accumulation is batch-row dependent. Two slots with
  // the same token prefix can therefore have different exact KV and KDA bytes.
  // Keep the slot in the durable namespace; cross-slot content sharing is a
  // value optimization and cannot precede byte-parity proof.
  return kv_namespace_hash_ ^ ((static_cast<std::uint64_t>(slot) + 1) *
                               0x9e3779b97f4a7c15ull);
}

kv::PrefixRecordKey DecodeEngine::kv_prefix_key(int slot, const int* tokens, int n_tokens,
                                                 std::uint32_t record_kind) const {
  if (!kv_nvme_backing_ || !tokens || n_tokens <= 0 ||
      n_tokens % kv_cache_->geometry().page_tokens != 0)
    fail("kv_prefix_key: invalid prefix boundary");
  const int P = kv_cache_->geometry().page_tokens;
  std::uint64_t seed = kv_hash_seed(slot), parent_record = 0, chain = 0;
  for (int at = 0; at < n_tokens; at += P) {
    chain = kv::PrefixTree::block_hash(seed, tokens + at, P);
    parent_record = at == 0 ? 0 : seed;
    seed = chain;
  }
  return kv::PrefixRecordKey{kv_namespace_hash_, parent_record, chain,
      static_cast<std::uint32_t>(kv_nvme_backing_->store().options().rank),
      static_cast<std::uint32_t>(n_tokens), record_kind, 0};
}

std::uint64_t DecodeEngine::kv_checkpoint(int slot, int next_token) {
  if (!kv_nvme_backing_) fail("kv_checkpoint: NVMe prefix cache is disabled");
  if (slot < 0 || slot >= max_batch_ || kv_seq_of_slot_[slot] < 0)
    fail("kv_checkpoint: slot holds no sequence");
  const int seq = kv_seq_of_slot_[slot];
  const int length = kv_cache_->info(seq).length;
  const int page_tokens = kv_cache_->geometry().page_tokens;
  if (length == 0 || length % page_tokens != 0)
    fail("kv_checkpoint: position must be a non-zero full-page boundary");
  if (!kv_cache_->persist(seq)) fail("kv_checkpoint: target page persistence failed");
  const std::uint64_t chain = kv_cache_->hash_at(seq, length);
  const std::uint64_t parent = length == page_tokens ? 0 :
      kv_cache_->hash_at(seq, length - page_tokens);
  kda_pack(slot, kda_stage_);
  auto* store = dynamic_cast<kv::NvmeKdaStateStore*>(kda_store_.get());
  if (!store) fail("kv_checkpoint: NVMe KDA store is unavailable");
  store->save_prefix(parent, chain, length, kda_stage_, kda_bytes_per_stream());
  cuda_check(cudaMemcpyAsync(prefix_token_dev_, &next_token, sizeof(next_token),
                             cudaMemcpyHostToDevice, stream_), "prefix token stage");
  cuda_check(cudaStreamSynchronize(stream_), "prefix token stage sync");
  const kv::PrefixRecordKey token_key{kv_namespace_hash_, parent, chain,
      static_cast<std::uint32_t>(kv_nvme_backing_->store().options().rank),
      static_cast<std::uint32_t>(length), /*record_kind=*/4, 0};
  kv_nvme_backing_->store().save(
      token_key, {{kv::PrefixComponent::kNextToken, prefix_token_dev_, sizeof(int)}});
  return chain;
}

int DecodeEngine::kv_match_shared(int slot, const int* tokens, int n_tokens) const {
  if (!kv_nvme_backing_ || !tokens || n_tokens < 0 || slot < 0 || slot >= max_batch_) return 0;
  auto* state = dynamic_cast<kv::NvmeKdaStateStore*>(kda_store_.get());
  if (!state) return 0;
  const int P = kv_cache_->geometry().page_tokens;
  std::uint64_t seed = kv_hash_seed(slot);
  int best = 0;
  for (int at = 0; at + P <= n_tokens; at += P) {
    const std::uint64_t chain = kv::PrefixTree::block_hash(seed, tokens + at, P);
    const std::uint64_t parent_record = at == 0 ? 0 : seed;
    if (!kv_nvme_backing_->holds_page(parent_record, chain, at + P)) break;
    const kv::PrefixRecordKey token_key{kv_namespace_hash_, parent_record, chain,
        static_cast<std::uint32_t>(kv_nvme_backing_->store().options().rank),
        static_cast<std::uint32_t>(at + P), /*record_kind=*/4, 0};
    if (state->holds_prefix(parent_record, chain, at + P) &&
        kv_nvme_backing_->store().holds(token_key)) best = at + P;
    seed = chain;
  }
  return best;
}

int DecodeEngine::kv_open_shared(int slot, const int* tokens, int n_tokens,
                                 int* next_token_out) {
  if (!kv_nvme_backing_) fail("kv_open_shared: NVMe prefix cache is disabled");
  if (slot < 0 || slot >= max_batch_) fail("kv_open_shared: slot out of range");
  if (!tokens || n_tokens < 0) fail("kv_open_shared: invalid tokens");
  auto* state = dynamic_cast<kv::NvmeKdaStateStore*>(kda_store_.get());
  if (!state) fail("kv_open_shared: NVMe KDA store is unavailable");
  const int P = kv_cache_->geometry().page_tokens;
  const int best = kv_match_shared(slot, tokens, n_tokens);
  std::uint64_t seed = kv_hash_seed(slot);
  std::uint64_t best_parent = 0;
  std::uint64_t best_chain = 0;
  for (int at = 0; at < best; at += P) {
    best_chain = kv::PrefixTree::block_hash(seed, tokens + at, P);
    best_parent = at == 0 ? 0 : seed;
    seed = best_chain;
  }
  if (kv_seq_of_slot_[slot] >= 0) kv_cache_->destroy(kv_seq_of_slot_[slot]);
  int matched = 0;
  const int seq = kv_cache_->open_shared(slot, tokens, best, &matched, kv_hash_seed(slot));
  kv_seq_of_slot_[slot] = seq;
  if (matched != best) {
    kv_cache_->destroy(seq);
    kv_seq_of_slot_[slot] = kv_cache_->open(slot, kv_hash_seed(slot));
    kv_arena_->upload_table(slot, {});
    pos_[slot] = 0;
    return 0;
  }
  if (best > 0) {
    const int source = max_batch_ >= 16 ? (slot / 8) * 8 : 0;
    if (source != slot && restored_kda_chain_[source] == best_chain) {
      kda_copy_slot(slot, source);
      if (next_token_out) *next_token_out = restored_next_token_[source];
    } else {
      if (!state->load_prefix(best_parent, best_chain, best, kda_stage_,
                              kda_bytes_per_stream())) {
        kv_cache_->destroy(seq);
        kv_seq_of_slot_[slot] = kv_cache_->open(slot, kv_hash_seed(slot));
        kv_arena_->upload_table(slot, {});
        pos_[slot] = 0;
        return 0;
      }
      kda_unpack(slot, kda_stage_);
      int cached_next = 0;
      const kv::PrefixRecordKey token_key{kv_namespace_hash_, best_parent, best_chain,
          static_cast<std::uint32_t>(kv_nvme_backing_->store().options().rank),
          static_cast<std::uint32_t>(best), /*record_kind=*/4, 0};
      if (!kv_nvme_backing_->store().load(
              token_key, {{kv::PrefixComponent::kNextToken, prefix_token_dev_, sizeof(int)}})) {
        kv_cache_->destroy(seq);
        kv_seq_of_slot_[slot] = kv_cache_->open(slot, kv_hash_seed(slot));
        kv_arena_->upload_table(slot, {});
        pos_[slot] = 0;
        return 0;
      }
      cuda_check(cudaMemcpyAsync(&cached_next, prefix_token_dev_, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream_), "prefix token restore");
      cuda_check(cudaStreamSynchronize(stream_), "prefix token restore sync");
      restored_kda_chain_[slot] = best_chain;
      restored_next_token_[slot] = cached_next;
      if (next_token_out) *next_token_out = cached_next;
    }
  }
  kv_arena_->upload_table(slot, kv_cache_->page_table(seq));
  pos_[slot] = best;
  return best;
}

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
  static const bool debug_kda = std::getenv("ROCKET_DEBUG_KDA") != nullptr;
  const bool kdbg = debug_kda && layer == 0;
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
  const int tc_batch = max_batch_ * kSpecMax;

  static const bool tc_proj = std::getenv("ROCKET_KDA_CUBLAS") != nullptr;
  if (tc_proj && k.qkv_fp4.packed == nullptr && k.qkv_fp8.packed == nullptr &&
      k.q_overlay.packed == nullptr) {
    gemm_bf16_cublas(q_raw_, k.qkv, normed_, tc_batch, qkv, H, stream_);
    gemm_bf16_cublas(k_raw_, k.qkv + static_cast<std::size_t>(qkv) * H, normed_, tc_batch, qkv, H,
                     stream_);
    gemm_bf16_cublas(v_raw_, k.qkv + static_cast<std::size_t>(2) * qkv * H, normed_, tc_batch, qkv, H,
                     stream_);
  } else if (k.qkv_fp4.packed != nullptr) {
    const std::size_t packed_stride = static_cast<std::size_t>(qkv) * H / 2;
    const std::size_t scale_stride = static_cast<std::size_t>(qkv) * H / 16;
    for (int m = 0; m < batch; ++m) {
      const bf16* x = normed_ + static_cast<std::size_t>(m) * H;
      gemv_nvfp4(q_raw_ + static_cast<std::size_t>(m) * qkv,
                 k.qkv_fp4.packed, k.qkv_fp4.scale, k.qkv_fp4.global, x, qkv, H, stream_);
      gemv_nvfp4(k_raw_ + static_cast<std::size_t>(m) * qkv,
                 k.qkv_fp4.packed + packed_stride, k.qkv_fp4.scale + scale_stride,
                 k.qkv_fp4.global, x, qkv, H, stream_);
      gemv_nvfp4(v_raw_ + static_cast<std::size_t>(m) * qkv,
                 k.qkv_fp4.packed + 2 * packed_stride, k.qkv_fp4.scale + 2 * scale_stride,
                 k.qkv_fp4.global, x, qkv, H, stream_);
    }
  } else if (k.qkv_fp8.packed != nullptr) {
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
    gemm_bf16(k_raw_, k.kv, normed_, batch, qkv, H, stream_);
    gemm_bf16(v_raw_, k.kv + static_cast<std::size_t>(qkv) * H, normed_, batch, qkv, H, stream_);
  } else {
    gemm_bf16(q_raw_, k.qkv, normed_, batch, qkv, H, stream_);
    gemm_bf16(k_raw_, k.qkv + static_cast<std::size_t>(qkv) * H, normed_, batch, qkv, H,
              stream_);
    gemm_bf16(v_raw_, k.qkv + static_cast<std::size_t>(2) * qkv * H, normed_, batch, qkv, H,
              stream_);
  }
  k_mark("qkv_gemm");
  if (telemetry_) {
    const std::string p = "layer" + std::to_string(layer) + ".";
    record_absmax(p + "kda_q.output", q_raw_, batch, qkv);
    record_absmax(p + "kda_k.output", k_raw_, batch, qkv);
    record_absmax(p + "kda_v.output", v_raw_, batch, qkv);
  }

  bf16* qstate = (kda_use_draft_state_ ? q_conv_state_draft_ : q_conv_state_) +
                 static_cast<std::size_t>(slot) * MB * qkv * taps;
  bf16* kstate = (kda_use_draft_state_ ? k_conv_state_draft_ : k_conv_state_) +
                 static_cast<std::size_t>(slot) * MB * qkv * taps;
  bf16* vstate = (kda_use_draft_state_ ? v_conv_state_draft_ : v_conv_state_) +
                 static_cast<std::size_t>(slot) * MB * qkv * taps;
  kda_conv_update(q_conv_, qstate, q_raw_, k.conv, batch, qkv, cfg_.kda_conv_kernel, stream_);
  kda_conv_update(k_conv_, kstate, k_raw_, k.conv + static_cast<std::size_t>(qkv) * cfg_.kda_conv_kernel,
                  batch, qkv, cfg_.kda_conv_kernel, stream_);
  kda_conv_update(v_conv_, vstate, v_raw_,
                  k.conv + static_cast<std::size_t>(2) * qkv * cfg_.kda_conv_kernel, batch, qkv,
                  cfg_.kda_conv_kernel, stream_);
  kda_norm_qk(q_conv_, k_conv_, batch, heads, hd, /*row_stride=*/qkv, stream_);
  k_mark("conv3+normqk");
  static bool lhcxp = std::getenv("ROCKET_LHC_CHECKSUM") != nullptr;
  const bool dbg3p = lhcxp && layer == 3;
  if (dbg3p) {
    print_streams_checksum(layer, "plain-kda1", batch, static_cast<std::size_t>(batch) * qkv,
                           q_raw_, qkv);
    print_streams_checksum(layer, "plain-kda1c", batch, static_cast<std::size_t>(batch) * qkv,
                           q_conv_, qkv);
  }
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
  if (dbg3p)
    print_streams_checksum(layer, "plain-kda2", batch, static_cast<std::size_t>(batch) * qkv,
                           gate_, qkv);
  k_mark("gates");

  gemm_bf16(beta_raw_, k.b_proj, normed_, batch, heads, H, stream_);
  kda_sigmoid(beta_, beta_raw_, batch * heads, stream_);
  k_mark("beta");

  float* state = (kda_use_draft_state_ ? kda_state_draft_ : kda_state_) +
                 static_cast<std::size_t>(slot) * MB * heads * hd * hd;
  kda_recurrent_step(state, kda_o_, q_conv_, k_conv_, v_conv_, gate_, beta_, batch, heads, hd,
                     /*row_stride=*/qkv, stream_);
  k_mark("recurrent");
  if (dbg3p) {
    print_streams_checksum(layer, "plain-kda3", batch, static_cast<std::size_t>(batch) * qkv,
                           kda_o_, qkv);
    static std::vector<float> hs(static_cast<std::size_t>(kSpecMax) * heads * hd * hd);
    cudaMemcpy(hs.data(), state, static_cast<std::size_t>(heads * hd * hd) * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream_);
    double ssum = 0.0;
    for (int i = 0; i < heads * hd * hd; ++i) ssum += static_cast<double>(hs[i]);
    fprintf(stderr, "[lhc] %-11s lay%d row%d state=%.8g\n", "plain-kda3s", layer, 0, ssum);
  }
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
  if (tc_proj && k.o_proj_fp8.packed == nullptr)
    gemm_bf16_cublas(sublayer_out_, k.o_proj, kda_on_, tc_batch, H, qkv, stream_);
  else if (k.o_proj_fp8.packed != nullptr)
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

// Spec KDA site: position-major [K * batch] rows, exactly as the verify
// batches them. With cut=nullptr the chained states end at the final position
// and land in the draft buffers; with a cut the kernels dump the committed
// state at the per-row accept boundary and the replay runs against the real
// buffers. Only the conv and the recurrence chain state; everything else is
// a plain [rows, width] batched call.
void DecodeEngine::run_kda_spec_site(int layer, int positions, int batch, const int* cut) {
  static const bool debug_kda = std::getenv("ROCKET_DEBUG_KDA") != nullptr;
  const bool kprof = debug_kda && layer == 0;
  const auto k_t0 = Clock::now();
  auto k_mark = [&](const char* what) {
    if (!kprof) return;
    cudaStreamSynchronize(stream_);
    std::printf("[kda-spec-prof] %-18s %8.3f ms\n", what,
                std::chrono::duration<double, std::milli>(Clock::now() - k_t0).count());
  };
  if (std::getenv("ROCKET_LHC_CHECKSUM") && layer == 3)
    fprintf(stderr, "[site-hit] L3 pos=%d batch=%d cut=%p\n", positions, batch, (const void*)cut);
  const KdaW& k = w_.layer(layer).kda;
  const int H = cfg_.hidden_size;
  const int qkv = cfg_.kda_qkv_dim();
  const int hd = cfg_.kda_head_dim;
  const int taps = cfg_.conv_state_taps();
  const int heads = cfg_.kda_heads;
  const int rows = positions * batch;
  const bool wide_prefill = positions > kSpecMax;
  const int tc_rows = max_batch_ * (wide_prefill ? max_work_k_ : kSpecMax);
  const int MB = max_batch_;

  const bool draft = (cut == nullptr);
  const int* chain_cut = cut;
  // Verification is transactional: read committed state and write the
  // speculative final state to the alternate arena. Cut/replay stays in place.
  bf16* qstate_in = q_conv_state_ + static_cast<std::size_t>(kda_slot_[layer]) * MB * qkv * taps;
  bf16* kstate_in = k_conv_state_ + static_cast<std::size_t>(kda_slot_[layer]) * MB * qkv * taps;
  bf16* vstate_in = v_conv_state_ + static_cast<std::size_t>(kda_slot_[layer]) * MB * qkv * taps;
  bf16* qstate_out = draft ? q_conv_state_draft_ + static_cast<std::size_t>(kda_slot_[layer]) * MB * qkv * taps : qstate_in;
  bf16* kstate_out = draft ? k_conv_state_draft_ + static_cast<std::size_t>(kda_slot_[layer]) * MB * qkv * taps : kstate_in;
  bf16* vstate_out = draft ? v_conv_state_draft_ + static_cast<std::size_t>(kda_slot_[layer]) * MB * qkv * taps : vstate_in;
  const int slot = kda_slot_[layer];

  const std::size_t rail_rows = static_cast<std::size_t>(kSpecMax) * max_batch_;
  const std::size_t rail_qkv = static_cast<std::size_t>(slot) * 3 * rail_rows * qkv;
  bf16* q_work = wide_prefill ? q_raw_ : (draft ? kda_spec_qkv_rail_ + rail_qkv : q_raw_);
  bf16* k_work =
      wide_prefill ? k_raw_ : (draft ? kda_spec_qkv_rail_ + rail_qkv + rail_rows * qkv : k_raw_);
  bf16* v_work = wide_prefill
                     ? v_raw_
                     : (draft ? kda_spec_qkv_rail_ + rail_qkv + 2 * rail_rows * qkv : v_raw_);

  static const bool tc_proj = std::getenv("ROCKET_KDA_CUBLAS") != nullptr;
  if (tc_proj && k.qkv_fp4.packed == nullptr && k.qkv_fp8.packed == nullptr &&
      k.q_overlay.packed == nullptr) {
    gemm_bf16_cublas(q_work, k.qkv, normed_, tc_rows, qkv, H, stream_);
    gemm_bf16_cublas(k_work, k.qkv + static_cast<std::size_t>(qkv) * H, normed_, tc_rows, qkv, H,
                     stream_);
    gemm_bf16_cublas(v_work, k.qkv + static_cast<std::size_t>(2) * qkv * H, normed_, tc_rows, qkv, H,
                     stream_);
  } else if (k.qkv_fp4.packed != nullptr) {
    const std::size_t packed_stride = static_cast<std::size_t>(qkv) * H / 2;
    const std::size_t scale_stride = static_cast<std::size_t>(qkv) * H / 16;
    for (int r = 0; r < rows; ++r) {
      const bf16* x = normed_ + static_cast<std::size_t>(r) * H;
      gemv_nvfp4(q_work + static_cast<std::size_t>(r) * qkv,
                 k.qkv_fp4.packed, k.qkv_fp4.scale, k.qkv_fp4.global, x, qkv, H, stream_);
      gemv_nvfp4(k_work + static_cast<std::size_t>(r) * qkv,
                 k.qkv_fp4.packed + packed_stride, k.qkv_fp4.scale + scale_stride,
                 k.qkv_fp4.global, x, qkv, H, stream_);
      gemv_nvfp4(v_work + static_cast<std::size_t>(r) * qkv,
                 k.qkv_fp4.packed + 2 * packed_stride, k.qkv_fp4.scale + 2 * scale_stride,
                 k.qkv_fp4.global, x, qkv, H, stream_);
    }
  } else if (k.qkv_fp8.packed != nullptr) {
    gemm_fp8_row(q_work, k.qkv_fp8.packed, k.qkv_fp8.scales, normed_, rows, qkv, H, 0, stream_);
    gemm_fp8_row(k_work, k.qkv_fp8.packed, k.qkv_fp8.scales, normed_, rows, qkv, H, qkv, stream_);
    gemm_fp8_row(v_work, k.qkv_fp8.packed, k.qkv_fp8.scales, normed_, rows, qkv, H, 2 * qkv, stream_);
  } else if (k.q_overlay.packed) {
    for (int r = 0; r < rows; ++r)
      gemv_nvfp4(q_work + static_cast<std::size_t>(r) * qkv, k.q_overlay.packed,
                 k.q_overlay.scale, k.q_overlay.global,
                 normed_ + static_cast<std::size_t>(r) * H, qkv, H, stream_);
    gemm_bf16(k_work, k.kv, normed_, rows, qkv, H, stream_);
    gemm_bf16(v_work, k.kv + static_cast<std::size_t>(qkv) * H, normed_, rows, qkv, H, stream_);
  } else {
    gemm_bf16(q_work, k.qkv, normed_, rows, qkv, H, stream_);
    gemm_bf16(k_work, k.qkv + static_cast<std::size_t>(qkv) * H, normed_, rows, qkv, H, stream_);
    gemm_bf16(v_work, k.qkv + static_cast<std::size_t>(2) * qkv * H, normed_, rows, qkv, H, stream_);
  }
  k_mark("qkv projections");
  kda_conv_update_chunk_oop(q_conv_, qstate_in, qstate_out, q_work, k.conv, batch, qkv,
                            cfg_.kda_conv_kernel, positions, stream_);
  kda_conv_update_chunk_oop(k_conv_, kstate_in, kstate_out, k_work,
                            k.conv + static_cast<std::size_t>(qkv) * cfg_.kda_conv_kernel,
                            batch, qkv, cfg_.kda_conv_kernel, positions, stream_);
  kda_conv_update_chunk_oop(v_conv_, vstate_in, vstate_out, v_work,
                            k.conv + static_cast<std::size_t>(2) * qkv * cfg_.kda_conv_kernel,
                            batch, qkv, cfg_.kda_conv_kernel, positions, stream_);
  kda_norm_qk(q_conv_, k_conv_, rows, heads, hd, /*row_stride=*/qkv, stream_);
  k_mark("conv3 + qk norm");
  static bool lhcx = std::getenv("ROCKET_LHC_CHECKSUM") != nullptr;
  const bool dbg3 = lhcx && layer == 3 && draft;
  if (dbg3) print_streams_checksum(layer, "spec-kda1", rows, static_cast<std::size_t>(rows) * qkv,
                                   q_work, qkv);
  const std::size_t rail_gate = static_cast<std::size_t>(slot) * rail_rows * qkv;
  const std::size_t rail_beta = static_cast<std::size_t>(slot) * rail_rows * heads;
  bf16* gate_work = wide_prefill ? gate_ : (draft ? kda_spec_gate_rail_ + rail_gate : gate_);
  bf16* beta_work = wide_prefill ? beta_ : (draft ? kda_spec_beta_rail_ + rail_beta : beta_);
  gemm_bf16(lr_a_, k.f_a, normed_, rows, hd, H, stream_);
  gemm_bf16(lr_b_, k.f_b, lr_a_, rows, qkv, hd, stream_);
  kda_forget_gate(gate_work, lr_b_, k.dt_bias, k.a_log, rows, heads, hd,
                  cfg_.kda_gate_lower_bound, stream_);
  k_mark("decay gate");
  if (dbg3) print_streams_checksum(layer, "spec-kda2", rows, static_cast<std::size_t>(rows) * qkv,
                                   gate_work, qkv);
  gemm_bf16(beta_raw_, k.b_proj, normed_, rows, heads, H, stream_);
  kda_sigmoid(beta_work, beta_raw_, rows * heads, stream_);
  k_mark("beta");
  float* state_in = kda_state_ + static_cast<std::size_t>(slot) * MB * heads * hd * hd;
  float* state_out = draft ? kda_state_draft_ + static_cast<std::size_t>(slot) * MB * heads * hd * hd
                           : state_in;
  if (draft) {
    kda_recurrent_chunk_oop(state_in, state_out, kda_o_, q_conv_, k_conv_, v_conv_, gate_work,
                            beta_work, batch, heads, hd, /*row_stride=*/qkv, positions, stream_);
  } else {
    kda_recurrent_cut(state_in, kda_o_, q_conv_, k_conv_, v_conv_, gate_work, beta_work,
                      chain_cut, batch, heads, hd, /*row_stride=*/qkv, positions, stream_);
  }
  k_mark("recurrent");
  if (dbg3) {
    print_streams_checksum(layer, "spec-kda3", rows, static_cast<std::size_t>(rows) * qkv,
                           kda_o_, qkv);
    if (batch == 1) {
      static std::vector<float> hs(static_cast<std::size_t>(kSpecMax) * heads * hd * hd);
      cudaMemcpy(hs.data(), state_out, hs.size() * sizeof(float), cudaMemcpyDeviceToHost);
      cudaStreamSynchronize(stream_);
      for (int r = 0; r < rows; ++r) {
        double s = 0.0;
        for (int i = 0; i < heads * hd * hd; ++i)
          s += static_cast<double>(hs[static_cast<std::size_t>(r) * heads * hd * hd + i]);
        fprintf(stderr, "[lhc] %-11s lay%d row%d state=%.8g\n", "spec-kda3s", layer, r, s);
      }
    }
  }
  gemm_bf16(lr_a_, k.g_a, normed_, rows, hd, H, stream_);
  gemm_bf16(lr_b_, k.g_b, lr_a_, rows, qkv, hd, stream_);
  kda_gated_norm(kda_on_, kda_o_, lr_b_, k.o_norm, rows, heads, hd, cfg_.rms_norm_eps, stream_);
  k_mark("output gate + norm");
  if (dbg3) print_streams_checksum(layer, "spec-kda4", rows, static_cast<std::size_t>(rows) * qkv,
                                   kda_on_, qkv);
  if (tc_proj && k.o_proj_fp8.packed == nullptr)
    gemm_bf16_cublas(sublayer_out_, k.o_proj, kda_on_, tc_rows, H, qkv, stream_);
  else if (k.o_proj_fp8.packed != nullptr)
    gemm_fp8_row(sublayer_out_, k.o_proj_fp8.packed, k.o_proj_fp8.scales, kda_on_, rows, H, qkv,
                 0, stream_);
  else
    gemm_bf16(sublayer_out_, k.o_proj, kda_on_, rows, H, qkv, stream_);
  k_mark("output projection");
}

void DecodeEngine::run_mla(int layer, int slot, int batch, const int* n_tokens_dev,
                           int n_pools_max, int n_streams) {
  const MlaW& m = w_.layer(layer).mla;
  const int H = cfg_.hidden_size;
  const int heads = cfg_.mla_heads;
  const int kvl = cfg_.kv_lora_rank;
  const int ihd = cfg_.index_head_dim;
  const int ih = cfg_.index_n_heads;
  const int kpool = cfg_.index_kpool;
  static const bool tc_all = std::getenv("ROCKET_CUBLAS_ALL") != nullptr;
  const int tc_batch = max_batch_ * (spec_k_ > kSpecMax ? max_work_k_ : kSpecMax);

  if (tc_all && m.q_a_fp8.packed == nullptr)
    gemm_bf16_cublas(q_resid_raw_, m.q_a, normed_, tc_batch, cfg_.q_lora_rank, H, stream_);
  else if (m.q_a_fp8.packed != nullptr)
    gemm_fp8_row(q_resid_raw_, m.q_a_fp8.packed, m.q_a_fp8.scales, normed_, batch,
                 cfg_.q_lora_rank, H, 0, stream_);
  else
    gemm_bf16(q_resid_raw_, m.q_a, normed_, batch, cfg_.q_lora_rank, H, stream_);
  rmsnorm(q_resid_, q_resid_raw_, m.q_a_norm, batch, cfg_.q_lora_rank, cfg_.rms_norm_eps, stream_);
  if (tc_all && m.q_b_fp8.packed == nullptr)
    gemm_bf16_cublas(q_, m.q_b, q_resid_, tc_batch, heads * cfg_.qk_head_dim(),
                     cfg_.q_lora_rank, stream_);
  else if (m.q_b_fp8.packed != nullptr)
    gemm_fp8_row(q_, m.q_b_fp8.packed, m.q_b_fp8.scales, q_resid_, batch,
                 heads * cfg_.qk_head_dim(), cfg_.q_lora_rank, 0, stream_);
  else
    gemm_bf16(q_, m.q_b, q_resid_, batch, heads * cfg_.qk_head_dim(), cfg_.q_lora_rank, stream_);
  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".mla_q.output", q_, batch,
                  heads * cfg_.qk_head_dim());

  if (tc_all && m.kv_a_fp8.packed == nullptr)
    gemm_bf16_cublas(ckv_, m.kv_a, normed_, tc_batch, kvl, H, stream_);
  else if (m.kv_a_fp8.packed != nullptr)
    gemm_fp8_row(ckv_, m.kv_a_fp8.packed, m.kv_a_fp8.scales, normed_, batch, kvl, H, 0, stream_);
  else
    gemm_bf16(ckv_, m.kv_a, normed_, batch, kvl, H, stream_);
  if (telemetry_)
    record_absmax("layer" + std::to_string(layer) + ".mla_kv.output", ckv_, batch, kvl);
  rmsnorm(latent_stage_, ckv_, m.kv_a_norm, batch, kvl, cfg_.rms_norm_eps, stream_);

  if (tc_all && m.idx_wq_b_fp8.packed == nullptr)
    gemm_bf16_cublas(q_idx_, m.idx_wq_b, q_resid_, tc_batch, ih * ihd, cfg_.q_lora_rank, stream_);
  else if (m.idx_wq_b_fp8.packed != nullptr)
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

  kv_write_latent(mla_kv_, latent_stage_, pos_dev_, batch, n_streams, slot, stream_);
  kv_write_index(mla_kv_, idx_k_stage_, idx_g_stage_, pos_dev_, batch, n_streams, slot, stream_);
  static bool lhcx = std::getenv("ROCKET_LHC_CHECKSUM") != nullptr;
  const bool dbg3 = lhcx && layer == 3;
  if (dbg3) {
    print_streams_checksum(layer, "mla-w", batch, static_cast<std::size_t>(batch) * kvl,
                           latent_stage_, kvl);
  }

  if (n_pools_max > 0) {
  indexer_pool_keys(pool_keys_, mla_kv_, m.idx_ape, n_pools_dev_, n_pools_max, pool_stride_,
                    batch, n_streams, slot, kpool, ihd, stream_);
    if (dbg3)
      print_streams_checksum(layer, "mla-pk", batch,
                             static_cast<std::size_t>(std::max(batch, 1)) * ihd, pool_keys_, ihd);
    indexer_scores(pool_scores_, q_idx_, pool_keys_, head_w_, n_pools_dev_, n_pools_max,
                   pool_stride_, batch, ih, ihd, stream_);
    if (dbg3) {
      std::vector<int> hnp(static_cast<std::size_t>(batch), 0);
      cudaMemcpy(hnp.data(), n_pools_dev_, hnp.size() * sizeof(int), cudaMemcpyDeviceToHost);
      cudaStreamSynchronize(stream_);
      for (int m = 0; m < batch; ++m)
        fprintf(stderr, "[lhc] %-11s lay%d row%d npool=%d npmax=%d\n", "mla-np", layer, m, hnp[m],
                n_pools_max);
    }
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
  if (!std::getenv("ROCKET_MLA_LEGACY_SPLIT")) {
    mla_fused_context(ctx_, q_abs_, mla_kv_, sel_tokens_, n_tok_, sel_stride_, batch,
                      n_streams, slot, heads, kvl, scaling, stream_);
  } else {
    mla_scores(scores_, q_abs_, mla_kv_, sel_tokens_, n_tok_, sel_stride_, sel_stride_, batch,
               n_streams, slot, heads, kvl, scaling, stream_);
    mla_softmax(scores_, n_tok_, sel_stride_, batch, heads, stream_);
    mla_context(ctx_, scores_, mla_kv_, sel_tokens_, n_tok_, sel_stride_, sel_stride_, batch,
                n_streams, slot, heads, kvl, stream_);
  }
  if (dbg3) {
    mla_dump_ints("mla-sel", layer, sel_tokens_, n_tok_, sel_stride_, batch);
    mla_dump_score("mla-ctx", layer, ctx_, batch, heads);
  }
  mla_expand_v(v_out_, m.kv_b, ctx_, batch, heads, cfg_.qk_nope_head_dim, cfg_.v_head_dim, kvl,
              stream_);
  if (tc_all && m.o_proj_fp8.packed == nullptr)
    gemm_bf16_cublas(sublayer_out_, m.o_proj, v_out_, tc_batch, H, heads * cfg_.v_head_dim, stream_);
  else if (m.o_proj_fp8.packed != nullptr)
    gemm_fp8_row(sublayer_out_, m.o_proj_fp8.packed, m.o_proj_fp8.scales, v_out_, batch, H,
                 heads * cfg_.v_head_dim, 0, stream_);
  else
    gemm_bf16(sublayer_out_, m.o_proj, v_out_, batch, H, heads * cfg_.v_head_dim, stream_);
  if (dbg3) {
    print_streams_checksum(layer, "mla-vout", batch,
                           static_cast<std::size_t>(batch) * heads * cfg_.v_head_dim, v_out_,
                           heads * cfg_.v_head_dim);
    print_streams_checksum(layer, "mla-out", batch, static_cast<std::size_t>(batch) * H,
                           sublayer_out_, H);
  }
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
    if (std::getenv("ROCKET_MOE_TRACE") && (gemv_fallback || layer == 3))
      std::fprintf(stderr, "[moe-gemv] ep=%d layer=%d rows=%d own_n=%d fallback=%d\n",
                   ep_ != nullptr, layer, rows, own_n, static_cast<int>(gemv_fallback));
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

void DecodeEngine::run_moe(int layer, int batch) {
  static const bool debug_moe = std::getenv("ROCKET_DEBUG_MOE") != nullptr;
  static const bool spec_debug = std::getenv("ROCKET_SPEC_DEBUG") != nullptr;
  const bool mdbg = debug_moe && layer == 4;
  const auto m_t0 = Clock::now();
  auto m_mark = [&](const char* what) {
    if (!mdbg) return;
    cudaStreamSynchronize(stream_);
    std::printf("[moe-prof] %-14s %8.3f ms\n", what,
                std::chrono::duration<double, std::milli>(Clock::now() - m_t0).count());
  };
  const MoeW& mo = w_.layer(layer).moe;
  const int H = cfg_.hidden_size;
  const int K = cfg_.num_experts_per_tok;

  static bool mtrace = std::getenv("ROCKET_MOE_TRACE") != nullptr;
  if (mtrace) {
    // Sentinel fill BEFORE the gemm: rows still holding 0xA5 bytes after the
    // gemm prove the gemm never wrote them.
    cudaMemsetAsync(router_logits_, 0xA5,
                    static_cast<std::size_t>(max_batch_ * max_work_k_) * cfg_.n_routed_experts *
                        sizeof(float),
                    stream_);
    cudaStreamSynchronize(stream_);
  }
  // Exactly one router projection per MoE layer. A second identical launch
  // here used to overwrite this result and cost 42 redundant GEMMs per full
  // GLM-5.3 forward.
  gemm_bf16_f32(router_logits_, mo.router, normed_, batch, cfg_.n_routed_experts, H, stream_);
  static bool rowwise_router = std::getenv("ROCKET_ROUTER_ROWWISE") != nullptr;
  if (rowwise_router) {
    // Band-aid probe: force the proven M=1 kernel per row for the router.
    for (int mp = 0; mp < batch; ++mp)
      gemm_bf16_f32(router_logits_ + static_cast<std::size_t>(mp) * cfg_.n_routed_experts,
                    mo.router, normed_ + static_cast<std::size_t>(mp) * H, 1,
                    cfg_.n_routed_experts, H, stream_);
  }

  moe_router(topk_idx_, topk_w_, router_logits_, mo.router_bias, batch, cfg_.n_routed_experts, K,
            cfg_.norm_topk_prob, cfg_.routed_scaling_factor, stream_);
  if (spec_debug) cuda_check(cudaGetLastError(), "spec moe router");
  if (mtrace) {
    static std::vector<float> rl(std::size_t(max_batch_) * cfg_.n_routed_experts, 0.f);
    static std::vector<bf16> nx(std::size_t(max_batch_ * max_work_k_) * cfg_.hidden_size);
    static std::vector<bf16> chead_buf(std::size_t(max_batch_ * max_work_k_) * cfg_.hidden_size);
    std::vector<int> tidx(std::size_t(batch) * K);
    std::vector<float> twts(std::size_t(batch) * K);
    cudaMemcpyAsync(rl.data(), router_logits_, rl.size() * sizeof(float), cudaMemcpyDeviceToHost,
                    stream_);
    cudaMemcpyAsync(nx.data(), normed_, nx.size() * sizeof(bf16), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(chead_buf.data(), collapsed_, chead_buf.size() * sizeof(bf16),
                    cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(rl.data(), router_logits_, rl.size() * sizeof(float), cudaMemcpyDeviceToHost,
                    stream_);
    cudaMemcpyAsync(tidx.data(), topk_idx_, tidx.size() * sizeof(int), cudaMemcpyDeviceToHost,
                    stream_);
    cudaMemcpyAsync(twts.data(), topk_w_, twts.size() * sizeof(float), cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronize(stream_);
    for (int m = 0; m < batch; ++m) {
      double s = 0.0, ns = 0.0, amax = 0.0;
      float lamax = 0.f;
      for (int e = 0; e < cfg_.n_routed_experts; ++e) {
        float v = rl[static_cast<std::size_t>(m) * cfg_.n_routed_experts + e];
        s += static_cast<double>(v);
        lamax = std::fmaxf(lamax, std::fabs(v));
      }
      for (int h = 0; h < H; ++h) {
        float v = __bfloat162float(nx[static_cast<std::size_t>(m) * H + h]);
        ns += v; amax = std::max(amax, double(std::fabs(v)));
      }
      std::string heads;
      for (int e = 0; e < 6; ++e) {
        heads += (e ? "," : "") +
                 std::to_string(rl[static_cast<std::size_t>(m) * cfg_.n_routed_experts + e])
                     .substr(0, 7);
      }
      std::string nhead, chead;
      for (int h = 0; h < 6; ++h)
        nhead += (h ? "," : "") +
                 std::to_string(__bfloat162float(nx[static_cast<std::size_t>(m) * H + h]))
                     .substr(0, 7);
      for (int h = 0; h < 6; ++h)
        chead += (h ? "," : "") +
                 std::to_string(__bfloat162float(chead_buf[static_cast<std::size_t>(m) * H + h]))
                     .substr(0, 7);
      std::string ids;
      for (int k = 0; k < K; ++k) {
        ids += (k ? "," : "") +
               std::to_string(tidx[static_cast<std::size_t>(m) * K + k]) + ":" +
               std::to_string(twts[static_cast<std::size_t>(m) * K + k]).substr(0, 6);
      }
      fprintf(stderr,
              "[moe] L%d row%d b=%d normed(s=%.4g amax=%.4g n0..5=[%s] c0..5=[%s]) "
              "llog(amax=%.4g e0..5=[%s]) lsum=%.8g ew=[%s]\n",
              layer, m, batch, ns, amax, nhead.c_str(), chead.c_str(), lamax, heads.c_str(), s,
              ids.c_str());
    }
  }

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

  cudaMemsetAsync(acc_, 0, static_cast<std::size_t>(batch) * H * sizeof(bf16), stream_);
  if (spec_debug) cuda_check(cudaGetLastError(), "spec moe memset");
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
  if (spec_debug) cuda_check(cudaGetLastError(), "spec moe grouped");

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
  if (spec_debug)
    cuda_check(cudaGetLastError(), ("spec moe L" + std::to_string(layer)).c_str());

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
    run_mla(layer, mla_slot_[layer], batch, n_tokens_dev_, n_pools_launch, batch);
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

// Shared experts remain BF16. The measured NVFP4 replacement increased
// WikiText perplexity by 12.1% for a 1.4% B1 throughput gain.
void DecodeEngine::run_shared_mlp(int layer, int batch) {
  const MoeW& mo = w_.layer(layer).moe;
  const int H = cfg_.hidden_size;
  const int MI = cfg_.moe_intermediate_size;
  const int SI = MI * cfg_.n_shared_experts;
  const DenseMlpW& sh = mo.shared;
  gemm_bf16(mlp_gate_, sh.gate, normed_, batch, SI, H, stream_);
  gemm_bf16(mlp_up_, sh.up, normed_, batch, SI, H, stream_);
  swiglu_clamped(mlp_h_, mlp_gate_, mlp_up_, batch * SI, cfg_.swiglu_limit, stream_);
  gemm_bf16(mlp_out_, sh.down, mlp_h_, batch, H, SI, stream_);
}

void DecodeEngine::run_moe_post_stage(int layer, int batch) {
  const MoeW& mo = w_.layer(layer).moe;
  const int H = cfg_.hidden_size;
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
  if (graph_batch_ == batch && !graph_execs_.empty()) return;
  // Warm the grouped-GEMM workspace outside capture: the dense fp4 path runs
  // inside captured segments, and a first-call workspace cudaMalloc cannot
  // happen during stream capture. The warm-up writes only scratch buffers
  // that every real layer overwrites.
  if (w_.layer(0).dense.fp4_gate.packed != nullptr) {
    for (int l = 0; l < cfg_.text_layers; ++l)
      if (cfg_.layers[l].mlp == fuel::MlpKind::kDense) run_dense_mlp(l, batch);
  }
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
    if (p >= max_tokens_) fail("position exceeds the compiled max_tokens");

  // Must run before any KV write: it is what decides which physical page
  // this step's position resolves to.
  kv_advance(tokens, batch);

  const int H = cfg_.hidden_size;
  const int hc = cfg_.hc_mult;
  stages_ = StageMs{};
  router_entropy_ = 0.0;

  std::vector<int> pos_h(batch), ntok_h(batch), npool_h(batch);
  int n_pools_max = 0;
  for (int m = 0; m < batch; ++m) {
    pos_h[m] = pos_[m];
    ntok_h[m] = pos_[m] + 1;
    npool_h[m] = ntok_h[m] / cfg_.index_kpool;
    n_pools_max = std::max(n_pools_max, npool_h[m]);
  }
  cudaMemcpyAsync(tokens_dev_, tokens.data(), batch * sizeof(int), cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(pos_dev_, pos_h.data(), batch * sizeof(int), cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(n_tokens_dev_, ntok_h.data(), batch * sizeof(int), cudaMemcpyHostToDevice,
                  stream_);
  cudaMemcpyAsync(n_pools_dev_, npool_h.data(), batch * sizeof(int), cudaMemcpyHostToDevice,
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
    static bool lhcx = std::getenv("ROCKET_LHC_CHECKSUM") != nullptr;
    if (lhcx && l == 3) {
      print_streams_checksum(l, "plain-l3a", batch, static_cast<std::size_t>(batch) * hc * H,
                             streams_, hc * H);
      print_streams_checksum(l, "plain-l3n", batch, static_cast<std::size_t>(batch) * H, normed_,
                             H);
    }
    if (cfg_.layers[l].attn == fuel::AttnKind::kKda) {
      StageTimer t(stream_, &stages_.kda, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      run_kda(l, kda_slot_[l], batch);
    } else {
      // The indexer runs inside the MLA layer; its cost is folded in here
      // rather than isolated by a sync, since a batched step has no
      // per-layer host readback of the selection count anymore.
      StageTimer t(stream_, &stages_.mla, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      run_mla(l, mla_slot_[l], batch, n_tokens_dev_, n_pools_max, batch);
    }
    if (lhcx && l == 3) {
      print_streams_checksum(l, "plain-l3b", batch, static_cast<std::size_t>(batch) * H,
                             sublayer_out_, H);
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
    if (std::getenv("ROCKET_DFLASH2_DIR")) {
      // Draft config ids are one-based, matching vLLM's idx + 1 test.
      static constexpr int taps[5] = {4, 13, 23, 32, 41};
      for (int ti = 0; ti < 5; ++ti)
        if (l == taps[ti])
          hc_head_mean(dflash_aux_hidden_ + static_cast<std::size_t>(ti) * max_work_k_ * max_batch_ * H,
                       streams_, batch, hc, H, stream_);
    }
    print_streams_checksum(l, "plain-lhc", batch, static_cast<std::size_t>(batch) * hc * H);

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
    gemm_bf16_f32(logits_, w_.lm_head(), normed_, max_batch_ * kSpecMax, cfg_.vocab_size, H, stream_);
    argmax_f32(argmax_i_, scratch_f_, logits_, batch, cfg_.vocab_size, stream_);
    cudaMemcpyAsync(next.data(), argmax_i_, batch * sizeof(int), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }

  

  finish_stage_events();

  int moe_layers = 0;
  for (const fuel::LayerSpec& s : cfg_.layers)
    if (s.mlp == fuel::MlpKind::kSparse) ++moe_layers;
  router_entropy_ /= (moe_layers > 0 ? moe_layers : 1);

  print_kda_checksum(0, "step-end");
  print_kv_checksum("step-kv", 0, pos_[0] + 1);
    for (int m = 0; m < batch; ++m) ++pos_[m];
  cuda_check(cudaGetLastError(), "decode step");
  out_tokens = std::move(next);
}

// Multi-token verify. tokens holds batch*spec_k entries in position-major
// order: tokens[j*batch + m] is stream m's token at verify position j
// (j=0 is the already-committed token, j>=1 are draft tokens). Every layer
// type sees the weights once; KDA and MLA layers walk positions sequentially
// because their state chains, while hc/norm/FFN/lm_head batch across all
// B*spec_k rows. out_tokens gets batch*spec_k argmax values in the same
// position-major order; out_tokens[j*batch+m] predicts the token at verify
// position j+1 of stream m.
void DecodeEngine::step_spec(const std::vector<int>& tokens, int spec_k,
                             std::vector<int>& out_tokens, bool collect_stages) {
  const int batch = static_cast<int>(tokens.size()) / spec_k;
  const int total = batch * spec_k;
  if (spec_k < 1 || spec_k > max_work_k_) fail("spec_k out of work-width range");
  if (batch <= 0 || batch > max_batch_) fail("batch out of [1, max_batch] range");
  for (int m = 0; m < batch; ++m)
    if (pos_[m] + spec_k > max_tokens_) fail("spec positions exceed max_tokens");

  stages_ = StageMs{};
  router_entropy_ = 0.0;
  spec_k_ = spec_k;
  spec_batch_ = batch;

  // Reserve every verify position's KV site up front; commit_positions
  // truncates the sequence back to the accepted length once the caller has
  // decided which drafts to keep.
  for (int m = 0; m < batch; ++m) {
    const int seq = kv_seq_of_slot_[m];
    if (seq < 0) fail("step_spec: stream slot holds no KV sequence");
    for (int j = 0; j < spec_k; ++j) {
      const kv::AppendSite site = kv_cache_->append_token(seq, tokens[j * batch + m]);
      if (site.page < 0) fail("step_spec: KV pool exhausted");
      if (site.grew_table || site.copied_on_extend)
        kv_arena_->upload_table(m, kv_cache_->page_table(seq));
    }
  }

  const int H = cfg_.hidden_size;
  const int hc = cfg_.hc_mult;
  const int taps = cfg_.conv_state_taps();
  const int qkv = cfg_.kda_qkv_dim();

  // Per-row position/n_tokens arrays cover all spec_k positions: row
  // j*batch+m sits at pos_[m]+j.
  std::vector<int> pos_spec(total), ntok_spec(total), npool_spec(total);
  int n_pools_max = 0;
  for (int j = 0; j < spec_k; ++j) {
    for (int m = 0; m < batch; ++m) {
      const int p = pos_[m] + j;
      pos_spec[j * batch + m] = p;
      ntok_spec[j * batch + m] = p + 1;
      npool_spec[j * batch + m] = (p + 1) / cfg_.index_kpool;
      n_pools_max = std::max(n_pools_max, npool_spec[j * batch + m]);
    }
  }
  cudaMemcpyAsync(pos_spec_dev_, pos_spec.data(), total * sizeof(int), cudaMemcpyHostToDevice,
                  stream_);
  cudaMemcpyAsync(ntok_spec_dev_, ntok_spec.data(), total * sizeof(int), cudaMemcpyHostToDevice,
                  stream_);
  cudaMemcpyAsync(npool_spec_dev_, npool_spec.data(), total * sizeof(int), cudaMemcpyHostToDevice,
                  stream_);
  cudaMemcpyAsync(tokens_spec_dev_, tokens.data(), total * sizeof(int), cudaMemcpyHostToDevice,
                  stream_);

  {
    StageTimer t(stream_, &stages_.embed, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
    embed_streams(streams_, w_.embed(), tokens_spec_dev_, total, hc, H, stream_);
  }
  cuda_check(cudaGetLastError(), "spec embed");
  if (std::getenv("ROCKET_SPEC_DEBUG")) {
    fprintf(stderr, "[ptrs] rig=%p gro=%p sfb=%p\n",
            static_cast<const void*>(dense_row_in_group_),
            static_cast<const void*>(dense_group_of_row_),
            static_cast<const void*>(dense_sf_base_));
    std::vector<int> rh(max_batch_ * DecodeEngine::kSpecMax),
        gh(max_batch_ * DecodeEngine::kSpecMax);
    cudaMemcpy(rh.data(), dense_row_in_group_, rh.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(gh.data(), dense_group_of_row_, gh.size() * sizeof(int), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[vals] row_in_group:");
    for (int m = 0; m < max_batch_ * DecodeEngine::kSpecMax; ++m) fprintf(stderr, " %d", rh[m]);
    fprintf(stderr, "\n[vals] group_of_row:");
    for (int m = 0; m < max_batch_ * DecodeEngine::kSpecMax; ++m) fprintf(stderr, " %d", gh[m]);
    fprintf(stderr, "\n");
  }
  print_kda_checksum(0, "spec-entry");
  print_kv_checksum("spec-kv", 0, pos_[0]);
  for (int l = 0; l < cfg_.text_layers; ++l) {
    const LayerW& lw = w_.layer(l);

    // ---- attention site (batched hyper-connection + norm) ----
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      cudaMemcpyAsync(residual_, streams_, static_cast<std::size_t>(total) * hc * H * sizeof(bf16),
                      cudaMemcpyDeviceToDevice, stream_);
      hc_mix_gemv(mix_, lw.attn_hc.fn, streams_, total, cfg_.hc_mix(), hc, H, cfg_.rms_norm_eps, stream_);
      hc_split(post_, comb_, collapsed_, mix_, lw.attn_hc.base, lw.attn_hc.scale, streams_, total,
               hc, H, cfg_.hc_eps, cfg_.hc_sinkhorn_iters, stream_);
    }
    {
      StageTimer t(stream_, &stages_.norms, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      rmsnorm(normed_, collapsed_, lw.input_norm, total, H, cfg_.rms_norm_eps, stream_);
    }
    cuda_check(cudaGetLastError(), "spec hc+norm attn");
    static bool lhcx = std::getenv("ROCKET_LHC_CHECKSUM") != nullptr;
    static bool brdbg = std::getenv("ROCKET_BRANCH_DEBUG") != nullptr;
    if (lhcx && l == 3) {
      print_streams_checksum(l, "spec-l3a", total, static_cast<std::size_t>(total) * hc * H,
                             streams_, hc * H);
      print_streams_checksum(l, "spec-l3n", total, static_cast<std::size_t>(total) * H, normed_,
                             H);
    }
    if (brdbg && l <= 6)
      fprintf(stderr, "[branch] spec L%d attn=%d -> %s\n", l, int(cfg_.layers[l].attn),
              cfg_.layers[l].attn == fuel::AttnKind::kKda ? "KDA" : "MLA");
    if (cfg_.layers[l].attn == fuel::AttnKind::kKda) {
      StageTimer t(stream_, &stages_.kda, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      // Whole site at spec_k*batch rows; the chained conv and recurrence run
      // against the draft state copies so a rejected prefix never touches the
      // committed state.
      run_kda_spec_site(l, spec_k, batch, /*cut=*/nullptr);
      cuda_check(cudaGetLastError(), "spec kda");
    } else {
      StageTimer t(stream_, &stages_.mla, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      // Whole site at total rows. kv_locate and the indexer read the
      // per-row device arrays uploaded above, so no per-position launches.
      int* saved_pos = pos_dev_;
      int* saved_tok = n_tokens_dev_;
      int* saved_pool = n_pools_dev_;
      pos_dev_ = reinterpret_cast<int*>(pos_spec_dev_);
      n_tokens_dev_ = reinterpret_cast<int*>(ntok_spec_dev_);
      n_pools_dev_ = reinterpret_cast<int*>(npool_spec_dev_);
      run_mla(l, mla_slot_[l], total, n_tokens_dev_, n_pools_max, batch);
      pos_dev_ = saved_pos;
      n_tokens_dev_ = saved_tok;
      n_pools_dev_ = saved_pool;
      cuda_check(cudaGetLastError(), "spec mla");
    }
    if (lhcx && l == 3) {
      print_streams_checksum(l, "spec-l3b", total, static_cast<std::size_t>(total) * H,
                             sublayer_out_, H);
    }
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      hc_combine(streams_, post_, sublayer_out_, comb_, residual_, total, hc, H, stream_);
    }

    // ---- feed-forward site ----
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      cudaMemcpyAsync(residual_, streams_, static_cast<std::size_t>(total) * hc * H * sizeof(bf16),
                      cudaMemcpyDeviceToDevice, stream_);
      hc_mix_gemv(mix_, lw.ffn_hc.fn, streams_, total, cfg_.hc_mix(), hc, H, cfg_.rms_norm_eps, stream_);
      hc_split(post_, comb_, collapsed_, mix_, lw.ffn_hc.base, lw.ffn_hc.scale, streams_, total, hc,
               H, cfg_.hc_eps, cfg_.hc_sinkhorn_iters, stream_);
    }
    {
      StageTimer t(stream_, &stages_.norms, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      rmsnorm(normed_, collapsed_, lw.post_attn_norm, total, H, cfg_.rms_norm_eps, stream_);
    }
    cuda_check(cudaGetLastError(), "spec hc+norm ffn");
    if (cfg_.layers[l].mlp == fuel::MlpKind::kDense) {
      StageTimer t(stream_, &stages_.dense_mlp, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      run_dense_mlp(l, total);
    } else {
      StageTimer t(stream_, &stages_.moe_experts, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      run_moe(l, total);
    }
    cuda_check(cudaGetLastError(), "spec ffn");
    {
      StageTimer t(stream_, &stages_.hyper_connection, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
      hc_combine(streams_, post_, sublayer_out_, comb_, residual_, total, hc, H, stream_);
      print_streams_checksum(l, "spec-lhc", total, static_cast<std::size_t>(total) * hc * H);
    }
    if (std::getenv("ROCKET_DFLASH2_DIR")) {
      // Draft config ids are one-based, matching vLLM's idx + 1 test.
      static constexpr int taps[5] = {4, 13, 23, 32, 41};
      for (int ti = 0; ti < 5; ++ti)
        if (l == taps[ti])
          hc_head_mean(dflash_aux_hidden_ + static_cast<std::size_t>(ti) * max_work_k_ * max_batch_ * H,
                       streams_, total, hc, H, stream_);
    }
  }

  std::vector<int> next(total, 0);
  {
    StageTimer t(stream_, &stages_.lm_head, collect_stages, &prof_starts_, &prof_stops_, &prof_sinks_);
    hc_head_mean(hmean_, streams_, total, hc, H, stream_);
    rmsnorm(normed_, hmean_, w_.final_norm(), total, H, cfg_.rms_norm_eps, stream_);
    const int tc_rows = max_batch_ * (spec_k > kSpecMax ? max_work_k_ : kSpecMax);
    gemm_bf16_f32(logits_, w_.lm_head(), normed_, tc_rows, cfg_.vocab_size, H, stream_);
    argmax_f32(argmax_i_, scratch_f_, logits_, total, cfg_.vocab_size, stream_);
    cudaMemcpyAsync(next.data(), argmax_i_, total * sizeof(int), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }

  print_kda_checksum(0, "spec-end");
  finish_stage_events();

  int moe_layers = 0;
  for (const fuel::LayerSpec& s : cfg_.layers)
    if (s.mlp == fuel::MlpKind::kSparse) ++moe_layers;
  router_entropy_ /= (moe_layers > 0 ? moe_layers : 1);

  cuda_check(cudaGetLastError(), "decode step_spec");
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

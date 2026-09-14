// Greedy DFlash2 draft engine. Equations and ordering follow vLLM's
// qwen3_dflash.py, qwen3_dflash2.py and DFlash2 selector walk.
#include "dflash2.h"
#include "fabric/expert_parallel.h"
#include "kernels.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rocket::engine {
namespace {
using bf16 = __nv_bfloat16;
__device__ __forceinline__ float f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 b(float x) { return __float2bfloat16(x); }

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("dflash2: ") + what + ": " + cudaGetErrorString(e));
}

__global__ void gather_aux_kernel(bf16* out, const bf16* aux, const int* src_row, int n,
                                  int aux_stride, int hidden) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (r >= n || c >= hidden) return;
  for (int t = 0; t < 5; ++t)
    out[(static_cast<long long>(r) * 5 + t) * hidden + c] =
        aux[(static_cast<long long>(t) * aux_stride + src_row[r]) * hidden + c];
}

__global__ void gather_embed_kernel(bf16* out, const bf16* embed, const int* ids, int rows,
                                    int hidden) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (r < rows && c < hidden)
    out[static_cast<long long>(r) * hidden + c] = embed[static_cast<long long>(ids[r]) * hidden + c];
}

__global__ void add_rms_kernel(bf16* normed, bf16* residual, const bf16* hidden,
                               const bf16* weight, int rows, int width, float eps, bool first) {
  __shared__ float sm[256];
  const int r = blockIdx.x;
  float ss = 0.0f;
  for (int c = threadIdx.x; c < width; c += blockDim.x) {
    const long long i = static_cast<long long>(r) * width + c;
    const float v = first ? f(hidden[i]) : f(b(f(residual[i]) + f(hidden[i])));
    residual[i] = b(v);
    ss += v * v;
  }
  sm[threadIdx.x] = ss;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (threadIdx.x < d) sm[threadIdx.x] += sm[threadIdx.x + d];
    __syncthreads();
  }
  const float inv = rsqrtf(sm[0] / width + eps);
  for (int c = threadIdx.x; c < width; c += blockDim.x) {
    const long long i = static_cast<long long>(r) * width + c;
    normed[i] = b(f(weight[c]) * f(b(f(residual[i]) * inv)));
  }
}

__global__ void swiglu_kernel(bf16* out, const bf16* gate, const bf16* up,
                              int rows, int width, int input_stride) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * width) {
    const int row = i / width, col = i % width;
    const float x = f(gate[static_cast<long long>(row) * input_stride + col]);
    out[i] = b((x / (1.0f + __expf(-x))) *
               f(up[static_cast<long long>(row) * input_stride + col]));
  }
}

__global__ void rms_rope_kernel(bf16* x, const bf16* weight, const int* positions,
                                int rows, int heads, int hd, int row_stride,
                                float eps, float theta) {
  __shared__ float sm[128];
  const int h = blockIdx.x;
  const int r = blockIdx.y;
  const int d = threadIdx.x;
  if (r >= rows || h >= heads) return;
  const long long base = static_cast<long long>(r) * row_stride + h * hd;
  const float raw = d < hd ? f(x[base + d]) : 0.0f;
  sm[d] = raw * raw;
  __syncthreads();
  for (int off = 64; off; off >>= 1) {
    if (d < off) sm[d] += sm[d + off];
    __syncthreads();
  }
  if (d < hd / 2) {
    const float inv = rsqrtf(sm[0] / hd + eps);
    const float x0 = f(b(raw * inv * f(weight[d])));
    const float x1 = f(b(f(x[base + d + hd / 2]) * inv * f(weight[d + hd / 2])));
    const float freq = __powf(theta, -2.0f * d / hd);
    float sn, cs;
    __sincosf(positions[r] * freq, &sn, &cs);
    x[base + d] = b(x0 * cs - x1 * sn);
    x[base + d + hd / 2] = b(x1 * cs + x0 * sn);
  }
}

__global__ void scatter_context_kv_kernel(bf16* cache, const bf16* src, const int* stream_id,
                                          const int* pos, int n, int max_tokens, int kv_width) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (r < n && c < kv_width)
    cache[(static_cast<long long>(stream_id[r]) * max_tokens + pos[r]) * kv_width + c] =
        src[static_cast<long long>(r) * kv_width + c];
}

__global__ void attention_scores_kernel(float* scores, const bf16* q, const bf16* ctx_k,
                                        const bf16* query_k, const int* base_pos, int rows,
                                        int qcount, int max_tokens, int heads, int kv_heads,
                                        int hd, int q_stride, int attend_stride, int window) {
  constexpr int kKeysPerBlock = 4;
  const int h = blockIdx.x % heads;
  const int key_tile = blockIdx.x / heads;
  const int row = blockIdx.y;
  if (row >= rows) return;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int key_i = key_tile * kKeysPerBlock + warp;
  const int req = row / qcount;
  const int qoff = row % qcount;
  const int ctx_end = base_pos[req];
  const int ctx_start = max(0, ctx_end + qoff + 1 - window);
  const int ctx_n = ctx_end - ctx_start;
  const int total = ctx_n + qcount;
  if (key_i >= total) return;
  const int kvh = h / (heads / kv_heads);
  const bf16* qp = q + static_cast<long long>(row) * q_stride + h * hd;
  const bf16* kp = key_i < ctx_n
      ? ctx_k + ((static_cast<long long>(req) * max_tokens + ctx_start + key_i) * kv_heads + kvh) * hd
      : query_k + ((static_cast<long long>(req) * qcount + key_i - ctx_n) * kv_heads + kvh) * hd;
  float sum = 0.0f;
  for (int d = lane; d < hd; d += 32) sum += f(qp[d]) * f(kp[d]);
  for (int off = 16; off; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
  if (lane == 0)
    scores[(static_cast<long long>(row) * heads + h) * attend_stride + key_i] =
        sum * rsqrtf(float(hd));
}

__global__ void attention_context_kernel(bf16* out, float* scores, const bf16* ctx_v,
                                         const bf16* query_v, const int* base_pos, int rows,
                                         int qcount, int max_tokens, int heads, int kv_heads,
                                         int hd, int query_v_stride, int attend_stride, int window) {
  const int h = blockIdx.x;
  const int row = blockIdx.y;
  const int d = threadIdx.x;
  const int req = row / qcount;
  const int qoff = row % qcount;
  const int ctx_end = base_pos[req];
  const int ctx_start = max(0, ctx_end + qoff + 1 - window);
  const int ctx_n = ctx_end - ctx_start;
  const int total = ctx_n + qcount;
  float* sc = scores + (static_cast<long long>(row) * heads + h) * attend_stride;
  __shared__ float red[128];
  float local_max = -INFINITY;
  for (int i = d; i < total; i += blockDim.x) local_max = fmaxf(local_max, sc[i]);
  red[d] = local_max;
  __syncthreads();
  for (int off = 64; off; off >>= 1) {
    if (d < off) red[d] = fmaxf(red[d], red[d + off]);
    __syncthreads();
  }
  const float mx = red[0];
  float local_sum = 0.0f;
  for (int i = d; i < total; i += blockDim.x) {
    const float e = __expf(sc[i] - mx);
    sc[i] = e;
    local_sum += e;
  }
  red[d] = local_sum;
  __syncthreads();
  for (int off = 64; off; off >>= 1) {
    if (d < off) red[d] += red[d + off];
    __syncthreads();
  }
  const float inv_sum = 1.0f / red[0];
  if (d < hd) {
    const int kvh = h / (heads / kv_heads);
    float acc = 0.0f;
    for (int i = 0; i < total; ++i) {
      const bf16* vp = i < ctx_n
          ? ctx_v + ((static_cast<long long>(req) * max_tokens + ctx_start + i) * kv_heads + kvh) * hd
          : query_v + static_cast<long long>(req * qcount + i - ctx_n) * query_v_stride + kvh * hd;
      acc += sc[i] * inv_sum * f(vp[d]);
    }
    out[(static_cast<long long>(row) * heads + h) * hd + d] = b(acc);
  }
}

__global__ void select_sample_rows_kernel(bf16* out, const bf16* hidden, int batch, int qcount,
                                          int draft, int width) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y;
  if (i < batch * draft && c < width) {
    const int req = i / draft, step = i % draft;
    out[static_cast<long long>(i) * width + c] =
        hidden[(static_cast<long long>(req) * qcount + step + 1) * width + c];
  }
}

__global__ void top16_kernel(int* ids, float* vals, const float* logits, int rows, int vocab) {
  const int r = blockIdx.x;
  __shared__ float sv[256 * 16];
  __shared__ int si[256 * 16];
  float lv[16]; int li[16];
  for (int j = 0; j < 16; ++j) { lv[j] = -INFINITY; li[j] = -1; }
  for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
    const float x = logits[static_cast<long long>(r) * vocab + v];
    if (x > lv[15] || (x == lv[15] && (li[15] < 0 || v < li[15]))) {
      int j = 15;
      while (j > 0 && (x > lv[j - 1] || (x == lv[j - 1] && v < li[j - 1]))) {
        lv[j] = lv[j - 1]; li[j] = li[j - 1]; --j;
      }
      lv[j] = x; li[j] = v;
    }
  }
  for (int j = 0; j < 16; ++j) { sv[threadIdx.x * 16 + j] = lv[j]; si[threadIdx.x * 16 + j] = li[j]; }
  __syncthreads();
  // Merge sorted per-thread lists as a tree. The old thread-0 walk performed
  // 4096 serial insertions per row; each level here merges two 16-entry lists.
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      float mv[16]; int mi[16];
      int a = 0, b = 0;
      const int lhs = threadIdx.x * 16;
      const int rhs = (threadIdx.x + stride) * 16;
      for (int j = 0; j < 16; ++j) {
        const float av = a < 16 ? sv[lhs + a] : -INFINITY;
        const float bv = b < 16 ? sv[rhs + b] : -INFINITY;
        const int ai = a < 16 ? si[lhs + a] : -1;
        const int bi = b < 16 ? si[rhs + b] : -1;
        const bool take_a = av > bv || (av == bv && ai <= bi);
        mv[j] = take_a ? av : bv;
        mi[j] = take_a ? ai : bi;
        a += take_a;
        b += !take_a;
      }
      for (int j = 0; j < 16; ++j) { sv[lhs + j] = mv[j]; si[lhs + j] = mi[j]; }
    }
    __syncthreads();
  }
  if (threadIdx.x < 16) {
    vals[r * 16 + threadIdx.x] = sv[threadIdx.x];
    ids[r * 16 + threadIdx.x] = si[threadIdx.x];
  }
}

__global__ void selector_walk_kernel(int* out, const int* candidates, const float* unary,
                                     const bf16* projected, const bf16* pred, const bf16* succ,
                                     const int* anchor, int batch, int draft, int rank) {
  const int req = blockIdx.x;
  const int lane = threadIdx.x;
  __shared__ float sc[16];
  __shared__ int previous;
  if (lane == 0) previous = -1;
  __syncthreads();
  for (int step = 0; step < draft; ++step) {
    const int row = req * draft + step;
    const int predecessor = step == 0 ? anchor[req] : candidates[(row - 1) * 16 + previous];
    const int candidate = candidates[row * 16 + lane];
    float edge = 0.0f;
    for (int d = 0; d < rank; ++d)
      edge += f(pred[static_cast<long long>(predecessor) * rank + d]) *
              f(projected[static_cast<long long>(row) * rank + d]) *
              f(succ[static_cast<long long>(candidate) * rank + d]);
    sc[lane] = unary[row * 16 + lane] + edge;
    __syncthreads();
    if (lane == 0) {
      int best = 0;
      for (int j = 1; j < 16; ++j) if (sc[j] > sc[best]) best = j;
      previous = best;
      out[step * batch + req] = candidates[row * 16 + best];
    }
    __syncthreads();
  }
}
}  // namespace

struct DFlash2DraftEngine::Impl {
  DFlash2Weights w;
  const bf16* embed;
  const bf16* lm_head;
  rocket::fabric::ExpertParallel* ep;
  int max_batch, max_tokens, max_draft, max_rows, attend_stride;
  cudaStream_t selector_stream = nullptr;
  cudaEvent_t selector_ready = nullptr;
  std::vector<void*> owned;
  bf16 *aux_cat, *ctx_hidden, *ctx_norm, *ctx_k, *ctx_v;
  bf16 *kv_k, *kv_v;
  bf16 *hidden, *residual, *normed, *tmp, *coeff, *q, *k, *v, *attn, *gate, *up, *mlp;
  bf16 *sample_hidden, *selector_hidden;
  float *scores, *logits, *top_vals;
  int *ids, *positions, *src_rows, *stream_ids, *ctx_positions, *top_ids, *draft_ids;

  void* alloc(std::size_t n) { void* p=nullptr; ck(cudaMalloc(&p,n),"workspace allocation"); owned.push_back(p); return p; }
  bf16* A(std::size_t n) { return static_cast<bf16*>(alloc(n*sizeof(bf16))); }
  float* F(std::size_t n) { return static_cast<float*>(alloc(n*sizeof(float))); }
  int* I(std::size_t n) { return static_cast<int*>(alloc(n*sizeof(int))); }

  Impl(const std::filesystem::path& dir, const bf16* e, const bf16* lm, int mb, int mt, int md,
       rocket::fabric::ExpertParallel* parallel)
      : embed(e), lm_head(lm), ep(parallel), max_batch(mb), max_tokens(mt), max_draft(md) {
    w.load(dir / "model.safetensors");
    ck(cudaStreamCreateWithFlags(&selector_stream,cudaStreamNonBlocking),"selector stream");
    ck(cudaEventCreateWithFlags(&selector_ready,cudaEventDisableTiming),"selector event");
    const int H=w.cfg.hidden_size, KV=w.cfg.num_kv_heads*w.cfg.head_dim;
    const int FF=w.cfg.intermediate_size, R=w.cfg.selector_rank;
    max_rows = mb * (md + 1);
    attend_stride = std::min(mt, w.cfg.sliding_window) + md + 1;
    aux_cat=A((std::size_t)max_rows*5*w.cfg.hidden_size); ctx_hidden=A((std::size_t)max_rows*w.cfg.hidden_size);
    ctx_norm=A((std::size_t)max_rows*H); ctx_k=A((std::size_t)max_rows*KV); ctx_v=A((std::size_t)max_rows*KV);
    kv_k=A((std::size_t)w.cfg.num_layers*mb*mt*KV); kv_v=A((std::size_t)w.cfg.num_layers*mb*mt*KV);
    hidden=A((std::size_t)max_rows*H); residual=A((std::size_t)max_rows*H); normed=A((std::size_t)max_rows*H);
    tmp=A((std::size_t)max_rows*H); coeff=A((std::size_t)max_rows*4*(H/w.cfg.conv_group_size));
    q=A((std::size_t)max_rows*(H+KV)); v=q+H;
    k=A((std::size_t)max_rows*KV); attn=A((std::size_t)max_rows*H);
    gate=A((std::size_t)max_rows*2*FF); up=gate+FF;
    mlp=A((std::size_t)max_rows*FF);
    sample_hidden=A((std::size_t)mb*md*H); selector_hidden=A((std::size_t)mb*md*R);
    scores=F((std::size_t)max_rows*w.cfg.num_heads*attend_stride); logits=F((std::size_t)mb*md*w.cfg.vocab_size); top_vals=F((std::size_t)mb*md*16);
    ids=I(max_rows); positions=I(max_rows); src_rows=I(max_rows); stream_ids=I(max_rows); ctx_positions=I(max_rows);
    top_ids=I((std::size_t)mb*md*16); draft_ids=I((std::size_t)mb*md);
    for (int l=0;l<5;++l) {
      const auto* qw=W(l,"self_attn.q_proj.weight");
      const auto* vw=W(l,"self_attn.v_proj.weight");
      const auto* gw=W(l,"mlp.gate_proj.weight");
      const auto* uw=W(l,"mlp.up_proj.weight");
      if (vw != qw + (std::size_t)H*H || uw != gw + (std::size_t)FF*H)
        throw std::runtime_error("dflash2 fused projection weights are not contiguous");
    }
    ck(cudaMemset(kv_k,0,(std::size_t)w.cfg.num_layers*mb*mt*KV*2),"zero K cache");
    ck(cudaMemset(kv_v,0,(std::size_t)w.cfg.num_layers*mb*mt*KV*2),"zero V cache");
  }
  ~Impl(){
    if(selector_ready) cudaEventDestroy(selector_ready);
    if(selector_stream) cudaStreamDestroy(selector_stream);
    for(void* p:owned) cudaFree(p);
  }
  const bf16* W(int l, const char* suffix) const {
    const std::string n="layers."+std::to_string(l)+"."+suffix;
    const void* p=w.tensor_data(n.c_str()); if(!p) throw std::runtime_error("dflash2 missing "+n); return static_cast<const bf16*>(p);
  }
};

DFlash2DraftEngine::DFlash2DraftEngine(const std::filesystem::path& dir, const bf16* embed,
                                       const bf16* lm, int mb, int mt, int md,
                                       rocket::fabric::ExpertParallel* ep)
    : p_(std::make_unique<Impl>(dir,embed,lm,mb,mt,md,ep)) {}
DFlash2DraftEngine::~DFlash2DraftEngine() = default;

void DFlash2DraftEngine::append_context(const bf16* aux, int aux_stride, int positions_n, int batch,
                                        const std::vector<int>& base_pos,
                                        const std::vector<int>& accepted, cudaStream_t s) {
  auto& z=*p_;
  const auto& c=z.w.cfg; const int H=c.hidden_size, KV=c.num_kv_heads*c.head_dim;
  if (batch < 1 || batch > z.max_batch || positions_n < 1 ||
      static_cast<long long>(positions_n) * batch > z.max_rows ||
      static_cast<int>(base_pos.size()) != batch || static_cast<int>(accepted.size()) != batch)
    throw std::runtime_error("dflash2 context shape");
  std::vector<int> sr, st, ps;
  for(int p=0;p<positions_n;++p) for(int m=0;m<batch;++m) if(p<accepted[m]) {
    const int pos = base_pos[m] + p;
    if (pos < 0 || pos >= z.max_tokens) throw std::runtime_error("dflash2 context position");
    sr.push_back(p*batch+m); st.push_back(m); ps.push_back(pos);
  }
  const int n=sr.size(); if(!n) return;
  ck(cudaMemcpyAsync(z.src_rows,sr.data(),n*4,cudaMemcpyHostToDevice,s),"context row map");
  ck(cudaMemcpyAsync(z.stream_ids,st.data(),n*4,cudaMemcpyHostToDevice,s),"context stream map");
  ck(cudaMemcpyAsync(z.ctx_positions,ps.data(),n*4,cudaMemcpyHostToDevice,s),"context positions");
  gather_aux_kernel<<<dim3(16,n),256,0,s>>>(z.aux_cat,aux,z.src_rows,n,aux_stride,H);
  gemm_bf16_cublas(z.ctx_hidden,static_cast<const bf16*>(z.w.tensor_data("fc.weight")),z.aux_cat,n,H,5*H,s);
  rmsnorm(z.ctx_norm,z.ctx_hidden,static_cast<const bf16*>(z.w.tensor_data("hidden_norm.weight")),n,H,z.w.cfg.rms_norm_eps,s);
  for(int l=0;l<c.num_layers;++l){
    gemm_bf16_cublas(z.ctx_k,z.W(l,"self_attn.k_proj.weight"),z.ctx_norm,n,KV,H,s);
    gemm_bf16_cublas(z.ctx_v,z.W(l,"self_attn.v_proj.weight"),z.ctx_norm,n,KV,H,s);
    rms_rope_kernel<<<dim3(c.num_kv_heads,n),c.head_dim,0,s>>>(z.ctx_k,z.W(l,"self_attn.k_norm.weight"),
                                            z.ctx_positions,n,c.num_kv_heads,c.head_dim,KV,c.rms_norm_eps,10000.0f);
    scatter_context_kv_kernel<<<dim3(4,n),256,0,s>>>(z.kv_k+(std::size_t)l*z.max_batch*z.max_tokens*KV,z.ctx_k,z.stream_ids,z.ctx_positions,n,z.max_tokens,KV);
    scatter_context_kv_kernel<<<dim3(4,n),256,0,s>>>(z.kv_v+(std::size_t)l*z.max_batch*z.max_tokens*KV,z.ctx_v,z.stream_ids,z.ctx_positions,n,z.max_tokens,KV);
  }
}

void DFlash2DraftEngine::propose(const std::vector<int>& anchor, const std::vector<int>& position,
                                 int batch, int draft, std::vector<int>& out, cudaStream_t s) {
  auto& z=*p_;
  const auto& c=z.w.cfg; const int H=c.hidden_size, KV=c.num_kv_heads*c.head_dim;
  const int NH=c.num_heads, NK=c.num_kv_heads, HD=c.head_dim, FF=c.intermediate_size;
  if(batch < 1 || batch > z.max_batch || static_cast<int>(anchor.size()) != batch ||
     static_cast<int>(position.size()) != batch)
    throw std::runtime_error("dflash2 proposal shape");
  if(draft < 1 || draft > z.max_draft)
    throw std::runtime_error("dflash2 draft width");
  for (int pos : position)
    if (pos < 0 || pos + draft >= z.max_tokens)
      throw std::runtime_error("dflash2 proposal position");
  const int qcount=draft+1, rows=batch*qcount;
  int max_attend = 0;
  for (int base : position) {
    for (int qoff = 0; qoff < qcount; ++qoff) {
      const int ctx_start = std::max(0, base + qoff + 1 - z.w.cfg.sliding_window);
      max_attend = std::max(max_attend, base - ctx_start + qcount);
    }
  }
  std::vector<int> hi(rows), hp(rows);
  for(int m=0;m<batch;++m) for(int j=0;j<qcount;++j){ hi[m*qcount+j]=j?z.w.cfg.mask_token_id:anchor[m]; hp[m*qcount+j]=position[m]+j; }
  ck(cudaMemcpyAsync(z.ids,hi.data(),rows*4,cudaMemcpyHostToDevice,s),"query ids");
  ck(cudaMemcpyAsync(z.positions,hp.data(),rows*4,cudaMemcpyHostToDevice,s),"query positions");
  ck(cudaMemcpyAsync(z.ctx_positions,position.data(),batch*4,cudaMemcpyHostToDevice,s),"base positions");
#ifdef ROCKET_DFLASH2_PROFILE
  cudaEvent_t ev0=nullptr, ev1=nullptr, ev2=nullptr, ev3=nullptr;
  cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2); cudaEventCreate(&ev3);
  cudaEventRecord(ev0,s);
#endif
  gather_embed_kernel<<<dim3(16,rows),256,0,s>>>(z.hidden,z.embed,z.ids,rows,H);
  for(int l=0;l<c.num_layers;++l){
    add_rms_kernel<<<rows,256,0,s>>>(z.normed,z.residual,z.hidden,z.W(l,"input_layernorm.weight"),rows,H,c.rms_norm_eps,l==0);
    dflash2_grouped_conv_prepare(z.tmp,z.coeff,z.normed,z.W(l,"attention_conv.kernel_projection.weight"),z.W(l,"attention_conv.base_kernel"),rows,H,qcount,c.conv_group_size,c.conv_kernel_size,s);
    gemm_bf16_cublas(z.q,z.W(l,"self_attn.q_proj.weight"),z.tmp,rows,H+KV,H,s);
    gemm_bf16_cublas(z.k,z.W(l,"self_attn.k_proj.weight"),z.tmp,rows,KV,H,s);
    rms_rope_kernel<<<dim3(NH,rows),HD,0,s>>>(z.q,z.W(l,"self_attn.q_norm.weight"),
                                               z.positions,rows,NH,HD,H+KV,c.rms_norm_eps,10000.0f);
    rms_rope_kernel<<<dim3(NK,rows),HD,0,s>>>(z.k,z.W(l,"self_attn.k_norm.weight"),
                                              z.positions,rows,NK,HD,KV,c.rms_norm_eps,10000.0f);
    const bf16* lk=z.kv_k+(std::size_t)l*z.max_batch*z.max_tokens*KV;
    const bf16* lv=z.kv_v+(std::size_t)l*z.max_batch*z.max_tokens*KV;
    const int score_tiles = (max_attend + 3) / 4;
    attention_scores_kernel<<<dim3(NH*score_tiles,rows),128,0,s>>>(z.scores,z.q,lk,z.k,z.ctx_positions,rows,qcount,z.max_tokens,NH,NK,HD,H+KV,z.attend_stride,c.sliding_window);
    attention_context_kernel<<<dim3(NH,rows),HD,0,s>>>(z.attn,z.scores,lv,z.v,z.ctx_positions,rows,qcount,z.max_tokens,NH,NK,HD,H+KV,z.attend_stride,c.sliding_window);
    gemm_bf16_cublas(z.tmp,z.W(l,"self_attn.o_proj.weight"),z.attn,rows,H,H,s);
    dflash2_grouped_conv_finish(z.hidden,z.tmp,z.coeff,z.W(l,"attention_conv.base_kernel"),rows,H,qcount,c.conv_group_size,c.conv_kernel_size,s);
    add_rms_kernel<<<rows,256,0,s>>>(z.normed,z.residual,z.hidden,z.W(l,"post_attention_layernorm.weight"),rows,H,c.rms_norm_eps,false);
    dflash2_grouped_conv_prepare(z.tmp,z.coeff,z.normed,z.W(l,"mlp_conv.kernel_projection.weight"),z.W(l,"mlp_conv.base_kernel"),rows,H,qcount,c.conv_group_size,c.conv_kernel_size,s);
    gemm_bf16_cublas(z.gate,z.W(l,"mlp.gate_proj.weight"),z.tmp,rows,2*FF,H,s);
    swiglu_kernel<<<(rows*FF+255)/256,256,0,s>>>(z.mlp,z.gate,z.up,rows,FF,2*FF);
    gemm_bf16_cublas(z.tmp,z.W(l,"mlp.down_proj.weight"),z.mlp,rows,H,FF,s);
    dflash2_grouped_conv_finish(z.hidden,z.tmp,z.coeff,z.W(l,"mlp_conv.base_kernel"),rows,H,qcount,c.conv_group_size,c.conv_kernel_size,s);
  }
  add_rms_kernel<<<rows,256,0,s>>>(z.normed,z.residual,z.hidden,static_cast<const bf16*>(z.w.tensor_data("norm.weight")),rows,H,c.rms_norm_eps,false);
#ifdef ROCKET_DFLASH2_PROFILE
  cudaEventRecord(ev1,s);
#endif
  select_sample_rows_kernel<<<dim3(16,batch*draft),256,0,s>>>(z.sample_hidden,z.normed,batch,qcount,draft,H);
  // Candidate logits use the target lm_head exactly as upstream.
  const int sample_rows=batch*draft;
  // Split the bandwidth-bound target head across the two expert-parallel
  // ranks. Each side returns 16 local candidates; 32 host records collapse
  // to the exact global top-16 before selector scoring.
  const int vocab_begin = z.ep ? z.ep->rank() * (z.w.cfg.vocab_size / 2) : 0;
  const int local_vocab = z.ep ? (z.ep->rank() == 0 ? z.w.cfg.vocab_size / 2
                                                   : z.w.cfg.vocab_size - vocab_begin)
                               : z.w.cfg.vocab_size;
  gemm_bf16_f32(z.logits,z.lm_head + static_cast<std::size_t>(vocab_begin) * H,
                 z.sample_hidden,sample_rows,local_vocab,H,s);
  top16_kernel<<<sample_rows,256,0,s>>>(z.top_ids,z.top_vals,z.logits,sample_rows,local_vocab);
  ck(cudaEventRecord(z.selector_ready,s),"selector input ready");
  ck(cudaStreamWaitEvent(z.selector_stream,z.selector_ready,0),"selector stream wait");
  gemm_bf16_cublas(z.selector_hidden,
                    static_cast<const bf16*>(z.w.tensor_data("candidate_selector.hidden_projection.weight")),
                    z.sample_hidden,sample_rows,c.selector_rank,H,z.selector_stream);
  ck(cudaEventRecord(z.selector_ready,z.selector_stream),"selector projection ready");
  if (z.ep) {
    const std::size_t count = static_cast<std::size_t>(sample_rows) * 16;
    std::vector<int> local_ids(count), peer_ids;
    std::vector<float> local_vals(count), peer_vals;
    ck(cudaMemcpyAsync(local_ids.data(),z.top_ids,count*sizeof(int),cudaMemcpyDeviceToHost,s),
       "local candidate ids");
    ck(cudaMemcpyAsync(local_vals.data(),z.top_vals,count*sizeof(float),cudaMemcpyDeviceToHost,s),
       "local candidate scores");
    ck(cudaStreamSynchronize(s),"local candidates synchronize");
    for (int& id : local_ids) id += vocab_begin;
    z.ep->exchange_draft_candidates(local_ids,local_vals,peer_ids,peer_vals);
    std::vector<int> merged_ids(count);
    std::vector<float> merged_vals(count);
    for (int r=0;r<sample_rows;++r) {
      std::vector<std::pair<float,int>> candidates;
      candidates.reserve(32);
      for(int j=0;j<16;++j) {
        candidates.emplace_back(local_vals[r*16+j],local_ids[r*16+j]);
        candidates.emplace_back(peer_vals[r*16+j],peer_ids[r*16+j]);
      }
      std::partial_sort(candidates.begin(),candidates.begin()+16,candidates.end(),
                        [](const auto& a,const auto& b) {
                          return a.first != b.first ? a.first > b.first : a.second < b.second;
                        });
      for(int j=0;j<16;++j) {
        merged_vals[r*16+j]=candidates[j].first; merged_ids[r*16+j]=candidates[j].second;
      }
    }
    ck(cudaMemcpyAsync(z.top_ids,merged_ids.data(),count*sizeof(int),cudaMemcpyHostToDevice,s),
       "merged candidate ids");
    ck(cudaMemcpyAsync(z.top_vals,merged_vals.data(),count*sizeof(float),cudaMemcpyHostToDevice,s),
       "merged candidate scores");
  }
#ifdef ROCKET_DFLASH2_PROFILE
  cudaEventRecord(ev2,s);
#endif
  ck(cudaStreamWaitEvent(s,z.selector_ready,0),"selector projection join");
  ck(cudaMemcpyAsync(z.ids,anchor.data(),batch*4,cudaMemcpyHostToDevice,s),"anchor ids");
  selector_walk_kernel<<<batch,16,0,s>>>(z.draft_ids,z.top_ids,z.top_vals,z.selector_hidden,
      static_cast<const bf16*>(z.w.tensor_data("candidate_selector.predecessor_codebook")),
      static_cast<const bf16*>(z.w.tensor_data("candidate_selector.successor_codebook")),z.ids,batch,draft,c.selector_rank);
  out.resize(batch*draft);
  ck(cudaMemcpyAsync(out.data(),z.draft_ids,out.size()*4,cudaMemcpyDeviceToHost,s),"draft ids out");
#ifdef ROCKET_DFLASH2_PROFILE
  cudaEventRecord(ev3,s);
#endif
  ck(cudaStreamSynchronize(s),"draft synchronize");
#ifdef ROCKET_DFLASH2_PROFILE
  float layers=0, candidates=0, selector=0;
  cudaEventElapsedTime(&layers,ev0,ev1); cudaEventElapsedTime(&candidates,ev1,ev2);
  cudaEventElapsedTime(&selector,ev2,ev3);
  std::fprintf(stderr,"[dflash2-profile] layers=%.3f ms candidates=%.3f ms selector=%.3f ms\n",
               layers,candidates,selector);
  cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaEventDestroy(ev2); cudaEventDestroy(ev3);
#endif
}

std::size_t DFlash2DraftEngine::prefix_state_bytes(int position) const {
  const auto& z = *p_;
  const int tokens = std::min(position, z.w.cfg.sliding_window);
  const std::size_t kv = static_cast<std::size_t>(z.w.cfg.num_kv_heads) * z.w.cfg.head_dim;
  return static_cast<std::size_t>(z.w.cfg.num_layers) * 2 * tokens * kv * sizeof(bf16);
}

std::uint64_t DFlash2DraftEngine::prefix_state_digest(int slot, int position) const {
  const auto& z = *p_;
  if (slot < 0 || slot >= z.max_batch || position < 0 || position > z.max_tokens)
    throw std::runtime_error("dflash2 prefix digest range");
  const int count = std::min(position, z.w.cfg.sliding_window);
  const int start = position - count;
  const std::size_t KV = static_cast<std::size_t>(z.w.cfg.num_kv_heads) * z.w.cfg.head_dim;
  std::vector<bf16> host(static_cast<std::size_t>(count) * KV);
  std::uint64_t hash = 0xcbf29ce484222325ull;
  auto add = [&](const bf16* src) {
    ck(cudaMemcpy(host.data(), src, host.size() * sizeof(bf16), cudaMemcpyDeviceToHost),
       "dflash2 digest copy");
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(host.data());
    for (std::size_t i = 0; i < host.size() * sizeof(bf16); ++i) {
      hash ^= bytes[i]; hash *= 0x100000001b3ull;
    }
  };
  for (int l = 0; l < z.w.cfg.num_layers; ++l) {
    const std::size_t off = (static_cast<std::size_t>(l) * z.max_batch + slot) * z.max_tokens * KV +
                            static_cast<std::size_t>(start) * KV;
    add(z.kv_k + off);
    add(z.kv_v + off);
  }
  return hash;
}

bool DFlash2DraftEngine::prefix_state_equal(int a, int b, int position) const {
  const auto& z = *p_;
  if (a < 0 || b < 0 || a >= z.max_batch || b >= z.max_batch ||
      position < 0 || position > z.max_tokens)
    throw std::runtime_error("dflash2 prefix equality range");
  const int count = std::min(position, z.w.cfg.sliding_window);
  const int start = position - count;
  const std::size_t KV = static_cast<std::size_t>(z.w.cfg.num_kv_heads) * z.w.cfg.head_dim;
  const std::size_t bytes = static_cast<std::size_t>(count) * KV * sizeof(bf16);
  std::vector<std::uint8_t> first(bytes), second(bytes);
  for (int l = 0; l < z.w.cfg.num_layers; ++l) {
    const std::size_t ao = (static_cast<std::size_t>(l) * z.max_batch + a) *
                               z.max_tokens * KV + static_cast<std::size_t>(start) * KV;
    const std::size_t bo = (static_cast<std::size_t>(l) * z.max_batch + b) *
                               z.max_tokens * KV + static_cast<std::size_t>(start) * KV;
    for (const auto pair : {std::pair{z.kv_k + ao, z.kv_k + bo},
                            std::pair{z.kv_v + ao, z.kv_v + bo}}) {
      ck(cudaMemcpy(first.data(), pair.first, bytes, cudaMemcpyDeviceToHost),
         "dflash2 exact first copy");
      ck(cudaMemcpy(second.data(), pair.second, bytes, cudaMemcpyDeviceToHost),
         "dflash2 exact second copy");
      if (first != second) return false;
    }
  }
  return true;
}

void DFlash2DraftEngine::save_prefix_state(kv::NvmePrefixStore& store,
                                           const kv::PrefixRecordKey& key,
                                           int slot, int position, cudaStream_t s) {
  auto& z = *p_;
  if (slot < 0 || slot >= z.max_batch || position < 0 || position > z.max_tokens)
    throw std::runtime_error("dflash2 save prefix range");
  const int count = std::min(position, z.w.cfg.sliding_window);
  const int start = position - count;
  const std::size_t KV = static_cast<std::size_t>(z.w.cfg.num_kv_heads) * z.w.cfg.head_dim;
  std::vector<kv::DeviceConstSpan> spans;
  spans.reserve(static_cast<std::size_t>(z.w.cfg.num_layers) * 2);
  for (int l = 0; l < z.w.cfg.num_layers; ++l) {
    const std::size_t off = (static_cast<std::size_t>(l) * z.max_batch + slot) * z.max_tokens * KV +
                            static_cast<std::size_t>(start) * KV;
    spans.push_back({static_cast<kv::PrefixComponent>(100 + l * 2), z.kv_k + off,
                     static_cast<std::size_t>(count) * KV * sizeof(bf16)});
    spans.push_back({static_cast<kv::PrefixComponent>(101 + l * 2), z.kv_v + off,
                     static_cast<std::size_t>(count) * KV * sizeof(bf16)});
  }
  store.save(key, spans);
  ck(cudaStreamSynchronize(s), "dflash2 save prefix sync");
}

bool DFlash2DraftEngine::load_prefix_state(kv::NvmePrefixStore& store,
                                           const kv::PrefixRecordKey& key,
                                           int slot, int position, cudaStream_t s) {
  auto& z = *p_;
  if (slot < 0 || slot >= z.max_batch || position < 0 || position > z.max_tokens)
    throw std::runtime_error("dflash2 load prefix range");
  const int count = std::min(position, z.w.cfg.sliding_window);
  const int start = position - count;
  const std::size_t KV = static_cast<std::size_t>(z.w.cfg.num_kv_heads) * z.w.cfg.head_dim;
  std::vector<kv::DeviceSpan> spans;
  spans.reserve(static_cast<std::size_t>(z.w.cfg.num_layers) * 2);
  for (int l = 0; l < z.w.cfg.num_layers; ++l) {
    const std::size_t off = (static_cast<std::size_t>(l) * z.max_batch + slot) * z.max_tokens * KV +
                            static_cast<std::size_t>(start) * KV;
    spans.push_back({static_cast<kv::PrefixComponent>(100 + l * 2), z.kv_k + off,
                     static_cast<std::size_t>(count) * KV * sizeof(bf16)});
    spans.push_back({static_cast<kv::PrefixComponent>(101 + l * 2), z.kv_v + off,
                     static_cast<std::size_t>(count) * KV * sizeof(bf16)});
  }
  const bool ok = store.load(key, spans);
  ck(cudaStreamSynchronize(s), "dflash2 load prefix sync");
  return ok;
}

void DFlash2DraftEngine::copy_prefix_state(int dst_slot, int src_slot, int position,
                                           cudaStream_t s) {
  auto& z = *p_;
  if (dst_slot < 0 || dst_slot >= z.max_batch || src_slot < 0 || src_slot >= z.max_batch ||
      position < 0 || position > z.max_tokens)
    throw std::runtime_error("dflash2 copy prefix range");
  if (dst_slot == src_slot) return;
  const int count = std::min(position, z.w.cfg.sliding_window);
  const int start = position - count;
  const std::size_t KV = static_cast<std::size_t>(z.w.cfg.num_kv_heads) * z.w.cfg.head_dim;
  const std::size_t bytes = static_cast<std::size_t>(count) * KV * sizeof(bf16);
  for (int l = 0; l < z.w.cfg.num_layers; ++l) {
    const std::size_t src = (static_cast<std::size_t>(l) * z.max_batch + src_slot) *
                                z.max_tokens * KV + static_cast<std::size_t>(start) * KV;
    const std::size_t dst = (static_cast<std::size_t>(l) * z.max_batch + dst_slot) *
                                z.max_tokens * KV + static_cast<std::size_t>(start) * KV;
    ck(cudaMemcpyAsync(z.kv_k + dst, z.kv_k + src, bytes,
                       cudaMemcpyDeviceToDevice, s), "dflash2 copy prefix k");
    ck(cudaMemcpyAsync(z.kv_v + dst, z.kv_v + src, bytes,
                       cudaMemcpyDeviceToDevice, s), "dflash2 copy prefix v");
  }
  ck(cudaStreamSynchronize(s), "dflash2 copy prefix sync");
}

void DFlash2DraftEngine::reset_slot(int slot, cudaStream_t s) {
  auto& z = *p_;
  if (slot < 0 || slot >= z.max_batch) throw std::runtime_error("dflash2 reset slot range");
  const std::size_t KV = static_cast<std::size_t>(z.w.cfg.num_kv_heads) * z.w.cfg.head_dim;
  const std::size_t bytes = static_cast<std::size_t>(z.max_tokens) * KV * sizeof(bf16);
  for (int l = 0; l < z.w.cfg.num_layers; ++l) {
    const std::size_t off = (static_cast<std::size_t>(l) * z.max_batch + slot) * z.max_tokens * KV;
    ck(cudaMemsetAsync(z.kv_k + off, 0, bytes, s), "dflash2 reset slot k");
    ck(cudaMemsetAsync(z.kv_v + off, 0, bytes, s), "dflash2 reset slot v");
  }
  ck(cudaStreamSynchronize(s), "dflash2 reset slot sync");
}

}  // namespace rocket::engine

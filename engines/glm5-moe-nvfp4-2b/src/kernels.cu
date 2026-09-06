#include "kernels.h"

#include <cuda_fp8.h>
#include <math_constants.h>

namespace rocket::engine {
namespace {

constexpr int kWarp = 32;

__device__ __forceinline__ float f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 b(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ float block_sum(float v, float* shared) {
  for (int off = kWarp / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + kWarp - 1) / kWarp;
  float total = 0.0f;
  if (threadIdx.x == 0) {
    for (int i = 0; i < warps; ++i) total += shared[i];
    shared[0] = total;
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float siluf(float x) { return x * sigmoidf(x); }

// Physical page + slot for logical token position `pos` of stream `m`.
__device__ __forceinline__ void kv_locate(const KvPages& kv, int m, int pos, int* page, int* slot) {
  const int logical_page = pos / kv.page_tokens;
  *page = kv.table[m * kv.max_pages + logical_page];
  *slot = pos % kv.page_tokens;
}

// --------------------------------------------------------------- batched GEMM

template <typename Out>
__global__ void gemm_kernel(Out* __restrict__ y, const bf16* __restrict__ w,
                            const bf16* __restrict__ x, int n_rows, int k) {
  __shared__ float red[32];
  const long long row = blockIdx.x;
  const int m = blockIdx.y;
  const bf16* wr = w + row * static_cast<long long>(k);
  const bf16* xr = x + static_cast<long long>(m) * k;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) acc += f(wr[i]) * f(xr[i]);
  const float total = block_sum(acc, red);
  if (threadIdx.x == 0) {
    Out* yr = y + static_cast<long long>(m) * n_rows;
    if constexpr (sizeof(Out) == 4) {
      yr[row] = total;
    } else {
      yr[row] = b(total);
    }
  }
}

// ------------------------------------------------------------------- norms

__global__ void rmsnorm_kernel(bf16* __restrict__ out, const bf16* __restrict__ x,
                               const bf16* __restrict__ weight, int n, float eps) {
  __shared__ float red[32];
  const int m = blockIdx.x;
  const bf16* xr = x + static_cast<long long>(m) * n;
  bf16* outr = out + static_cast<long long>(m) * n;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = f(xr[i]);
    acc += v * v;
  }
  const float inv = rsqrtf(block_sum(acc, red) / static_cast<float>(n) + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    outr[i] = b(f(weight[i]) * f(b(f(xr[i]) * inv)));
}

__global__ void layernorm_kernel(bf16* __restrict__ out, const bf16* __restrict__ x,
                                 const bf16* __restrict__ w, const bf16* __restrict__ bias, int n,
                                 float eps) {
  __shared__ float red[32];
  __shared__ float mean_s;
  const int m = blockIdx.x;
  const bf16* xr = x + static_cast<long long>(m) * n;
  bf16* outr = out + static_cast<long long>(m) * n;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) acc += f(xr[i]);
  const float mean = block_sum(acc, red) / static_cast<float>(n);
  if (threadIdx.x == 0) mean_s = mean;
  __syncthreads();
  float var = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float d = f(xr[i]) - mean_s;
    var += d * d;
  }
  const float inv = rsqrtf(block_sum(var, red) / static_cast<float>(n) + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    outr[i] = b((f(xr[i]) - mean_s) * inv * f(w[i]) + f(bias[i]));
}

// --------------------------------------------------------------- embedding

__global__ void embed_streams_kernel(bf16* __restrict__ streams, const bf16* __restrict__ embed,
                                     const int* __restrict__ tokens, int hc, int hidden) {
  const int m = blockIdx.y;
  const int i = blockIdx.x;  // which of the hc hyper-connection streams
  const bf16* src = embed + static_cast<long long>(tokens[m]) * hidden;
  bf16* dst = streams + (static_cast<long long>(m) * hc + i) * hidden;
  for (int d = threadIdx.x; d < hidden; d += blockDim.x) dst[d] = src[d];
}

// ------------------------------------------------------ hyper-connections

__global__ void hc_mix_kernel(float* __restrict__ mix, const bf16* __restrict__ fn,
                              const bf16* __restrict__ streams, int hc_mix, int flat, float eps) {
  __shared__ float red[32];
  const int row = blockIdx.x;
  const int m = blockIdx.y;
  const bf16* s = streams + static_cast<long long>(m) * flat;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < flat; i += blockDim.x) {
    const float v = f(s[i]);
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum(ss, red) / static_cast<float>(flat) + eps);
  const bf16* wr = fn + static_cast<long long>(row) * flat;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < flat; i += blockDim.x) acc += f(wr[i]) * (f(s[i]) * inv);
  const float total = block_sum(acc, red);
  if (threadIdx.x == 0) mix[static_cast<long long>(m) * hc_mix + row] = total;
}

// One block per stream. hc is 4, so the comb work is 16 scalars and stays on
// thread 0; the stream collapse is the only part wide enough to parallelise.
__global__ void hc_split_kernel(float* __restrict__ post, float* __restrict__ comb,
                                bf16* __restrict__ collapsed, const float* __restrict__ mix,
                                const float* __restrict__ base, const float* __restrict__ scale,
                                const bf16* __restrict__ streams, int hc_mix_n, int hc, int hidden,
                                float hc_eps, int iters) {
  extern __shared__ float sh[];
  float* pre = sh;          // hc
  float* cm = sh + hc;      // hc*hc
  const int m = blockIdx.x;
  const float* mx_row = mix + static_cast<long long>(m) * hc_mix_n;
  const bf16* s = streams + static_cast<long long>(m) * hc * hidden;
  float* post_row = post + static_cast<long long>(m) * hc;
  float* comb_row = comb + static_cast<long long>(m) * hc * hc;
  bf16* col_row = collapsed + static_cast<long long>(m) * hidden;

  if (threadIdx.x == 0) {
    const float pre_s = scale[0], post_s = scale[1], comb_s = scale[2];
    for (int i = 0; i < hc; ++i) pre[i] = sigmoidf(mx_row[i] * pre_s + base[i]) + hc_eps;
    for (int i = 0; i < hc; ++i)
      post_row[i] = 2.0f * sigmoidf(mx_row[hc + i] * post_s + base[hc + i]);
    for (int r = 0; r < hc; ++r) {
      float mxv = -CUDART_INF_F;
      for (int c = 0; c < hc; ++c) {
        const int o = r * hc + c;
        cm[o] = mx_row[2 * hc + o] * comb_s + base[2 * hc + o];
        mxv = fmaxf(mxv, cm[o]);
      }
      float sum = 0.0f;
      for (int c = 0; c < hc; ++c) {
        cm[r * hc + c] = __expf(cm[r * hc + c] - mxv);
        sum += cm[r * hc + c];
      }
      for (int c = 0; c < hc; ++c) cm[r * hc + c] = cm[r * hc + c] / sum + hc_eps;
    }
    // Sinkhorn-Knopp: the reference normalises columns once, then alternates
    // rows and columns for iters-1 more rounds.
    for (int c = 0; c < hc; ++c) {
      float s2 = 0.0f;
      for (int r = 0; r < hc; ++r) s2 += cm[r * hc + c];
      for (int r = 0; r < hc; ++r) cm[r * hc + c] /= (s2 + hc_eps);
    }
    for (int it = 0; it < iters - 1; ++it) {
      for (int r = 0; r < hc; ++r) {
        float s2 = 0.0f;
        for (int c = 0; c < hc; ++c) s2 += cm[r * hc + c];
        for (int c = 0; c < hc; ++c) cm[r * hc + c] /= (s2 + hc_eps);
      }
      for (int c = 0; c < hc; ++c) {
        float s2 = 0.0f;
        for (int r = 0; r < hc; ++r) s2 += cm[r * hc + c];
        for (int r = 0; r < hc; ++r) cm[r * hc + c] /= (s2 + hc_eps);
      }
    }
    for (int i = 0; i < hc * hc; ++i) comb_row[i] = cm[i];
  }
  __syncthreads();
  for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < hc; ++i) acc += pre[i] * f(s[i * hidden + d]);
    col_row[d] = b(acc);
  }
}

__global__ void hc_combine_kernel(bf16* __restrict__ out, const float* __restrict__ post,
                                  const bf16* __restrict__ y, const float* __restrict__ comb,
                                  const bf16* __restrict__ residual, int hc, int hidden) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= hidden) return;
  const int m = blockIdx.y;
  const float* post_row = post + static_cast<long long>(m) * hc;
  const float* comb_row = comb + static_cast<long long>(m) * hc * hc;
  const bf16* y_row = y + static_cast<long long>(m) * hidden;
  const bf16* res_row = residual + static_cast<long long>(m) * hc * hidden;
  bf16* out_row = out + static_cast<long long>(m) * hc * hidden;
  const float yv = f(y_row[d]);
  for (int i = 0; i < hc; ++i) {
    float acc = post_row[i] * yv;
    for (int j = 0; j < hc; ++j) acc += comb_row[j * hc + i] * f(res_row[j * hidden + d]);
    out_row[i * hidden + d] = b(acc);
  }
}

__global__ void hc_head_mean_kernel(bf16* __restrict__ out, const bf16* __restrict__ streams,
                                    int hc, int hidden) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= hidden) return;
  const int m = blockIdx.y;
  const bf16* s = streams + static_cast<long long>(m) * hc * hidden;
  float acc = 0.0f;
  for (int i = 0; i < hc; ++i) acc += f(s[i * hidden + d]);
  out[static_cast<long long>(m) * hidden + d] = b(acc / static_cast<float>(hc));
}

// ----------------------------------------------------------------------- KDA

__global__ void kda_conv_kernel(bf16* __restrict__ out, bf16* __restrict__ state,
                                const bf16* __restrict__ in, const bf16* __restrict__ weight,
                                int channels, int kernel) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const int m = blockIdx.y;
  const int taps = kernel - 1;
  const bf16* w = weight + static_cast<long long>(c) * kernel;
  bf16* st = state + (static_cast<long long>(m) * channels + c) * taps;
  const bf16* inr = in + static_cast<long long>(m) * channels;
  bf16* outr = out + static_cast<long long>(m) * channels;
  float acc = 0.0f;
  for (int t = 0; t < taps; ++t) acc += f(w[t]) * f(st[t]);
  const float xv = f(inr[c]);
  acc += f(w[taps]) * xv;
  for (int t = 0; t < taps - 1; ++t) st[t] = st[t + 1];
  st[taps - 1] = b(xv);
  outr[c] = b(siluf(acc));
}

__global__ void kda_forget_kernel(bf16* __restrict__ g, const bf16* __restrict__ fb,
                                  const float* __restrict__ dt_bias,
                                  const float* __restrict__ a_log, int heads, int head_dim,
                                  float lower_bound) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = heads * head_dim;
  if (i >= n) return;
  const int m = blockIdx.y;
  const int h = i / head_dim;
  const float decay = __expf(a_log[h]);
  g[static_cast<long long>(m) * n + i] =
      b(lower_bound * sigmoidf(decay * (f(fb[static_cast<long long>(m) * n + i]) + dt_bias[i])));
}

__global__ void kda_norm_qk_kernel(bf16* __restrict__ q, bf16* __restrict__ k, int head_dim,
                                   int row_stride, float q_scale) {
  const int h = blockIdx.x;
  const int m = blockIdx.y;
  const long long base = static_cast<long long>(m) * row_stride + static_cast<long long>(h) * head_dim;
  __shared__ float red[32];
  float sq = 0.0f, sk = 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    const float a = f(q[base + i]);
    const float c = f(k[base + i]);
    sq += a * a;
    sk += c * c;
  }
  const float nq = block_sum(sq, red);
  __syncthreads();
  const float nk = block_sum(sk, red);
  // FLA divides by sqrt(sum + eps) rather than clamping the norm.
  const float iq = 1.0f / sqrtf(nq + 1e-6f);
  const float ik = 1.0f / sqrtf(nk + 1e-6f);
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    q[base + i] = b(f(q[base + i]) * iq * q_scale);
    k[base + i] = b(f(k[base + i]) * ik);
  }
}

__global__ void sigmoid_kernel(bf16* __restrict__ out, const bf16* __restrict__ in, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = b(sigmoidf(f(in[i])));
}

// One block per (head, stream). Thread j owns value column j; the state tile
// is staged in shared so the recurrence reads it once and writes it once.
__global__ void kda_step_kernel(float* __restrict__ state, bf16* __restrict__ o,
                                const bf16* __restrict__ q, const bf16* __restrict__ k,
                                const bf16* __restrict__ v, const bf16* __restrict__ g,
                                const bf16* __restrict__ beta, int heads, int head_dim,
                                int row_stride) {
  extern __shared__ float tile[];
  float* sq = tile + head_dim * head_dim;
  float* sk = sq + head_dim;
  float* sg = sk + head_dim;

  const int h = blockIdx.x;
  const int m = blockIdx.y;
  const int j = threadIdx.x;
  const long long sbase =
      (static_cast<long long>(m) * heads + h) * head_dim * head_dim;
  const long long vbase = static_cast<long long>(m) * row_stride + static_cast<long long>(h) * head_dim;

  for (int t = j; t < head_dim * head_dim; t += blockDim.x) tile[t] = state[sbase + t];
  sq[j] = f(q[vbase + j]);
  sk[j] = f(k[vbase + j]);
  sg[j] = __expf(f(g[vbase + j]));
  __syncthreads();

  const float beta_h = f(beta[static_cast<long long>(m) * heads + h]);
  float u = 0.0f;
  for (int i = 0; i < head_dim; ++i) u += sg[i] * tile[i * head_dim + j] * sk[i];
  const float d = beta_h * (f(v[vbase + j]) - u);

  float acc = 0.0f;
  for (int i = 0; i < head_dim; ++i) {
    const float s = sg[i] * tile[i * head_dim + j] + sk[i] * d;
    tile[i * head_dim + j] = s;
    acc += s * sq[i];
  }
  __syncthreads();
  for (int t = j; t < head_dim * head_dim; t += blockDim.x) state[sbase + t] = tile[t];
  o[vbase + j] = b(acc);
}

__global__ void kda_gated_norm_kernel(bf16* __restrict__ out, const bf16* __restrict__ x,
                                      const bf16* __restrict__ gate, const bf16* __restrict__ w,
                                      int head_dim, float eps) {
  __shared__ float red[32];
  const int h = blockIdx.x;
  const int m = blockIdx.y;
  const long long base = (static_cast<long long>(m) * gridDim.x + h) * head_dim;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    const float v = f(x[base + i]);
    acc += v * v;
  }
  const float inv = rsqrtf(block_sum(acc, red) / static_cast<float>(head_dim) + eps);
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    out[base + i] = b(f(x[base + i]) * inv * f(w[i]) * sigmoidf(f(gate[base + i])));
}

// -------------------------------------------------------------- paged KV writes

__global__ void kv_write_latent_kernel(KvPages kv, const bf16* __restrict__ src,
                                       const int* __restrict__ pos, int layer_slot) {
  const int m = blockIdx.x;
  int page = 0, slot = 0;
  kv_locate(kv, m, pos[m], &page, &slot);
  const long long dst_base =
      ((static_cast<long long>(page) * kv.layers + layer_slot) * kv.page_tokens + slot) * kv.kv_lora;
  const bf16* s = src + static_cast<long long>(m) * kv.kv_lora;
  for (int c = threadIdx.x; c < kv.kv_lora; c += blockDim.x) kv.latent[dst_base + c] = s[c];
}

__global__ void kv_write_index_kernel(KvPages kv, const bf16* __restrict__ key_src,
                                      const bf16* __restrict__ gate_src, const int* __restrict__ pos,
                                      int layer_slot) {
  const int m = blockIdx.x;
  int page = 0, slot = 0;
  kv_locate(kv, m, pos[m], &page, &slot);
  const long long dst_base = ((static_cast<long long>(page) * kv.layers + layer_slot) *
                              kv.page_tokens + slot) * kv.index_head_dim;
  const bf16* ks = key_src + static_cast<long long>(m) * kv.index_head_dim;
  const bf16* gs = gate_src + static_cast<long long>(m) * kv.index_head_dim;
  for (int c = threadIdx.x; c < kv.index_head_dim; c += blockDim.x) {
    kv.key[dst_base + c] = ks[c];
    kv.gate[dst_base + c] = gs[c];
  }
}

// ----------------------------------------------------------------------- MLA

__global__ void mla_absorb_q_kernel(float* __restrict__ q_abs, const bf16* __restrict__ kv_b,
                                    const bf16* __restrict__ q, int nope, int v_dim, int kv_lora) {
  const int h = blockIdx.x;
  const int m = blockIdx.y;
  const int c = threadIdx.x;
  const long long row0 = static_cast<long long>(h) * (nope + v_dim);
  const bf16* q_row = q + (static_cast<long long>(m) * gridDim.x + h) * nope;
  float acc = 0.0f;
  for (int d = 0; d < nope; ++d) acc += f(kv_b[(row0 + d) * kv_lora + c]) * f(q_row[d]);
  q_abs[(static_cast<long long>(m) * gridDim.x + h) * kv_lora + c] = acc;
}

__global__ void mla_scores_kernel(float* __restrict__ scores, const float* __restrict__ q_abs,
                                  KvPages kv, const int* __restrict__ sel,
                                  const int* __restrict__ n_sel, int sel_stride, int layer_slot,
                                  int heads, int kv_lora, float scaling) {
  extern __shared__ float lat[];
  const int i = blockIdx.x;
  const int m = blockIdx.y;
  if (i >= n_sel[m]) return;
  const int tok = sel[static_cast<long long>(m) * sel_stride + i];
  int page = 0, slot = 0;
  kv_locate(kv, m, tok, &page, &slot);
  const long long src =
      ((static_cast<long long>(page) * kv.layers + layer_slot) * kv.page_tokens + slot) * kv_lora;
  for (int c = threadIdx.x; c < kv_lora; c += blockDim.x) lat[c] = f(kv.latent[src + c]);
  __syncthreads();
  __shared__ float red[32];
  const float* qa = q_abs + static_cast<long long>(m) * heads * kv_lora;
  float* out_row = scores + static_cast<long long>(m) * heads * sel_stride;
  for (int h = 0; h < heads; ++h) {
    float acc = 0.0f;
    for (int c = threadIdx.x; c < kv_lora; c += blockDim.x) acc += qa[h * kv_lora + c] * lat[c];
    const float total = block_sum(acc, red);
    if (threadIdx.x == 0) out_row[h * sel_stride + i] = total * scaling;
    __syncthreads();
  }
}

__global__ void mla_softmax_kernel(float* __restrict__ scores, const int* __restrict__ n_sel,
                                   int sel_stride, int heads) {
  __shared__ float red[32];
  __shared__ float mx_s;
  const int h = blockIdx.x;
  const int m = blockIdx.y;
  const int n = n_sel[m];
  float* row = scores + (static_cast<long long>(m) * heads + h) * sel_stride;
  float mx = -CUDART_INF_F;
  for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, row[i]);
  for (int off = kWarp / 2; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
  if (threadIdx.x % kWarp == 0) red[threadIdx.x / kWarp] = mx;
  __syncthreads();
  if (threadIdx.x == 0) {
    float mm = -CUDART_INF_F;
    const int warps = (blockDim.x + kWarp - 1) / kWarp;
    for (int i = 0; i < warps; ++i) mm = fmaxf(mm, red[i]);
    mx_s = mm;
  }
  __syncthreads();
  float sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float e = __expf(row[i] - mx_s);
    row[i] = e;
    sum += e;
  }
  const float total = block_sum(sum, red);
  for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] /= total;
}

__global__ void mla_context_kernel(float* __restrict__ ctx, const float* __restrict__ scores,
                                   KvPages kv, const int* __restrict__ sel,
                                   const int* __restrict__ n_sel, int sel_stride, int layer_slot,
                                   int kv_lora) {
  const int h = blockIdx.x;
  const int m = blockIdx.y;
  const int c = threadIdx.x;
  const int n = n_sel[m];
  float acc = 0.0f;
  const float* row = scores + (static_cast<long long>(m) * gridDim.x + h) * sel_stride;
  const int* sel_row = sel + static_cast<long long>(m) * sel_stride;
  for (int i = 0; i < n; ++i) {
    int page = 0, slot = 0;
    kv_locate(kv, m, sel_row[i], &page, &slot);
    const long long src =
        ((static_cast<long long>(page) * kv.layers + layer_slot) * kv.page_tokens + slot) * kv_lora;
    acc += row[i] * f(kv.latent[src + c]);
  }
  ctx[(static_cast<long long>(m) * gridDim.x + h) * kv_lora + c] = acc;
}

__global__ void mla_expand_v_kernel(bf16* __restrict__ out, const bf16* __restrict__ kv_b,
                                    const float* __restrict__ ctx, int nope, int v_dim,
                                    int kv_lora) {
  __shared__ float red[32];
  const int h = blockIdx.x;
  const int vd = blockIdx.y;
  const int m = blockIdx.z;
  const long long row = (static_cast<long long>(h) * (nope + v_dim)) + nope + vd;
  const float* ctx_row = ctx + (static_cast<long long>(m) * gridDim.x + h) * kv_lora;
  float acc = 0.0f;
  for (int c = threadIdx.x; c < kv_lora; c += blockDim.x)
    acc += f(kv_b[row * kv_lora + c]) * ctx_row[c];
  const float total = block_sum(acc, red);
  if (threadIdx.x == 0) out[(static_cast<long long>(m) * gridDim.x + h) * v_dim + vd] = b(total);
}

// ------------------------------------------------------------------- indexer

__global__ void indexer_pool_kernel(bf16* __restrict__ pool_keys, KvPages kv,
                                    const bf16* __restrict__ ape, const int* __restrict__ n_pools,
                                    int pool_stride, int layer_slot, int kpool, int head_dim) {
  const int p = blockIdx.x;
  const int m = blockIdx.y;
  if (p >= n_pools[m]) return;
  const int c = threadIdx.x;
  if (c >= head_dim) return;
  float mx = -CUDART_INF_F;
  float logit[8];
  for (int i = 0; i < kpool; ++i) {
    const int tok = p * kpool + i;
    int page = 0, slot = 0;
    kv_locate(kv, m, tok, &page, &slot);
    const long long src =
        ((static_cast<long long>(page) * kv.layers + layer_slot) * kv.page_tokens + slot) * head_dim;
    logit[i] = f(kv.gate[src + c]) + f(ape[i * head_dim + c]);
    mx = fmaxf(mx, logit[i]);
  }
  float sum = 0.0f;
  for (int i = 0; i < kpool; ++i) {
    logit[i] = __expf(logit[i] - mx);
    sum += logit[i];
  }
  float acc = 0.0f;
  for (int i = 0; i < kpool; ++i) {
    const int tok = p * kpool + i;
    int page = 0, slot = 0;
    kv_locate(kv, m, tok, &page, &slot);
    const long long src =
        ((static_cast<long long>(page) * kv.layers + layer_slot) * kv.page_tokens + slot) * head_dim;
    acc += (logit[i] / sum) * f(kv.key[src + c]);
  }
  pool_keys[(static_cast<long long>(m) * pool_stride + p) * head_dim + c] = b(acc);
}

__global__ void indexer_scores_kernel(float* __restrict__ scores, const bf16* __restrict__ q,
                                      const bf16* __restrict__ pool_keys,
                                      const float* __restrict__ head_w,
                                      const int* __restrict__ n_pools, int pool_stride, int heads,
                                      int head_dim, float softmax_scale) {
  extern __shared__ float pk[];
  const int p = blockIdx.x;
  const int m = blockIdx.y;
  if (p >= n_pools[m]) return;
  const bf16* pk_src = pool_keys + (static_cast<long long>(m) * pool_stride + p) * head_dim;
  for (int c = threadIdx.x; c < head_dim; c += blockDim.x) pk[c] = f(pk_src[c]);
  __syncthreads();
  __shared__ float red[32];
  const bf16* q_row = q + static_cast<long long>(m) * heads * head_dim;
  float total = 0.0f;
  for (int h = 0; h < heads; ++h) {
    float acc = 0.0f;
    for (int c = threadIdx.x; c < head_dim; c += blockDim.x)
      acc += f(q_row[h * head_dim + c]) * pk[c];
    const float dot = block_sum(acc, red);
    if (threadIdx.x == 0) total += head_w[h] * fmaxf(dot * softmax_scale, 0.0f);
    __syncthreads();
  }
  if (threadIdx.x == 0) scores[static_cast<long long>(m) * pool_stride + p] = total;
}

__device__ __forceinline__ unsigned int order_key(float v) {
  unsigned int u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One block per stream. Finds the select_k-th largest score by four 8-bit
// radix passes, then emits every pool above the threshold followed by as many
// ties as the budget still allows.
__global__ void indexer_select_kernel(int* __restrict__ sel, int* __restrict__ n_out,
                                      const float* __restrict__ scores,
                                      const int* __restrict__ n_pools, int pool_stride,
                                      int sel_pool_stride, int select_k) {
  __shared__ unsigned int hist[256];
  __shared__ unsigned int prefix_hi;
  __shared__ int remaining;
  __shared__ int count;

  const int m = blockIdx.x;
  const int n_p = n_pools[m];
  const float* row = scores + static_cast<long long>(m) * pool_stride;
  int* sel_row = sel + static_cast<long long>(m) * sel_pool_stride;

  if (n_p <= select_k) {
    for (int i = threadIdx.x; i < n_p; i += blockDim.x) sel_row[i] = i;
    if (threadIdx.x == 0) n_out[m] = n_p;
    return;
  }
  if (threadIdx.x == 0) {
    prefix_hi = 0u;
    remaining = select_k;
  }
  __syncthreads();

  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
    __syncthreads();
    const unsigned int mask = (shift == 24) ? 0u : (0xffffffffu << (shift + 8));
    for (int i = threadIdx.x; i < n_p; i += blockDim.x) {
      const unsigned int key = order_key(row[i]);
      if ((key & mask) == prefix_hi) atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int left = remaining;
      int bucket = 255;
      for (; bucket >= 0; --bucket) {
        if (static_cast<int>(hist[bucket]) >= left) break;
        left -= static_cast<int>(hist[bucket]);
      }
      remaining = left;
      prefix_hi = prefix_hi | (static_cast<unsigned int>(bucket) << shift);
    }
    __syncthreads();
  }

  // prefix_hi is now the order-key of the select_k-th largest score.
  if (threadIdx.x == 0) count = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < n_p; i += blockDim.x) {
    if (order_key(row[i]) > prefix_hi) {
      const int slot = atomicAdd(&count, 1);
      if (slot < select_k) sel_row[slot] = i;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n_p; i += blockDim.x) {
    if (order_key(row[i]) == prefix_hi) {
      const int slot = atomicAdd(&count, 1);
      if (slot < select_k) sel_row[slot] = i;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) n_out[m] = min(count, select_k);
}

__global__ void indexer_expand_kernel(int* __restrict__ tokens, int* __restrict__ n_out,
                                      const int* __restrict__ sel_pools,
                                      const int* __restrict__ n_sel,
                                      const int* __restrict__ n_tokens, int sel_pool_stride,
                                      int sel_stride, int kpool) {
  const int m = blockIdx.y;
  const int ns = n_sel[m];
  const int n_tok = n_tokens[m];
  const int tail_count = n_tok % kpool;
  const int tail_start = n_tok - tail_count;
  const int total = ns * kpool + tail_count;
  const int* sel_row = sel_pools + static_cast<long long>(m) * sel_pool_stride;
  int* tok_row = tokens + static_cast<long long>(m) * sel_stride;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
    if (i < ns * kpool) {
      tok_row[i] = sel_row[i / kpool] * kpool + (i % kpool);
    } else {
      tok_row[i] = tail_start + (i - ns * kpool);
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) n_out[m] = total;
}

// ------------------------------------------------------------------- MoE

__global__ void moe_router_kernel(int* __restrict__ topk_idx, float* __restrict__ topk_w,
                                  const float* __restrict__ logits, const float* __restrict__ bias,
                                  int n_experts, int top_k, bool norm_topk, float scale) {
  extern __shared__ float sc[];
  float* choice = sc + n_experts;
  const int m = blockIdx.x;
  const float* logits_row = logits + static_cast<long long>(m) * n_experts;
  int* idx_row = topk_idx + static_cast<long long>(m) * top_k;
  float* w_row = topk_w + static_cast<long long>(m) * top_k;
  for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
    const float s = sigmoidf(logits_row[i]);
    sc[i] = s;
    choice[i] = s + bias[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int t = 0; t < top_k; ++t) {
      int best = -1;
      float bestv = -CUDART_INF_F;
      for (int i = 0; i < n_experts; ++i) {
        if (choice[i] > bestv) {
          bestv = choice[i];
          best = i;
        }
      }
      choice[best] = -CUDART_INF_F;
      idx_row[t] = best;
      w_row[t] = sc[best];
      sum += sc[best];
    }
    const float denom = norm_topk ? (sum + 1e-20f) : 1.0f;
    for (int t = 0; t < top_k; ++t) w_row[t] = w_row[t] / denom * scale;
  }
}

__global__ void swiglu_kernel(bf16* __restrict__ out, const bf16* __restrict__ gate,
                              const bf16* __restrict__ up, int n, float limit) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = fminf(f(gate[i]), limit);
  const float u = fminf(fmaxf(f(up[i]), -limit), limit);
  out[i] = b(siluf(g) * u);
}

__constant__ float kE2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f,  3.0f,  4.0f,  6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// --------------------------------------------------- NVFP4 grouped-GEMM glue
//
// Device-side mirrors of nvfp4.cc's host codecs and CUTLASS SFA/SFB swizzle
// (nvfp4.h::SfLayout::offset), so activation quantization for the grouped
// GEMM path (nvfp4_quantize_rows) matches the loader's weight quantization
// bit-for-bit under test, without a host round trip per decode step. Kept as
// a plain 127-code linear search, same as nvfp4.cc::float_to_e4m3: rows here
// are gathered activations, at most a few hundred per step, so the search
// cost is nothing next to the GEMM it feeds.

__device__ __forceinline__ float e4m3_to_float_dev(std::uint8_t bits) {
  const std::uint32_t sign = (bits & 0x80u) ? 0x80000000u : 0u;
  const std::uint32_t exp = (bits >> 3) & 0x0Fu;
  const std::uint32_t mant = bits & 0x07u;
  if (exp == 0) {
    if (mant == 0) return __uint_as_float(sign);
    const float v = static_cast<float>(mant) * (1.0f / 8.0f) * 0.015625f;
    return (sign != 0u) ? -v : v;
  }
  return __uint_as_float(sign | ((exp + 120u) << 23) | (mant << 20));
}

__device__ __forceinline__ std::uint8_t float_to_e4m3_dev(float v) {
  const std::uint8_t sign = (v < 0.0f) ? 0x80 : 0x00;
  const float a = fabsf(v);
  if (a >= 464.0f) return static_cast<std::uint8_t>(sign | 0x7E);
  std::uint8_t best = sign;
  float best_err = a;
  for (unsigned bits = 0; bits <= 0x7Eu; ++bits) {
    const float c = e4m3_to_float_dev(static_cast<std::uint8_t>(bits));
    const float err = fabsf(a - c);
    if (err < best_err || (err == best_err && (bits & 1u) == 0u)) {
      best_err = err;
      best = static_cast<std::uint8_t>(sign | bits);
    }
  }
  return best;
}

__device__ __forceinline__ std::uint8_t float_to_e2m1_dev(float v) {
  const std::uint8_t sign = (v < 0.0f) ? 0x08 : 0x00;
  const float a = fabsf(v);
  int best = 0;
  float best_err = a;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float mag = kE2M1[i];
    const float err = fabsf(a - mag);
    if (err < best_err || (err == best_err && (i & 1) == 0)) {
      best_err = err;
      best = i;
    }
  }
  return static_cast<std::uint8_t>(sign | best);
}

// nvfp4.h::SfLayout::offset, device side. Blk_MN=128, Blk_SF=4, 512 B/atom.
__device__ __forceinline__ std::size_t sf_swizzle_offset(long long mn_idx, long long sf_idx,
                                                         long long k_tiles) {
  constexpr int kBlkMN = 128, kBlkSF = 4, kAtomBytes = 512;
  const long long mn_tile = mn_idx / kBlkMN;
  const long long mn_in = mn_idx % kBlkMN;
  const long long k_tile = sf_idx / kBlkSF;
  const long long ssub = sf_idx % kBlkSF;
  const long long atom = mn_tile * k_tiles + k_tile;
  return static_cast<std::size_t>(atom) * kAtomBytes +
         static_cast<std::size_t>((mn_in % 32) * 16 + (mn_in / 32) * 4 + ssub);
}

__global__ void nvfp4_quantize_rows_kernel(std::uint8_t* __restrict__ packed,
                                           std::uint8_t* __restrict__ sf_swizzled,
                                           const bf16* __restrict__ x,
                                           const int* __restrict__ row_in_group,
                                           const int* __restrict__ group_of_row,
                                           const long long* __restrict__ group_sf_base, int k) {
  const long long row = blockIdx.x;
  const long long k_sf = k / 16;
  const long long k_tiles = (k_sf + 3) / 4;
  const long long mn_idx = row_in_group[row];
  const long long sf_base = group_sf_base[group_of_row[row]];
  const bf16* xr = x + row * k;
  std::uint8_t* pr = packed + row * (k / 2);
  for (long long blk = threadIdx.x; blk < k_sf; blk += blockDim.x) {
    const long long base = blk * 16;
    float vals[16];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      vals[j] = f(xr[base + j]);
      amax = fmaxf(amax, fabsf(vals[j]));
    }
    std::uint8_t sbyte = 0;
    float sdeq = 0.0f;
    if (amax > 0.0f) {
      sbyte = float_to_e4m3_dev(amax * (1.0f / 6.0f));
      sdeq = e4m3_to_float_dev(sbyte);
    }
    sf_swizzled[sf_base + sf_swizzle_offset(mn_idx, blk, k_tiles)] = sbyte;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      std::uint8_t lo = 0, hi = 0;
      if (sdeq > 0.0f) {
        lo = float_to_e2m1_dev(vals[2 * j] / sdeq);
        hi = float_to_e2m1_dev(vals[2 * j + 1] / sdeq);
      }
      pr[blk * 8 + j] = static_cast<std::uint8_t>(lo | (hi << 4));
    }
  }
}

__global__ void swiglu_grouped_kernel(bf16* __restrict__ out, const bf16* __restrict__ gu,
                                      const float* __restrict__ gate_global,
                                      const float* __restrict__ up_global,
                                      const int* __restrict__ group_of_row, int inter,
                                      float limit) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= inter) return;
  const long long row = blockIdx.y;
  const int g = group_of_row[row];
  const bf16* gr = gu + row * (2LL * inter);
  const float gate = fminf(f(gr[i]) * gate_global[g], limit);
  const float up = fminf(fmaxf(f(gr[inter + i]) * up_global[g], -limit), limit);
  out[row * inter + i] = b(siluf(gate) * up);
}

__global__ void moe_gather_rows_kernel(bf16* __restrict__ out, const bf16* __restrict__ x,
                                       const int* __restrict__ row_of, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long r = blockIdx.y;
  const long long src = row_of[r];
  out[r * n + i] = x[src * n + i];
}

// See kernels.h: the sum for a given accumulator row must not depend on
// thread scheduling, so this assigns one block per accumulator row (not per
// gathered row) and has it scan every gathered row itself, in ascending i.
// rows is at most a few hundred (max_batch * top_k), so the scan is cheap
// next to the column width it runs per row_of match.
__global__ void moe_scatter_add_kernel(bf16* __restrict__ acc, const bf16* __restrict__ y,
                                       const int* __restrict__ row_of,
                                       const float* __restrict__ weight_of, int rows, int n) {
  extern __shared__ unsigned char smem[];
  int* s_row = reinterpret_cast<int*>(smem);
  float* s_w = reinterpret_cast<float*>(s_row + rows);
  for (int i = threadIdx.x; i < rows; i += blockDim.x) {
    s_row[i] = row_of[i];
    s_w[i] = weight_of[i];
  }
  __syncthreads();
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= n) return;
  const int m = blockIdx.y;
  float sum = f(acc[static_cast<long long>(m) * n + col]);
  for (int i = 0; i < rows; ++i)
    if (s_row[i] == m) sum += s_w[i] * f(y[static_cast<long long>(i) * n + col]);
  acc[static_cast<long long>(m) * n + col] = b(sum);
}


__global__ void gemv_nvfp4_kernel(bf16* __restrict__ y, const std::uint8_t* __restrict__ packed,
                                  const std::uint8_t* __restrict__ block_scale, float global_scale,
                                  const bf16* __restrict__ x, int k) {
  __shared__ float red[32];
  const long long row = blockIdx.x;
  const std::uint8_t* pr = packed + row * (k / 2);
  const std::uint8_t* sr = block_scale + row * (k / 16);
  float acc = 0.0f;
  // One iteration covers a 16-element scale block: 8 packed bytes.
  for (int blk = threadIdx.x; blk < k / 16; blk += blockDim.x) {
    const float sf = static_cast<float>(reinterpret_cast<const __nv_fp8_e4m3*>(sr)[blk]);
    float part = 0.0f;
    const int base = blk * 16;
    for (int j = 0; j < 8; ++j) {
      const std::uint8_t byte = pr[blk * 8 + j];
      part += kE2M1[byte & 0x0Fu] * f(x[base + 2 * j]);
      part += kE2M1[byte >> 4] * f(x[base + 2 * j + 1]);
    }
    acc += part * sf;
  }
  const float total = block_sum(acc, red);
  if (threadIdx.x == 0) y[row] = b(total * global_scale);
}

// -------------------------------------------------------------------- misc

__global__ void axpy_kernel(bf16* __restrict__ acc, const bf16* __restrict__ v,
                            const float* __restrict__ w, int widx, int w_stride, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int m = blockIdx.y;
  bf16* acc_row = acc + static_cast<long long>(m) * n;
  const bf16* v_row = v + static_cast<long long>(m) * n;
  acc_row[i] = b(f(acc_row[i]) + w[static_cast<long long>(m) * w_stride + widx] * f(v_row[i]));
}

__global__ void add_kernel(bf16* __restrict__ acc, const bf16* __restrict__ v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) acc[i] = b(f(acc[i]) + f(v[i]));
}

__global__ void argmax_kernel(int* __restrict__ idx_out, float* __restrict__ val_out,
                              const float* __restrict__ x, int n) {
  __shared__ float vs[1024];
  __shared__ int is[1024];
  const int m = blockIdx.x;
  const float* xr = x + static_cast<long long>(m) * n;
  float best = -CUDART_INF_F;
  int bi = -1;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (xr[i] > best) {
      best = xr[i];
      bi = i;
    }
  }
  vs[threadIdx.x] = best;
  is[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s && vs[threadIdx.x + s] > vs[threadIdx.x]) {
      vs[threadIdx.x] = vs[threadIdx.x + s];
      is[threadIdx.x] = is[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    idx_out[m] = is[0];
    val_out[m] = vs[0];
  }
}

__global__ void sumsq_kernel(float* __restrict__ out, const bf16* __restrict__ x, int n) {
  __shared__ float red[32];
  const int m = blockIdx.x;
  const bf16* xr = x + static_cast<long long>(m) * n;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = f(xr[i]);
    acc += v * v;
  }
  const float total = block_sum(acc, red);
  if (threadIdx.x == 0) out[m] = total;
}

__global__ void absmax_kernel(float* __restrict__ out, const bf16* __restrict__ x, int n) {
  __shared__ float red[32];
  const int m = blockIdx.x;
  const bf16* xr = x + static_cast<long long>(m) * n;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) acc = fmaxf(acc, fabsf(f(xr[i])));
  for (int off = kWarp / 2; off > 0; off >>= 1)
    acc = fmaxf(acc, __shfl_down_sync(0xffffffffu, acc, off));
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + kWarp - 1) / kWarp;
    float mx = 0.0f;
    for (int i = 0; i < warps; ++i) mx = fmaxf(mx, red[i]);
    out[m] = mx;
  }
}

}  // namespace

// --------------------------------------------------------------- launchers

void gemm_bf16(bf16* y, const bf16* w, const bf16* x, int batch, int n_rows, int k, cudaStream_t s) {
  gemm_kernel<bf16><<<dim3(n_rows, batch), 256, 0, s>>>(y, w, x, n_rows, k);
}
void gemm_bf16_f32(float* y, const bf16* w, const bf16* x, int batch, int n_rows, int k,
                   cudaStream_t s) {
  gemm_kernel<float><<<dim3(n_rows, batch), 256, 0, s>>>(y, w, x, n_rows, k);
}
void rmsnorm(bf16* out, const bf16* x, const bf16* weight, int batch, int n, float eps,
            cudaStream_t s) {
  rmsnorm_kernel<<<batch, 256, 0, s>>>(out, x, weight, n, eps);
}
void layernorm(bf16* out, const bf16* x, const bf16* w, const bf16* bias, int batch, int n,
              float eps, cudaStream_t s) {
  layernorm_kernel<<<batch, 128, 0, s>>>(out, x, w, bias, n, eps);
}
void embed_streams(bf16* streams, const bf16* embed, const int* tokens, int batch, int hc,
                   int hidden, cudaStream_t s) {
  embed_streams_kernel<<<dim3(hc, batch), 256, 0, s>>>(streams, embed, tokens, hc, hidden);
}
void hc_mix_gemv(float* mix, const bf16* fn, const bf16* streams, int batch, int hc_mix, int hc,
                 int hidden, float eps, cudaStream_t s) {
  hc_mix_kernel<<<dim3(hc_mix, batch), 256, 0, s>>>(mix, fn, streams, hc_mix, hc * hidden, eps);
}
void hc_split(float* post, float* comb, bf16* collapsed, const float* mix, const float* base,
              const float* scale, const bf16* streams, int batch, int hc, int hidden, float hc_eps,
              int sinkhorn_iters, cudaStream_t s) {
  const std::size_t shmem = sizeof(float) * static_cast<std::size_t>(hc + hc * hc);
  const int hc_mix_n = (2 + hc) * hc;
  hc_split_kernel<<<batch, 256, shmem, s>>>(post, comb, collapsed, mix, base, scale, streams,
                                            hc_mix_n, hc, hidden, hc_eps, sinkhorn_iters);
}
void hc_combine(bf16* streams_out, const float* post, const bf16* y, const float* comb,
                const bf16* residual, int batch, int hc, int hidden, cudaStream_t s) {
  hc_combine_kernel<<<dim3((hidden + 255) / 256, batch), 256, 0, s>>>(streams_out, post, y, comb,
                                                                      residual, hc, hidden);
}
void hc_head_mean(bf16* out, const bf16* streams, int batch, int hc, int hidden, cudaStream_t s) {
  hc_head_mean_kernel<<<dim3((hidden + 255) / 256, batch), 256, 0, s>>>(out, streams, hc, hidden);
}
void kda_conv_update(bf16* out, bf16* state, const bf16* in, const bf16* weight, int batch,
                     int channels, int kernel, cudaStream_t s) {
  kda_conv_kernel<<<dim3((channels + 255) / 256, batch), 256, 0, s>>>(out, state, in, weight,
                                                                      channels, kernel);
}
void kda_forget_gate(bf16* g, const bf16* fb, const float* dt_bias, const float* a_log, int batch,
                     int heads, int head_dim, float lower_bound, cudaStream_t s) {
  const int n = heads * head_dim;
  kda_forget_kernel<<<dim3((n + 255) / 256, batch), 256, 0, s>>>(g, fb, dt_bias, a_log, heads,
                                                                 head_dim, lower_bound);
}
void kda_norm_qk(bf16* q, bf16* k, int batch, int heads, int head_dim, int row_stride,
                 cudaStream_t s) {
  kda_norm_qk_kernel<<<dim3(heads, batch), 128, 0, s>>>(q, k, head_dim, row_stride,
                                                        rsqrtf(static_cast<float>(head_dim)));
}
void kda_sigmoid(bf16* out, const bf16* in, int n, cudaStream_t s) {
  sigmoid_kernel<<<(n + 255) / 256, 256, 0, s>>>(out, in, n);
}
void kda_recurrent_step(float* state, bf16* o, const bf16* q, const bf16* k, const bf16* v,
                        const bf16* g, const bf16* beta, int batch, int heads, int head_dim,
                        int row_stride, cudaStream_t s) {
  const std::size_t shmem =
      sizeof(float) * (static_cast<std::size_t>(head_dim) * head_dim + 3 * head_dim);
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kda_step_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shmem));
    configured = true;
  }
  kda_step_kernel<<<dim3(heads, batch), head_dim, shmem, s>>>(state, o, q, k, v, g, beta, heads,
                                                              head_dim, row_stride);
}
void kda_gated_norm(bf16* out, const bf16* x, const bf16* gate, const bf16* weight, int batch,
                    int heads, int head_dim, float eps, cudaStream_t s) {
  kda_gated_norm_kernel<<<dim3(heads, batch), 128, 0, s>>>(out, x, gate, weight, head_dim, eps);
}
void kv_write_latent(const KvPages& kv, const bf16* src, const int* pos, int batch, int layer_slot,
                     cudaStream_t s) {
  kv_write_latent_kernel<<<batch, 128, 0, s>>>(kv, src, pos, layer_slot);
}
void kv_write_index(const KvPages& kv, const bf16* key_src, const bf16* gate_src, const int* pos,
                    int batch, int layer_slot, cudaStream_t s) {
  kv_write_index_kernel<<<batch, 128, 0, s>>>(kv, key_src, gate_src, pos, layer_slot);
}
void mla_absorb_q(float* q_abs, const bf16* kv_b, const bf16* q, int batch, int heads, int nope,
                  int v_dim, int kv_lora, cudaStream_t s) {
  mla_absorb_q_kernel<<<dim3(heads, batch), kv_lora, 0, s>>>(q_abs, kv_b, q, nope, v_dim, kv_lora);
}
void mla_scores(float* scores, const float* q_abs, const KvPages& kv, const int* sel,
                const int* n_sel, int n_sel_max, int sel_stride, int batch, int layer_slot,
                int heads, int kv_lora, float scaling, cudaStream_t s) {
  mla_scores_kernel<<<dim3(n_sel_max, batch), 256, sizeof(float) * kv_lora, s>>>(
      scores, q_abs, kv, sel, n_sel, sel_stride, layer_slot, heads, kv_lora, scaling);
}
void mla_softmax(float* scores, const int* n_sel, int sel_stride, int batch, int heads,
                 cudaStream_t s) {
  mla_softmax_kernel<<<dim3(heads, batch), 256, 0, s>>>(scores, n_sel, sel_stride, heads);
}
void mla_context(float* ctx, const float* scores, const KvPages& kv, const int* sel,
                 const int* n_sel, int n_sel_max, int sel_stride, int batch, int layer_slot,
                 int heads, int kv_lora, cudaStream_t s) {
  (void)n_sel_max;
  mla_context_kernel<<<dim3(heads, batch), kv_lora, 0, s>>>(ctx, scores, kv, sel, n_sel, sel_stride,
                                                            layer_slot, kv_lora);
}
void mla_expand_v(bf16* out, const bf16* kv_b, const float* ctx, int batch, int heads, int nope,
                  int v_dim, int kv_lora, cudaStream_t s) {
  mla_expand_v_kernel<<<dim3(heads, v_dim, batch), 128, 0, s>>>(out, kv_b, ctx, nope, v_dim, kv_lora);
}
void indexer_pool_keys(bf16* pool_keys, const KvPages& kv, const bf16* ape, const int* n_pools,
                       int n_pools_max, int pool_stride, int batch, int layer_slot, int kpool,
                       int head_dim, cudaStream_t s) {
  indexer_pool_kernel<<<dim3(n_pools_max, batch), head_dim, 0, s>>>(pool_keys, kv, ape, n_pools,
                                                                    pool_stride, layer_slot, kpool,
                                                                    head_dim);
}
void indexer_scores(float* scores, const bf16* q, const bf16* pool_keys, const float* head_w,
                    const int* n_pools, int n_pools_max, int pool_stride, int batch, int heads,
                    int head_dim, cudaStream_t s) {
  indexer_scores_kernel<<<dim3(n_pools_max, batch), 128, sizeof(float) * head_dim, s>>>(
      scores, q, pool_keys, head_w, n_pools, pool_stride, heads, head_dim,
      rsqrtf(static_cast<float>(head_dim)));
}
void indexer_select(int* sel_pools, int* n_sel_out, const float* scores, const int* n_pools,
                    int n_pools_max, int pool_stride, int sel_pool_stride, int batch, int select_k,
                    cudaStream_t s) {
  (void)n_pools_max;
  indexer_select_kernel<<<batch, 256, 0, s>>>(sel_pools, n_sel_out, scores, n_pools, pool_stride,
                                              sel_pool_stride, select_k);
}
void indexer_expand(int* sel_tokens, int* n_tokens_out, const int* sel_pools, const int* n_sel,
                    const int* n_tokens, int sel_pool_stride, int sel_stride, int batch, int kpool,
                    cudaStream_t s) {
  const int blocks_x = (sel_stride + 255) / 256;
  indexer_expand_kernel<<<dim3(blocks_x, batch), 256, 0, s>>>(
      sel_tokens, n_tokens_out, sel_pools, n_sel, n_tokens, sel_pool_stride, sel_stride, kpool);
}
void moe_router(int* topk_idx, float* topk_w, const float* logits, const float* bias, int batch,
                int n_experts, int top_k, bool norm_topk, float scale, cudaStream_t s) {
  moe_router_kernel<<<batch, 256, sizeof(float) * 2 * n_experts, s>>>(
      topk_idx, topk_w, logits, bias, n_experts, top_k, norm_topk, scale);
}
void swiglu_clamped(bf16* out, const bf16* gate, const bf16* up, int n, float limit,
                    cudaStream_t s) {
  swiglu_kernel<<<(n + 255) / 256, 256, 0, s>>>(out, gate, up, n, limit);
}
void gemv_nvfp4(bf16* y, const std::uint8_t* packed, const std::uint8_t* block_scale,
                float global_scale, const bf16* x, int n_rows, int k, cudaStream_t s) {
  gemv_nvfp4_kernel<<<n_rows, 256, 0, s>>>(y, packed, block_scale, global_scale, x, k);
}
void nvfp4_quantize_rows(std::uint8_t* packed, std::uint8_t* sf_swizzled, const bf16* x,
                         const int* row_in_group, const int* group_of_row,
                         const long long* group_sf_base, int rows, int k, cudaStream_t s) {
  if (rows <= 0) return;
  nvfp4_quantize_rows_kernel<<<rows, 256, 0, s>>>(packed, sf_swizzled, x, row_in_group,
                                                  group_of_row, group_sf_base, k);
}
void swiglu_grouped(bf16* out, const bf16* gu, const float* gate_global, const float* up_global,
                    const int* group_of_row, int rows, int inter, float limit, cudaStream_t s) {
  if (rows <= 0) return;
  swiglu_grouped_kernel<<<dim3((inter + 255) / 256, rows), 256, 0, s>>>(
      out, gu, gate_global, up_global, group_of_row, inter, limit);
}
void moe_gather_rows(bf16* out, const bf16* x, const int* row_of, int rows, int n, cudaStream_t s) {
  if (rows <= 0) return;
  moe_gather_rows_kernel<<<dim3((n + 255) / 256, rows), 256, 0, s>>>(out, x, row_of, n);
}
void moe_scatter_add(bf16* acc, const bf16* y, const int* row_of, const float* weight_of, int rows,
                     int batch, int n, cudaStream_t s) {
  if (rows <= 0) return;
  const std::size_t shmem = static_cast<std::size_t>(rows) * (sizeof(int) + sizeof(float));
  moe_scatter_add_kernel<<<dim3((n + 255) / 256, batch), 256, shmem, s>>>(acc, y, row_of, weight_of,
                                                                          rows, n);
}
void axpy_bf16(bf16* acc, const bf16* v, const float* w, int widx, int w_stride, int batch, int n,
              cudaStream_t s) {
  axpy_kernel<<<dim3((n + 255) / 256, batch), 256, 0, s>>>(acc, v, w, widx, w_stride, n);
}
void add_bf16(bf16* acc, const bf16* v, int n, cudaStream_t s) {
  add_kernel<<<(n + 255) / 256, 256, 0, s>>>(acc, v, n);
}
void argmax_f32(int* idx_out, float* val_out, const float* x, int batch, int n, cudaStream_t s) {
  argmax_kernel<<<batch, 1024, 0, s>>>(idx_out, val_out, x, n);
}
void sumsq_bf16(float* out, const bf16* x, int batch, int n, cudaStream_t s) {
  sumsq_kernel<<<batch, 256, 0, s>>>(out, x, n);
}
void absmax_bf16(float* out, const bf16* x, int batch, int n, cudaStream_t s) {
  absmax_kernel<<<batch, 256, 0, s>>>(out, x, n);
}

}  // namespace rocket::engine

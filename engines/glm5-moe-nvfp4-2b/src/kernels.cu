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

// --------------------------------------------------------------------- gemv

template <typename Out>
__global__ void gemv_kernel(Out* __restrict__ y, const bf16* __restrict__ w,
                            const bf16* __restrict__ x, int k) {
  __shared__ float red[32];
  const long long row = blockIdx.x;
  const bf16* wr = w + row * static_cast<long long>(k);
  float acc = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) acc += f(wr[i]) * f(x[i]);
  const float total = block_sum(acc, red);
  if (threadIdx.x == 0) {
    if constexpr (sizeof(Out) == 4) {
      y[row] = total;
    } else {
      y[row] = b(total);
    }
  }
}

// ------------------------------------------------------------------- norms

__global__ void rmsnorm_kernel(bf16* __restrict__ out, const bf16* __restrict__ x,
                               const bf16* __restrict__ weight, int n, float eps) {
  __shared__ float red[32];
  float acc = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = f(x[i]);
    acc += v * v;
  }
  const float inv = rsqrtf(block_sum(acc, red) / static_cast<float>(n) + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = b(f(weight[i]) * f(b(f(x[i]) * inv)));
}

__global__ void layernorm_kernel(bf16* __restrict__ out, const bf16* __restrict__ x,
                                 const bf16* __restrict__ w, const bf16* __restrict__ bias, int n,
                                 float eps) {
  __shared__ float red[32];
  __shared__ float mean_s;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) acc += f(x[i]);
  const float mean = block_sum(acc, red) / static_cast<float>(n);
  if (threadIdx.x == 0) mean_s = mean;
  __syncthreads();
  float var = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float d = f(x[i]) - mean_s;
    var += d * d;
  }
  const float inv = rsqrtf(block_sum(var, red) / static_cast<float>(n) + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    out[i] = b((f(x[i]) - mean_s) * inv * f(w[i]) + f(bias[i]));
}

// ------------------------------------------------------ hyper-connections

__global__ void hc_mix_kernel(float* __restrict__ mix, const bf16* __restrict__ fn,
                              const bf16* __restrict__ streams, int flat, float eps) {
  __shared__ float red[32];
  const int row = blockIdx.x;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < flat; i += blockDim.x) {
    const float v = f(streams[i]);
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum(ss, red) / static_cast<float>(flat) + eps);
  const bf16* wr = fn + static_cast<long long>(row) * flat;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < flat; i += blockDim.x) acc += f(wr[i]) * (f(streams[i]) * inv);
  const float total = block_sum(acc, red);
  if (threadIdx.x == 0) mix[row] = total;
}

// One block. hc is 4, so the comb work is 16 scalars and stays on thread 0;
// the stream collapse is the only part wide enough to parallelise.
__global__ void hc_split_kernel(float* __restrict__ post, float* __restrict__ comb,
                                bf16* __restrict__ collapsed, const float* __restrict__ mix,
                                const float* __restrict__ base, const float* __restrict__ scale,
                                const bf16* __restrict__ streams, int hc, int hidden, float hc_eps,
                                int iters) {
  extern __shared__ float sh[];
  float* pre = sh;          // hc
  float* cm = sh + hc;      // hc*hc
  if (threadIdx.x == 0) {
    const float pre_s = scale[0], post_s = scale[1], comb_s = scale[2];
    for (int i = 0; i < hc; ++i) pre[i] = sigmoidf(mix[i] * pre_s + base[i]) + hc_eps;
    for (int i = 0; i < hc; ++i) post[i] = 2.0f * sigmoidf(mix[hc + i] * post_s + base[hc + i]);
    for (int r = 0; r < hc; ++r) {
      float mx = -CUDART_INF_F;
      for (int c = 0; c < hc; ++c) {
        const int o = r * hc + c;
        cm[o] = mix[2 * hc + o] * comb_s + base[2 * hc + o];
        mx = fmaxf(mx, cm[o]);
      }
      float sum = 0.0f;
      for (int c = 0; c < hc; ++c) {
        cm[r * hc + c] = __expf(cm[r * hc + c] - mx);
        sum += cm[r * hc + c];
      }
      for (int c = 0; c < hc; ++c) cm[r * hc + c] = cm[r * hc + c] / sum + hc_eps;
    }
    // Sinkhorn-Knopp: the reference normalises columns once, then alternates
    // rows and columns for iters-1 more rounds.
    for (int c = 0; c < hc; ++c) {
      float s = 0.0f;
      for (int r = 0; r < hc; ++r) s += cm[r * hc + c];
      for (int r = 0; r < hc; ++r) cm[r * hc + c] /= (s + hc_eps);
    }
    for (int it = 0; it < iters - 1; ++it) {
      for (int r = 0; r < hc; ++r) {
        float s = 0.0f;
        for (int c = 0; c < hc; ++c) s += cm[r * hc + c];
        for (int c = 0; c < hc; ++c) cm[r * hc + c] /= (s + hc_eps);
      }
      for (int c = 0; c < hc; ++c) {
        float s = 0.0f;
        for (int r = 0; r < hc; ++r) s += cm[r * hc + c];
        for (int r = 0; r < hc; ++r) cm[r * hc + c] /= (s + hc_eps);
      }
    }
    for (int i = 0; i < hc * hc; ++i) comb[i] = cm[i];
  }
  __syncthreads();
  for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < hc; ++i) acc += pre[i] * f(streams[i * hidden + d]);
    collapsed[d] = b(acc);
  }
}

__global__ void hc_combine_kernel(bf16* __restrict__ out, const float* __restrict__ post,
                                  const bf16* __restrict__ y, const float* __restrict__ comb,
                                  const bf16* __restrict__ residual, int hc, int hidden) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= hidden) return;
  const float yv = f(y[d]);
  for (int i = 0; i < hc; ++i) {
    float acc = post[i] * yv;
    for (int j = 0; j < hc; ++j) acc += comb[j * hc + i] * f(residual[j * hidden + d]);
    out[i * hidden + d] = b(acc);
  }
}

__global__ void hc_head_mean_kernel(bf16* __restrict__ out, const bf16* __restrict__ streams,
                                    int hc, int hidden) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= hidden) return;
  float acc = 0.0f;
  for (int i = 0; i < hc; ++i) acc += f(streams[i * hidden + d]);
  out[d] = b(acc / static_cast<float>(hc));
}

// ----------------------------------------------------------------------- KDA

__global__ void kda_conv_kernel(bf16* __restrict__ out, bf16* __restrict__ state,
                                const bf16* __restrict__ in, const bf16* __restrict__ weight,
                                int channels, int kernel) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const int taps = kernel - 1;
  const bf16* w = weight + static_cast<long long>(c) * kernel;
  bf16* st = state + static_cast<long long>(c) * taps;
  float acc = 0.0f;
  for (int t = 0; t < taps; ++t) acc += f(w[t]) * f(st[t]);
  const float xv = f(in[c]);
  acc += f(w[taps]) * xv;
  for (int t = 0; t < taps - 1; ++t) st[t] = st[t + 1];
  st[taps - 1] = b(xv);
  out[c] = b(siluf(acc));
}

__global__ void kda_forget_kernel(bf16* __restrict__ g, const bf16* __restrict__ fb,
                                  const float* __restrict__ dt_bias,
                                  const float* __restrict__ a_log, int heads, int head_dim,
                                  float lower_bound) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= heads * head_dim) return;
  const int h = i / head_dim;
  const float decay = __expf(a_log[h]);
  g[i] = b(lower_bound * sigmoidf(decay * (f(fb[i]) + dt_bias[i])));
}

__global__ void kda_norm_qk_kernel(bf16* __restrict__ q, bf16* __restrict__ k, int head_dim,
                                   float q_scale) {
  const int h = blockIdx.x;
  const int base = h * head_dim;
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

// One block per head. Thread j owns value column j; the state tile is staged
// in shared so the recurrence reads it once and writes it once.
__global__ void kda_step_kernel(float* __restrict__ state, bf16* __restrict__ o,
                                const bf16* __restrict__ q, const bf16* __restrict__ k,
                                const bf16* __restrict__ v, const bf16* __restrict__ g,
                                const bf16* __restrict__ beta, int head_dim) {
  extern __shared__ float tile[];
  float* sq = tile + head_dim * head_dim;
  float* sk = sq + head_dim;
  float* sg = sk + head_dim;

  const int h = blockIdx.x;
  const int j = threadIdx.x;
  const long long sbase = static_cast<long long>(h) * head_dim * head_dim;
  const int vbase = h * head_dim;

  for (int t = j; t < head_dim * head_dim; t += blockDim.x) tile[t] = state[sbase + t];
  sq[j] = f(q[vbase + j]);
  sk[j] = f(k[vbase + j]);
  sg[j] = __expf(f(g[vbase + j]));
  __syncthreads();

  float u = 0.0f;
  for (int i = 0; i < head_dim; ++i) u += sg[i] * tile[i * head_dim + j] * sk[i];
  const float d = f(beta[h]) * (f(v[vbase + j]) - u);

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
  const int base = h * head_dim;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    const float v = f(x[base + i]);
    acc += v * v;
  }
  const float inv = rsqrtf(block_sum(acc, red) / static_cast<float>(head_dim) + eps);
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    out[base + i] = b(f(x[base + i]) * inv * f(w[i]) * sigmoidf(f(gate[base + i])));
}

// ----------------------------------------------------------------------- MLA

__global__ void mla_absorb_q_kernel(float* __restrict__ q_abs, const bf16* __restrict__ kv_b,
                                    const bf16* __restrict__ q, int nope, int v_dim, int kv_lora) {
  const int h = blockIdx.x;
  const int c = threadIdx.x;
  const long long row0 = static_cast<long long>(h) * (nope + v_dim);
  float acc = 0.0f;
  for (int d = 0; d < nope; ++d)
    acc += f(kv_b[(row0 + d) * kv_lora + c]) * f(q[h * nope + d]);
  q_abs[h * kv_lora + c] = acc;
}

__global__ void mla_scores_kernel(float* __restrict__ scores, const float* __restrict__ q_abs,
                                  const bf16* __restrict__ latent, const int* __restrict__ sel,
                                  int n_sel, int heads, int kv_lora, float scaling) {
  extern __shared__ float lat[];
  const int i = blockIdx.x;
  const int t = sel[i];
  for (int c = threadIdx.x; c < kv_lora; c += blockDim.x)
    lat[c] = f(latent[static_cast<long long>(t) * kv_lora + c]);
  __syncthreads();
  __shared__ float red[32];
  for (int h = 0; h < heads; ++h) {
    float acc = 0.0f;
    for (int c = threadIdx.x; c < kv_lora; c += blockDim.x) acc += q_abs[h * kv_lora + c] * lat[c];
    const float total = block_sum(acc, red);
    if (threadIdx.x == 0) scores[static_cast<long long>(h) * n_sel + i] = total * scaling;
    __syncthreads();
  }
}

__global__ void mla_softmax_kernel(float* __restrict__ scores, int n_sel) {
  __shared__ float red[32];
  __shared__ float mx_s;
  const int h = blockIdx.x;
  float* row = scores + static_cast<long long>(h) * n_sel;
  float mx = -CUDART_INF_F;
  for (int i = threadIdx.x; i < n_sel; i += blockDim.x) mx = fmaxf(mx, row[i]);
  for (int off = kWarp / 2; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
  if (threadIdx.x % kWarp == 0) red[threadIdx.x / kWarp] = mx;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = -CUDART_INF_F;
    const int warps = (blockDim.x + kWarp - 1) / kWarp;
    for (int i = 0; i < warps; ++i) m = fmaxf(m, red[i]);
    mx_s = m;
  }
  __syncthreads();
  float sum = 0.0f;
  for (int i = threadIdx.x; i < n_sel; i += blockDim.x) {
    const float e = __expf(row[i] - mx_s);
    row[i] = e;
    sum += e;
  }
  const float total = block_sum(sum, red);
  for (int i = threadIdx.x; i < n_sel; i += blockDim.x) row[i] /= total;
}

__global__ void mla_context_kernel(float* __restrict__ ctx, const float* __restrict__ scores,
                                   const bf16* __restrict__ latent, const int* __restrict__ sel,
                                   int n_sel, int kv_lora) {
  const int h = blockIdx.x;
  const int c = threadIdx.x;
  float acc = 0.0f;
  const float* row = scores + static_cast<long long>(h) * n_sel;
  for (int i = 0; i < n_sel; ++i)
    acc += row[i] * f(latent[static_cast<long long>(sel[i]) * kv_lora + c]);
  ctx[h * kv_lora + c] = acc;
}

__global__ void mla_expand_v_kernel(bf16* __restrict__ out, const bf16* __restrict__ kv_b,
                                    const float* __restrict__ ctx, int nope, int v_dim,
                                    int kv_lora) {
  __shared__ float red[32];
  const int h = blockIdx.x;
  const int vd = blockIdx.y;
  const long long row = (static_cast<long long>(h) * (nope + v_dim)) + nope + vd;
  float acc = 0.0f;
  for (int c = threadIdx.x; c < kv_lora; c += blockDim.x)
    acc += f(kv_b[row * kv_lora + c]) * ctx[h * kv_lora + c];
  const float total = block_sum(acc, red);
  if (threadIdx.x == 0) out[h * v_dim + vd] = b(total);
}

// ------------------------------------------------------------------- indexer

__global__ void indexer_pool_kernel(bf16* __restrict__ pool_keys, const bf16* __restrict__ keys,
                                    const bf16* __restrict__ gates, const bf16* __restrict__ ape,
                                    int kpool, int head_dim) {
  const int p = blockIdx.x;
  const int c = threadIdx.x;
  if (c >= head_dim) return;
  float mx = -CUDART_INF_F;
  float logit[8];
  for (int i = 0; i < kpool; ++i) {
    const long long t = static_cast<long long>(p) * kpool + i;
    logit[i] = f(gates[t * head_dim + c]) + f(ape[i * head_dim + c]);
    mx = fmaxf(mx, logit[i]);
  }
  float sum = 0.0f;
  for (int i = 0; i < kpool; ++i) {
    logit[i] = __expf(logit[i] - mx);
    sum += logit[i];
  }
  float acc = 0.0f;
  for (int i = 0; i < kpool; ++i) {
    const long long t = static_cast<long long>(p) * kpool + i;
    acc += (logit[i] / sum) * f(keys[t * head_dim + c]);
  }
  pool_keys[static_cast<long long>(p) * head_dim + c] = b(acc);
}

__global__ void indexer_scores_kernel(float* __restrict__ scores, const bf16* __restrict__ q,
                                      const bf16* __restrict__ pool_keys,
                                      const float* __restrict__ head_w, int heads, int head_dim,
                                      float softmax_scale) {
  extern __shared__ float pk[];
  const int p = blockIdx.x;
  for (int c = threadIdx.x; c < head_dim; c += blockDim.x)
    pk[c] = f(pool_keys[static_cast<long long>(p) * head_dim + c]);
  __syncthreads();
  __shared__ float red[32];
  float total = 0.0f;
  for (int h = 0; h < heads; ++h) {
    float acc = 0.0f;
    for (int c = threadIdx.x; c < head_dim; c += blockDim.x) acc += f(q[h * head_dim + c]) * pk[c];
    const float dot = block_sum(acc, red);
    if (threadIdx.x == 0) total += head_w[h] * fmaxf(dot * softmax_scale, 0.0f);
    __syncthreads();
  }
  if (threadIdx.x == 0) scores[p] = total;
}

__device__ __forceinline__ unsigned int order_key(float v) {
  unsigned int u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One block. Finds the select_k-th largest score by four 8-bit radix passes,
// then emits every pool above the threshold followed by as many ties as the
// budget still allows.
__global__ void indexer_select_kernel(int* __restrict__ sel, int* __restrict__ n_out,
                                      const float* __restrict__ scores, int n_pools,
                                      int select_k) {
  __shared__ unsigned int hist[256];
  __shared__ unsigned int prefix_hi;
  __shared__ int remaining;
  __shared__ int count;

  if (n_pools <= select_k) {
    for (int i = threadIdx.x; i < n_pools; i += blockDim.x) sel[i] = i;
    if (threadIdx.x == 0) *n_out = n_pools;
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
    for (int i = threadIdx.x; i < n_pools; i += blockDim.x) {
      const unsigned int key = order_key(scores[i]);
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
  for (int i = threadIdx.x; i < n_pools; i += blockDim.x) {
    if (order_key(scores[i]) > prefix_hi) {
      const int slot = atomicAdd(&count, 1);
      if (slot < select_k) sel[slot] = i;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n_pools; i += blockDim.x) {
    if (order_key(scores[i]) == prefix_hi) {
      const int slot = atomicAdd(&count, 1);
      if (slot < select_k) sel[slot] = i;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) *n_out = min(count, select_k);
}

__global__ void indexer_expand_kernel(int* __restrict__ tokens, int* __restrict__ n_out,
                                      const int* __restrict__ sel_pools,
                                      const int* __restrict__ n_sel, int kpool, int n_tokens) {
  const int ns = *n_sel;
  const int tail_count = n_tokens % kpool;
  const int tail_start = n_tokens - tail_count;
  const int total = ns * kpool + tail_count;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
    if (i < ns * kpool) {
      tokens[i] = sel_pools[i / kpool] * kpool + (i % kpool);
    } else {
      tokens[i] = tail_start + (i - ns * kpool);
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) *n_out = total;
}

// ------------------------------------------------------------------- MoE

__global__ void moe_router_kernel(int* __restrict__ topk_idx, float* __restrict__ topk_w,
                                  const float* __restrict__ logits, const float* __restrict__ bias,
                                  int n_experts, int top_k, bool norm_topk, float scale) {
  extern __shared__ float sc[];
  float* choice = sc + n_experts;
  for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
    const float s = sigmoidf(logits[i]);
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
      topk_idx[t] = best;
      topk_w[t] = sc[best];
      sum += sc[best];
    }
    const float denom = norm_topk ? (sum + 1e-20f) : 1.0f;
    for (int t = 0; t < top_k; ++t) topk_w[t] = topk_w[t] / denom * scale;
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
                            const float* __restrict__ w, int widx, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) acc[i] = b(f(acc[i]) + w[widx] * f(v[i]));
}

__global__ void add_kernel(bf16* __restrict__ acc, const bf16* __restrict__ v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) acc[i] = b(f(acc[i]) + f(v[i]));
}

__global__ void argmax_kernel(int* __restrict__ idx_out, float* __restrict__ val_out,
                              const float* __restrict__ x, int n) {
  __shared__ float vs[1024];
  __shared__ int is[1024];
  float best = -CUDART_INF_F;
  int bi = -1;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (x[i] > best) {
      best = x[i];
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
    *idx_out = is[0];
    *val_out = vs[0];
  }
}

__global__ void sumsq_kernel(float* __restrict__ out, const bf16* __restrict__ x, int n) {
  __shared__ float red[32];
  float acc = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = f(x[i]);
    acc += v * v;
  }
  const float total = block_sum(acc, red);
  if (threadIdx.x == 0) *out = total;
}

}  // namespace

// --------------------------------------------------------------- launchers

void gemv_bf16(bf16* y, const bf16* w, const bf16* x, int n_rows, int k, cudaStream_t s) {
  gemv_kernel<bf16><<<n_rows, 256, 0, s>>>(y, w, x, k);
}
void gemv_bf16_f32(float* y, const bf16* w, const bf16* x, int n_rows, int k, cudaStream_t s) {
  gemv_kernel<float><<<n_rows, 256, 0, s>>>(y, w, x, k);
}
void rmsnorm(bf16* out, const bf16* x, const bf16* weight, int n, float eps, cudaStream_t s) {
  rmsnorm_kernel<<<1, 256, 0, s>>>(out, x, weight, n, eps);
}
void layernorm(bf16* out, const bf16* x, const bf16* w, const bf16* bias, int n, float eps,
               cudaStream_t s) {
  layernorm_kernel<<<1, 128, 0, s>>>(out, x, w, bias, n, eps);
}
void hc_mix_gemv(float* mix, const bf16* fn, const bf16* streams, int hc_mix, int hc, int hidden,
                 float eps, cudaStream_t s) {
  hc_mix_kernel<<<hc_mix, 256, 0, s>>>(mix, fn, streams, hc * hidden, eps);
}
void hc_split(float* post, float* comb, bf16* collapsed, const float* mix, const float* base,
              const float* scale, const bf16* streams, int hc, int hidden, float hc_eps,
              int sinkhorn_iters, cudaStream_t s) {
  const std::size_t shmem = sizeof(float) * static_cast<std::size_t>(hc + hc * hc);
  hc_split_kernel<<<1, 256, shmem, s>>>(post, comb, collapsed, mix, base, scale, streams, hc, hidden,
                                        hc_eps, sinkhorn_iters);
}
void hc_combine(bf16* streams_out, const float* post, const bf16* y, const float* comb,
                const bf16* residual, int hc, int hidden, cudaStream_t s) {
  hc_combine_kernel<<<(hidden + 255) / 256, 256, 0, s>>>(streams_out, post, y, comb, residual, hc,
                                                         hidden);
}
void hc_head_mean(bf16* out, const bf16* streams, int hc, int hidden, cudaStream_t s) {
  hc_head_mean_kernel<<<(hidden + 255) / 256, 256, 0, s>>>(out, streams, hc, hidden);
}
void kda_conv_update(bf16* out, bf16* state, const bf16* in, const bf16* weight, int channels,
                     int kernel, cudaStream_t s) {
  kda_conv_kernel<<<(channels + 255) / 256, 256, 0, s>>>(out, state, in, weight, channels, kernel);
}
void kda_forget_gate(bf16* g, const bf16* fb, const float* dt_bias, const float* a_log, int heads,
                     int head_dim, float lower_bound, cudaStream_t s) {
  const int n = heads * head_dim;
  kda_forget_kernel<<<(n + 255) / 256, 256, 0, s>>>(g, fb, dt_bias, a_log, heads, head_dim,
                                                    lower_bound);
}
void kda_norm_qk(bf16* q, bf16* k, int heads, int head_dim, cudaStream_t s) {
  kda_norm_qk_kernel<<<heads, 128, 0, s>>>(q, k, head_dim, rsqrtf(static_cast<float>(head_dim)));
}
void kda_sigmoid(bf16* out, const bf16* in, int n, cudaStream_t s) {
  sigmoid_kernel<<<(n + 255) / 256, 256, 0, s>>>(out, in, n);
}
void kda_recurrent_step(float* state, bf16* o, const bf16* q, const bf16* k, const bf16* v,
                        const bf16* g, const bf16* beta, int heads, int head_dim, cudaStream_t s) {
  const std::size_t shmem = sizeof(float) * (static_cast<std::size_t>(head_dim) * head_dim + 3 * head_dim);
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kda_step_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shmem));
    configured = true;
  }
  kda_step_kernel<<<heads, head_dim, shmem, s>>>(state, o, q, k, v, g, beta, head_dim);
}
void kda_gated_norm(bf16* out, const bf16* x, const bf16* gate, const bf16* weight, int heads,
                    int head_dim, float eps, cudaStream_t s) {
  kda_gated_norm_kernel<<<heads, 128, 0, s>>>(out, x, gate, weight, head_dim, eps);
}
void mla_absorb_q(float* q_abs, const bf16* kv_b, const bf16* q, int heads, int nope, int v_dim,
                  int kv_lora, cudaStream_t s) {
  mla_absorb_q_kernel<<<heads, kv_lora, 0, s>>>(q_abs, kv_b, q, nope, v_dim, kv_lora);
}
void mla_scores(float* scores, const float* q_abs, const bf16* latent, const int* sel, int n_sel,
                int heads, int kv_lora, float scaling, cudaStream_t s) {
  mla_scores_kernel<<<n_sel, 256, sizeof(float) * kv_lora, s>>>(scores, q_abs, latent, sel, n_sel,
                                                                heads, kv_lora, scaling);
}
void mla_softmax(float* scores, int heads, int n_sel, cudaStream_t s) {
  mla_softmax_kernel<<<heads, 256, 0, s>>>(scores, n_sel);
}
void mla_context(float* ctx, const float* scores, const bf16* latent, const int* sel, int n_sel,
                 int heads, int kv_lora, cudaStream_t s) {
  mla_context_kernel<<<heads, kv_lora, 0, s>>>(ctx, scores, latent, sel, n_sel, kv_lora);
}
void mla_expand_v(bf16* out, const bf16* kv_b, const float* ctx, int heads, int nope, int v_dim,
                  int kv_lora, cudaStream_t s) {
  mla_expand_v_kernel<<<dim3(heads, v_dim), 128, 0, s>>>(out, kv_b, ctx, nope, v_dim, kv_lora);
}
void indexer_pool_keys(bf16* pool_keys, const bf16* keys, const bf16* gates, const bf16* ape,
                       int n_pools, int kpool, int head_dim, cudaStream_t s) {
  indexer_pool_kernel<<<n_pools, head_dim, 0, s>>>(pool_keys, keys, gates, ape, kpool, head_dim);
}
void indexer_scores(float* scores, const bf16* q, const bf16* pool_keys, const float* head_w,
                    int n_pools, int heads, int head_dim, cudaStream_t s) {
  indexer_scores_kernel<<<n_pools, 128, sizeof(float) * head_dim, s>>>(
      scores, q, pool_keys, head_w, heads, head_dim, rsqrtf(static_cast<float>(head_dim)));
}
void indexer_select(int* sel_pools, int* n_sel_out, const float* scores, int n_pools, int select_k,
                    cudaStream_t s) {
  indexer_select_kernel<<<1, 256, 0, s>>>(sel_pools, n_sel_out, scores, n_pools, select_k);
}
void indexer_expand(int* sel_tokens, int* n_tokens_out, const int* sel_pools, const int* n_sel,
                    int kpool, int n_tokens, cudaStream_t s) {
  indexer_expand_kernel<<<64, 256, 0, s>>>(sel_tokens, n_tokens_out, sel_pools, n_sel, kpool,
                                           n_tokens);
}
void moe_router(int* topk_idx, float* topk_w, const float* logits, const float* bias, int n_experts,
                int top_k, bool norm_topk, float scale, cudaStream_t s) {
  moe_router_kernel<<<1, 256, sizeof(float) * 2 * n_experts, s>>>(topk_idx, topk_w, logits, bias,
                                                                  n_experts, top_k, norm_topk, scale);
}
void swiglu_clamped(bf16* out, const bf16* gate, const bf16* up, int n, float limit,
                    cudaStream_t s) {
  swiglu_kernel<<<(n + 255) / 256, 256, 0, s>>>(out, gate, up, n, limit);
}
void gemv_nvfp4(bf16* y, const std::uint8_t* packed, const std::uint8_t* block_scale,
                float global_scale, const bf16* x, int n_rows, int k, cudaStream_t s) {
  gemv_nvfp4_kernel<<<n_rows, 256, 0, s>>>(y, packed, block_scale, global_scale, x, k);
}
void axpy_bf16(bf16* acc, const bf16* v, const float* w, int widx, int n, cudaStream_t s) {
  axpy_kernel<<<(n + 255) / 256, 256, 0, s>>>(acc, v, w, widx, n);
}
void add_bf16(bf16* acc, const bf16* v, int n, cudaStream_t s) {
  add_kernel<<<(n + 255) / 256, 256, 0, s>>>(acc, v, n);
}
void argmax_f32(int* idx_out, float* val_out, const float* x, int n, cudaStream_t s) {
  argmax_kernel<<<1, 1024, 0, s>>>(idx_out, val_out, x, n);
}
void sumsq_bf16(float* out, const bf16* x, int n, cudaStream_t s) {
  sumsq_kernel<<<1, 256, 0, s>>>(out, x, n);
}

}  // namespace rocket::engine

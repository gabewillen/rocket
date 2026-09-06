// KDA recurrent decode step, register-split. Drop-in replacement for the
// kernel behind src/kernels.cu kda_recurrent_step().
//
// Measured against the current kernel in
// blog/posts/kernels/2026-09-08-kda-step-is-occupancy-bound-not-bandwidth-bound/.
//
// What changes against src/kernels.cu kda_step_kernel: nothing in the math and
// nothing in the bytes moved. The state is still read once and written once,
// still one block per (head, stream). The FP32 128x128 tile no longer goes to
// shared memory. Each thread keeps the part of the column it needs in
// registers, so shared memory per block falls from 67072 B to 3584 B and the
// SM can hold more than one block.
//
// The block splits two ways:
//   C columns per thread   -> (128 / C) threads in x, vectorised state access
//   R slices of the key axis -> R threads in y, partial sums reduced in shared
// State registers per thread are 128 * C / R. R=4, C=1 is what the bench
// picked at the engine's dims; the template is left open because the trade is
// the only tuning knob here.
//
// T is the state storage type. float is the engine's current contract. bf16
// halves the traffic and is a separate decision owned by the storage lane; the
// kernel is correct either way and the arithmetic is FP32 in both cases.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace rocket_kda {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(bf16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }
template <> __device__ __forceinline__ bf16 from_f<bf16>(float x) {
  return __float2bfloat16(x);
}

template <typename T, int C> struct VecT;
template <> struct VecT<float, 1> { using type = float; };
template <> struct VecT<float, 4> { using type = float4; };
template <> struct VecT<bf16, 1> { using type = bf16; };
template <> struct VecT<bf16, 4> { using type = float2; };  // 4 x 2 B

template <typename T, int C>
__device__ __forceinline__ void load_cols(const T* p, float* out) {
  using V = typename VecT<T, C>::type;
  V raw = *reinterpret_cast<const V*>(p);
  const T* e = reinterpret_cast<const T*>(&raw);
#pragma unroll
  for (int c = 0; c < C; ++c) out[c] = to_f(e[c]);
}

template <typename T, int C>
__device__ __forceinline__ void store_cols(T* p, const float* in) {
  using V = typename VecT<T, C>::type;
  V raw;
  T* e = reinterpret_cast<T*>(&raw);
#pragma unroll
  for (int c = 0; c < C; ++c) e[c] = from_f<T>(in[c]);
  *reinterpret_cast<V*>(p) = raw;
}

// S = diag(exp(g)) S; delta = beta * (v - S^T k); S += k delta^T; o = S^T q,
// which is what kda_step_kernel computes, term for term.
//
// HEADS and DK are compile-time because the checkpoint fixes them at 64 and
// 128 (fuels/glm-5.3-flash/attention.yaml). row_stride stays a parameter
// because the caller passes the q/k/v row width.
template <typename T, int HEADS, int DK, int R, int C>
__global__ __launch_bounds__((DK / C) * R) void step_split(
    T* __restrict__ S, bf16* __restrict__ o, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const bf16* __restrict__ v, const bf16* __restrict__ g,
    const bf16* __restrict__ beta, int row_stride) {
  constexpr int TX = DK / C;
  constexpr int RPT = DK / R;

  __shared__ float sq[DK], sk[DK], sg[DK];
  __shared__ float red[R][DK];

  const int h = blockIdx.x;
  const int m = blockIdx.y;
  const int lane = threadIdx.x;
  const int r = threadIdx.y;
  const int col0 = lane * C;

  const size_t sbase = (static_cast<size_t>(m) * HEADS + h) * DK * DK;
  const size_t vbase =
      static_cast<size_t>(m) * row_stride + static_cast<size_t>(h) * DK;

  for (int i = r * TX + lane; i < DK; i += TX * R) {
    sq[i] = __bfloat162float(q[vbase + i]);
    sk[i] = __bfloat162float(k[vbase + i]);
    sg[i] = __expf(__bfloat162float(g[vbase + i]));
  }
  __syncthreads();

  float col[RPT][C];
  float up[C];
#pragma unroll
  for (int c = 0; c < C; ++c) up[c] = 0.0f;

#pragma unroll
  for (int t = 0; t < RPT; ++t) {
    const int i = t * R + r;
    load_cols<T, C>(S + sbase + static_cast<size_t>(i) * DK + col0, col[t]);
    const float gk = sg[i] * sk[i];
#pragma unroll
    for (int c = 0; c < C; ++c) up[c] += gk * col[t][c];
  }
#pragma unroll
  for (int c = 0; c < C; ++c) red[r][col0 + c] = up[c];
  __syncthreads();

  const float beta_h = __bfloat162float(beta[static_cast<size_t>(m) * HEADS + h]);
  float d[C];
#pragma unroll
  for (int c = 0; c < C; ++c) {
    float u = 0.0f;
#pragma unroll
    for (int rr = 0; rr < R; ++rr) u += red[rr][col0 + c];
    d[c] = beta_h * (__bfloat162float(v[vbase + col0 + c]) - u);
  }
  __syncthreads();

  float ap[C];
#pragma unroll
  for (int c = 0; c < C; ++c) ap[c] = 0.0f;

#pragma unroll
  for (int t = 0; t < RPT; ++t) {
    const int i = t * R + r;
    float s[C];
#pragma unroll
    for (int c = 0; c < C; ++c) {
      s[c] = sg[i] * col[t][c] + sk[i] * d[c];
      ap[c] += s[c] * sq[i];
    }
    store_cols<T, C>(S + sbase + static_cast<size_t>(i) * DK + col0, s);
  }
#pragma unroll
  for (int c = 0; c < C; ++c) red[r][col0 + c] = ap[c];
  __syncthreads();

  if (r == 0) {
    float acc[C];
#pragma unroll
    for (int c = 0; c < C; ++c) {
      float a = 0.0f;
#pragma unroll
      for (int rr = 0; rr < R; ++rr) a += red[rr][col0 + c];
      acc[c] = a;
    }
    store_cols<bf16, C>(o + vbase + col0, acc);
  }
}

// Same parameter list as src/kernels.cu kda_recurrent_step(), so the engine
// lane replaces that function body with this call and changes nothing else.
// Two entry points because the state storage type is a live decision: the
// engine held FP32 at e0dcfa9 and the storage lane is moving it to BF16. The
// arithmetic is FP32 in both.
void recurrent_step_f32(float* state, bf16* o, const bf16* q, const bf16* k,
                        const bf16* v, const bf16* g, const bf16* beta, int batch,
                        int heads, int head_dim, int row_stride, cudaStream_t s);

void recurrent_step_bf16(bf16* state, bf16* o, const bf16* q, const bf16* k,
                         const bf16* v, const bf16* g, const bf16* beta, int batch,
                         int heads, int head_dim, int row_stride, cudaStream_t s);

}  // namespace rocket_kda

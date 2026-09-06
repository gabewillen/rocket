// KDA decode step on sm_121 (GB10): recurrent state update + readout, M streams.
//
// Question: GLM-5.3-Flash spends 34 of its 45 text layers on KDA linear
// attention. A KDA layer carries a fixed-size recurrent state per stream that
// must be read and written on every decode step, whatever the context length.
// Is that step bandwidth-bound or latency-bound, and what does it cost per
// layer per step at M streams?
//
// Dims are read from the on-disk NVFP4 checkpoint, not assumed
// (scripts/attention/dump-attention-shapes.py):
//   q_proj / k_proj / v_proj   [8192, 4096]   -> 64 heads x 128, K and V both full width
//   q_conv1d / k_conv1d / v_conv1d [8192,1,4] -> depthwise, kernel 4, 3 taps of state
//   f_a_proj [128,4096] f_b_proj [8192,128]   -> per-channel decay, low rank 128
//   g_a_proj / g_b_proj  same shapes          -> per-channel output gate
//   b_proj   [64, 4096]                       -> per-head beta
//   A_log    [64]  dt_bias [8192]  o_norm [128]
// so the state is S[h] in R^{128x128} for 64 heads = 1,048,576 elements per
// layer per stream.
//
// The exact recurrence lives in modeling code, not in the checkpoint. What is
// implemented here is the gated delta rule that the tensor set describes
// (per-key-channel decay g, per-head step size beta, delta correction against
// the current state, gated readout):
//   u   = (g .* S)^T k
//   d   = beta * (v - u)
//   S'  = diag(g) S + k d^T
//   o   = S'^T q
// Cost is what is being measured, and cost is set by the state shape, which is
// on disk. A different ordering of the same terms reads and writes the same
// 2 MiB per layer per stream.
//
// Bytes per layer per step, BF16 state:
//   state       M * 64 * 128 * 128 * 2 B, read once and written once
//   conv state  M * 3 tensors * 8192 ch * 3 taps * 2 B, read once and written once
// Roofline is 238 GB/s (blog/posts/hardware/2026-09-06-measured-bandwidth/).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err_ = (x);                                                    \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                   \
                   cudaGetErrorString(err_));                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

constexpr int kHeads = 64;
constexpr int kDk = 128;
constexpr int kDv = 128;
constexpr int kHidden = kHeads * kDk;  // 8192
constexpr int kConvK = 4;
constexpr int kConvTaps = kConvK - 1;  // 3 stored positions
constexpr int kConvTensors = 3;        // q, k, v each have their own conv1d
constexpr int kKdaLayers = 34;         // layer_types in config.json

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float f(bf16 x) { return __bfloat162float(x); }

// One block per (head, stream). 128 threads, thread j owns column j of the
// state. Reductions run along the key axis inside a thread, so no cross-thread
// reduction is needed. The state tile is staged in shared memory so the two
// passes over it cost one HBM read and one HBM write.
__global__ void kda_step_smem(bf16* __restrict__ S, const bf16* __restrict__ q,
                              const bf16* __restrict__ k,
                              const bf16* __restrict__ v,
                              const bf16* __restrict__ g,
                              const bf16* __restrict__ beta,
                              bf16* __restrict__ o) {
  __shared__ bf16 tile[kDk * kDv];
  __shared__ float sq[kDk], sk[kDk], sg[kDk];

  const int h = blockIdx.x;
  const int m = blockIdx.y;
  const int j = threadIdx.x;

  const size_t sbase = (static_cast<size_t>(m) * kHeads + h) * kDk * kDv;
  const size_t vbase = (static_cast<size_t>(m) * kHeads + h) * kDk;

  for (int t = j; t < kDk * kDv; t += kDv) tile[t] = S[sbase + t];
  sq[j] = f(q[vbase + j]);
  sk[j] = f(k[vbase + j]);
  sg[j] = f(g[vbase + j]);
  __syncthreads();

  float u = 0.f;
  for (int i = 0; i < kDk; ++i) u += sg[i] * f(tile[i * kDv + j]) * sk[i];

  const float d = f(beta[m * kHeads + h]) * (f(v[vbase + j]) - u);

  float acc = 0.f;
  for (int i = 0; i < kDk; ++i) {
    const float s = sg[i] * f(tile[i * kDv + j]) + sk[i] * d;
    tile[i * kDv + j] = __float2bfloat16(s);
    acc += s * sq[i];
  }
  __syncthreads();

  for (int t = j; t < kDk * kDv; t += kDv) S[sbase + t] = tile[t];
  o[vbase + j] = __float2bfloat16(acc);
}

// Same math with no shared-memory staging: the state is read from global twice.
// Kept to show what the staging is worth.
__global__ void kda_step_global(bf16* __restrict__ S, const bf16* __restrict__ q,
                                const bf16* __restrict__ k,
                                const bf16* __restrict__ v,
                                const bf16* __restrict__ g,
                                const bf16* __restrict__ beta,
                                bf16* __restrict__ o) {
  __shared__ float sq[kDk], sk[kDk], sg[kDk];
  const int h = blockIdx.x;
  const int m = blockIdx.y;
  const int j = threadIdx.x;
  const size_t sbase = (static_cast<size_t>(m) * kHeads + h) * kDk * kDv;
  const size_t vbase = (static_cast<size_t>(m) * kHeads + h) * kDk;

  sq[j] = f(q[vbase + j]);
  sk[j] = f(k[vbase + j]);
  sg[j] = f(g[vbase + j]);
  __syncthreads();

  float u = 0.f;
  for (int i = 0; i < kDk; ++i) u += sg[i] * f(S[sbase + i * kDv + j]) * sk[i];
  const float d = f(beta[m * kHeads + h]) * (f(v[vbase + j]) - u);
  float acc = 0.f;
  for (int i = 0; i < kDk; ++i) {
    const float s = sg[i] * f(S[sbase + i * kDv + j]) + sk[i] * d;
    S[sbase + i * kDv + j] = __float2bfloat16(s);
    acc += s * sq[i];
  }
  o[vbase + j] = __float2bfloat16(acc);
}

// Causal depthwise conv over the token stream. At decode the layer holds the
// last 3 inputs per channel for each of q, k, v.
__global__ void kda_conv_step(bf16* __restrict__ state,      // [M][3][8192][3]
                              const bf16* __restrict__ x,    // [M][3][8192]
                              const bf16* __restrict__ w,    // [3][8192][4]
                              bf16* __restrict__ y) {        // [M][3][8192]
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = gridDim.x * blockDim.x;
  (void)total;
  const int lane = idx % (kConvTensors * kHidden);
  const int m = idx / (kConvTensors * kHidden);
  const size_t sb = (static_cast<size_t>(m) * kConvTensors * kHidden + lane) * kConvTaps;
  const size_t wb = static_cast<size_t>(lane) * kConvK;

  const float s0 = f(state[sb + 0]);
  const float s1 = f(state[sb + 1]);
  const float s2 = f(state[sb + 2]);
  const float xv = f(x[idx]);
  const float acc = f(w[wb + 0]) * s0 + f(w[wb + 1]) * s1 + f(w[wb + 2]) * s2 +
                    f(w[wb + 3]) * xv;
  state[sb + 0] = __float2bfloat16(s1);
  state[sb + 1] = __float2bfloat16(s2);
  state[sb + 2] = __float2bfloat16(xv);
  y[idx] = __float2bfloat16(acc);
}

// Same math, two columns per thread through bfloat162. One block covers two
// heads, so a warp touches 128 contiguous bytes of the state instead of 64.
__global__ void kda_step_vec2(bf16* __restrict__ S, const bf16* __restrict__ q,
                              const bf16* __restrict__ k,
                              const bf16* __restrict__ v,
                              const bf16* __restrict__ g,
                              const bf16* __restrict__ beta,
                              bf16* __restrict__ o) {
  __shared__ float sq[2][kDk], sk[2][kDk], sg[2][kDk];
  const int sub = threadIdx.x / 64;
  const int j2 = threadIdx.x % 64;
  const int h = blockIdx.x * 2 + sub;
  const int m = blockIdx.y;
  const size_t sbase = (static_cast<size_t>(m) * kHeads + h) * kDk * kDv;
  const size_t vbase = (static_cast<size_t>(m) * kHeads + h) * kDk;

  for (int i = j2; i < kDk; i += 64) {
    sq[sub][i] = f(q[vbase + i]);
    sk[sub][i] = f(k[vbase + i]);
    sg[sub][i] = f(g[vbase + i]);
  }
  __syncthreads();

  __nv_bfloat162* S2 = reinterpret_cast<__nv_bfloat162*>(S + sbase);
  float ux = 0.f, uy = 0.f;
  for (int i = 0; i < kDk; ++i) {
    const float2 s = __bfloat1622float2(S2[i * 64 + j2]);
    const float gk = sg[sub][i] * sk[sub][i];
    ux += s.x * gk;
    uy += s.y * gk;
  }
  const float bt = f(beta[m * kHeads + h]);
  const float dx = bt * (f(v[vbase + 2 * j2 + 0]) - ux);
  const float dy = bt * (f(v[vbase + 2 * j2 + 1]) - uy);

  float ax = 0.f, ay = 0.f;
  for (int i = 0; i < kDk; ++i) {
    const float2 s = __bfloat1622float2(S2[i * 64 + j2]);
    const float gi = sg[sub][i], ki = sk[sub][i], qi = sq[sub][i];
    const float nx = gi * s.x + ki * dx;
    const float ny = gi * s.y + ki * dy;
    S2[i * 64 + j2] = __floats2bfloat162_rn(nx, ny);
    ax += nx * qi;
    ay += ny * qi;
  }
  reinterpret_cast<__nv_bfloat162*>(o + vbase)[j2] = __floats2bfloat162_rn(ax, ay);
}

// One pass over the state instead of two.
//
// The update is rank one, so the readout does not need the updated state:
//   S' = diag(g) S + k d^T
//   o  = S'^T q = (diag(g) S)^T q + d (k . q)
// so u = (diag(g) S)^T k and w = (diag(g) S)^T q come out of the same sweep,
// and o = w + d (k . q) closes without touching S again.
//
// The rank-one term still has to reach memory. Rather than a second pass, it
// is carried as a pending pair (k_prev, d_prev) and folded in when the next
// step reads the state:
//   S_true = S_stored + k_prev d_prev^T
// The pending pair is 256 values per head against 16384 stored, so the step
// costs one read and one write of the state and nothing else.
__global__ void kda_step_lazy(bf16* __restrict__ S, const bf16* __restrict__ q,
                              const bf16* __restrict__ k,
                              const bf16* __restrict__ v,
                              const bf16* __restrict__ g,
                              const bf16* __restrict__ beta,
                              bf16* __restrict__ o, bf16* __restrict__ kprev,
                              bf16* __restrict__ dprev) {
  __shared__ float sq[kDk], sk[kDk], sg[kDk], skp[kDk];
  const int h = blockIdx.x;
  const int m = blockIdx.y;
  const int j = threadIdx.x;
  const size_t sbase = (static_cast<size_t>(m) * kHeads + h) * kDk * kDv;
  const size_t vbase = (static_cast<size_t>(m) * kHeads + h) * kDk;

  sq[j] = f(q[vbase + j]);
  sk[j] = f(k[vbase + j]);
  sg[j] = f(g[vbase + j]);
  skp[j] = f(kprev[vbase + j]);
  const float dpj = f(dprev[vbase + j]);
  __syncthreads();

  float u = 0.f, w = 0.f;
  for (int i = 0; i < kDk; ++i) {
    const float val = sg[i] * (f(S[sbase + i * kDv + j]) + skp[i] * dpj);
    S[sbase + i * kDv + j] = __float2bfloat16(val);
    u += val * sk[i];
    w += val * sq[i];
  }
  float kq = 0.f;
  for (int i = 0; i < kDk; ++i) kq += sk[i] * sq[i];

  const float d = f(beta[m * kHeads + h]) * (f(v[vbase + j]) - u);
  o[vbase + j] = __float2bfloat16(w + d * kq);
  kprev[vbase + j] = k[vbase + j];
  dprev[vbase + j] = __float2bfloat16(d);
}

struct Host {
  std::vector<float> S, q, k, v, g, beta, o;
};

void reference(Host& h, int M) {
  std::vector<float> Sn(h.S.size());
  for (int m = 0; m < M; ++m) {
    for (int hd = 0; hd < kHeads; ++hd) {
      const size_t sb = (static_cast<size_t>(m) * kHeads + hd) * kDk * kDv;
      const size_t vb = (static_cast<size_t>(m) * kHeads + hd) * kDk;
      for (int j = 0; j < kDv; ++j) {
        double u = 0.0;
        for (int i = 0; i < kDk; ++i)
          u += static_cast<double>(h.g[vb + i]) * h.S[sb + i * kDv + j] * h.k[vb + i];
        const double d = static_cast<double>(h.beta[m * kHeads + hd]) * (h.v[vb + j] - u);
        double acc = 0.0;
        for (int i = 0; i < kDk; ++i) {
          const double s = static_cast<double>(h.g[vb + i]) * h.S[sb + i * kDv + j] +
                           static_cast<double>(h.k[vb + i]) * d;
          Sn[sb + i * kDv + j] = static_cast<float>(s);
          acc += s * h.q[vb + i];
        }
        h.o[vb + j] = static_cast<float>(acc);
      }
    }
  }
  h.S.swap(Sn);
}

float bench(void (*kern)(bf16*, const bf16*, const bf16*, const bf16*,
                         const bf16*, const bf16*, bf16*),
            bf16* S, bf16* q, bf16* k, bf16* v, bf16* g, bf16* beta, bf16* o,
            int M, int iters, bool two_heads_per_block = false) {
  dim3 grid(two_heads_per_block ? kHeads / 2 : kHeads, M);
  for (int i = 0; i < 10; ++i) kern<<<grid, kDv>>>(S, q, k, v, g, beta, o);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) kern<<<grid, kDv>>>(S, q, k, v, g, beta, o);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? std::atoi(argv[1]) : 200;
  const std::vector<int> Ms = {8, 16, 24, 32, 48, 64};
  const int Mmax = Ms.back();

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s sm_%d%d, %d SMs, %.0f KiB smem/SM\n", prop.name,
              prop.major, prop.minor, prop.multiProcessorCount,
              prop.sharedMemPerMultiprocessor / 1024.0);
  std::printf("KDA dims H=%d Dk=%d Dv=%d, state %d elem/layer/stream, "
              "conv %d taps x %d ch\n\n",
              kHeads, kDk, kDv, kHeads * kDk * kDv, kConvTaps,
              kConvTensors * kHidden);

  std::mt19937 rng(20260906);
  std::uniform_real_distribution<float> unit(-1.f, 1.f);
  std::uniform_real_distribution<float> decay(0.90f, 0.999f);
  std::uniform_real_distribution<float> bdist(0.f, 1.f);

  Host hst;
  hst.S.resize(static_cast<size_t>(Mmax) * kHeads * kDk * kDv);
  hst.q.resize(static_cast<size_t>(Mmax) * kHidden);
  hst.k.resize(hst.q.size());
  hst.v.resize(hst.q.size());
  hst.g.resize(hst.q.size());
  hst.o.resize(hst.q.size());
  hst.beta.resize(static_cast<size_t>(Mmax) * kHeads);
  for (auto& x : hst.S) x = unit(rng) * 0.1f;
  for (auto& x : hst.q) x = unit(rng);
  for (auto& x : hst.k) x = unit(rng) * 0.1f;
  for (auto& x : hst.v) x = unit(rng);
  for (auto& x : hst.g) x = decay(rng);
  for (auto& x : hst.beta) x = bdist(rng);

  auto to_bf16 = [](const std::vector<float>& src) {
    std::vector<bf16> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) out[i] = __float2bfloat16(src[i]);
    return out;
  };
  // Round the host copy through BF16 too, so the reference sees the same inputs.
  auto round_trip = [](std::vector<float>& x) {
    for (auto& e : x) e = __bfloat162float(__float2bfloat16(e));
  };
  round_trip(hst.S);
  round_trip(hst.q);
  round_trip(hst.k);
  round_trip(hst.v);
  round_trip(hst.g);
  round_trip(hst.beta);

  bf16 *dS, *dq, *dk, *dv, *dg, *dbeta, *doo, *dkprev, *ddprev;
  CUDA_CHECK(cudaMalloc(&dS, hst.S.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dq, hst.q.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dk, hst.k.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dv, hst.v.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dg, hst.g.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dbeta, hst.beta.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&doo, hst.o.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dkprev, hst.q.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&ddprev, hst.q.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMemset(dkprev, 0, hst.q.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMemset(ddprev, 0, hst.q.size() * sizeof(bf16)));

  auto upload = [&]() {
    auto S = to_bf16(hst.S), q = to_bf16(hst.q), k = to_bf16(hst.k),
         v = to_bf16(hst.v), g = to_bf16(hst.g), b = to_bf16(hst.beta);
    CUDA_CHECK(cudaMemcpy(dS, S.data(), S.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, k.data(), k.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, v.data(), v.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg, g.data(), g.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbeta, b.data(), b.size() * sizeof(bf16), cudaMemcpyHostToDevice));
  };

  // Correctness: one step at M=8, both kernels against a double-precision
  // reference at the real dims.
  {
    const int M = 8;
    Host ref = hst;
    reference(ref, M);
    for (const char* which : {"smem", "global", "vec2", "lazy"}) {
      upload();
      dim3 grid(kHeads, M);
      const std::string w(which);
      if (w == "smem")
        kda_step_smem<<<grid, kDv>>>(dS, dq, dk, dv, dg, dbeta, doo);
      else if (w == "global")
        kda_step_global<<<grid, kDv>>>(dS, dq, dk, dv, dg, dbeta, doo);
      else if (w == "vec2")
        kda_step_vec2<<<dim3(kHeads / 2, M), kDv>>>(dS, dq, dk, dv, dg, dbeta, doo);
      else {
        CUDA_CHECK(cudaMemset(dkprev, 0, hst.q.size() * sizeof(bf16)));
        CUDA_CHECK(cudaMemset(ddprev, 0, hst.q.size() * sizeof(bf16)));
        kda_step_lazy<<<grid, kDv>>>(dS, dq, dk, dv, dg, dbeta, doo, dkprev, ddprev);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<bf16> go(hst.o.size()), gS(hst.S.size());
      CUDA_CHECK(cudaMemcpy(go.data(), doo, go.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gS.data(), dS, gS.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
      std::vector<float> full(gS.size());
      for (size_t i = 0; i < gS.size(); ++i) full[i] = __bfloat162float(gS[i]);
      if (w == "lazy") {
        std::vector<bf16> kp(hst.q.size()), dp(hst.q.size());
        CUDA_CHECK(cudaMemcpy(kp.data(), dkprev, kp.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dp.data(), ddprev, dp.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
        for (int mm = 0; mm < M; ++mm)
          for (int hh = 0; hh < kHeads; ++hh) {
            const size_t sb = (static_cast<size_t>(mm) * kHeads + hh) * kDk * kDv;
            const size_t vb = (static_cast<size_t>(mm) * kHeads + hh) * kDk;
            for (int i = 0; i < kDk; ++i)
              for (int jj = 0; jj < kDv; ++jj)
                full[sb + i * kDv + jj] +=
                    __bfloat162float(kp[vb + i]) * __bfloat162float(dp[vb + jj]);
          }
      }
      double omax = 0.0, smax = 0.0, oden = 0.0, sden = 0.0;
      for (size_t i = 0; i < static_cast<size_t>(M) * kHidden; ++i) {
        omax = std::max(omax, static_cast<double>(std::fabs(__bfloat162float(go[i]) - ref.o[i])));
        oden = std::max(oden, std::fabs(static_cast<double>(ref.o[i])));
      }
      for (size_t i = 0; i < static_cast<size_t>(M) * kHeads * kDk * kDv; ++i) {
        smax = std::max(smax, static_cast<double>(std::fabs(full[i] - ref.S[i])));
        sden = std::max(sden, std::fabs(static_cast<double>(ref.S[i])));
      }
      std::printf("correctness %-6s M=%d  o rel %.3e   state rel %.3e\n", which, M,
                  omax / oden, smax / sden);
    }
    std::printf("\n");
  }

  // Conv state step, sized for the largest M.
  bf16 *dcs, *dcx, *dcw, *dcy;
  const size_t conv_state_n = static_cast<size_t>(Mmax) * kConvTensors * kHidden * kConvTaps;
  const size_t conv_io_n = static_cast<size_t>(Mmax) * kConvTensors * kHidden;
  CUDA_CHECK(cudaMalloc(&dcs, conv_state_n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dcx, conv_io_n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dcw, static_cast<size_t>(kConvTensors) * kHidden * kConvK * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dcy, conv_io_n * sizeof(bf16)));
  CUDA_CHECK(cudaMemset(dcs, 0, conv_state_n * sizeof(bf16)));
  CUDA_CHECK(cudaMemset(dcx, 0, conv_io_n * sizeof(bf16)));
  CUDA_CHECK(cudaMemset(dcw, 0, static_cast<size_t>(kConvTensors) * kHidden * kConvK * sizeof(bf16)));

  std::printf("| M | kernel | ms/layer | GB/s | state MiB | 34 layers ms |\n");
  std::printf("| --- | --- | --- | --- | --- | --- |\n");
  for (int M : Ms) {
    upload();
    const double state_bytes =
        2.0 * static_cast<double>(M) * kHeads * kDk * kDv * sizeof(bf16);
    const double conv_bytes =
        2.0 * static_cast<double>(M) * kConvTensors * kHidden * kConvTaps * sizeof(bf16);
    const float ms_smem = bench(kda_step_smem, dS, dq, dk, dv, dg, dbeta, doo, M, iters);
    upload();
    const float ms_glob = bench(kda_step_global, dS, dq, dk, dv, dg, dbeta, doo, M, iters);
    upload();
    const float ms_vec2 =
        bench(kda_step_vec2, dS, dq, dk, dv, dg, dbeta, doo, M, iters, true);
    upload();
    CUDA_CHECK(cudaMemset(dkprev, 0, hst.q.size() * sizeof(bf16)));
    CUDA_CHECK(cudaMemset(ddprev, 0, hst.q.size() * sizeof(bf16)));
    float ms_lazy = 0.f;
    {
      dim3 grid(kHeads, M);
      for (int i = 0; i < 10; ++i)
        kda_step_lazy<<<grid, kDv>>>(dS, dq, dk, dv, dg, dbeta, doo, dkprev, ddprev);
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaEvent_t a, b;
      CUDA_CHECK(cudaEventCreate(&a));
      CUDA_CHECK(cudaEventCreate(&b));
      CUDA_CHECK(cudaEventRecord(a));
      for (int i = 0; i < iters; ++i)
        kda_step_lazy<<<grid, kDv>>>(dS, dq, dk, dv, dg, dbeta, doo, dkprev, ddprev);
      CUDA_CHECK(cudaEventRecord(b));
      CUDA_CHECK(cudaEventSynchronize(b));
      CUDA_CHECK(cudaEventElapsedTime(&ms_lazy, a, b));
      ms_lazy /= iters;
      CUDA_CHECK(cudaEventDestroy(a));
      CUDA_CHECK(cudaEventDestroy(b));
    }

    const int conv_threads = 256;
    const int conv_blocks = (M * kConvTensors * kHidden + conv_threads - 1) / conv_threads;
    for (int i = 0; i < 10; ++i)
      kda_conv_step<<<conv_blocks, conv_threads>>>(dcs, dcx, dcw, dcy);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i)
      kda_conv_step<<<conv_blocks, conv_threads>>>(dcs, dcx, dcw, dcy);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms_conv = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_conv, a, b));
    ms_conv /= iters;
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));

    const double total_smem = ms_smem + ms_conv;
    std::printf("| %d | smem | %.4f | %.1f | %.2f | %.2f |\n", M, total_smem,
                (state_bytes + conv_bytes) / (total_smem * 1e6),
                state_bytes / 2.0 / (1024.0 * 1024.0), total_smem * kKdaLayers);
    std::printf("| %d | global 2-pass | %.4f | %.1f | %.2f | %.2f |\n", M,
                ms_glob + ms_conv,
                (state_bytes + conv_bytes) / ((ms_glob + ms_conv) * 1e6),
                state_bytes / 2.0 / (1024.0 * 1024.0), (ms_glob + ms_conv) * kKdaLayers);
    std::printf("| %d | lazy 1-pass | %.4f | %.1f | %.2f | %.2f |\n", M, ms_lazy + ms_conv,
                (state_bytes + conv_bytes) / ((ms_lazy + ms_conv) * 1e6),
                state_bytes / 2.0 / (1024.0 * 1024.0), (ms_lazy + ms_conv) * kKdaLayers);
    std::printf("| %d | vec2 | %.4f | %.1f | %.2f | %.2f |\n", M, ms_vec2 + ms_conv,
                (state_bytes + conv_bytes) / ((ms_vec2 + ms_conv) * 1e6),
                state_bytes / 2.0 / (1024.0 * 1024.0), (ms_vec2 + ms_conv) * kKdaLayers);
  }
  std::printf("\nstate + conv bytes counted as one read and one write; "
              "roofline 238 GB/s\n");
  return 0;
}

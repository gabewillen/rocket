// KDA recurrent step as the engine actually launches it, against the state
// roofline, at M = 1, 8, 16, 32.
//
// blog/posts/attention/2026-09-06-dsa-indexer-is-the-attention-cost/ measured a
// standalone KDA step with a BF16 state and a 32 KiB shared tile. The engine
// does neither: src/kernels.cu kda_step_kernel keeps the state in FP32 and
// stages a full FP32 128x128 tile in dynamic shared memory, 67072 B per block.
// This bench holds the engine kernel next to variants that change one thing at
// a time, at the checkpoint's real dims (fuels/glm-5.3-flash/attention.yaml):
//   64 heads, Dk = Dv = 128, row_stride 8192, 34 KDA layers, conv kernel 4.
//
// Bytes per layer per step, counting the state read once and written once:
//   FP32 state  M * 64 * 128 * 128 * 4 B each way   = 4 MiB/stream each way
//   BF16 state  same shape at 2 B                   = 2 MiB/stream each way
//   conv state  M * 3 tensors * 8192 ch * 3 taps * 2 B each way = 144 KiB/stream
// Roofline is 238 GB/s (blog/posts/hardware/2026-09-06-measured-bandwidth/).
//
// The engine kernel's own math is the definition of correct here, so every
// variant is checked against a double-precision evaluation of it.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kda_step_dropin.cuh"

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

using bf16 = __nv_bfloat16;

constexpr int kHeads = 64;
constexpr int kDk = 128;
constexpr int kDv = 128;
constexpr int kHidden = kHeads * kDk;  // 8192, the q/k/v row stride
constexpr int kConvK = 4;
constexpr int kConvTaps = kConvK - 1;
constexpr int kConvTensors = 3;
constexpr int kKdaLayers = 34;

__device__ __forceinline__ float f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 b(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float siluf(float x) { return x / (1.0f + __expf(-x)); }

// ------------------------------------------------------------------ engine

// Verbatim from engines/glm5-moe-nvfp4-2b/src/kernels.cu kda_step_kernel, the
// kernel kda_recurrent_step() launches for every KDA layer of every decode
// step. One block per (head, stream), 128 threads, thread j owns column j, and
// the whole FP32 state tile is staged in dynamic shared memory.
__global__ void kda_step_engine(float* __restrict__ state, bf16* __restrict__ o,
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
  const long long sbase = (static_cast<long long>(m) * heads + h) * head_dim * head_dim;
  const long long vbase =
      static_cast<long long>(m) * row_stride + static_cast<long long>(h) * head_dim;

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

// Verbatim from src/kernels.cu kda_conv_kernel. The engine calls it three
// times per KDA layer, once each for q, k and v.
__global__ void kda_conv_engine(bf16* __restrict__ out, bf16* __restrict__ state,
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

// Diagnostic, not a candidate: the engine kernel with one thing changed, the
// staged tile held as BF16 instead of FP32. The global state stays FP32, the
// block stays 128 threads, and the bytes moved are identical. All that moves
// is shared memory per block, 67072 B -> 34304 B, and with it how many blocks
// fit on an SM. It exists to separate "shared capacity caps residency" from
// "registers beat shared memory".
__global__ void kda_step_engine_bf16tile(float* __restrict__ state, bf16* __restrict__ o,
                                         const bf16* __restrict__ q,
                                         const bf16* __restrict__ k,
                                         const bf16* __restrict__ v,
                                         const bf16* __restrict__ g,
                                         const bf16* __restrict__ beta, int heads,
                                         int head_dim, int row_stride) {
  __shared__ bf16 tile[kDk * kDv];
  __shared__ float sq[kDk], sk[kDk], sg[kDk];

  const int h = blockIdx.x;
  const int m = blockIdx.y;
  const int j = threadIdx.x;
  const long long sbase = (static_cast<long long>(m) * heads + h) * head_dim * head_dim;
  const long long vbase =
      static_cast<long long>(m) * row_stride + static_cast<long long>(h) * head_dim;

  for (int t = j; t < head_dim * head_dim; t += blockDim.x) tile[t] = b(state[sbase + t]);
  sq[j] = f(q[vbase + j]);
  sk[j] = f(k[vbase + j]);
  sg[j] = __expf(f(g[vbase + j]));
  __syncthreads();

  const float beta_h = f(beta[static_cast<long long>(m) * heads + h]);
  float u = 0.0f;
  for (int i = 0; i < head_dim; ++i) u += sg[i] * f(tile[i * head_dim + j]) * sk[i];
  const float d = beta_h * (f(v[vbase + j]) - u);

  float acc = 0.0f;
  for (int i = 0; i < head_dim; ++i) {
    const float s = sg[i] * f(tile[i * head_dim + j]) + sk[i] * d;
    tile[i * head_dim + j] = b(s);
    acc += s * sq[i];
  }
  __syncthreads();
  for (int t = j; t < head_dim * head_dim; t += blockDim.x)
    state[sbase + t] = f(tile[t]);
  o[vbase + j] = b(acc);
}

// --------------------------------------------------------------- reference

struct Host {
  std::vector<float> S, q, k, v, g, beta, o;
};

// Double-precision evaluation of exactly what kda_step_engine computes,
// including the exp() on the gate.
void reference(Host& h, int M) {
  std::vector<float> Sn(h.S.size());
  for (int m = 0; m < M; ++m) {
    for (int hd = 0; hd < kHeads; ++hd) {
      const size_t sb = (static_cast<size_t>(m) * kHeads + hd) * kDk * kDv;
      const size_t vb = static_cast<size_t>(m) * kHidden + static_cast<size_t>(hd) * kDk;
      std::vector<double> ge(kDk);
      for (int i = 0; i < kDk; ++i) ge[i] = std::exp(static_cast<double>(h.g[vb + i]));
      for (int j = 0; j < kDv; ++j) {
        double u = 0.0;
        for (int i = 0; i < kDk; ++i)
          u += ge[i] * static_cast<double>(h.S[sb + i * kDv + j]) *
               static_cast<double>(h.k[vb + i]);
        const double d =
            static_cast<double>(h.beta[static_cast<size_t>(m) * kHeads + hd]) *
            (static_cast<double>(h.v[vb + j]) - u);
        double acc = 0.0;
        for (int i = 0; i < kDk; ++i) {
          const double s = ge[i] * static_cast<double>(h.S[sb + i * kDv + j]) +
                           static_cast<double>(h.k[vb + i]) * d;
          Sn[sb + i * kDv + j] = static_cast<float>(s);
          acc += s * static_cast<double>(h.q[vb + i]);
        }
        h.o[vb + j] = static_cast<float>(acc);
      }
    }
  }
  h.S.swap(Sn);
}

// ------------------------------------------------------------------ harness

using LaunchFn = void (*)(dim3, int, void*, bf16*, const bf16*, const bf16*, const bf16*,
                          const bf16*, const bf16*);

struct Variant {
  const char* name;
  const char* storage;  // "fp32" or "bf16"
  int threads;
  int smem;
  double tol_state;
  LaunchFn launch;
  const void* fn;  // for the occupancy API
};

void launch_engine(dim3 grid, int smem, void* S, bf16* o, const bf16* q, const bf16* k,
                   const bf16* v, const bf16* g, const bf16* beta) {
  kda_step_engine<<<grid, kDv, smem>>>(static_cast<float*>(S), o, q, k, v, g, beta, kHeads,
                                       kDk, kHidden);
}

void launch_engine_bf16tile(dim3 grid, int, void* S, bf16* o, const bf16* q,
                            const bf16* k, const bf16* v, const bf16* g,
                            const bf16* beta) {
  kda_step_engine_bf16tile<<<grid, kDv>>>(static_cast<float*>(S), o, q, k, v, g, beta,
                                          kHeads, kDk, kHidden);
}

template <typename T, int R, int C>
void launch_split(dim3 grid, int, void* S, bf16* o, const bf16* q, const bf16* k,
                  const bf16* v, const bf16* g, const bf16* beta) {
  rocket_kda::step_split<T, kHeads, kDk, R, C><<<grid, dim3(kDv / C, R)>>>(
      static_cast<T*>(S), o, q, k, v, g, beta, kHidden);
}

// The shipped entry points, launched exactly as the engine would call them.
void launch_dropin_f32(dim3 grid, int, void* S, bf16* o, const bf16* q, const bf16* k,
                       const bf16* v, const bf16* g, const bf16* beta) {
  rocket_kda::recurrent_step_f32(static_cast<float*>(S), o, q, k, v, g, beta,
                                 static_cast<int>(grid.y), kHeads, kDk, kHidden, nullptr);
}

void launch_dropin_bf16(dim3 grid, int, void* S, bf16* o, const bf16* q, const bf16* k,
                        const bf16* v, const bf16* g, const bf16* beta) {
  rocket_kda::recurrent_step_bf16(static_cast<bf16*>(S), o, q, k, v, g, beta,
                                  static_cast<int>(grid.y), kHeads, kDk, kHidden, nullptr);
}

double time_kernel(const Variant& var, dim3 grid, void* S, bf16* o, const bf16* q,
                   const bf16* k, const bf16* v, const bf16* g, const bf16* beta,
                   int iters) {
  for (int i = 0; i < 10; ++i) var.launch(grid, var.smem, S, o, q, k, v, g, beta);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, e;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) var.launch(grid, var.smem, S, o, q, k, v, g, beta);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, e));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(e));
  return static_cast<double>(ms) / iters;
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? std::atoi(argv[1]) : 100;
  const int reps = (argc > 2) ? std::atoi(argv[2]) : 5;
  const std::vector<int> Ms = {1, 8, 16, 32};
  const int Mmax = Ms.back();

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s sm_%d%d, %d SMs, %.0f KiB smem/SM, %d regs/SM\n", prop.name,
              prop.major, prop.minor, prop.multiProcessorCount,
              prop.sharedMemPerMultiprocessor / 1024.0, prop.regsPerMultiprocessor);
  std::printf("L2 %.1f MiB, max %d warps/SM\n", prop.l2CacheSize / (1024.0 * 1024.0),
              prop.maxThreadsPerMultiProcessor / 32);
  std::printf("dims H=%d Dk=%d Dv=%d row_stride=%d, %d KDA layers\n\n", kHeads, kDk, kDv,
              kHidden, kKdaLayers);

  const int engine_smem = static_cast<int>(sizeof(float) * (kDk * kDv + 3 * kDk));
  CUDA_CHECK(cudaFuncSetAttribute(kda_step_engine,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, engine_smem));

  const std::vector<Variant> vars = {
      {"engine smem fp32", "fp32", kDv, engine_smem, 1e-5, launch_engine,
       reinterpret_cast<const void*>(kda_step_engine)},
      {"engine bf16 tile", "fp32", kDv, 0, 4e-3, launch_engine_bf16tile,
       reinterpret_cast<const void*>(kda_step_engine_bf16tile)},
      {"split fp32 R1 C1", "fp32", kDv * 1, 0, 1e-5, launch_split<float, 1, 1>,
       reinterpret_cast<const void*>(
           rocket_kda::step_split<float, kHeads, kDk, 1, 1>)},
      {"split fp32 R2 C1", "fp32", kDv * 2, 0, 1e-5, launch_split<float, 2, 1>,
       reinterpret_cast<const void*>(
           rocket_kda::step_split<float, kHeads, kDk, 2, 1>)},
      {"split fp32 R4 C1", "fp32", kDv * 4, 0, 1e-5, launch_split<float, 4, 1>,
       reinterpret_cast<const void*>(
           rocket_kda::step_split<float, kHeads, kDk, 4, 1>)},
      {"split fp32 R8 C1", "fp32", kDv * 8, 0, 1e-5, launch_split<float, 8, 1>,
       reinterpret_cast<const void*>(
           rocket_kda::step_split<float, kHeads, kDk, 8, 1>)},
      {"split fp32 R16 C4", "fp32", (kDv / 4) * 16, 0, 1e-5, launch_split<float, 16, 4>,
       reinterpret_cast<const void*>(
           rocket_kda::step_split<float, kHeads, kDk, 16, 4>)},
      {"split fp32 R32 C4", "fp32", (kDv / 4) * 32, 0, 1e-5, launch_split<float, 32, 4>,
       reinterpret_cast<const void*>(
           rocket_kda::step_split<float, kHeads, kDk, 32, 4>)},
      {"split bf16 R8 C1", "bf16", kDv * 8, 0, 4e-3, launch_split<bf16, 8, 1>,
       reinterpret_cast<const void*>(
           rocket_kda::step_split<bf16, kHeads, kDk, 8, 1>)},
      {"split bf16 R16 C4", "bf16", (kDv / 4) * 16, 0, 4e-3, launch_split<bf16, 16, 4>,
       reinterpret_cast<const void*>(
           rocket_kda::step_split<bf16, kHeads, kDk, 16, 4>)},
      {"split bf16 R32 C4", "bf16", (kDv / 4) * 32, 0, 4e-3, launch_split<bf16, 32, 4>,
       reinterpret_cast<const void*>(
           rocket_kda::step_split<bf16, kHeads, kDk, 32, 4>)},
      {"dropin fp32 (R4 C1)", "fp32", kDv * 4, 0, 1e-5, launch_dropin_f32,
       reinterpret_cast<const void*>(
           rocket_kda::step_split<float, kHeads, kDk, 4, 1>)},
      {"dropin bf16 (R16 C4)", "bf16", (kDv / 4) * 16, 0, 4e-3, launch_dropin_bf16,
       reinterpret_cast<const void*>(
           rocket_kda::step_split<bf16, kHeads, kDk, 16, 4>)},
  };

  // Inputs. The state is rounded to BF16 on the host before upload so the FP32
  // and BF16 storage variants start from bit-identical values and one
  // reference serves both.
  std::mt19937 rng(20260907);
  std::uniform_real_distribution<float> unit(-1.f, 1.f);
  // g is the pre-exp gate. The engine's forget gate is lower_bound * sigmoid()
  // with lower_bound -5, so g is negative and exp(g) < 1.
  std::uniform_real_distribution<float> gate(-0.5f, -0.001f);
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
  for (auto& x : hst.g) x = gate(rng);
  for (auto& x : hst.beta) x = bdist(rng);
  auto round_trip = [](std::vector<float>& x) {
    for (auto& e : x) e = __bfloat162float(__float2bfloat16(e));
  };
  round_trip(hst.S);
  round_trip(hst.q);
  round_trip(hst.k);
  round_trip(hst.v);
  round_trip(hst.g);
  round_trip(hst.beta);

  const size_t state_n = hst.S.size();
  const size_t io_n = hst.q.size();
  float* dS32 = nullptr;
  bf16* dS16 = nullptr;
  bf16 *dq, *dk, *dv, *dg, *dbeta, *doo;
  CUDA_CHECK(cudaMalloc(&dS32, state_n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dS16, state_n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dq, io_n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dk, io_n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dv, io_n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dg, io_n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dbeta, hst.beta.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&doo, io_n * sizeof(bf16)));

  std::vector<bf16> tmp16(std::max(state_n, io_n));
  auto up_bf16 = [&](bf16* dst, const std::vector<float>& src) {
    for (size_t i = 0; i < src.size(); ++i) tmp16[i] = __float2bfloat16(src[i]);
    CUDA_CHECK(cudaMemcpy(dst, tmp16.data(), src.size() * sizeof(bf16),
                          cudaMemcpyHostToDevice));
  };
  auto upload_state = [&]() {
    CUDA_CHECK(cudaMemcpy(dS32, hst.S.data(), state_n * sizeof(float),
                          cudaMemcpyHostToDevice));
    up_bf16(dS16, hst.S);
  };
  up_bf16(dq, hst.q);
  up_bf16(dk, hst.k);
  up_bf16(dv, hst.v);
  up_bf16(dg, hst.g);
  up_bf16(dbeta, hst.beta);

  // Occupancy from the CUDA occupancy API rather than a guess.
  std::printf("| variant | threads/block | smem B/block | regs/thread | spill B | "
              "blocks/SM | warps/SM |\n");
  std::printf("| --- | --- | --- | --- | --- | --- | --- |\n");
  for (const auto& var : vars) {
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, var.fn, var.threads,
                                                             var.smem));
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, var.fn));
    std::printf("| %s | %d | %d | %d | %d | %d | %d |\n", var.name, var.threads,
                var.smem + static_cast<int>(attr.sharedSizeBytes), attr.numRegs,
                static_cast<int>(attr.localSizeBytes), blocks,
                blocks * var.threads / 32);
  }
  std::printf("\n");

  // Correctness at M=8, every variant, against the double reference.
  {
    const int M = 8;
    Host ref = hst;
    reference(ref, M);
    double refo = 0.0, refs = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(M) * kHidden; ++i)
      refo = std::max(refo, std::fabs(static_cast<double>(ref.o[i])));
    for (size_t i = 0; i < static_cast<size_t>(M) * kHeads * kDk * kDv; ++i)
      refs = std::max(refs, std::fabs(static_cast<double>(ref.S[i])));

    // BF16 carries 8 mantissa bits, so one rounding of a value at the top of
    // this range costs at most 2^-9 = 1.95e-3 relative to that maximum. The
    // thresholds are that half-ulp with 2x slack for the FP32 accumulation
    // underneath. An FP32-stored state rounds at 2^-24, so 1e-5 is loose.
    const double tol_o = 4e-3;

    std::printf("| variant | o rel | state rel | tol o | tol state | check |\n");
    std::printf("| --- | --- | --- | --- | --- | --- |\n");
    std::vector<bf16> go(io_n);
    std::vector<float> gs(state_n);
    std::vector<bf16> gs16(state_n);
    bool all_ok = true;
    for (const auto& var : vars) {
      const bool fp32 = std::string(var.storage) == "fp32";
      upload_state();
      CUDA_CHECK(cudaMemset(doo, 0, io_n * sizeof(bf16)));
      void* S = fp32 ? static_cast<void*>(dS32) : static_cast<void*>(dS16);
      var.launch(dim3(kHeads, M), var.smem, S, doo, dq, dk, dv, dg, dbeta);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(go.data(), doo, io_n * sizeof(bf16), cudaMemcpyDeviceToHost));
      if (fp32) {
        CUDA_CHECK(cudaMemcpy(gs.data(), dS32, state_n * sizeof(float),
                              cudaMemcpyDeviceToHost));
      } else {
        CUDA_CHECK(cudaMemcpy(gs16.data(), dS16, state_n * sizeof(bf16),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < state_n; ++i) gs[i] = __bfloat162float(gs16[i]);
      }
      double omax = 0.0, smax = 0.0;
      for (size_t i = 0; i < static_cast<size_t>(M) * kHidden; ++i)
        omax = std::max(omax, std::fabs(static_cast<double>(__bfloat162float(go[i])) -
                                        static_cast<double>(ref.o[i])));
      for (size_t i = 0; i < static_cast<size_t>(M) * kHeads * kDk * kDv; ++i)
        smax = std::max(smax, std::fabs(static_cast<double>(gs[i]) -
                                        static_cast<double>(ref.S[i])));
      const double orel = omax / refo;
      const double srel = smax / refs;
      const double tol_s = var.tol_state;
      const bool ok = orel <= tol_o && srel <= tol_s;
      all_ok = all_ok && ok;
      std::printf("| %s | %.3e | %.3e | %.1e | %.1e | %s |\n", var.name, orel, srel, tol_o,
                  tol_s, ok ? "pass" : "FAIL");
    }
    std::printf("\ncorrectness %s\n\n", all_ok ? "all pass" : "FAILURES PRESENT");
    if (!all_ok) return 1;
  }

  // Conv state advance, three calls per layer as the engine issues them.
  bf16 *dcs, *dcx, *dcw, *dcy;
  const size_t conv_state_n =
      static_cast<size_t>(Mmax) * kConvTensors * kHidden * kConvTaps;
  const size_t conv_io_n = static_cast<size_t>(Mmax) * kConvTensors * kHidden;
  CUDA_CHECK(cudaMalloc(&dcs, conv_state_n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dcx, conv_io_n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dcy, conv_io_n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dcw, static_cast<size_t>(kConvTensors) * kHidden * kConvK *
                                  sizeof(bf16)));
  CUDA_CHECK(cudaMemset(dcs, 0, conv_state_n * sizeof(bf16)));
  CUDA_CHECK(cudaMemset(dcx, 0, conv_io_n * sizeof(bf16)));
  CUDA_CHECK(cudaMemset(dcw, 0, static_cast<size_t>(kConvTensors) * kHidden * kConvK *
                                    sizeof(bf16)));

  auto time_conv = [&](int M) {
    const dim3 grid((kHidden + 255) / 256, M);
    auto issue = [&]() {
      for (int t = 0; t < kConvTensors; ++t) {
        kda_conv_engine<<<grid, 256>>>(
            dcy + static_cast<size_t>(t) * Mmax * kHidden,
            dcs + static_cast<size_t>(t) * Mmax * kHidden * kConvTaps,
            dcx + static_cast<size_t>(t) * Mmax * kHidden,
            dcw + static_cast<size_t>(t) * kHidden * kConvK, kHidden, kConvK);
      }
    };
    for (int i = 0; i < 10; ++i) issue();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, e;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) issue();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, e));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(e));
    return static_cast<double>(ms) / iters;
  };

  std::printf("| M | variant | blocks | ms/layer | spread %% | GB/s state | GB/s "
              "state+conv | 34 layers ms |\n");
  std::printf("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
  for (int M : Ms) {
    const double conv_ms = time_conv(M);
    const double conv_bytes = 2.0 * static_cast<double>(M) * kConvTensors * kHidden *
                              kConvTaps * sizeof(bf16);
    for (const auto& var : vars) {
      const bool fp32 = std::string(var.storage) == "fp32";
      const double elem = 2.0 * static_cast<double>(M) * kHeads * kDk * kDv;
      const double state_bytes = elem * (fp32 ? 4.0 : 2.0);
      void* S = fp32 ? static_cast<void*>(dS32) : static_cast<void*>(dS16);
      upload_state();
      double best = 1e30, worst = 0.0;
      for (int rep = 0; rep < reps; ++rep) {
        const double ms =
            time_kernel(var, dim3(kHeads, M), S, doo, dq, dk, dv, dg, dbeta, iters);
        best = std::min(best, ms);
        worst = std::max(worst, ms);
      }
      const double total = best + conv_ms;
      std::printf("| %d | %s | %d | %.4f | %.1f | %.1f | %.1f | %.2f |\n", M, var.name,
                  kHeads * M, best, 100.0 * (worst - best) / best,
                  state_bytes / (best * 1e6), (state_bytes + conv_bytes) / (total * 1e6),
                  total * kKdaLayers);
    }
  }
  std::printf("\nms/layer is the recurrent step alone (best of %d reps of %d iters); "
              "the state+conv and 34-layer columns add the three conv calls the "
              "engine issues per layer.\n",
              reps, iters);
  return 0;
}

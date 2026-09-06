// DSA indexer scoring pass on sm_121 (GB10): score every candidate in a 262144
// token context for M streams, which is what has to happen before sparse MLA
// can select its 2048 entries.
//
// Question: sparse attention reads 2048 entries, but something has to rank the
// whole context to find them. At the 262k serving cap, what does that ranking
// pass cost per layer per decode step, and how does it compare to the sparse
// attention it feeds?
//
// Dims from the on-disk checkpoint (scripts/attention/dump-attention-shapes.py):
//   indexer.wk           [128, 4096]   -> one 128 wide key per token, shared by heads
//   indexer.wq_b         [4096, 1536]  -> 32 heads x 128 query
//   indexer.weights_proj [32, 4096]    -> one scalar per head
//   indexer.k_norm       [128] weight + bias
//   indexer.index_kpool_compress_gate [128, 4096], _ape [4, 128]
// config.json: index_n_heads 32, index_head_dim 128, index_topk 2048,
// index_kpool 4, index_kpool_compress true.
//
// Scored form:
//   score_t = sum_h w_h * relu(q_h . k_t)
// The ReLU is the one term here that is not on disk. It is in modeling code,
// and it is assumed because that is the DSA indexer's published form. It does
// not change the cost: the pass reads 128 BF16 per candidate whatever the
// nonlinearity, and the measurement below is bandwidth-bound. It changes only
// whether the 32 head dots could collapse into one, which is a correctness
// question, not a cost one.
//
// Bytes per layer per step: M * N * 128 * 2 B of keys, plus M * N * 4 B of
// scores written. At N = 262144 that is 64 MiB of keys per stream per layer.
// With index_kpool 4 compressing keys 4:1 the candidate count is 65536 and the
// keys are 16 MiB. Both are measured, because which one the engine pays is set
// by whether pooled keys are cached instead of per-token keys, and that is not
// established from on-disk sources.
//
// Selection of the top 2048 is not included. It reads the N fp32 scores once
// more, 1 MiB per stream at N = 262144, against 64 MiB of key traffic.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

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

constexpr int kIdxHeads = 32;   // index_n_heads
constexpr int kIdxDim = 128;    // index_head_dim
constexpr int kCtx = 262144;    // serving_context_cap
constexpr int kPool = 4;        // index_kpool
constexpr int kMlaLayers = 11;
constexpr int kTile = 64;       // candidates per block
constexpr int kThreads = 128;

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float f(bf16 x) { return __bfloat162float(x); }

// One block scores kTile candidates against all 32 heads. Each thread owns a
// 4x4 tile of (head, candidate), so shared memory is read four times per
// sixteen FMAs instead of eight.
//
// Both tiles are held transposed, [dim][entry], and padded by 2. Held the
// natural way, [entry][dim], a fixed dim strides by 128 BF16 = 256 words, so
// every candidate group in a warp lands on one bank and the pass runs at
// 15 GB/s. The pad makes the stride odd in words and the conflicts go away.
__global__ void indexer_scan(const bf16* __restrict__ q,   // [M][32][128]
                             const bf16* __restrict__ keys, // [M][N][128]
                             const float* __restrict__ w,   // [M][32]
                             float* __restrict__ score,     // [M][N]
                             int n_cand) {
  constexpr int kQPad = kIdxHeads + 2;
  constexpr int kKPad = kTile + 2;
  __shared__ bf16 sq[kIdxDim * kQPad];
  __shared__ bf16 sk[kIdxDim * kKPad];
  __shared__ float sw[kIdxHeads];
  __shared__ float red[kThreads / 16][kTile];

  const int m = blockIdx.y;
  const int base_cand = blockIdx.x * kTile;
  const int tid = threadIdx.x;

  for (int i = tid; i < kIdxHeads * kIdxDim; i += kThreads)
    sq[(i % kIdxDim) * kQPad + i / kIdxDim] =
        q[static_cast<size_t>(m) * kIdxHeads * kIdxDim + i];
  for (int i = tid; i < kIdxHeads; i += kThreads)
    sw[i] = w[static_cast<size_t>(m) * kIdxHeads + i];
  const size_t kbase = (static_cast<size_t>(m) * n_cand + base_cand) * kIdxDim;
  for (int i = tid; i < kTile * kIdxDim; i += kThreads)
    sk[(i % kIdxDim) * kKPad + i / kIdxDim] = keys[kbase + i];
  __syncthreads();

  const int hg = tid / 16;   // 0..7  -> heads hg*4 .. hg*4+3
  const int cg = tid % 16;   // 0..15 -> candidates cg*4 .. cg*4+3

  float acc[4][4] = {};
  for (int d = 0; d < kIdxDim; ++d) {
    const __nv_bfloat162* qp =
        reinterpret_cast<const __nv_bfloat162*>(sq + d * kQPad + hg * 4);
    const __nv_bfloat162* kp =
        reinterpret_cast<const __nv_bfloat162*>(sk + d * kKPad + cg * 4);
    const float2 q01 = __bfloat1622float2(qp[0]);
    const float2 q23 = __bfloat1622float2(qp[1]);
    const float2 k01 = __bfloat1622float2(kp[0]);
    const float2 k23 = __bfloat1622float2(kp[1]);
    const float qv[4] = {q01.x, q01.y, q23.x, q23.y};
    const float kv[4] = {k01.x, k01.y, k23.x, k23.y};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(qv[i], kv[j], acc[i][j]);
  }

  float part[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float wh = sw[hg * 4 + i];
#pragma unroll
    for (int j = 0; j < 4; ++j) part[j] += wh * fmaxf(acc[i][j], 0.f);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) red[hg][cg * 4 + j] = part[j];
  __syncthreads();

  for (int c = tid; c < kTile; c += kThreads) {
    float s = 0.f;
#pragma unroll
    for (int g = 0; g < kThreads / 16; ++g) s += red[g][c];
    score[static_cast<size_t>(m) * n_cand + base_cand + c] = s;
  }
}

// The same scan on tensor cores. The scored quantity is a GEMM, 32 heads by
// kTile candidates by 128 dims, so the CUDA-core version above is paying for
// FFMA issue and BF16 to float conversion that HMMA does not.
//
// Four warps, output 32x64 in eight 16x16 tiles, two per warp. Shared tiles are
// padded to a multiple of 8 because that is what wmma's leading dimension needs
// for BF16 alignment.
__global__ void indexer_scan_mma(const bf16* __restrict__ q,
                                 const bf16* __restrict__ keys,
                                 const float* __restrict__ w,
                                 float* __restrict__ score, int n_cand) {
  using namespace nvcuda;
  constexpr int kLdq = kIdxDim + 8;  // 136
  constexpr int kLdk = kTile + 8;    // 72
  __shared__ bf16 sq[kIdxHeads * kLdq];
  __shared__ bf16 sk[kIdxDim * kLdk];
  __shared__ float sc[kIdxHeads * kLdk];
  __shared__ float sw[kIdxHeads];

  const int m = blockIdx.y;
  const int base_cand = blockIdx.x * kTile;
  const int tid = threadIdx.x;

  for (int i = tid; i < kIdxHeads * kIdxDim; i += kThreads)
    sq[(i / kIdxDim) * kLdq + (i % kIdxDim)] =
        q[static_cast<size_t>(m) * kIdxHeads * kIdxDim + i];
  for (int i = tid; i < kIdxHeads; i += kThreads)
    sw[i] = w[static_cast<size_t>(m) * kIdxHeads + i];
  const size_t kbase = (static_cast<size_t>(m) * n_cand + base_cand) * kIdxDim;
  for (int i = tid; i < kTile * kIdxDim; i += kThreads)
    sk[(i % kIdxDim) * kLdk + (i / kIdxDim)] = keys[kbase + i];
  __syncthreads();

  const int warp = tid / 32;
  const int mt = warp % 2;        // 0..1 -> head rows mt*16
  const int nt0 = (warp / 2) * 2; // 0,2  -> candidate cols nt*16

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
  wmma::fill_fragment(c0, 0.f);
  wmma::fill_fragment(c1, 0.f);
  for (int k = 0; k < kIdxDim; k += 16) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b0, b1;
    wmma::load_matrix_sync(a, sq + mt * 16 * kLdq + k, kLdq);
    wmma::load_matrix_sync(b0, sk + k * kLdk + nt0 * 16, kLdk);
    wmma::load_matrix_sync(b1, sk + k * kLdk + (nt0 + 1) * 16, kLdk);
    wmma::mma_sync(c0, a, b0, c0);
    wmma::mma_sync(c1, a, b1, c1);
  }
  wmma::store_matrix_sync(sc + mt * 16 * kLdk + nt0 * 16, c0, kLdk,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(sc + mt * 16 * kLdk + (nt0 + 1) * 16, c1, kLdk,
                          wmma::mem_row_major);
  __syncthreads();

  for (int c = tid; c < kTile; c += kThreads) {
    float s = 0.f;
    for (int h = 0; h < kIdxHeads; ++h) s += sw[h] * fmaxf(sc[h * kLdk + c], 0.f);
    score[static_cast<size_t>(m) * n_cand + base_cand + c] = s;
  }
}

// Floor: read the key array and emit one score per candidate, with no scoring
// math. Anything the scan does costs at least this, and the gap to it says
// whether the scan is limited by memory or by the kernel.
__global__ void indexer_stream_only(const bf16* __restrict__ keys,
                                    float* __restrict__ score, int n_cand) {
  // One block per kThreads candidates. Threads walk the block's key run
  // contiguously, so a warp reads 512 B at a time.
  const int m = blockIdx.y;
  const size_t base =
      (static_cast<size_t>(m) * n_cand + static_cast<size_t>(blockIdx.x) * kThreads) *
      kIdxDim;
  const uint4* k4 = reinterpret_cast<const uint4*>(keys + base);
  const int per_block = kThreads * kIdxDim / 8;  // uint4 per block
  float acc = 0.f;
  for (int i = threadIdx.x; i < per_block; i += kThreads) {
    const uint4 v = k4[i];
    acc += static_cast<float>(v.x ^ v.y ^ v.z ^ v.w);
  }
  score[static_cast<size_t>(m) * n_cand + blockIdx.x * kThreads + threadIdx.x] = acc;
}

__global__ void fill_keys(bf16* __restrict__ p, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    unsigned int h = static_cast<unsigned int>(i) * 2246822519u + 374761393u;
    h ^= h >> 13;
    p[i] = __float2bfloat16((static_cast<float>(h & 0xffffu) / 32768.f - 1.f) * 0.5f);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? std::atoi(argv[1]) : 20;
  const std::vector<int> Ms = {8, 16, 32, 64};
  const int Mmax = Ms.back();

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);
  std::printf("indexer heads=%d dim=%d, ctx=%d, kpool=%d -> pooled candidates %d\n\n",
              kIdxHeads, kIdxDim, kCtx, kPool, kCtx / kPool);

  std::mt19937 rng(20260906);
  std::uniform_real_distribution<float> unit(-1.f, 1.f);

  bf16* d_keys = nullptr;
  const size_t key_elems = static_cast<size_t>(Mmax) * kCtx * kIdxDim;
  CUDA_CHECK(cudaMalloc(&d_keys, key_elems * sizeof(bf16)));
  fill_keys<<<4096, 256>>>(d_keys, key_elems);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<bf16> hq(static_cast<size_t>(Mmax) * kIdxHeads * kIdxDim);
  for (auto& x : hq) x = __float2bfloat16(unit(rng) * 0.2f);
  std::vector<float> hw(static_cast<size_t>(Mmax) * kIdxHeads);
  for (auto& x : hw) x = unit(rng);
  bf16* d_q = nullptr;
  float* d_w = nullptr;
  float* d_score = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, hq.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_w, hw.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_score, static_cast<size_t>(Mmax) * kCtx * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_q, hq.data(), hq.size() * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Correctness: stream 0, first 4096 candidates, double reference.
  for (int variant = 0; variant < 2; ++variant) {
    const int n = kCtx;
    CUDA_CHECK(cudaMemset(d_score, 0, static_cast<size_t>(Mmax) * kCtx * sizeof(float)));
    if (variant == 0)
      indexer_scan<<<dim3(n / kTile, 1), kThreads>>>(d_q, d_keys, d_w, d_score, n);
    else
      indexer_scan_mma<<<dim3(n / kTile, 1), kThreads>>>(d_q, d_keys, d_w, d_score, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    const int probe = 4096;
    std::vector<bf16> hk(static_cast<size_t>(probe) * kIdxDim);
    std::vector<float> hs(probe);
    CUDA_CHECK(cudaMemcpy(hk.data(), d_keys, hk.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hs.data(), d_score, probe * sizeof(float), cudaMemcpyDeviceToHost));
    double mx = 0.0, den = 0.0;
    for (int t = 0; t < probe; ++t) {
      double s = 0.0;
      for (int h = 0; h < kIdxHeads; ++h) {
        double a = 0.0;
        for (int d = 0; d < kIdxDim; ++d)
          a += static_cast<double>(__bfloat162float(hq[h * kIdxDim + d])) *
               __bfloat162float(hk[static_cast<size_t>(t) * kIdxDim + d]);
        s += hw[h] * std::max(a, 0.0);
      }
      mx = std::max(mx, std::fabs(s - hs[t]));
      den = std::max(den, std::fabs(s));
    }
    std::printf("correctness %-5s (4096 candidates vs double reference): rel %.3e\n",
                variant == 0 ? "fma" : "mma", mx / den);
  }

  std::printf("\n");
  std::printf("| M | candidates | kernel | ms/layer | key MiB | GB/s | GFLOP/s | 11 layers ms |\n");
  std::printf("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
  for (int M : Ms) {
    for (int n : {kCtx, kCtx / kPool}) {
     for (int variant = 0; variant < 3; ++variant) {
      auto launch = [&](dim3 g) {
        if (variant == 0)
          indexer_scan<<<g, kThreads>>>(d_q, d_keys, d_w, d_score, n);
        else if (variant == 1)
          indexer_scan_mma<<<g, kThreads>>>(d_q, d_keys, d_w, d_score, n);
        else
          indexer_stream_only<<<dim3(n / kThreads, M), kThreads>>>(d_keys, d_score, n);
      };
      dim3 grid(n / kTile, M);
      for (int i = 0; i < 3; ++i) launch(grid);
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaEvent_t a, b;
      CUDA_CHECK(cudaEventCreate(&a));
      CUDA_CHECK(cudaEventCreate(&b));
      CUDA_CHECK(cudaEventRecord(a));
      for (int i = 0; i < iters; ++i) launch(grid);
      CUDA_CHECK(cudaEventRecord(b));
      CUDA_CHECK(cudaEventSynchronize(b));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      ms /= iters;
      CUDA_CHECK(cudaEventDestroy(a));
      CUDA_CHECK(cudaEventDestroy(b));

      const double kb = static_cast<double>(M) * n * kIdxDim * sizeof(bf16);
      const double bytes = kb + static_cast<double>(M) * n * sizeof(float);
      const double flops = 2.0 * static_cast<double>(M) * n * kIdxHeads * kIdxDim;
      std::printf("| %d | %d | %s | %.4f | %.1f | %.1f | %.0f | %.2f |\n", M, n,
                  variant == 0 ? "fma" : (variant == 1 ? "mma" : "stream only"), ms,
                  kb / (1024.0 * 1024.0),
                  bytes / (ms * 1e6), flops / (ms * 1e6), ms * kMlaLayers);
     }
    }
  }
  std::printf("\nread roofline 238 GB/s; top-2048 selection not included\n");
  return 0;
}

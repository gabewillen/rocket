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
// Selection of the top 2048 is included, because the engine needs the index set
// and not the score array. It is a four digit radix select over the order
// preserving uint32 form of the score; the first digit is produced by the scan
// itself and the last two read only the boundary group.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
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

// Register tiled variant. Four threads per candidate, 64 candidates per
// sub-tile, eight sub-tiles per block, so one block covers 512 candidates and
// loads the query once for all of them.
constexpr int kRegThreads = 256;
constexpr int kRegCand = kRegThreads / 4;
constexpr int kRegTiles = 8;
constexpr int kRegBlock = kRegCand * kRegTiles;
constexpr int kQLd = kIdxDim + 4;

// One thread per candidate: 256 threads cover 256 candidates per sub-tile.
constexpr int kReg1Threads = 256;
constexpr int kReg1Tiles = 4;
constexpr int kReg1Block = kReg1Threads * kReg1Tiles;

// Tensor core variant with the transpose removed. 256 threads, 8 warps, each
// warp owning 16 candidates of a 128 candidate tile.
constexpr int kM2Threads = 256;
constexpr int kM2Cand = 128;
constexpr int kM2Tiles = 4;
constexpr int kM2Block = kM2Cand * kM2Tiles;
constexpr int kQLd2 = kIdxDim + 8;    // 136, wmma wants a multiple of 8
constexpr int kCLd2 = kIdxHeads + 4;  // 36, multiple of 4 for the fp32 store

// Same shape, but the key tile is staged in shared memory and the result tile
// is aliased on top of it, so the block pays for one of the two and not both.
constexpr int kM3Key = kM2Cand * kQLd2 * 2;
constexpr int kM3Res = kM2Cand * kCLd2 * 4;
constexpr int kM3Pool = kM3Key > kM3Res ? kM3Key : kM3Res;

// Top-k selection. index_topk is 2048. Selection is a two digit radix select on
// the order preserving uint32 form of the score, so it reads the scores three
// times: one 8 bit histogram, one 8 bit histogram restricted to the boundary
// bin, and one emit pass. The first histogram can be produced by the scan
// itself, since the scan already holds every score in a register.
constexpr int kTopK = 2048;
constexpr int kSelThreads = 256;
constexpr int kSelBins = 256;
constexpr int kSelBlocks = 1024;
// Two 8 bit digits find the boundary group; the remaining two digits refine it.
// The last two passes read only the boundary group, which is why exactness is
// nearly free. kTieCap bounds that group; a stream whose scores collide beyond
// it is reported rather than silently truncated.
constexpr int kTieCap = 65536;

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

// Register tiled, four threads per candidate. No key staging at all: each
// owning 32 of its 128 dims, and each reads its share as one uint4 per 32 dims
// so a warp issues 512 B per load instruction instead of 64 B.
//
// What is left in shared memory is the query, held as fp32 so the inner loop is
// FFMA against LDS.128 with no BF16 conversion. Row stride is padded to 132 so
// the four lane offsets 0, 32, 64, 96 B land in different bank quads, and the
// eight candidates of a warp read the same address, which broadcasts.
//
// The head sum cannot be folded before the ReLU, so the four lanes have to be
// reduced. Doing it as a full butterfly over 32 accumulators costs 64 shuffles.
// Halving the live set at each step instead costs 16 + 8, after which each lane
// owns 8 finished heads, applies its own weights, and two more shuffles finish
// the score: 26 shuffles per candidate against 1024 FFMA.
__global__ __launch_bounds__(kRegThreads) void indexer_scan_reg4(
    const bf16* __restrict__ q, const bf16* __restrict__ keys,
    const float* __restrict__ w, float* __restrict__ score, int n_cand) {
  __align__(16) __shared__ float sqf[kIdxHeads * kQLd];
  __shared__ float sw[kIdxHeads];

  const int m = blockIdx.y;
  const int tid = threadIdx.x;

  const uint4* q4 = reinterpret_cast<const uint4*>(
      q + static_cast<size_t>(m) * kIdxHeads * kIdxDim);
  for (int i = tid; i < kIdxHeads * kIdxDim / 8; i += kRegThreads) {
    const uint4 v = q4[i];
    const __nv_bfloat162* qb = reinterpret_cast<const __nv_bfloat162*>(&v);
    const int e = i * 8;
    float* dst = sqf + (e / kIdxDim) * kQLd + (e % kIdxDim);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 p = __bfloat1622float2(qb[j]);
      dst[2 * j] = p.x;
      dst[2 * j + 1] = p.y;
    }
  }
  for (int i = tid; i < kIdxHeads; i += kRegThreads)
    sw[i] = w[static_cast<size_t>(m) * kIdxHeads + i];
  __syncthreads();

  const int lane4 = tid & 3;
  const int cl = tid >> 2;  // candidate within the 64 wide sub-tile
  const int hbase = (lane4 & 1) * 16 + ((lane4 >> 1) & 1) * 8;
  const float* qrow = sqf + lane4 * 8;
  const uint4* k4 = reinterpret_cast<const uint4*>(keys) +
                    static_cast<size_t>(m) * n_cand * (kIdxDim / 8);

  for (int t = 0; t < kRegTiles; ++t) {
    const int cand = (blockIdx.x * kRegTiles + t) * kRegCand + cl;
    const uint4* kp = k4 + static_cast<size_t>(cand) * (kIdxDim / 8) + lane4;

    float acc[kIdxHeads];
#pragma unroll
    for (int h = 0; h < kIdxHeads; ++h) acc[h] = 0.f;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 v = __ldg(kp + 4 * i);
      const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&v);
      float kf[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 p = __bfloat1622float2(kb[j]);
        kf[2 * j] = p.x;
        kf[2 * j + 1] = p.y;
      }
      const float* qi = qrow + i * 32;
#pragma unroll
      for (int h = 0; h < kIdxHeads; ++h) {
        const float4* qp = reinterpret_cast<const float4*>(qi + h * kQLd);
        const float4 a = qp[0];
        const float4 b = qp[1];
        float s = acc[h];
        s = fmaf(a.x, kf[0], s);
        s = fmaf(a.y, kf[1], s);
        s = fmaf(a.z, kf[2], s);
        s = fmaf(a.w, kf[3], s);
        s = fmaf(b.x, kf[4], s);
        s = fmaf(b.y, kf[5], s);
        s = fmaf(b.z, kf[6], s);
        s = fmaf(b.w, kf[7], s);
        acc[h] = s;
      }
    }

    float r1[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float send = (lane4 & 1) ? acc[j] : acc[16 + j];
      const float mine = (lane4 & 1) ? acc[16 + j] : acc[j];
      r1[j] = mine + __shfl_xor_sync(0xffffffffu, send, 1);
    }
    float r2[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float send = (lane4 & 2) ? r1[j] : r1[8 + j];
      const float mine = (lane4 & 2) ? r1[8 + j] : r1[j];
      r2[j] = mine + __shfl_xor_sync(0xffffffffu, send, 2);
    }

    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) s += sw[hbase + j] * fmaxf(r2[j], 0.f);
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    if (lane4 == 0) score[static_cast<size_t>(m) * n_cand + cand] = s;
  }
}

// Register tiled, one thread per candidate. Same idea, but the four lanes of a
// group no longer read four different query offsets: every lane of the warp is
// on the same dim chunk, so each LDS.128 has one distinct address and returns
// as a broadcast instead of four phases. That is the whole difference between
// this and the variant above.
//
// It costs global load width. A thread owns one candidate's whole 128 dims, so
// a warp's load instruction asks for 32 x 16 B at a 256 B stride and touches 32
// cache lines for 512 B. The other 3.5 KiB of those lines is consumed by the
// next seven chunk iterations, so DRAM traffic is unchanged and only L1 tag
// throughput pays. No shuffle reduction survives: one thread holds all 32 head
// dots for its candidate, and the score write is 32 consecutive floats.
__global__ __launch_bounds__(kReg1Threads) void indexer_scan_reg1(
    const bf16* __restrict__ q, const bf16* __restrict__ keys,
    const float* __restrict__ w, float* __restrict__ score, int n_cand) {
  __align__(16) __shared__ float sqf[kIdxHeads * kQLd];
  __shared__ float sw[kIdxHeads];

  const int m = blockIdx.y;
  const int tid = threadIdx.x;

  const uint4* q4 = reinterpret_cast<const uint4*>(
      q + static_cast<size_t>(m) * kIdxHeads * kIdxDim);
  for (int i = tid; i < kIdxHeads * kIdxDim / 8; i += kReg1Threads) {
    const uint4 v = q4[i];
    const __nv_bfloat162* qb = reinterpret_cast<const __nv_bfloat162*>(&v);
    const int e = i * 8;
    float* dst = sqf + (e / kIdxDim) * kQLd + (e % kIdxDim);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 p = __bfloat1622float2(qb[j]);
      dst[2 * j] = p.x;
      dst[2 * j + 1] = p.y;
    }
  }
  for (int i = tid; i < kIdxHeads; i += kReg1Threads)
    sw[i] = w[static_cast<size_t>(m) * kIdxHeads + i];
  __syncthreads();

  const uint4* k4 = reinterpret_cast<const uint4*>(keys) +
                    static_cast<size_t>(m) * n_cand * (kIdxDim / 8);

  for (int t = 0; t < kReg1Tiles; ++t) {
    const int cand = (blockIdx.x * kReg1Tiles + t) * kReg1Threads + tid;
    const uint4* kp = k4 + static_cast<size_t>(cand) * (kIdxDim / 8);

    float acc[kIdxHeads];
#pragma unroll
    for (int h = 0; h < kIdxHeads; ++h) acc[h] = 0.f;

#pragma unroll 2
    for (int c = 0; c < kIdxDim / 8; ++c) {
      const uint4 v = __ldg(kp + c);
      const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&v);
      float kf[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 p = __bfloat1622float2(kb[j]);
        kf[2 * j] = p.x;
        kf[2 * j + 1] = p.y;
      }
      const float* qc = sqf + c * 8;
#pragma unroll
      for (int h = 0; h < kIdxHeads; ++h) {
        const float4* qp = reinterpret_cast<const float4*>(qc + h * kQLd);
        const float4 a = qp[0];
        const float4 b = qp[1];
        float s = acc[h];
        s = fmaf(a.x, kf[0], s);
        s = fmaf(a.y, kf[1], s);
        s = fmaf(a.z, kf[2], s);
        s = fmaf(a.w, kf[3], s);
        s = fmaf(b.x, kf[4], s);
        s = fmaf(b.y, kf[5], s);
        s = fmaf(b.z, kf[6], s);
        s = fmaf(b.w, kf[7], s);
        acc[h] = s;
      }
    }

    float s = 0.f;
#pragma unroll
    for (int h = 0; h < kIdxHeads; ++h) s += sw[h] * fmaxf(acc[h], 0.f);
    score[static_cast<size_t>(m) * n_cand + cand] = s;
  }
}

// Floor: read the key array and emit one score per candidate, with no scoring
// math. Anything the scan does costs at least this, and the gap to it says
// whether the scan is limited by memory or by the kernel.
// mma2 with the key tile staged. wmma::load_matrix_sync on a global pointer
// compiles to 96 generic LD.E per tile, and the kernel stalls 49 cycles per
// instruction on a full LG queue. Reading the same tile with eight coalesced
// uint4 loads per thread and letting wmma load A out of shared memory turns
// those into 8 LDG.128 plus 8 LDSM.
//
// The [candidate][head] result tile is aliased onto the key tile, which is dead
// by the time the accumulators are stored, so 128 candidates cost 34 KiB and
// not 53 KiB.
// Order preserving map from float to uint32: larger float, larger key. The
// radix select below works on this, so it never compares floats.
__device__ __forceinline__ unsigned int score_key(float f) {
  const unsigned int u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <bool kWithHist>
__global__ __launch_bounds__(kM2Threads) void indexer_scan_mma3_t(
    const bf16* __restrict__ q, const bf16* __restrict__ keys,
    const float* __restrict__ w, float* __restrict__ score, int n_cand,
    unsigned int* __restrict__ hist) {
  using namespace nvcuda;
  __align__(32) __shared__ bf16 sq[kIdxHeads * kQLd2];
  __align__(32) __shared__ char pool[kM3Pool];
  __shared__ float sw[kIdxHeads];
  __shared__ unsigned int shist[kWithHist ? kSelBins : 1];
  bf16* sk = reinterpret_cast<bf16*>(pool);
  float* sc = reinterpret_cast<float*>(pool);

  const int m = blockIdx.y;
  const int tid = threadIdx.x;

  if (kWithHist)
    for (int i = tid; i < kSelBins; i += kM2Threads) shist[i] = 0u;

  const uint4* q4 = reinterpret_cast<const uint4*>(
      q + static_cast<size_t>(m) * kIdxHeads * kIdxDim);
  for (int i = tid; i < kIdxHeads * kIdxDim / 8; i += kM2Threads) {
    const int e = i * 8;
    *reinterpret_cast<uint4*>(sq + (e / kIdxDim) * kQLd2 + (e % kIdxDim)) = q4[i];
  }
  for (int i = tid; i < kIdxHeads; i += kM2Threads)
    sw[i] = w[static_cast<size_t>(m) * kIdxHeads + i];

  const int warp = tid / 32;
  constexpr int kU4PerRow = kIdxDim / 8;              // 16
  constexpr int kStage = kM2Cand * kU4PerRow / kM2Threads;  // 8 uint4 per thread

  for (int t = 0; t < kM2Tiles; ++t) {
    const int base = (blockIdx.x * kM2Tiles + t) * kM2Cand;
    const uint4* g4 = reinterpret_cast<const uint4*>(keys) +
                      (static_cast<size_t>(m) * n_cand + base) * kU4PerRow;
    __syncthreads();
#pragma unroll
    for (int s = 0; s < kStage; ++s) {
      const int i = s * kM2Threads + tid;
      *reinterpret_cast<uint4*>(sk + (i / kU4PerRow) * kQLd2 +
                                (i % kU4PerRow) * 8) = g4[i];
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
    wmma::fill_fragment(c0, 0.f);
    wmma::fill_fragment(c1, 0.f);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b0, b1;
      wmma::load_matrix_sync(a, sk + warp * 16 * kQLd2 + k * 16, kQLd2);
      wmma::load_matrix_sync(b0, sq + k * 16, kQLd2);
      wmma::load_matrix_sync(b1, sq + 16 * kQLd2 + k * 16, kQLd2);
      wmma::mma_sync(c0, a, b0, c0);
      wmma::mma_sync(c1, a, b1, c1);
    }
    __syncthreads();
    wmma::store_matrix_sync(sc + warp * 16 * kCLd2, c0, kCLd2,
                            wmma::mem_row_major);
    wmma::store_matrix_sync(sc + warp * 16 * kCLd2 + 16, c1, kCLd2,
                            wmma::mem_row_major);
    __syncthreads();

    for (int c = tid; c < kM2Cand; c += kM2Threads) {
      const float4* r = reinterpret_cast<const float4*>(sc + c * kCLd2);
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < kIdxHeads / 4; ++j) {
        const float4 v = r[j];
        s += sw[4 * j] * fmaxf(v.x, 0.f);
        s += sw[4 * j + 1] * fmaxf(v.y, 0.f);
        s += sw[4 * j + 2] * fmaxf(v.z, 0.f);
        s += sw[4 * j + 3] * fmaxf(v.w, 0.f);
      }
      score[static_cast<size_t>(m) * n_cand + base + c] = s;
      if (kWithHist) atomicAdd(&shist[score_key(s) >> 24], 1u);
    }
  }

  if (kWithHist) {
    __syncthreads();
    unsigned int* g = hist + static_cast<size_t>(m) * kSelBins;
    for (int i = tid; i < kSelBins; i += kM2Threads)
      if (shist[i]) atomicAdd(g + i, shist[i]);
  }
}

// One 8 bit digit of a radix select. shift 24 is the first pass and looks at
// every score; shift 16 is the second and looks only at the scores whose top
// byte equals the boundary bin the first pass found.
__global__ __launch_bounds__(kSelThreads) void topk_hist(
    const float* __restrict__ score, unsigned int* __restrict__ hist,
    const unsigned int* __restrict__ state, int n_cand, int shift) {
  __shared__ unsigned int h[kSelBins];
  for (int i = threadIdx.x; i < kSelBins; i += kSelThreads) h[i] = 0u;
  __syncthreads();

  const int m = blockIdx.y;
  const float* s = score + static_cast<size_t>(m) * n_cand;
  const unsigned int prefix = (shift == 24) ? 0u : state[m * 4 + 0];
  for (int i = blockIdx.x * kSelThreads + threadIdx.x; i < n_cand;
       i += gridDim.x * kSelThreads) {
    const unsigned int k = score_key(s[i]);
    if (shift == 24 || (k >> 24) == prefix)
      atomicAdd(&h[(k >> shift) & 0xffu], 1u);
  }
  __syncthreads();

  unsigned int* g = hist + static_cast<size_t>(m) * kSelBins;
  for (int i = threadIdx.x; i < kSelBins; i += kSelThreads)
    if (h[i]) atomicAdd(g + i, h[i]);
}

// Walk the 256 bins from the top until the running count would reach 2048. The
// bin that crosses is the boundary; everything above it is definitely selected.
// state is [prefix, above, need, boundary bin] per stream.
__global__ void topk_scan(const unsigned int* __restrict__ hist,
                          unsigned int* __restrict__ state, int shift) {
  const int m = blockIdx.x;
  if (threadIdx.x != 0) return;
  const unsigned int* h = hist + static_cast<size_t>(m) * kSelBins;
  unsigned int above = (shift == 24) ? 0u : state[m * 4 + 1];
  int b = 0;
  for (b = kSelBins - 1; b > 0; --b) {
    if (above + h[b] >= static_cast<unsigned int>(kTopK)) break;
    above += h[b];
  }
  state[m * 4 + 0] =
      (shift == 24) ? static_cast<unsigned int>(b)
                    : ((state[m * 4 + 0] << 8) | static_cast<unsigned int>(b));
  state[m * 4 + 1] = above;
  state[m * 4 + 2] = static_cast<unsigned int>(kTopK) - above;
  state[m * 4 + 3] = static_cast<unsigned int>(b);
}

// First emit. Everything strictly above the 16 bit prefix is in by construction
// and there are fewer than kTopK of them, so it takes slots [0, above). The
// boundary group is compacted into tie so the last two digits can refine it
// without reading the whole score array again.
__global__ __launch_bounds__(kSelThreads) void topk_emit(
    const float* __restrict__ score, const unsigned int* __restrict__ state,
    unsigned int* __restrict__ cnt, int* __restrict__ out,
    int* __restrict__ tie, int n_cand) {
  const int m = blockIdx.y;
  const unsigned int prefix = state[m * 4 + 0];
  const float* s = score + static_cast<size_t>(m) * n_cand;
  int* o = out + static_cast<size_t>(m) * kTopK;
  int* tg = tie + static_cast<size_t>(m) * kTieCap;
  for (int i = blockIdx.x * kSelThreads + threadIdx.x; i < n_cand;
       i += gridDim.x * kSelThreads) {
    const unsigned int p = score_key(s[i]) >> 16;
    if (p > prefix) {
      o[atomicAdd(cnt + m * 2, 1u)] = i;
    } else if (p == prefix) {
      const unsigned int j = atomicAdd(cnt + m * 2 + 1, 1u);
      if (j < kTieCap) tg[j] = i;
    }
  }
}

// Digits three and four, over the compacted boundary group only.
__global__ __launch_bounds__(kSelThreads) void topk_hist_sub(
    const float* __restrict__ score, const int* __restrict__ tie,
    const unsigned int* __restrict__ cnt, unsigned int* __restrict__ hist,
    const unsigned int* __restrict__ state, int n_cand, int shift) {
  __shared__ unsigned int h[kSelBins];
  for (int i = threadIdx.x; i < kSelBins; i += kSelThreads) h[i] = 0u;
  __syncthreads();

  const int m = blockIdx.y;
  const unsigned int prefix = state[m * 4 + 0];
  const unsigned int have = min(cnt[m * 2 + 1], static_cast<unsigned int>(kTieCap));
  const float* s = score + static_cast<size_t>(m) * n_cand;
  const int* tg = tie + static_cast<size_t>(m) * kTieCap;
  for (unsigned int i = blockIdx.x * kSelThreads + threadIdx.x; i < have;
       i += gridDim.x * kSelThreads) {
    const unsigned int k = score_key(s[tg[i]]);
    if ((k >> (shift + 8)) == prefix) atomicAdd(&h[(k >> shift) & 0xffu], 1u);
  }
  __syncthreads();

  unsigned int* g = hist + static_cast<size_t>(m) * kSelBins;
  for (int i = threadIdx.x; i < kSelBins; i += kSelThreads)
    if (h[i]) atomicAdd(g + i, h[i]);
}

// Final emit. After four digits the prefix is the whole ordered key, so the
// remaining group is scores that are bit for bit equal and any choice among
// them is the exact top-k.
__global__ __launch_bounds__(kSelThreads) void topk_emit_sub(
    const float* __restrict__ score, const int* __restrict__ tie,
    const unsigned int* __restrict__ cnt, const unsigned int* __restrict__ state,
    unsigned int* __restrict__ cnt2, int* __restrict__ out, int n_cand) {
  const int m = blockIdx.y;
  const unsigned int prefix = state[m * 4 + 0];
  const unsigned int above = state[m * 4 + 1];
  const unsigned int need = state[m * 4 + 2];
  const unsigned int base = cnt[m * 2 + 0];  // slots [0, base) already written
  const unsigned int have = min(cnt[m * 2 + 1], static_cast<unsigned int>(kTieCap));
  const float* s = score + static_cast<size_t>(m) * n_cand;
  const int* tg = tie + static_cast<size_t>(m) * kTieCap;
  int* o = out + static_cast<size_t>(m) * kTopK;
  for (unsigned int i = blockIdx.x * kSelThreads + threadIdx.x; i < have;
       i += gridDim.x * kSelThreads) {
    const unsigned int k = score_key(s[tg[i]]);
    if (k > prefix) {
      o[base + atomicAdd(cnt2 + m * 2, 1u)] = tg[i];
    } else if (k == prefix) {
      const unsigned int j = atomicAdd(cnt2 + m * 2 + 1, 1u);
      if (j < need) o[above + j] = tg[i];
    }
  }
}

// Floor: read the key array and emit one score per candidate, with no scoring
// math. Anything the scan does costs at least this, and the gap to it says
// whether the scan is limited by memory or by the kernel.
// Tensor cores, with the key transpose deleted. The existing mma kernel holds
// the key tile as [dim][candidate] so it can be the B operand, which is why it
// pays a transposed scalar store into shared memory. Scoring is symmetric: make
// the keys the A operand in their natural [candidate][dim] order and the query
// the B operand, read column major straight out of its own [head][dim] layout
// with ldm = 136. Nothing is transposed and nothing is staged.
//
// Keys are then never staged either. wmma loads the A fragment from global at
// ld = 128, which is 32 B aligned for every 16 dim step, so the key tile costs
// no shared memory and no staging instructions. What is left in shared memory
// is the query and the [candidate][head] result, which the block reduces over
// heads with the weights and the ReLU.
__global__ __launch_bounds__(kM2Threads) void indexer_scan_mma2(
    const bf16* __restrict__ q, const bf16* __restrict__ keys,
    const float* __restrict__ w, float* __restrict__ score, int n_cand) {
  using namespace nvcuda;
  __align__(32) __shared__ bf16 sq[kIdxHeads * kQLd2];
  __align__(16) __shared__ float sc[kM2Cand * kCLd2];
  __shared__ float sw[kIdxHeads];

  const int m = blockIdx.y;
  const int tid = threadIdx.x;

  const uint4* q4 = reinterpret_cast<const uint4*>(
      q + static_cast<size_t>(m) * kIdxHeads * kIdxDim);
  for (int i = tid; i < kIdxHeads * kIdxDim / 8; i += kM2Threads) {
    const int e = i * 8;
    *reinterpret_cast<uint4*>(sq + (e / kIdxDim) * kQLd2 + (e % kIdxDim)) = q4[i];
  }
  for (int i = tid; i < kIdxHeads; i += kM2Threads)
    sw[i] = w[static_cast<size_t>(m) * kIdxHeads + i];
  __syncthreads();

  const int warp = tid / 32;
  for (int t = 0; t < kM2Tiles; ++t) {
    const int base = (blockIdx.x * kM2Tiles + t) * kM2Cand;
    const bf16* krow =
        keys + (static_cast<size_t>(m) * n_cand + base + warp * 16) * kIdxDim;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
    wmma::fill_fragment(c0, 0.f);
    wmma::fill_fragment(c1, 0.f);
    // All eight A fragments are issued before the first mma so the warp has
    // eight independent global loads in flight. Leaving the load next to its
    // mma leaves the kernel latency bound at 35% of memory throughput.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
      wmma::load_matrix_sync(a[k], krow + k * 16, kIdxDim);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b0, b1;
      wmma::load_matrix_sync(b0, sq + k * 16, kQLd2);
      wmma::load_matrix_sync(b1, sq + 16 * kQLd2 + k * 16, kQLd2);
      wmma::mma_sync(c0, a[k], b0, c0);
      wmma::mma_sync(c1, a[k], b1, c1);
    }
    wmma::store_matrix_sync(sc + warp * 16 * kCLd2, c0, kCLd2,
                            wmma::mem_row_major);
    wmma::store_matrix_sync(sc + warp * 16 * kCLd2 + 16, c1, kCLd2,
                            wmma::mem_row_major);
    __syncthreads();

    for (int c = tid; c < kM2Cand; c += kM2Threads) {
      const float4* r = reinterpret_cast<const float4*>(sc + c * kCLd2);
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < kIdxHeads / 4; ++j) {
        const float4 v = r[j];
        s += sw[4 * j] * fmaxf(v.x, 0.f);
        s += sw[4 * j + 1] * fmaxf(v.y, 0.f);
        s += sw[4 * j + 2] * fmaxf(v.z, 0.f);
        s += sw[4 * j + 3] * fmaxf(v.w, 0.f);
      }
      score[static_cast<size_t>(m) * n_cand + base + c] = s;
    }
    __syncthreads();
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

const char* kScanName[] = {"fma",  "mma",  "reg4",       "reg1",
                           "mma2", "mma3", "stream only"};
constexpr int kScanVariants = 7;

void launch_scan(int variant, int M, int n, const bf16* q, const bf16* keys,
                 const float* w, float* score) {
  switch (variant) {
    case 0:
      indexer_scan<<<dim3(n / kTile, M), kThreads>>>(q, keys, w, score, n);
      break;
    case 1:
      indexer_scan_mma<<<dim3(n / kTile, M), kThreads>>>(q, keys, w, score, n);
      break;
    case 2:
      indexer_scan_reg4<<<dim3(n / kRegBlock, M), kRegThreads>>>(q, keys, w,
                                                                 score, n);
      break;
    case 3:
      indexer_scan_reg1<<<dim3(n / kReg1Block, M), kReg1Threads>>>(q, keys, w,
                                                                   score, n);
      break;
    case 4:
      indexer_scan_mma2<<<dim3(n / kM2Block, M), kM2Threads>>>(q, keys, w,
                                                               score, n);
      break;
    case 5:
      indexer_scan_mma3_t<false><<<dim3(n / kM2Block, M), kM2Threads>>>(
          q, keys, w, score, n, nullptr);
      break;
    default:
      indexer_stream_only<<<dim3(n / kThreads, M), kThreads>>>(keys, score, n);
      break;
  }
}

// Registers and shared memory per block decide how many blocks an SM can hold,
// which is the whole question here, so the numbers are printed rather than
// asserted from a comment.
void report_occupancy(const char* name, const void* fn, int threads) {
  cudaFuncAttributes a{};
  CUDA_CHECK(cudaFuncGetAttributes(&a, fn));
  int blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, threads,
                                                           0));
  std::printf("| %s | %d | %d | %zu | %d | %d |\n", name, threads,
              a.numRegs, a.sharedSizeBytes, blocks, blocks * threads);
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
  for (int variant = 0; variant < kScanVariants - 1; ++variant) {
    const int n = kCtx;
    CUDA_CHECK(cudaMemset(d_score, 0, static_cast<size_t>(Mmax) * kCtx * sizeof(float)));
    launch_scan(variant, 1, n, d_q, d_keys, d_w, d_score);
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
                kScanName[variant], mx / den);
  }

  std::printf("\n");
  std::printf("| kernel | threads/block | regs | smem B | blocks/SM | threads/SM |\n");
  std::printf("| --- | --- | --- | --- | --- | --- |\n");
  report_occupancy("fma", reinterpret_cast<const void*>(indexer_scan), kThreads);
  report_occupancy("mma", reinterpret_cast<const void*>(indexer_scan_mma),
                   kThreads);
  report_occupancy("reg4", reinterpret_cast<const void*>(indexer_scan_reg4),
                   kRegThreads);
  report_occupancy("reg1", reinterpret_cast<const void*>(indexer_scan_reg1),
                   kReg1Threads);
  report_occupancy("mma2", reinterpret_cast<const void*>(indexer_scan_mma2),
                   kM2Threads);
  report_occupancy("mma3",
                   reinterpret_cast<const void*>(indexer_scan_mma3_t<false>),
                   kM2Threads);
  report_occupancy("mma3+hist",
                   reinterpret_cast<const void*>(indexer_scan_mma3_t<true>),
                   kM2Threads);
  report_occupancy("stream only",
                   reinterpret_cast<const void*>(indexer_stream_only), kThreads);

  std::printf("\n");
  std::printf("| M | candidates | kernel | ms/layer | key MiB | GB/s | GFLOP/s | 11 layers ms |\n");
  std::printf("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
  for (int M : Ms) {
    for (int n : {kCtx, kCtx / kPool}) {
     for (int variant = 0; variant < kScanVariants; ++variant) {
      auto launch = [&]() { launch_scan(variant, M, n, d_q, d_keys, d_w, d_score); };
      for (int i = 0; i < 3; ++i) launch();
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaEvent_t a, b;
      CUDA_CHECK(cudaEventCreate(&a));
      CUDA_CHECK(cudaEventCreate(&b));
      CUDA_CHECK(cudaEventRecord(a));
      for (int i = 0; i < iters; ++i) launch();
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
                  kScanName[variant], ms,
                  kb / (1024.0 * 1024.0),
                  bytes / (ms * 1e6), flops / (ms * 1e6), ms * kMlaLayers);
     }
    }
  }
  // Selection. The scan is only half the pass: the engine needs the 2048 index
  // set, not the score array.
  unsigned int* d_hist = nullptr;
  unsigned int* d_state = nullptr;
  unsigned int* d_cnt = nullptr;
  int* d_out = nullptr;
  unsigned int* d_cnt2 = nullptr;
  int* d_tie = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hist, static_cast<size_t>(Mmax) * kSelBins * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_state, static_cast<size_t>(Mmax) * 4 * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_cnt, static_cast<size_t>(Mmax) * 2 * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(Mmax) * kTopK * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cnt2, static_cast<size_t>(Mmax) * 2 * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_tie, static_cast<size_t>(Mmax) * kTieCap * sizeof(int)));
  const size_t hist_bytes = static_cast<size_t>(Mmax) * kSelBins * sizeof(unsigned int);
  cudaStream_t sel_stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&sel_stream));

  // fused == the first histogram came from the scan, so selection is two passes
  // over the scores instead of three.
  // Selection is nine kernels and five memsets. On a stream so it can be
  // captured into a graph, which is the only way to see how much of it is
  // launch overhead rather than work.
  auto run_select = [&](int M, int n, bool fused, cudaStream_t st) {
    const int blocks = std::min(kSelBlocks, n / kSelThreads);
    const int sub_blocks = 64;
    CUDA_CHECK(cudaMemsetAsync(d_state, 0, static_cast<size_t>(M) * 4 * sizeof(unsigned int), st));
    CUDA_CHECK(cudaMemsetAsync(d_cnt, 0, static_cast<size_t>(M) * 2 * sizeof(unsigned int), st));
    CUDA_CHECK(cudaMemsetAsync(d_cnt2, 0, static_cast<size_t>(M) * 2 * sizeof(unsigned int), st));
    if (!fused) {
      CUDA_CHECK(cudaMemsetAsync(d_hist, 0, hist_bytes, st));
      topk_hist<<<dim3(blocks, M), kSelThreads, 0, st>>>(d_score, d_hist, d_state, n, 24);
    }
    topk_scan<<<M, 32, 0, st>>>(d_hist, d_state, 24);
    CUDA_CHECK(cudaMemsetAsync(d_hist, 0, hist_bytes, st));
    topk_hist<<<dim3(blocks, M), kSelThreads, 0, st>>>(d_score, d_hist, d_state, n, 16);
    topk_scan<<<M, 32, 0, st>>>(d_hist, d_state, 16);
    topk_emit<<<dim3(blocks, M), kSelThreads, 0, st>>>(d_score, d_state, d_cnt,
                                                       d_out, d_tie, n);
    for (int shift : {8, 0}) {
      CUDA_CHECK(cudaMemsetAsync(d_hist, 0, hist_bytes, st));
      topk_hist_sub<<<dim3(sub_blocks, M), kSelThreads, 0, st>>>(
          d_score, d_tie, d_cnt, d_hist, d_state, n, shift);
      topk_scan<<<M, 32, 0, st>>>(d_hist, d_state, shift);
    }
    topk_emit_sub<<<dim3(sub_blocks, M), kSelThreads, 0, st>>>(
        d_score, d_tie, d_cnt, d_state, d_cnt2, d_out, n);
  };
  auto run_scan_hist = [&](int M, int n) {
    CUDA_CHECK(cudaMemsetAsync(d_hist, 0, hist_bytes));
    indexer_scan_mma3_t<true><<<dim3(n / kM2Block, M), kM2Threads>>>(
        d_q, d_keys, d_w, d_score, n, d_hist);
  };

  // Correctness: stream 0 at the full context. Every selected score must be at
  // least the true 2048th largest, the set must be exactly 2048, and no index
  // may repeat.
  {
    const int n = kCtx;
    run_scan_hist(1, n);
    run_select(1, n, true, sel_stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hs(n);
    std::vector<int> ho(kTopK);
    std::vector<unsigned int> hc(2);
    CUDA_CHECK(cudaMemcpy(hs.data(), d_score, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ho.data(), d_out, kTopK * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hc.data(), d_cnt, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    std::vector<float> ref = hs;
    std::nth_element(ref.begin(), ref.begin() + (kTopK - 1), ref.end(),
                     std::greater<float>());
    const float kth = ref[kTopK - 1];
    std::vector<char> seen(n, 0);
    int bad = 0, dup = 0;
    for (int i = 0; i < kTopK; ++i) {
      const int idx = ho[i];
      if (idx < 0 || idx >= n) { ++bad; continue; }
      if (seen[idx]) ++dup;
      seen[idx] = 1;
      if (hs[idx] < kth) ++bad;
    }
    std::printf("\nselection top-%d on stream 0 of %d: %d below the true %dth "
                "largest, %d duplicates, threshold %.6f, boundary group %u\n",
                kTopK, n, bad, kTopK, dup, kth, hc[1]);
  }

  std::printf("\n| M | candidates | scan ms | scan+hist ms | select 3-pass ms | "
              "select 2-pass ms | select graph ms | scan+select ms | select share |\n");
  std::printf("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
  for (int M : Ms) {
    for (int n : {kCtx, kCtx / kPool}) {
      auto time_it = [&](const std::function<void()>& fn, cudaStream_t st) {
        for (int i = 0; i < 3; ++i) fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        CUDA_CHECK(cudaEventCreate(&a));
        CUDA_CHECK(cudaEventCreate(&b));
        CUDA_CHECK(cudaEventRecord(a, st));
        for (int i = 0; i < iters; ++i) fn();
        CUDA_CHECK(cudaEventRecord(b, st));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        CUDA_CHECK(cudaEventDestroy(a));
        CUDA_CHECK(cudaEventDestroy(b));
        return ms / iters;
      };
      const float t_scan =
          time_it([&] { launch_scan(5, M, n, d_q, d_keys, d_w, d_score); }, 0);
      const float t_scanh = time_it([&] { run_scan_hist(M, n); }, 0);
      const float t_sel3 =
          time_it([&] { run_select(M, n, false, sel_stream); }, sel_stream);
      run_scan_hist(M, n);
      CUDA_CHECK(cudaDeviceSynchronize());
      const float t_sel2 =
          time_it([&] { run_select(M, n, true, sel_stream); }, sel_stream);

      cudaGraph_t graph;
      cudaGraphExec_t gexec;
      CUDA_CHECK(cudaStreamBeginCapture(sel_stream, cudaStreamCaptureModeThreadLocal));
      run_select(M, n, true, sel_stream);
      CUDA_CHECK(cudaStreamEndCapture(sel_stream, &graph));
      CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
      const float t_graph =
          time_it([&] { CUDA_CHECK(cudaGraphLaunch(gexec, sel_stream)); }, sel_stream);
      CUDA_CHECK(cudaGraphExecDestroy(gexec));
      CUDA_CHECK(cudaGraphDestroy(graph));

      const float total = t_scanh + t_graph;
      std::printf("| %d | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.0f%% |\n",
                  M, n, t_scan, t_scanh, t_sel3, t_sel2, t_graph, total,
                  100.0 * (total - t_scan) / total);
    }
  }

  std::printf("\nread roofline 238 GB/s\n");
  return 0;
}

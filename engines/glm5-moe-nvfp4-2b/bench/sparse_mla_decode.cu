// Sparse MLA decode step on sm_121 (GB10): gather top-k KV from a paged pool
// sized for 262144 tokens, then run the MLA attention math at real dims.
//
// Question: 11 of the 45 text layers are sparse MLA with a DSA indexer that
// selects 2048 entries out of the context. What does one such layer cost per
// decode step for M streams at the 262k serving cap, and how much does the
// sparsity actually buy against dense attention?
//
// Dims from the on-disk checkpoint (scripts/attention/dump-attention-shapes.py):
//   kv_a_proj_with_mqa [512, 4096]   -> cached latent is 512 wide
//   kv_b_proj          [32768, 512]  -> 64 heads x (256 nope + 256 v)
//   q_b_proj           [16384, 1536] -> 64 heads x 256, no rope split
//   o_proj             [4096, 16384]
// config.json: qk_rope_head_dim 0, mla_use_nope true, so the cached latent is
// 512 wide and nothing else. 262144 tokens * 512 * 2 B = 256 MiB per stream
// per layer.
//
// The attention runs in absorbed form, which is what makes MLA cheap at decode:
// W^K_b is folded into the query, so all 64 heads score against the same 512
// wide latent and the cache is read once per selected entry.
//   scores[h][t] = (q_latent[h] . c_t) / sqrt(256)
//   o_latent[h]  = sum_t softmax(scores)[h][t] * c_t
//   o[h]         = W^V_b[h]^T o_latent[h]     (512 -> 256)
//
// The pool is paged at 64 KiB, the quantum this project aligns to
// (blog/rocket.qmd#alignment): 64 tokens * 512 * 2 B = 65536 B per page, so a
// 262144 token stream is 4096 pages. Pages are handed out from one shuffled
// pool shared by all streams, so the gather is as fragmented as it would be
// after real prefix forking.
//
// Two selection patterns are measured, because the checkpoint says the indexer
// pools keys 4:1 (index_kpool 4, index_kpool_compress true) and that decides
// the gather granularity:
//   grouped   512 runs of 4 consecutive tokens, 4 KiB per run
//   scattered 2048 independent tokens, 1 KiB per run
// Dense reference is the same pipeline over a contiguous 8192 token context.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err_ = (x);                                                    \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                   \
                   cudaGetErrorString(err_));                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublasStatus_t st_ = (x);                                                  \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "%s:%d cublas %d\n", __FILE__, __LINE__, (int)st_); \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

constexpr int kHeads = 64;
constexpr int kLatent = 512;   // kv_lora_rank
constexpr int kVHead = 256;    // v_head_dim
constexpr int kQkHead = 256;   // qk_nope_head_dim
constexpr int kTopK = 2048;    // index_topk
constexpr int kCtx = 262144;   // serving_context_cap
constexpr int kDenseCtx = 8192;
constexpr int kPageTokens = 64;  // 64 KiB page / (512 * 2 B)
constexpr int kPagesPerStream = kCtx / kPageTokens;  // 4096
constexpr int kMlaLayers = 11;

using bf16 = __nv_bfloat16;

// Deterministic device-side fill, so a 16 GiB pool does not cross the bus.
__global__ void fill_pool(bf16* __restrict__ p, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    unsigned int h = static_cast<unsigned int>(i) * 2654435761u + 1013904223u;
    h ^= h >> 15;
    p[i] = __float2bfloat16((static_cast<float>(h & 0xffffu) / 32768.f - 1.f) * 0.1f);
  }
}

// Gather selected tokens out of the paged pool into a compact [M][K][512] buffer.
// One block per (stream, token); 128 threads move 512 bf16 as float4.
__global__ void gather_paged(const bf16* __restrict__ pool,
                             const int* __restrict__ block_table,
                             const int* __restrict__ sel, bf16* __restrict__ out,
                             int k_sel) {
  const int t = blockIdx.x;
  const int m = blockIdx.y;
  const int token = sel[static_cast<size_t>(m) * k_sel + t];
  const int page = block_table[static_cast<size_t>(m) * kPagesPerStream + token / kPageTokens];
  const size_t src =
      (static_cast<size_t>(page) * kPageTokens + token % kPageTokens) * kLatent;
  const size_t dst = (static_cast<size_t>(m) * k_sel + t) * kLatent;
  const float4* s4 = reinterpret_cast<const float4*>(pool + src);
  float4* d4 = reinterpret_cast<float4*>(out + dst);
  for (int i = threadIdx.x; i < kLatent / 8; i += blockDim.x) d4[i] = s4[i];
}

// Softmax over the selected axis, then emit BF16 probabilities for the second
// GEMM. Layout is [M][H][k_sel], k_sel contiguous.
__global__ void softmax_rows(const float* __restrict__ in, bf16* __restrict__ out,
                             int k_sel, float scale) {
  const int row = blockIdx.x;
  const size_t base = static_cast<size_t>(row) * k_sel;
  __shared__ float red[256];
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < k_sel; i += blockDim.x) mx = fmaxf(mx, in[base + i] * scale);
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();
  float sum = 0.f;
  for (int i = threadIdx.x; i < k_sel; i += blockDim.x) sum += __expf(in[base + i] * scale - mx);
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.f / red[0];
  __syncthreads();
  for (int i = threadIdx.x; i < k_sel; i += blockDim.x)
    out[base + i] = __float2bfloat16(__expf(in[base + i] * scale - mx) * inv);
}

struct Timer {
  cudaEvent_t a, b;
  Timer() {
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
  }
  ~Timer() {
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
  void start() { CUDA_CHECK(cudaEventRecord(a)); }
  float stop() {
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
  }
};

}  // namespace

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? std::atoi(argv[1]) : 50;
  const std::vector<int> Ms = {8, 16, 32, 64};
  const int Mmax = Ms.back();

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);
  std::printf("MLA latent=%d heads=%d v_head=%d, ctx=%d, page=%d tokens (%zu B), "
              "topk=%d\n",
              kLatent, kHeads, kVHead, kCtx, kPageTokens,
              sizeof(bf16) * kPageTokens * kLatent, kTopK);
  std::printf("pool for M=%d: %.2f GiB\n\n", Mmax,
              static_cast<double>(Mmax) * kCtx * kLatent * sizeof(bf16) /
                  (1024.0 * 1024.0 * 1024.0));

  cublasHandle_t cb;
  CUBLAS_CHECK(cublasCreate(&cb));
  CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));

  std::mt19937 rng(20260906);
  std::uniform_real_distribution<float> unit(-1.f, 1.f);

  // Paged pool, one shared arena of M * 4096 pages, shuffled so a stream's
  // pages are scattered through it.
  const size_t total_pages = static_cast<size_t>(Mmax) * kPagesPerStream;
  const size_t pool_elems = total_pages * kPageTokens * kLatent;
  bf16* d_pool = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pool, pool_elems * sizeof(bf16)));
  fill_pool<<<4096, 256>>>(d_pool, pool_elems);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> pages(total_pages);
  std::iota(pages.begin(), pages.end(), 0);
  std::shuffle(pages.begin(), pages.end(), rng);
  int* d_table = nullptr;
  CUDA_CHECK(cudaMalloc(&d_table, total_pages * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_table, pages.data(), total_pages * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Selection patterns.
  std::vector<int> sel_grouped(static_cast<size_t>(Mmax) * kTopK);
  std::vector<int> sel_scattered(static_cast<size_t>(Mmax) * kTopK);
  {
    std::uniform_int_distribution<int> grp(0, kCtx / 4 - 1);
    std::uniform_int_distribution<int> tok(0, kCtx - 1);
    for (int m = 0; m < Mmax; ++m) {
      std::vector<int> g(kTopK / 4);
      for (auto& x : g) x = grp(rng);
      std::sort(g.begin(), g.end());
      for (size_t i = 0; i < g.size(); ++i)
        for (int j = 0; j < 4; ++j)
          sel_grouped[static_cast<size_t>(m) * kTopK + i * 4 + j] = g[i] * 4 + j;
      std::vector<int> t(kTopK);
      for (auto& x : t) x = tok(rng);
      std::sort(t.begin(), t.end());
      for (int i = 0; i < kTopK; ++i) sel_scattered[static_cast<size_t>(m) * kTopK + i] = t[i];
    }
  }
  // Dense reference: the first 8192 tokens, contiguous.
  std::vector<int> sel_dense(static_cast<size_t>(Mmax) * kDenseCtx);
  for (int m = 0; m < Mmax; ++m)
    for (int i = 0; i < kDenseCtx; ++i)
      sel_dense[static_cast<size_t>(m) * kDenseCtx + i] = i;

  int *d_sel_g, *d_sel_s, *d_sel_d;
  CUDA_CHECK(cudaMalloc(&d_sel_g, sel_grouped.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sel_s, sel_scattered.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sel_d, sel_dense.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_sel_g, sel_grouped.data(), sel_grouped.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sel_s, sel_scattered.data(), sel_scattered.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sel_d, sel_dense.data(), sel_dense.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Query in latent space (W^K_b already absorbed), and the value-side
  // projection W^V_b.
  std::vector<bf16> hq(static_cast<size_t>(Mmax) * kHeads * kLatent);
  for (auto& x : hq) x = __float2bfloat16(unit(rng) * 0.05f);
  std::vector<bf16> hwv(static_cast<size_t>(kHeads) * kLatent * kVHead);
  for (auto& x : hwv) x = __float2bfloat16(unit(rng) * 0.05f);
  bf16 *d_q, *d_wv;
  CUDA_CHECK(cudaMalloc(&d_q, hq.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_wv, hwv.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMemcpy(d_q, hq.data(), hq.size() * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wv, hwv.data(), hwv.size() * sizeof(bf16), cudaMemcpyHostToDevice));

  const int kmax = kDenseCtx;
  bf16 *d_kg, *d_p;
  float* d_s;
  bf16 *d_olat, *d_out;
  CUDA_CHECK(cudaMalloc(&d_kg, static_cast<size_t>(Mmax) * kmax * kLatent * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_s, static_cast<size_t>(Mmax) * kHeads * kmax * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_p, static_cast<size_t>(Mmax) * kHeads * kmax * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_olat, static_cast<size_t>(Mmax) * kHeads * kLatent * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(Mmax) * kHeads * kVHead * sizeof(bf16)));

  const float scale = 1.0f / std::sqrt(static_cast<float>(kQkHead));
  const float one = 1.f, zero = 0.f;

  auto run_attn = [&](int M, int k_sel, const int* sel, bool do_gather) {
    if (do_gather)
      gather_paged<<<dim3(k_sel, M), 128>>>(d_pool, d_table, sel, d_kg, k_sel);
    // scores: C_cm[k_sel][H] = Kc^T * Qc, Kc is [512 x k_sel] col-major
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cb, CUBLAS_OP_T, CUBLAS_OP_N, k_sel, kHeads, kLatent, &one, d_kg,
        CUDA_R_16BF, kLatent, static_cast<long long>(k_sel) * kLatent, d_q,
        CUDA_R_16BF, kLatent, static_cast<long long>(kHeads) * kLatent, &zero,
        d_s, CUDA_R_32F, k_sel, static_cast<long long>(kHeads) * k_sel, M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    softmax_rows<<<M * kHeads, 256>>>(d_s, d_p, k_sel, scale);
    // o_latent: C_cm[512][H] = Kc * Pc, Pc is [k_sel x H] col-major
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cb, CUBLAS_OP_N, CUBLAS_OP_N, kLatent, kHeads, k_sel, &one, d_kg,
        CUDA_R_16BF, kLatent, static_cast<long long>(k_sel) * kLatent, d_p,
        CUDA_R_16BF, k_sel, static_cast<long long>(kHeads) * k_sel, &zero,
        d_olat, CUDA_R_16BF, kLatent, static_cast<long long>(kHeads) * kLatent, M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    // out[m][h] = W^V_b[h]^T o_latent[m][h]. W^V_b is shared by all streams, so
    // the batch runs over heads and the M streams of a head are the GEMM's n.
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cb, CUBLAS_OP_N, CUBLAS_OP_N, kVHead, M, kLatent, &one, d_wv, CUDA_R_16BF,
        kVHead, static_cast<long long>(kLatent) * kVHead, d_olat, CUDA_R_16BF,
        static_cast<long long>(kHeads) * kLatent, kLatent, &zero, d_out,
        CUDA_R_16BF, static_cast<long long>(kHeads) * kVHead, kVHead, kHeads,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };

  // Correctness for the grouped gather, stream 0, first 4 heads, in double.
  {
    const int M = 8;
    run_attn(M, kTopK, d_sel_g, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<bf16> gk(static_cast<size_t>(kTopK) * kLatent);
    CUDA_CHECK(cudaMemcpy(gk.data(), d_kg, gk.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::vector<bf16> gout(static_cast<size_t>(kHeads) * kVHead);
    CUDA_CHECK(cudaMemcpy(gout.data(), d_out, gout.size() * sizeof(bf16),
                          cudaMemcpyDeviceToHost));
    // Independently verify the gather against the pool.
    std::vector<bf16> probe(kLatent);
    double gmax = 0.0;
    for (int t = 0; t < kTopK; t += 97) {
      const int token = sel_grouped[t];
      const int page = pages[token / kPageTokens];
      const size_t src = (static_cast<size_t>(page) * kPageTokens + token % kPageTokens) * kLatent;
      CUDA_CHECK(cudaMemcpy(probe.data(), d_pool + src, kLatent * sizeof(bf16),
                            cudaMemcpyDeviceToHost));
      for (int d = 0; d < kLatent; ++d)
        gmax = std::max(gmax, static_cast<double>(std::fabs(
                                  __bfloat162float(probe[d]) -
                                  __bfloat162float(gk[static_cast<size_t>(t) * kLatent + d]))));
    }
    std::printf("gather exactness (max abs diff vs pool): %.1e\n", gmax);

    double omax = 0.0, oden = 0.0;
    for (int h = 0; h < 4; ++h) {
      std::vector<double> sc(kTopK);
      double mx = -1e300;
      for (int t = 0; t < kTopK; ++t) {
        double a = 0.0;
        for (int d = 0; d < kLatent; ++d)
          a += static_cast<double>(__bfloat162float(hq[static_cast<size_t>(h) * kLatent + d])) *
               __bfloat162float(gk[static_cast<size_t>(t) * kLatent + d]);
        sc[t] = a * scale;
        mx = std::max(mx, sc[t]);
      }
      double sum = 0.0;
      for (auto& x : sc) { x = std::exp(x - mx); sum += x; }
      std::vector<double> ol(kLatent, 0.0);
      for (int t = 0; t < kTopK; ++t)
        for (int d = 0; d < kLatent; ++d)
          ol[d] += (sc[t] / sum) * __bfloat162float(gk[static_cast<size_t>(t) * kLatent + d]);
      for (int d = 0; d < kVHead; ++d) {
        double a = 0.0;
        for (int l = 0; l < kLatent; ++l)
          a += ol[l] * __bfloat162float(hwv[(static_cast<size_t>(h) * kLatent + l) * kVHead + d]);
        omax = std::max(omax, std::fabs(a - __bfloat162float(gout[static_cast<size_t>(h) * kVHead + d])));
        oden = std::max(oden, std::fabs(a));
      }
    }
    std::printf("correctness (4 heads vs double reference): o rel %.3e\n\n", omax / oden);
  }

  std::printf("| M | variant | k | ms/layer | gather ms | KV MiB read | GB/s | 11 layers ms |\n");
  std::printf("| --- | --- | --- | --- | --- | --- | --- | --- |\n");

  struct Case { const char* name; int k; int* sel; bool gather; };
  for (int M : Ms) {
    const Case cases[] = {
        {"sparse grouped", kTopK, d_sel_g, true},
        {"sparse scattered", kTopK, d_sel_s, true},
        {"dense 8k", kDenseCtx, d_sel_d, true},
    };
    for (const Case& c : cases) {
      for (int i = 0; i < 5; ++i) run_attn(M, c.k, c.sel, c.gather);
      CUDA_CHECK(cudaDeviceSynchronize());
      Timer t;
      t.start();
      for (int i = 0; i < iters; ++i) run_attn(M, c.k, c.sel, c.gather);
      const float ms = t.stop() / iters;

      Timer tg;
      for (int i = 0; i < 5; ++i)
        gather_paged<<<dim3(c.k, M), 128>>>(d_pool, d_table, c.sel, d_kg, c.k);
      CUDA_CHECK(cudaDeviceSynchronize());
      tg.start();
      for (int i = 0; i < iters; ++i)
        gather_paged<<<dim3(c.k, M), 128>>>(d_pool, d_table, c.sel, d_kg, c.k);
      const float msg = tg.stop() / iters;

      const double kv_bytes =
          static_cast<double>(M) * c.k * kLatent * sizeof(bf16);
      std::printf("| %d | %s | %d | %.4f | %.4f | %.1f | %.1f | %.2f |\n", M,
                  c.name, c.k, ms, msg, kv_bytes / (1024.0 * 1024.0),
                  2.0 * kv_bytes / (msg * 1e6), ms * kMlaLayers);
    }
  }
  std::printf("\ngather GB/s counts one read from the pool and one write to the "
              "compact buffer; copy roofline 216 GB/s\n");
  CUBLAS_CHECK(cublasDestroy(cb));
  return 0;
}

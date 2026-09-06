// What page size should the KV pool use?
//
// Page size trades two things against each other. Small pages share at finer
// granularity, so a fork at an arbitrary token position wastes less of a
// boundary page and a partially-diverged subagent pins fewer whole pages.
// Large pages keep a stream's logical token range in fewer physical extents,
// which is what the gather has to walk.
//
// The gather is the side that has to be measured, because the sharing side is
// arithmetic. Two kernels read the cache through the block table every decode
// step of every MLA layer:
//   indexer scan   indexer_pool_keys + indexer_scores, over EVERY cached
//                  token (the DSA indexer scores all of them), reading
//                  key+gate in runs of kpool=4;
//   sparse gather  mla_scores + mla_softmax + mla_context, over the 2048
//                  tokens top-k selected, read as 512 scattered runs of 4.
//
// Two physical layouts are measured at every page size, because a page-size
// sweep that cannot see layout at all would report "flat" for the wrong
// reason:
//   rr    round-robin, what a real pool produces when M streams grow
//         together: stream m's logical page i is physical page i*M + m, so
//         consecutive logical pages are a page-stride apart;
//   seq   stream-contiguous, physical page i*pages_per_stream + m, the
//         best case and what stage 1's one-page-per-stream allocator gives.
// If rr and seq time the same, the gather is not layout-sensitive and page
// size cannot be chosen on gather grounds. `--sel dense` is the control that
// says whether this bench can see locality at all: it selects 512 adjacent
// pools instead of 512 scattered ones, which is a strictly better access
// pattern. A bench where dense and scattered also tie is measuring something
// other than memory and its flat page-size sweep proves nothing.
//
// Synthetic KV throughout: random latents, keys and gates, random top-k
// selection. No checkpoint is loaded and no weights are touched.
//
// Usage:
//   bench/kv-page-gather [--ctx 65536] [--streams 8] [--iters 20] [--repeat 1]
//                        [--sel scatter|dense] [--page N]
//
// One 8.25 GiB arena is allocated and freed per configuration, and running
// several configurations in one process makes the later ones slower by up to
// 30% as the allocator recycles that much unified memory. Drive it one page
// size per process with scripts/cache/kv-page-gather.sh, which is how the
// numbers in the log were produced.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kernels.h"
#include "kv/kv_arena.h"
#include "kv/page_pool.h"

using rocket::engine::bf16;
using rocket::engine::KvPages;
namespace kv = rocket::engine::kv;

namespace {

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}

// Deterministic fill: an LCG per element, so a rerun at the same page size
// sees the same bytes and two page sizes see the same values per (stream,
// logical token), which is what makes the timings comparable.
__global__ void fill_lcg(bf16* p, long long n, unsigned seed) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned x = seed + (unsigned)(i * 2654435761u);
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    p[i] = __float2bfloat16(((float)(x & 0xffff) / 32768.0f) - 1.0f);
  }
}

__global__ void fill_lcg_f(float* p, long long n, unsigned seed) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    unsigned x = seed + (unsigned)(i * 2654435761u);
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    p[i] = ((float)(x & 0xffff) / 32768.0f) - 1.0f;
  }
}

struct Timing {
  double scan_ms = 0;
  double gather_ms = 0;
};

Timing run_one(int page_tokens, int ctx, int streams, int iters, bool round_robin,
               bool dense_sel, double* arena_gib) {
  // Fuel geometry, fuels/glm-5.3-flash/attention.yaml.
  kv::KvGeometry g;
  g.page_tokens = page_tokens;
  g.layers = 11;
  g.kv_lora = 512;
  g.index_head_dim = 128;
  g.kpool = 4;
  const std::string bad = g.why_invalid();
  if (!bad.empty()) {
    std::fprintf(stderr, "page_tokens=%d rejected: %s\n", page_tokens, bad.c_str());
    std::exit(1);
  }

  const int pages_per_stream = ctx / page_tokens;
  const int num_pages = pages_per_stream * streams;
  cudaStream_t s = nullptr;
  ck(cudaStreamCreate(&s), "stream");
  kv::KvArena arena(g, num_pages, streams, pages_per_stream, s);
  *arena_gib = (double)arena.bytes() / (1024.0 * 1024.0 * 1024.0);

  for (int m = 0; m < streams; ++m) {
    std::vector<int> table(pages_per_stream);
    for (int i = 0; i < pages_per_stream; ++i)
      table[i] = round_robin ? i * streams + m : m * pages_per_stream + i;
    arena.upload_table(m, table);
  }

  const KvPages& kvp = arena.pages();
  const long long slots = (long long)num_pages * g.layers * page_tokens;
  fill_lcg<<<1024, 256, 0, s>>>(kvp.latent, slots * g.kv_lora, 1u);
  fill_lcg<<<1024, 256, 0, s>>>(kvp.key, slots * g.index_head_dim, 2u);
  fill_lcg<<<1024, 256, 0, s>>>(kvp.gate, slots * g.index_head_dim, 3u);

  const int heads = 64, kv_lora = 512, ihd = 128, ih = 32, kpool = 4;
  const int layer_slot = 3;
  const int topk = 2048;
  const int sel_stride = topk + kpool - 1;
  const int n_pools = ctx / kpool;
  const int pool_stride = n_pools + 1;

  void* owned[16];
  int n_owned = 0;
  auto A = [&](std::size_t bytes) {
    void* p = nullptr;
    ck(cudaMalloc(&p, bytes), "cudaMalloc bench");
    ck(cudaMemset(p, 0, bytes), "memset bench");
    owned[n_owned++] = p;
    return p;
  };
  auto* q_abs = (float*)A((std::size_t)streams * heads * kv_lora * sizeof(float));
  auto* scores = (float*)A((std::size_t)streams * heads * sel_stride * sizeof(float));
  auto* ctxbuf = (float*)A((std::size_t)streams * heads * kv_lora * sizeof(float));
  auto* sel = (int*)A((std::size_t)streams * sel_stride * sizeof(int));
  auto* n_sel = (int*)A((std::size_t)streams * sizeof(int));
  auto* n_pools_d = (int*)A((std::size_t)streams * sizeof(int));
  auto* pool_keys = (bf16*)A((std::size_t)streams * pool_stride * ihd * sizeof(bf16));
  auto* pool_scores = (float*)A((std::size_t)streams * pool_stride * sizeof(float));
  auto* q_idx = (bf16*)A((std::size_t)streams * ih * ihd * sizeof(bf16));
  auto* head_w = (float*)A((std::size_t)streams * ih * sizeof(float));
  auto* ape = (bf16*)A((std::size_t)kpool * ihd * sizeof(bf16));

  fill_lcg_f<<<256, 256, 0, s>>>(q_abs, (long long)streams * heads * kv_lora, 4u);
  fill_lcg<<<64, 256, 0, s>>>(q_idx, (long long)streams * ih * ihd, 5u);
  fill_lcg_f<<<8, 256, 0, s>>>(head_w, (long long)streams * ih, 6u);
  fill_lcg<<<8, 256, 0, s>>>(ape, (long long)kpool * ihd, 7u);

  // Top-k selection: 512 pools per stream expanded to their 4 member tokens.
  // `scatter` spreads them over the whole context (what the DSA top-k
  // actually produces); `dense` takes 512 adjacent pools and exists only as
  // the locality control. Same logical token ids at every page size, so only
  // the physical layout differs between configurations.
  {
    std::vector<int> h_sel((std::size_t)streams * sel_stride, 0);
    std::vector<int> h_n(streams, topk);
    unsigned rng = 12345u;
    for (int m = 0; m < streams; ++m) {
      for (int i = 0; i < topk / kpool; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        const int pool = dense_sel ? i : (int)(rng % (unsigned)n_pools);
        for (int k = 0; k < kpool; ++k)
          h_sel[(std::size_t)m * sel_stride + i * kpool + k] = pool * kpool + k;
      }
    }
    std::vector<int> h_np(streams, n_pools);
    ck(cudaMemcpyAsync(sel, h_sel.data(), h_sel.size() * sizeof(int), cudaMemcpyHostToDevice, s),
       "sel upload");
    ck(cudaMemcpyAsync(n_sel, h_n.data(), h_n.size() * sizeof(int), cudaMemcpyHostToDevice, s),
       "n_sel upload");
    ck(cudaMemcpyAsync(n_pools_d, h_np.data(), h_np.size() * sizeof(int), cudaMemcpyHostToDevice, s),
       "n_pools upload");
  }
  ck(cudaStreamSynchronize(s), "setup sync");

  cudaEvent_t e0, e1;
  ck(cudaEventCreate(&e0), "ev"); ck(cudaEventCreate(&e1), "ev");
  // The GB10 idles at P8 and this bench alternates host-side setup with short
  // measured bursts, so a single pass measures the clock ramp as much as the
  // kernel. One full-length warmup pass to reach steady clocks, then best of
  // three: the minimum is the run that was not interrupted by a clock drop.
  auto time_it = [&](auto&& body) {
    for (int i = 0; i < iters; ++i) body();
    ck(cudaStreamSynchronize(s), "warmup");
    double best = 1e30;
    for (int pass = 0; pass < 3; ++pass) {
      ck(cudaEventRecord(e0, s), "rec");
      for (int i = 0; i < iters; ++i) body();
      ck(cudaEventRecord(e1, s), "rec");
      ck(cudaEventSynchronize(e1), "sync");
      float ms = 0;
      ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
      const double per = (double)ms / iters;
      if (per < best) best = per;
    }
    return best;
  };

  Timing t;
  t.scan_ms = time_it([&] {
    rocket::engine::indexer_pool_keys(pool_keys, kvp, ape, n_pools_d, n_pools, pool_stride,
                                      streams, layer_slot, kpool, ihd, s);
    rocket::engine::indexer_scores(pool_scores, q_idx, pool_keys, head_w, n_pools_d, n_pools,
                                   pool_stride, streams, ih, ihd, s);
  });
  t.gather_ms = time_it([&] {
    rocket::engine::mla_scores(scores, q_abs, kvp, sel, n_sel, topk, sel_stride, streams,
                               layer_slot, heads, kv_lora, 0.0625f, s);
    rocket::engine::mla_softmax(scores, n_sel, sel_stride, streams, heads, s);
    rocket::engine::mla_context(ctxbuf, scores, kvp, sel, n_sel, topk, sel_stride, streams,
                                layer_slot, heads, kv_lora, s);
  });

  cudaEventDestroy(e0); cudaEventDestroy(e1);
  for (int i = 0; i < n_owned; ++i) cudaFree(owned[i]);
  ck(cudaStreamSynchronize(s), "teardown");
  cudaStreamDestroy(s);
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  int ctx = 65536, streams = 8, iters = 20, repeat = 1;
  bool dense = false;
  int only_page = 0;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--streams") && i + 1 < argc) streams = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--sel") && i + 1 < argc) dense = !std::strcmp(argv[++i], "dense");
    else if (!std::strcmp(argv[i], "--page") && i + 1 < argc) only_page = std::atoi(argv[++i]);
  }
  const int candidates[] = {128, 256, 512, 1024, 2048};
  if (!only_page) {
    std::printf("ctx=%d streams=%d iters=%d repeat=%d sel=%s, layer 3 of 11, synthetic KV\n", ctx,
                streams, iters, repeat, dense ? "dense" : "scatter");
    std::printf("%10s %10s %8s %7s %12s %12s %14s\n", "page_tok", "page_KiB", "pages", "layout",
                "scan_ms", "gather_ms", "arena_GiB");
  }
  for (int r = 0; r < repeat; ++r) {
    for (int pt : candidates) {
      if (ctx % pt != 0) continue;
      if (only_page && pt != only_page) continue;
      for (int lay = 0; lay < 2; ++lay) {
        double gib = 0;
        const Timing t = run_one(pt, ctx, streams, iters, lay == 0, dense, &gib);
        kv::KvGeometry g{pt, 11, 512, 128, 4};
        std::printf("%10d %10.0f %8d %7s %12.3f %12.3f %14.2f\n", pt,
                    (double)g.page_bytes() / 1024.0, (ctx / pt) * streams,
                    lay == 0 ? "rr" : "seq", t.scan_ms, t.gather_ms, gib);
        std::fflush(stdout);
      }
    }
  }
  return 0;
}

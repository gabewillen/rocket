// Properties of the radix-tree paged KV cache (src/kv/page_pool.h,
// src/kv/kv_arena.h):
//
//   1. fork shares every full page below the fork point and refcounts it
//   2. deep fork chains keep refcounts and page tables consistent, and
//      copy on extend privatises only the shared boundary page
//   3. detach/resume is a roundtrip: pages survive, the KDA state leaves the
//      device and comes back byte for byte
//   4. the indexer scan and the sparse-MLA gather over a forked stream are
//      bit-for-bit equal to the same kernels over a materialised copy of the
//      same logical sequence in private, contiguous pages
//   5. a page is not evictable while any sequence references it
//   6. the page geometry reproduces the byte arithmetic in
//      fuels/glm-5.3-flash/attention.yaml
//   7. a multi-slot decode loop that only reuploads a block table when the
//      table changed still reads every logical position correctly, across a
//      fork and a detach/resume
//
// Synthetic KV throughout: latents, keys and gates are a deterministic
// function of (owner sequence, logical position, channel). No checkpoint is
// opened and no weights are loaded, so this runs in seconds.
//
// Property 4 is the one that matters. The forked stream's pages are whatever
// the pool handed out, in whatever order, including one page privatised
// mid-sequence by copy on extend; the reference stream's pages are private
// and contiguous. Both are filled from the same host-side model of what each
// logical token holds, never from each other, so agreement is evidence the
// block table is right rather than evidence the copy was faithful. Both slots
// are scored in the same kernel launch, so the reduction order is identical
// and a bit-for-bit comparison is the right one.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "kernels.h"
#include "kv/kv_arena.h"
#include "kv/page_pool.h"

namespace {

using rocket::engine::bf16;
namespace kv = rocket::engine::kv;

int failures = 0;

void check(const std::string& what, bool ok) {
  std::printf("  %-64s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}

// Fuel geometry, fuels/glm-5.3-flash/attention.yaml. page_tokens is the value
// bench/kv_page_gather.cu settled on.
constexpr int kPageTokens = 128;
constexpr int kLayers = 11;
constexpr int kKvLora = 512;
constexpr int kIhd = 128;
constexpr int kKpool = 4;
constexpr int kLayerSlot = 3;  // only this layer is populated; see below

kv::KvGeometry geometry() { return kv::KvGeometry{kPageTokens, kLayers, kKvLora, kIhd, kKpool}; }

// The host-side model of the cache contents: what logical position `pos` of
// the sequence owned by `owner` holds in channel `c` of `which` (0 latent,
// 1 key, 2 gate). Deterministic, so a forked stream and a materialised copy
// can be filled independently and still be expected to agree.
float model_value(int owner, int pos, int c, int which) {
  std::uint32_t h = 2166136261u;
  for (int v : {owner, pos, c, which}) {
    h ^= static_cast<std::uint32_t>(v);
    h *= 16777619u;
  }
  return (static_cast<float>(h & 0xffffu) / 32768.0f) - 1.0f;
}

// Writes one logical token's slice of `layer` into whatever physical page the
// table says. Only kLayerSlot is populated: the read kernels under test are
// launched for one layer, and filling the other ten would multiply the test's
// setup cost for nothing.
void write_token(kv::KvArena& arena, int page, int slot, int owner, int pos) {
  const rocket::engine::KvPages& kvp = arena.pages();
  std::vector<bf16> lat(kKvLora), key(kIhd), gate(kIhd);
  for (int c = 0; c < kKvLora; ++c) lat[c] = __float2bfloat16(model_value(owner, pos, c, 0));
  for (int c = 0; c < kIhd; ++c) {
    key[c] = __float2bfloat16(model_value(owner, pos, c, 1));
    gate[c] = __float2bfloat16(model_value(owner, pos, c, 2));
  }
  const std::size_t base =
      (static_cast<std::size_t>(page) * kLayers + kLayerSlot) * kPageTokens + slot;
  ck(cudaMemcpy(kvp.latent + base * kKvLora, lat.data(), lat.size() * sizeof(bf16),
                cudaMemcpyHostToDevice), "write latent");
  ck(cudaMemcpy(const_cast<bf16*>(kvp.key) + base * kIhd, key.data(), key.size() * sizeof(bf16),
                cudaMemcpyHostToDevice), "write key");
  ck(cudaMemcpy(const_cast<bf16*>(kvp.gate) + base * kIhd, gate.data(), gate.size() * sizeof(bf16),
                cudaMemcpyHostToDevice), "write gate");
}

// Appends `n` tokens to `seq`, writing each one's KV as owned by `owner`.
void grow(kv::KvCache& cache, kv::KvArena& arena, int seq, int owner, int n) {
  for (int i = 0; i < n; ++i) {
    const int pos = cache.info(seq).length;
    const kv::AppendSite site = cache.append_token(seq, owner * 100003 + pos);
    if (site.page < 0) {
      std::fprintf(stderr, "pool exhausted at pos %d\n", pos);
      std::exit(1);
    }
    write_token(arena, site.page, site.slot, owner, pos);
  }
}

// ------------------------------------------------------------------ 1, 2, 5

void test_pool_and_forks() {
  std::printf("fork sharing, deep chains, refcounts, eviction\n");
  const kv::KvGeometry g = geometry();
  cudaStream_t s = nullptr;
  ck(cudaStreamCreate(&s), "stream");
  kv::KvArena arena(g, /*num_pages=*/96, /*max_streams=*/8, /*max_pages_per_stream=*/24, s);
  kv::PagePool pool(96, kPageTokens);
  kv::PrefixTree tree;
  kv::KvCache cache(g, &pool, &tree, &arena);

  // Parent: 5 full pages plus 40 tokens of a sixth.
  const int parent = cache.open(0);
  grow(cache, arena, parent, /*owner=*/1, 5 * kPageTokens + 40);
  check("parent holds 6 pages for 5.31 pages of tokens",
        cache.info(parent).pages == 6 && cache.info(parent).length == 5 * kPageTokens + 40);
  check("pinned pages equal the parent's page count", pool.pinned_pages() == 6);

  // Fork on a page boundary: 4 full pages shared, no partial page.
  const int childA = cache.fork(parent, 4 * kPageTokens, 1);
  check("boundary fork shares exactly 4 pages", cache.info(childA).pages == 4);
  check("boundary fork allocates nothing new", pool.pinned_pages() == 6);
  for (int i = 0; i < 4; ++i)
    if (cache.page_table(childA)[i] != cache.page_table(parent)[i])
      check("boundary fork points at the parent's pages", false);
  check("shared pages carry refcount 2", pool.refcount(cache.page_table(parent)[0]) == 2 &&
                                             pool.refcount(cache.page_table(parent)[3]) == 2);
  check("the parent's 5th page is still private", pool.refcount(cache.page_table(parent)[4]) == 1);

  // Fork mid-page: the boundary page is shared and must be copied on extend.
  const int fork_pos = 5 * kPageTokens + 20;
  const int childB = cache.fork(parent, fork_pos, 2);
  const int boundary = cache.page_table(parent)[5];
  check("mid-page fork shares the partial boundary page too",
        cache.info(childB).pages == 6 && cache.page_table(childB)[5] == boundary);
  check("boundary page carries refcount 2", pool.refcount(boundary) == 2);

  check("a referenced page refuses eviction", !pool.evict(boundary));
  check("an unreferenced page is not yet in the pool's evictable set",
        pool.evictable_pages() == 0);

  const int pinned_before = pool.pinned_pages();
  grow(cache, arena, childB, /*owner=*/2, 1);
  check("copy on extend privatises the boundary page",
        cache.page_table(childB)[5] != boundary && pool.pinned_pages() == pinned_before + 1);
  check("the parent keeps the original boundary page at refcount 1",
        cache.page_table(parent)[5] == boundary && pool.refcount(boundary) == 1);

  // Deep chain: fork the fork, four levels down, each extending a little.
  int chain = childB;
  std::vector<int> chain_ids{childB};
  for (int d = 0; d < 4; ++d) {
    chain = cache.fork(chain, cache.info(chain).length, 3 + d);
    chain_ids.push_back(chain);
    grow(cache, arena, chain, /*owner=*/10 + d, 30);
  }
  bool chain_ok = true;
  for (std::size_t i = 1; i < chain_ids.size(); ++i) {
    const auto& lo = cache.page_table(chain_ids[i - 1]);
    const auto& hi = cache.page_table(chain_ids[i]);
    if (hi.size() < lo.size()) chain_ok = false;
    // Every page the descendant did not privatise must be its ancestor's.
    for (std::size_t p = 0; p + 1 < lo.size(); ++p)
      if (hi[p] != lo[p]) chain_ok = false;
  }
  check("a 4-deep fork chain shares every unwritten ancestor page", chain_ok);

  // Refcount bookkeeping: destroy the whole chain and the parent, and the
  // pool must come back to nothing pinned.
  for (int id : chain_ids) cache.destroy(id);
  cache.destroy(childA);
  cache.destroy(parent);
  check("destroying every sequence unpins every page", pool.pinned_pages() == 0);
  check("released pages stay cached rather than freed",
        pool.evictable_pages() > 0 && pool.free_pages() + pool.evictable_pages() == 96);

  const int victim = pool.lru_victim();
  check("an unreferenced page evicts", victim >= 0 && pool.evict(victim));

  cudaStreamDestroy(s);
}

// ---------------------------------------------------------------------- 3

void test_detach_resume() {
  std::printf("detach and resume\n");
  const kv::KvGeometry g = geometry();
  cudaStream_t s = nullptr;
  ck(cudaStreamCreate(&s), "stream");
  kv::KvArena arena(g, 32, 4, 16, s);
  kv::PagePool pool(32, kPageTokens);
  kv::PrefixTree tree;
  kv::KvCache cache(g, &pool, &tree, &arena);
  kv::HostKdaStateStore kda(s);

  const int seq = cache.open(0);
  grow(cache, arena, seq, 7, 3 * kPageTokens);
  const std::vector<int> before = cache.page_table(seq);
  const int pinned = pool.pinned_pages();

  // The KDA slice for one stream, at this fuel's real size.
  const std::size_t kda_bytes = 76316672;  // attention.yaml bytes.kda_state_per_stream.total
  void* dev = nullptr;
  ck(cudaMalloc(&dev, kda_bytes), "kda alloc");
  std::vector<std::uint8_t> pattern(kda_bytes);
  for (std::size_t i = 0; i < kda_bytes; ++i) pattern[i] = static_cast<std::uint8_t>(i * 31 + 7);
  ck(cudaMemcpy(dev, pattern.data(), kda_bytes, cudaMemcpyHostToDevice), "kda seed");

  kda.save(/*session=*/seq, dev, kda_bytes);
  cache.detach(seq);
  ck(cudaMemset(dev, 0, kda_bytes), "kda clobber");

  check("detach frees the stream slot", cache.slot_of(seq) == -1 && cache.seq_in_slot(0) == -1);
  check("detach keeps every page pinned", pool.pinned_pages() == pinned);
  check("the KDA state left the device", kda.holds(seq) &&
                                             kda.resident_bytes() == kda_bytes);

  cache.attach(seq, 2);
  kda.load(seq, dev, kda_bytes);
  std::vector<std::uint8_t> back(kda_bytes);
  ck(cudaMemcpy(back.data(), dev, kda_bytes, cudaMemcpyDeviceToHost), "kda readback");
  check("resume restores the KDA state byte for byte",
        std::memcmp(back.data(), pattern.data(), kda_bytes) == 0);
  check("resume restores the page table unchanged", cache.page_table(seq) == before);
  check("resume takes the new slot", cache.slot_of(seq) == 2 && cache.seq_in_slot(2) == seq);

  cudaFree(dev);
  cudaStreamDestroy(s);
}

// ---------------------------------------------------------------------- 4

void test_forked_read_equals_materialised() {
  std::printf("indexer scan and sparse-MLA gather over a forked stream\n");
  const kv::KvGeometry g = geometry();
  cudaStream_t s = nullptr;
  ck(cudaStreamCreate(&s), "stream");
  const int max_pages_per_stream = 16;
  kv::KvArena arena(g, /*num_pages=*/64, /*max_streams=*/4, max_pages_per_stream, s);
  kv::PagePool pool(64, kPageTokens);
  kv::PrefixTree tree;
  kv::KvCache cache(g, &pool, &tree, &arena);

  // Parent runs to 900, the child forks mid-page at 700 and runs to 1024.
  const int kParentOwner = 1, kChildOwner = 2;
  const int fork_pos = 700, child_len = 1024;
  const int parent = cache.open(3);
  grow(cache, arena, parent, kParentOwner, 900);
  const int child = cache.fork(parent, fork_pos, 0);
  grow(cache, arena, child, kChildOwner, child_len - fork_pos);
  check("child length is the fork point plus its own tokens",
        cache.info(child).length == child_len);

  // The reference: a private sequence of the same length, whose pages come
  // out of the same pool but are filled straight from the host model of the
  // child's logical contents. Nothing is copied from the child's pages.
  const int ref = cache.open(1);
  for (int pos = 0; pos < child_len; ++pos) {
    const kv::AppendSite site = cache.append_token(ref, pos);
    if (site.page < 0) { std::fprintf(stderr, "pool exhausted\n"); std::exit(1); }
    write_token(arena, site.page, site.slot,
                pos < fork_pos ? kParentOwner : kChildOwner, pos);
  }
  bool disjoint = true;
  for (int p : cache.page_table(ref))
    for (int q : cache.page_table(child))
      if (p == q) disjoint = false;
  check("the reference shares no physical page with the forked child", disjoint);

  arena.upload_table(0, cache.page_table(child));
  arena.upload_table(1, cache.page_table(ref));

  // Both slots are scored in one launch, so any difference is the block
  // table and not the reduction order.
  const int batch = 2, heads = 64, ih = 32;
  const int n_pools = child_len / kKpool;
  const int pool_stride = n_pools + 1;
  const int n_gather = 256;
  const int sel_stride = n_gather + kKpool - 1;

  std::vector<void*> owned;
  auto A = [&](std::size_t bytes) {
    void* p = nullptr;
    ck(cudaMalloc(&p, bytes), "alloc");
    ck(cudaMemset(p, 0, bytes), "memset");
    owned.push_back(p);
    return p;
  };
  auto* pool_keys = (bf16*)A((std::size_t)batch * pool_stride * kIhd * sizeof(bf16));
  auto* pool_scores = (float*)A((std::size_t)batch * pool_stride * sizeof(float));
  auto* q_idx = (bf16*)A((std::size_t)batch * ih * kIhd * sizeof(bf16));
  auto* head_w = (float*)A((std::size_t)batch * ih * sizeof(float));
  auto* ape = (bf16*)A((std::size_t)kKpool * kIhd * sizeof(bf16));
  auto* q_abs = (float*)A((std::size_t)batch * heads * kKvLora * sizeof(float));
  auto* scores = (float*)A((std::size_t)batch * heads * sel_stride * sizeof(float));
  auto* ctx = (float*)A((std::size_t)batch * heads * kKvLora * sizeof(float));
  auto* sel = (int*)A((std::size_t)batch * sel_stride * sizeof(int));
  auto* n_sel = (int*)A((std::size_t)batch * sizeof(int));
  auto* n_pools_d = (int*)A((std::size_t)batch * sizeof(int));

  // Identical queries in both slots, so identical inputs meet different page
  // tables over the same logical sequence.
  {
    std::vector<bf16> hq((std::size_t)batch * ih * kIhd);
    for (int m = 0; m < batch; ++m)
      for (int i = 0; i < ih * kIhd; ++i)
        hq[(std::size_t)m * ih * kIhd + i] = __float2bfloat16(model_value(99, i, 0, 3));
    std::vector<float> hw((std::size_t)batch * ih);
    for (int m = 0; m < batch; ++m)
      for (int i = 0; i < ih; ++i) hw[(std::size_t)m * ih + i] = model_value(98, i, 0, 4);
    std::vector<bf16> hape((std::size_t)kKpool * kIhd);
    for (int i = 0; i < kKpool * kIhd; ++i) hape[i] = __float2bfloat16(model_value(97, i, 0, 5));
    std::vector<float> hqa((std::size_t)batch * heads * kKvLora);
    for (int m = 0; m < batch; ++m)
      for (int i = 0; i < heads * kKvLora; ++i)
        hqa[(std::size_t)m * heads * kKvLora + i] = model_value(96, i, 0, 6);
    // 64 pools spread over the context, expanded to their 4 member tokens.
    std::vector<int> hsel((std::size_t)batch * sel_stride, 0);
    for (int m = 0; m < batch; ++m)
      for (int i = 0; i < n_gather / kKpool; ++i) {
        const int p = (i * 7919) % n_pools;
        for (int k = 0; k < kKpool; ++k)
          hsel[(std::size_t)m * sel_stride + i * kKpool + k] = p * kKpool + k;
      }
    std::vector<int> hn(batch, n_gather), hnp(batch, n_pools);
    ck(cudaMemcpy(q_idx, hq.data(), hq.size() * sizeof(bf16), cudaMemcpyHostToDevice), "q");
    ck(cudaMemcpy(head_w, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice), "w");
    ck(cudaMemcpy(ape, hape.data(), hape.size() * sizeof(bf16), cudaMemcpyHostToDevice), "ape");
    ck(cudaMemcpy(q_abs, hqa.data(), hqa.size() * sizeof(float), cudaMemcpyHostToDevice), "qa");
    ck(cudaMemcpy(sel, hsel.data(), hsel.size() * sizeof(int), cudaMemcpyHostToDevice), "sel");
    ck(cudaMemcpy(n_sel, hn.data(), hn.size() * sizeof(int), cudaMemcpyHostToDevice), "n");
    ck(cudaMemcpy(n_pools_d, hnp.data(), hnp.size() * sizeof(int), cudaMemcpyHostToDevice), "np");
  }

  const rocket::engine::KvPages& kvp = arena.pages();
  rocket::engine::indexer_pool_keys(pool_keys, kvp, ape, n_pools_d, n_pools, pool_stride, batch,
                                    kLayerSlot, kKpool, kIhd, s);
  rocket::engine::indexer_scores(pool_scores, q_idx, pool_keys, head_w, n_pools_d, n_pools,
                                 pool_stride, batch, ih, kIhd, s);
  rocket::engine::mla_scores(scores, q_abs, kvp, sel, n_sel, n_gather, sel_stride, batch,
                             kLayerSlot, heads, kKvLora, 0.0625f, s);
  rocket::engine::mla_softmax(scores, n_sel, sel_stride, batch, heads, s);
  rocket::engine::mla_context(ctx, scores, kvp, sel, n_sel, n_gather, sel_stride, batch,
                              kLayerSlot, heads, kKvLora, s);
  ck(cudaStreamSynchronize(s), "kernels");

  auto slots_equal = [&](const char* what, const void* base, std::size_t elem, std::size_t per_slot) {
    std::vector<std::uint8_t> h(elem * per_slot * 2);
    ck(cudaMemcpy(h.data(), base, h.size(), cudaMemcpyDeviceToHost), "readback");
    const bool ok = std::memcmp(h.data(), h.data() + elem * per_slot, elem * per_slot) == 0;
    check(what, ok);
  };
  slots_equal("pooled indexer keys match a materialised copy bit for bit", pool_keys,
              sizeof(bf16), (std::size_t)pool_stride * kIhd);
  slots_equal("indexer pool scores match a materialised copy bit for bit", pool_scores,
              sizeof(float), pool_stride);
  slots_equal("MLA scores match a materialised copy bit for bit", scores, sizeof(float),
              (std::size_t)heads * sel_stride);
  slots_equal("MLA context matches a materialised copy bit for bit", ctx, sizeof(float),
              (std::size_t)heads * kKvLora);

  // Negative control: if the child's table were wrong the comparison above
  // would have to fail. Point slot 0 at the parent's pages instead, which is
  // the same prefix but the wrong tail, and confirm the gather notices.
  arena.upload_table(0, cache.page_table(parent));
  rocket::engine::mla_scores(scores, q_abs, kvp, sel, n_sel, n_gather, sel_stride, batch,
                             kLayerSlot, heads, kKvLora, 0.0625f, s);
  ck(cudaStreamSynchronize(s), "control");
  {
    std::vector<float> h((std::size_t)batch * heads * sel_stride);
    ck(cudaMemcpy(h.data(), scores, h.size() * sizeof(float), cudaMemcpyDeviceToHost), "control rb");
    const std::size_t per = (std::size_t)heads * sel_stride;
    check("a wrong page table does change the gather (negative control)",
          std::memcmp(h.data(), h.data() + per, per * sizeof(float)) != 0);
  }

  for (void* p : owned) cudaFree(p);
  cudaStreamDestroy(s);
}

// ---------------------------------------------------------------------- 6

// Pulls a scalar out of attention.yaml so an edit there breaks this test
// rather than silently disagreeing with the pool's arithmetic.
long long yaml_scalar(const std::string& path, const std::string& key) {
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t at = line.find(key + ":");
    if (at == std::string::npos) continue;
    if (line.find_first_not_of(" \t") != at) continue;  // must be the field, not a mention
    const std::size_t v = line.find_first_of("0123456789", at + key.size() + 1);
    if (v == std::string::npos) continue;
    return std::atoll(line.c_str() + v);
  }
  return -1;
}

void test_capacity_math() {
  std::printf("page geometry against fuels/glm-5.3-flash/attention.yaml\n");
  const std::string yaml = std::string(ROCKET_REPO_ROOT) + "/fuels/glm-5.3-flash/attention.yaml";
  const long long mla_bpt = yaml_scalar(yaml, "mla_kv_per_token_per_layer");
  const long long idx_bpt = yaml_scalar(yaml, "indexer_key_per_token_per_layer");
  const long long mla_count = yaml_scalar(yaml, "mla_count");
  const long long ctx = yaml_scalar(yaml, "context_tokens");
  check("attention.yaml still carries the four fields this pool is sized from",
        mla_bpt > 0 && idx_bpt > 0 && mla_count > 0 && ctx > 0);

  const kv::KvGeometry g = geometry();
  check("geometry layer count matches the fuel's MLA layer count", g.layers == mla_count);
  check("bytes per token match (mla + indexer) * layers",
        (long long)g.bytes_per_token() == (mla_bpt + idx_bpt) * mla_count);

  const long long per_stream = (long long)g.bytes_per_token() * ctx;
  check("a full-context stream costs the yaml's mla_latent + indexer_keys total",
        per_stream == 2952790016LL + 1476395008LL);
  check("a full context is a whole number of pages",
        ctx % g.page_tokens == 0 && ctx / g.page_tokens == 2048);
  check("a page slab is a whole number of 64 KiB platform pages",
        g.page_bytes() % 65536 == 0 && g.page_bytes() == 2162688);
  check("a page holds a whole number of indexer pools", g.page_tokens % g.kpool == 0);

  // The geometry guard has to reject what the kernels cannot index.
  kv::KvGeometry bad = g;
  bad.page_tokens = 130;  // multiple of 2 but not of kpool
  check("a page size that splits an indexer pool is rejected", !bad.valid());
  bad.page_tokens = 64;  // multiple of kpool but not 64 KiB aligned
  check("a page size that leaves the slab 64 KiB unaligned is rejected", !bad.valid());
}

// ---------------------------------------------------------------------- 7

// DecodeEngine::kv_advance (model.cu) appends one token per active slot per
// step and reuploads a slot's block table only when append_token reports the
// table grew or a page was privatised. That conditional upload is the part
// most likely to be wrong, because a missed upload leaves the device reading
// a stale row and the symptom is silent. This mirrors the rule (it cannot
// call it: DecodeEngine's constructor opens a checkpoint) and then checks the
// result the same way property 4 does.
void test_decode_loop_table_discipline() {
  std::printf("decode loop, conditional table uploads, fork and detach/resume\n");
  const kv::KvGeometry g = geometry();
  cudaStream_t s = nullptr;
  ck(cudaStreamCreate(&s), "stream");
  kv::KvArena arena(g, 64, 4, 16, s);
  kv::PagePool pool(64, kPageTokens);
  kv::PrefixTree tree;
  kv::KvCache cache(g, &pool, &tree, &arena);
  kv::HostKdaStateStore kda(s);

  int seq_of_slot[3] = {cache.open(0), cache.open(1), cache.open(2)};
  int owner_of_slot[3] = {11, 12, 13};
  long long uploads = 0;

  // The rule under test, copied from model.cu::kv_advance.
  auto advance = [&](int slot) {
    const int seq = seq_of_slot[slot];
    const int pos = cache.info(seq).length;
    const kv::AppendSite site = cache.append_token(seq, owner_of_slot[slot] * 100003 + pos);
    if (site.page < 0) { std::fprintf(stderr, "pool exhausted\n"); std::exit(1); }
    write_token(arena, site.page, site.slot, owner_of_slot[slot], pos);
    if (site.grew_table || site.copied_on_extend) {
      arena.upload_table(slot, cache.page_table(seq));
      ++uploads;
    }
  };

  const int steps_before_fork = 300, steps_after = 260;
  for (int t = 0; t < steps_before_fork; ++t)
    for (int slot = 0; slot < 3; ++slot) advance(slot);

  // Slot 1 forks from slot 0 mid-page and diverges from there.
  const int fork_pos = cache.info(seq_of_slot[0]).length;
  cache.destroy(seq_of_slot[1]);
  seq_of_slot[1] = cache.fork(seq_of_slot[0], fork_pos, 1);
  arena.upload_table(1, cache.page_table(seq_of_slot[1]));
  owner_of_slot[1] = 21;

  // Slot 2 detaches, its KDA leaves the device, and it comes back later.
  const std::size_t kda_bytes = 1 << 20;  // shape only; the real size is tested above
  void* dev = nullptr;
  ck(cudaMalloc(&dev, kda_bytes), "kda alloc");
  ck(cudaMemset(dev, 0xab, kda_bytes), "kda seed");
  kda.save(seq_of_slot[2], dev, kda_bytes);
  cache.detach(seq_of_slot[2]);
  const int detached = seq_of_slot[2];

  for (int t = 0; t < steps_after; ++t) {
    advance(0);
    advance(1);
  }

  cache.attach(detached, 2);
  kda.load(detached, dev, kda_bytes);
  arena.upload_table(2, cache.page_table(detached));
  for (int t = 0; t < 40; ++t) advance(2);

  check("the table was reuploaded far less often than once per step",
        uploads > 0 && uploads < (steps_before_fork + steps_after) * 3 / 8);

  // Slot 1's logical contents: it forked off slot 0, so everything below the
  // fork point is slot 0's owner (11), and everything above is its own (21).
  const int child_len = (int)cache.info(seq_of_slot[1]).length;
  const int ref = cache.open(3);
  for (int pos = 0; pos < child_len; ++pos) {
    const kv::AppendSite site = cache.append_token(ref, pos);
    if (site.page < 0) { std::fprintf(stderr, "pool exhausted\n"); std::exit(1); }
    write_token(arena, site.page, site.slot, pos < fork_pos ? 11 : 21, pos);
  }
  arena.upload_table(3, cache.page_table(ref));

  // Score slot 1 and the reference in one launch over the same logical ids.
  const int heads = 64;
  const int n_gather = 256;
  const int sel_stride = n_gather + kKpool - 1;
  std::vector<void*> owned;
  auto A = [&](std::size_t bytes) {
    void* p = nullptr;
    ck(cudaMalloc(&p, bytes), "alloc");
    ck(cudaMemset(p, 0, bytes), "memset");
    owned.push_back(p);
    return p;
  };
  const int batch = 4;  // slots 0..3; only 1 and 3 are compared
  auto* q_abs = (float*)A((std::size_t)batch * heads * kKvLora * sizeof(float));
  auto* scores = (float*)A((std::size_t)batch * heads * sel_stride * sizeof(float));
  auto* sel = (int*)A((std::size_t)batch * sel_stride * sizeof(int));
  auto* n_sel = (int*)A((std::size_t)batch * sizeof(int));
  {
    std::vector<float> hqa((std::size_t)batch * heads * kKvLora);
    for (int m = 0; m < batch; ++m)
      for (int i = 0; i < heads * kKvLora; ++i)
        hqa[(std::size_t)m * heads * kKvLora + i] = model_value(96, i, 0, 6);
    std::vector<int> hsel((std::size_t)batch * sel_stride, 0);
    const int n_pools = child_len / kKpool;
    for (int m = 0; m < batch; ++m)
      for (int i = 0; i < n_gather / kKpool; ++i) {
        const int p = (i * 7919) % n_pools;
        for (int k = 0; k < kKpool; ++k)
          hsel[(std::size_t)m * sel_stride + i * kKpool + k] = p * kKpool + k;
      }
    std::vector<int> hn(batch, n_gather);
    ck(cudaMemcpy(q_abs, hqa.data(), hqa.size() * sizeof(float), cudaMemcpyHostToDevice), "qa");
    ck(cudaMemcpy(sel, hsel.data(), hsel.size() * sizeof(int), cudaMemcpyHostToDevice), "sel");
    ck(cudaMemcpy(n_sel, hn.data(), hn.size() * sizeof(int), cudaMemcpyHostToDevice), "n");
  }
  rocket::engine::mla_scores(scores, q_abs, arena.pages(), sel, n_sel, n_gather, sel_stride, batch,
                             kLayerSlot, heads, kKvLora, 0.0625f, s);
  ck(cudaStreamSynchronize(s), "kernels");
  {
    const std::size_t per = (std::size_t)heads * sel_stride;
    std::vector<float> h((std::size_t)batch * per);
    ck(cudaMemcpy(h.data(), scores, h.size() * sizeof(float), cudaMemcpyDeviceToHost), "rb");
    check("a forked slot driven by the decode loop matches a materialised copy",
          std::memcmp(h.data() + per, h.data() + 3 * per, per * sizeof(float)) == 0);
    check("an unrelated slot does not match it (negative control)",
          std::memcmp(h.data(), h.data() + 3 * per, per * sizeof(float)) != 0);
  }
  check("the resumed slot kept its length across detach and resume",
        cache.info(detached).length == steps_before_fork + 40);

  for (void* p : owned) cudaFree(p);
  cudaFree(dev);
  cudaStreamDestroy(s);
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("no CUDA device, skipping\n");
    return 77;
  }
  test_pool_and_forks();
  test_detach_resume();
  test_forked_read_equals_materialised();
  test_capacity_math();
  test_decode_loop_table_discipline();
  std::printf("%s\n", failures ? "FAILED" : "all ok");
  return failures ? 1 : 0;
}

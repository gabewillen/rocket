// Radix-pool fork against the real checkpoint (src/kv/page_pool.h,
// src/kv/kv_arena.h, wired into DecodeEngine as kv_fork/kv_detach/kv_resume
// in src/model.cu). tests/test_kv_radix.cu already proves the pool's
// bookkeeping and its kernels against synthetic KV; this test proves the
// same fork against the real model's own activations and its own step()
// loop, where a real-weight-only bug (grouped-GEMM row grouping, a real
// KDA-state copy) could still hide.
//
// Shape: decode a prompt to 30 total tokens (prefill + greedy), fork the
// stream into two children, decode each 20 more tokens with a forced
// divergent first token so the two continuations actually differ, then
// check each child's tail token-for-token against an unforked reference run
// of that exact full sequence (fed as forced tokens, no forking, no shared
// pages). Refcounts are checked at the fork (pinned pages must not grow,
// since nothing has extended into a shared boundary page yet) and after the
// children are destroyed (pinned pages must fall back toward the parent's
// own footprint).
//
// Returns 77 when the checkpoint is not on this node, matching the other
// real-weight tests.
#include <cstdio>
#include <string>
#include <vector>

#include "model.h"
#include "model_config.h"
#include "tokenizer.h"

namespace {

int failures = 0;
void check(const std::string& what, bool ok) {
  std::printf("  %-64s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

const int kSharedTokens = 30;
const int kForkTokens = 20;

}  // namespace

int main() {
  const std::filesystem::path snapshot = rocket::fuel::default_nvfp4_snapshot_dir();
  const std::filesystem::path attn = rocket::fuel::default_attention_yaml();
  if (snapshot.empty() || !std::filesystem::exists(snapshot / "config.json") || attn.empty() ||
      !std::filesystem::exists(attn)) {
    std::printf("no checkpoint or attention.yaml on this node\n");
    return 77;
  }

  const rocket::fuel::ModelConfig cfg = rocket::fuel::load_model_config(attn, snapshot);
  const rocket::fuel::Tokenizer tok(snapshot / "tokenizer.json");
  const std::string prompt = "The quick brown fox jumps over the lazy";
  const std::vector<int> prompt_ids = tok.encode(prompt);

  // Small pool (16 pages of 128 tokens) shared by 3 slots (parent, child A,
  // child B): 2048 tokens/stream if unshared, plenty for a 70-token
  // sequence, but small enough that a fork actually has to share pages
  // rather than each slot getting its own private full-size arena the way
  // the auto-sized default would.
  const int max_tokens = 2048;
  const int page_tokens = 128;
  const int max_batch = 3;
  const int pool_pages = 16;
  std::printf("loading engine (pool_pages=%d, page_tokens=%d)...\n", pool_pages, page_tokens);
  rocket::engine::DecodeEngine engine(cfg, snapshot, static_cast<std::size_t>(20) * (1ull << 30),
                                      max_tokens, max_batch, pool_pages, page_tokens);
  check("FP32 KDA detach image is exactly 147619840 bytes",
        engine.kda_bytes_per_stream() == 147619840ull);

  const int kParent = 0, kChildA = 1, kChildB = 2;

  // --- shared prefix on the parent slot -------------------------------
  std::vector<int> next_batch;
  int next = 0;
  for (const int id : prompt_ids) {
    engine.step(std::vector<int>{id}, next_batch, false);
    next = next_batch[0];
  }
  // kSharedTokens step() calls, each appending exactly one token to the
  // parent's KV sequence (matches test_batch_parity.cu's ref-loop pattern):
  // shared[t] is fed as input at step t and is what fork_pos below counts.
  std::vector<int> shared;
  for (int t = 0; t < kSharedTokens; ++t) {
    shared.push_back(next);
    engine.step(std::vector<int>{next}, next_batch, false);
    next = next_batch[0];
  }
  std::printf("  shared prefix: %zu prompt tokens + %d generated\n", prompt_ids.size(),
              kSharedTokens);

  const int pinned_before_fork = engine.kv_pinned_pages();

  // --- fork into two children at the shared prefix's end --------------
  const int fork_pos = static_cast<int>(prompt_ids.size()) + kSharedTokens;
  engine.kv_fork(kParent, fork_pos, kChildA);
  engine.kv_fork(kParent, fork_pos, kChildB);
  const int pinned_after_fork = engine.kv_pinned_pages();
  check("fork does not grow pinned pages (pages shared, not copied)",
        pinned_after_fork == pinned_before_fork);

  // --- diverge: force each child's first post-fork token to the parent's
  // top-2 next-token candidates at the fork point (both legal continuations
  // of the shared prefix, and different from each other by construction),
  // then continue greedy. -----------------------------------------------
  const std::vector<float> logits = engine.last_logits(kParent);
  int best = 0, second = 0;
  float best_v = -1e30f, second_v = -1e30f;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (logits[i] > best_v) {
      second = best; second_v = best_v; best = static_cast<int>(i); best_v = logits[i];
    } else if (logits[i] > second_v) {
      second = static_cast<int>(i); second_v = logits[i];
    }
  }
  const int forced_a = best, forced_b = second;
  check("forced continuations actually differ", forced_a != forced_b);

  // step() batches every active slot in one call; drive A and B together
  // and feed the parent slot a don't-care token it never reads back. Same
  // push-then-step order as test_batch_parity.cu's reference loop: seq_a[t]
  // is the t-th token fed to child A after the fork, seq_a[0] the forced
  // one, kForkTokens entries total.
  std::vector<int> seq_a, seq_b;
  std::vector<int> toks(max_batch, 0);
  int na = forced_a, nb = forced_b;
  for (int t = 0; t < kForkTokens; ++t) {
    seq_a.push_back(na);
    seq_b.push_back(nb);
    toks[kChildA] = na;
    toks[kChildB] = nb;
    engine.step(toks, next_batch, false);
    na = next_batch[kChildA];
    nb = next_batch[kChildB];
  }

  const int session_a = engine.kv_session_in_slot(kChildA);
  const int session_b = engine.kv_session_in_slot(kChildB);
  const int pinned_after_generate = engine.kv_pinned_pages();
  engine.kv_destroy(session_a);
  engine.kv_destroy(session_b);
  const int pinned_after_destroy = engine.kv_pinned_pages();
  check("destroying both children drops pinned pages back down",
        pinned_after_destroy < pinned_after_generate && pinned_after_destroy >= pinned_before_fork);

  // --- unforked references: the exact same two full sequences, fed as
  // forced tokens on a fresh slot with no forking and no shared pages. The
  // forced first post-fork token (seq_a[0]/seq_b[0]) is baked into the fed
  // history -- an unforced greedy run from the shared prefix alone would
  // reproduce forced_a (the parent's own top-1) but never forced_b (its
  // second-best), so the reference has to be told it explicitly, same as
  // the forked child was. Only the remaining kForkTokens-1 tokens are the
  // reference's own greedy continuation. -------------------------------
  std::vector<int> hist_a = prompt_ids, hist_b = prompt_ids;
  for (int t : shared) { hist_a.push_back(t); hist_b.push_back(t); }
  hist_a.push_back(seq_a[0]);
  hist_b.push_back(seq_b[0]);

  auto run_reference_tail = [&](const std::vector<int>& history) {
    engine.reset();
    std::vector<int> nb;
    int n = 0;
    for (const int id : history) {
      engine.step(std::vector<int>{id}, nb, false);
      n = nb[0];
    }
    std::vector<int> tail;
    for (int t = 1; t < kForkTokens; ++t) {
      tail.push_back(n);
      engine.step(std::vector<int>{n}, nb, false);
      n = nb[0];
    }
    return tail;
  };

  std::vector<int> ref_a = {seq_a[0]};
  for (int t : run_reference_tail(hist_a)) ref_a.push_back(t);
  std::vector<int> ref_b = {seq_b[0]};
  for (int t : run_reference_tail(hist_b)) ref_b.push_back(t);

  bool ok_a = ref_a.size() == seq_a.size();
  for (std::size_t t = 0; ok_a && t < seq_a.size(); ++t) ok_a = ok_a && ref_a[t] == seq_a[t];
  check("forked child A matches its unforked reference, token for token", ok_a);
  if (!ok_a)
    for (std::size_t t = 0; t < seq_a.size(); ++t)
      if (t >= ref_a.size() || ref_a[t] != seq_a[t])
        std::printf("    token %zu: forked=%d reference=%d\n", t, seq_a[t],
                    t < ref_a.size() ? ref_a[t] : -1);

  bool ok_b = ref_b.size() == seq_b.size();
  for (std::size_t t = 0; ok_b && t < seq_b.size(); ++t) ok_b = ok_b && ref_b[t] == seq_b[t];
  check("forked child B matches its unforked reference, token for token", ok_b);
  if (!ok_b)
    for (std::size_t t = 0; t < seq_b.size(); ++t)
      if (t >= ref_b.size() || ref_b[t] != seq_b[t])
        std::printf("    token %zu: forked=%d reference=%d\n", t, seq_b[t],
                    t < ref_b.size() ? ref_b[t] : -1);

  std::printf("\n%s: %d check(s) failed\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}

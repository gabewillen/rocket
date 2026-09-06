// Batch parity: M=8 concurrent streams, 8 different prompts, must decode
// token-for-token identically to running each prompt alone at M=1 on the
// same engine instance (same resident weights, same expert cache). A second
// check repeats the same 8 prompts 4x to fill M=32 concurrent streams (the
// engine is built with max_batch=32; see blog/posts/runtime/2026-09-07-
// grouped-gemm-beats-gemv-at-every-m/'s Next: "batch-parity must stay green
// at the larger max_batch"), reusing the same 8 M=1 references rather than
// recomputing them 4 times, since a duplicate prompt on an independent
// stream slot is a parity check of KV/state isolation, not a new reference.
// Returns 77 when the checkpoint is not on this node, matching the other
// real-weight tests.
#include <cstdio>
#include <string>
#include <vector>

#include "model.h"
#include "model_config.h"
#include "tokenizer.h"

namespace {

const int kNewTokens = 20;

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

  const std::vector<std::string> prompts = {
      "The capital of France is",
      "def add(a, b):\n    return",
      "Water boils at a temperature of",
      "The opposite of hot is",
      "In 1969, humans first landed on the",
      "Two plus two equals",
      "The largest planet in the solar system is",
      "Roses are red, violets are",
  };
  const int M = static_cast<int>(prompts.size());  // 8 unique prompts
  const int kWideBatch = 32;                        // 4x duplication of the 8 above

  std::printf("loading engine (max_batch=%d)...\n", kWideBatch);
  const int max_tokens = 512;
  rocket::engine::DecodeEngine engine(cfg, snapshot, static_cast<std::size_t>(20) * (1ull << 30),
                                      max_tokens, kWideBatch);

  std::vector<std::vector<int>> prompt_ids(M);
  for (int i = 0; i < M; ++i) prompt_ids[i] = tok.encode(prompts[i]);

  // --- M=1 reference: each prompt run alone, sequentially -------------------
  std::vector<std::vector<int>> ref(M);
  for (int i = 0; i < M; ++i) {
    engine.reset();
    std::vector<int> next_batch;
    int next = 0;
    for (const int id : prompt_ids[i]) {
      engine.step(std::vector<int>{id}, next_batch, false);
      next = next_batch[0];
    }
    for (int t = 0; t < kNewTokens; ++t) {
      ref[i].push_back(next);
      engine.step(std::vector<int>{next}, next_batch, false);
      next = next_batch[0];
    }
    std::string text = tok.decode(ref[i]);
    for (char& c : text) if (c == '\n') c = ' ';
    std::printf("  M=1 stream %d: %s\n", i, text.c_str());
  }

  // --- M=8 concurrent: every stream steps together, staggered prefill -------
  engine.reset();
  const std::size_t max_prompt_len = [&] {
    std::size_t m = 0;
    for (const auto& p : prompt_ids) m = std::max(m, p.size());
    return m;
  }();
  std::vector<int> last(M, 0);
  std::vector<std::vector<int>> got(M);
  std::vector<int> next_batch;
  const std::size_t total_steps = max_prompt_len - 1 + kNewTokens;
  for (std::size_t t = 0; t < total_steps; ++t) {
    std::vector<int> tokens(M);
    for (int i = 0; i < M; ++i) {
      tokens[i] = (t < prompt_ids[i].size()) ? prompt_ids[i][t] : last[i];
    }
    engine.step(tokens, next_batch, false);
    for (int i = 0; i < M; ++i) {
      last[i] = next_batch[i];
      // Once past this stream's own prompt, every step generates one of its
      // tokens (the same offset as the M=1 reference above).
      if (t + 1 >= prompt_ids[i].size() && got[i].size() < static_cast<std::size_t>(kNewTokens))
        got[i].push_back(next_batch[i]);
    }
  }

  int failures = 0;
  for (int i = 0; i < M; ++i) {
    bool ok = got[i].size() == ref[i].size();
    for (std::size_t t = 0; ok && t < got[i].size(); ++t) ok = ok && got[i][t] == ref[i][t];
    std::printf("  M=8 stream %d vs M=1: %s\n", i, ok ? "identical" : "MISMATCH");
    if (!ok) {
      ++failures;
      for (std::size_t t = 0; t < got[i].size(); ++t) {
        const int a = ref[i][t];
        const int b = t < got[i].size() ? got[i][t] : -1;
        if (a != b) std::printf("    token %zu: M=1=%d M=8=%d\n", t, a, b);
      }
    }
  }

  std::printf("\n%s: %d/%d streams mismatched\n", failures == 0 ? "PASS" : "FAIL", failures, M);

  // --- kWideBatch=32 concurrent: each of the 8 prompts run on 4 independent
  // stream slots, every slot compared against its own prompt's M=1 reference
  // above (ref[i % M], not a new M=1 run) -------------------------------
  engine.reset();
  std::vector<std::vector<int>> prompt_ids_wide(kWideBatch);
  for (int i = 0; i < kWideBatch; ++i) prompt_ids_wide[i] = prompt_ids[i % M];
  const std::size_t max_prompt_len_wide = [&] {
    std::size_t m = 0;
    for (const auto& p : prompt_ids_wide) m = std::max(m, p.size());
    return m;
  }();
  std::vector<int> last_wide(kWideBatch, 0);
  std::vector<std::vector<int>> got_wide(kWideBatch);
  const std::size_t total_steps_wide = max_prompt_len_wide - 1 + kNewTokens;
  for (std::size_t t = 0; t < total_steps_wide; ++t) {
    std::vector<int> tokens(kWideBatch);
    for (int i = 0; i < kWideBatch; ++i)
      tokens[i] = (t < prompt_ids_wide[i].size()) ? prompt_ids_wide[i][t] : last_wide[i];
    engine.step(tokens, next_batch, false);
    for (int i = 0; i < kWideBatch; ++i) {
      last_wide[i] = next_batch[i];
      if (t + 1 >= prompt_ids_wide[i].size() &&
          got_wide[i].size() < static_cast<std::size_t>(kNewTokens))
        got_wide[i].push_back(next_batch[i]);
    }
  }
  int failures_wide = 0;
  for (int i = 0; i < kWideBatch; ++i) {
    const std::vector<int>& r = ref[i % M];
    bool ok = got_wide[i].size() == r.size();
    for (std::size_t t = 0; ok && t < got_wide[i].size(); ++t) ok = ok && got_wide[i][t] == r[t];
    if (!ok) ++failures_wide;
  }
  std::printf("\nM=%d stream vs M=1 (prompt i %% %d): %d/%d streams mismatched\n", kWideBatch, M,
             failures_wide, kWideBatch);

  const int total_failures = failures + failures_wide;
  std::printf("\n%s: %d failure(s) (%d at M=%d, %d at M=%d)\n",
             total_failures == 0 ? "PASS" : "FAIL", total_failures, failures, M, failures_wide,
             kWideBatch);
  return total_failures == 0 ? 0 : 1;
}

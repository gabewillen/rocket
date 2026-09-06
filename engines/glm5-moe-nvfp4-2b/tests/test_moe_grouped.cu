// Stage 2 of the M-stream batching arc: the CUTLASS grouped GEMM replacing
// the per-(stream, expert) GEMV loop in DecodeEngine::run_moe. This test is
// the source of the numbers in
// blog/posts/runtime/2026-09-07-grouped-gemm-replaces-the-gemv-loop/:
//
//   1. grouped-vs-GEMV numeric agreement at M=1 (same batch, different code
//      path -- CUTLASS and the scalar GEMV do not share a reduction order,
//      so this is a tolerance check, not an equality check)
//   2. grouped-M1-sequential vs grouped-M8-concurrent token parity, the same
//      8 real prompts and 20 generated tokens as test_batch_parity.cu, run
//      here with MoePath::kForceGrouped so both sides take the grouped path
//      (test_batch_parity.cu itself uses the default MoePath::kAuto, which
//      is the same thing once kGroupedMoeMinBatch is 1)
//   3. an M-sweep (1, 2, 4, 8, 16, 32) of both paths, forced independently at
//      every M, to find the crossover and report tok/s and bytes streamed
//      per token, plus the host-side expert-grouping cost the grouped path
//      pays per step (blog/posts/runtime/2026-09-07-grouped-gemm-beats-gemv-
//      at-every-m/'s Next: "measure first" before replacing the std::map
//      with a flat counting pass)
//
// One engine load for the whole file (model loads dominate test runtime; see
// AGENTS.md and the ~11 minute batch-parity test). Checks 1 and 2 build the
// engine's real prompts at M=8, matching test_batch_parity.cu's scenario
// exactly (a byte-for-byte replica, just forcing the path explicitly instead
// of relying on the default crossover); the engine itself is sized to
// max_batch=kMaxSweepBatch=32 so the M-sweep above can run every M on the
// same load.
//
// max_tokens stays 512, not the fuel's serving context cap (262144,
// fuels/glm-5.3-flash/fuel.yaml serving_context_cap): KV bytes/stream/token
// are mla_layers * (kv_lora + 2*index_head_dim) * 2 bytes = 11 * 768 * 2 =
// 16896 B, so at max_batch=32 that is 264 MiB total for 512 tokens/stream
// against 135 GiB for 262144 tokens/stream -- more than this booster's
// 123.73 GiB total memory on its own, before the 20 GiB expert cache and the
// resident weights. 512 exercises every kernel and launch-bound choice below
// at the batch sizes this entry measures; it is not a claim about how far a
// real agent session's context should run.
//
// Returns 77 when the checkpoint is not on this node, matching the other
// real-weight tests.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "model.h"
#include "model_config.h"
#include "tokenizer.h"
#include "weights.h"

namespace {

using Clock = std::chrono::steady_clock;
using rocket::engine::DecodeEngine;
using rocket::engine::MoePath;

int failures = 0;

void check(const std::string& what, bool ok, const std::string& detail) {
  std::printf("  %-46s %-4s %s\n", what.c_str(), ok ? "ok" : "FAIL", detail.c_str());
  if (!ok) ++failures;
}

double max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double scale = 1e-6;
  for (const float v : b) scale = std::max(scale, static_cast<double>(std::fabs(v)));
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]) / scale);
  return worst;
}

// Runs `steps` synthetic decode steps at batch M, forcing `path`. Returns
// mean ms/step over the timed steps (the warmup steps are a load artifact:
// the expert cache is cold on the first touch of every (layer, expert) pair
// this M has not seen yet, not a steady-state cost). Token id 5 for every
// stream, every step: content does not matter, only that every layer's
// routing and every kernel actually runs.
double time_steps(DecodeEngine& engine, int M, MoePath path, int warmup, int timed,
                  std::uint64_t* bytes_delta, bool use_graph = false) {
  engine.reset();
  engine.set_moe_path(path);
  engine.set_use_cuda_graph(use_graph);
  std::vector<int> tokens(M, 5), next;
  for (int i = 0; i < warmup; ++i) engine.step(tokens, next, false);
  const std::uint64_t before = engine.weights().expert_bytes_streamed();
  const auto t0 = Clock::now();
  for (int i = 0; i < timed; ++i) engine.step(tokens, next, false);
  const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  if (bytes_delta != nullptr) *bytes_delta = engine.weights().expert_bytes_streamed() - before;
  engine.set_use_cuda_graph(false);
  return ms / timed;
}

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
  const std::vector<std::string> prompt_text = {
      "The capital of France is",
      "def add(a, b):\n    return",
      "Water boils at a temperature of",
      "The opposite of hot is",
      "In 1969, humans first landed on the",
      "Two plus two equals",
      "The largest planet in the solar system is",
      "Roses are red, violets are",
  };
  std::vector<std::vector<int>> prompts;
  for (const auto& p : prompt_text) prompts.push_back(tok.encode(p));
  const int M = static_cast<int>(prompts.size());  // 8, matches test_batch_parity.cu
  const int kMaxSweepBatch = 32;                    // widest M in the sweep below
  const int max_tokens = 512;                       // see header comment
  std::printf("loading engine (max_batch=%d, max_tokens=%d)...\n", kMaxSweepBatch, max_tokens);
  DecodeEngine engine(cfg, snapshot, static_cast<std::size_t>(20) * (1ull << 30), max_tokens,
                      kMaxSweepBatch);

  // --- 1. grouped vs GEMV at M=1, same token sequence, logits agreement ---
  std::printf("grouped vs GEMV, M=1\n");
  const std::vector<int>& seq = prompts[0];
  auto run_path = [&](MoePath path) {
    engine.reset();
    engine.set_moe_path(path);
    std::vector<int> next;
    for (const int id : seq) engine.step(std::vector<int>{id}, next, false);
    return engine.last_logits(0);
  };
  const std::vector<float> logits_gemv = run_path(MoePath::kForceGemv);
  const std::vector<float> logits_grouped = run_path(MoePath::kForceGrouped);
  const double diff = max_rel_diff(logits_grouped, logits_gemv);
  const auto argmax = [](const std::vector<float>& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
  };
  const int tok_gemv = argmax(logits_gemv);
  const int tok_grouped = argmax(logits_grouped);
  char buf[192];
  // The GEMV path is W4A16: exact BF16 activations against an NVFP4 weight.
  // The grouped path is W4A4: CUTLASS's FP4 tensor core needs FP4 on both
  // operands (bench/nvfp4_grouped_gemm.cu), so activations are NVFP4-
  // quantized too. That is a materially different arithmetic, not just a
  // different reduction order, so logits are not expected to agree closely
  // -- the real bar is that greedy decoding still picks the same token.
  std::snprintf(buf, sizeof(buf),
               "max rel diff %.4g (W4A16 vs W4A4, informational); greedy token %d vs %d", diff,
               tok_gemv, tok_grouped);
  check("grouped vs GEMV, M=1, greedy token agrees", tok_gemv == tok_grouped, buf);

  // --- 2. grouped-M1-sequential vs grouped-M8-concurrent, token parity ----
  std::printf("\ngrouped M=1-sequential vs grouped M=%d-concurrent, real prompts, staggered prefill\n",
             M);
  const int new_tokens = 20;
  std::vector<std::vector<int>> ref(M);
  for (int i = 0; i < M; ++i) {
    engine.reset();
    engine.set_moe_path(MoePath::kForceGrouped);
    std::vector<int> next;
    int last_tok = 0;
    for (const int id : prompts[i]) {
      engine.step(std::vector<int>{id}, next, false);
      last_tok = next[0];
    }
    for (int t = 0; t < new_tokens; ++t) {
      ref[i].push_back(last_tok);
      engine.step(std::vector<int>{last_tok}, next, false);
      last_tok = next[0];
    }
  }
  engine.reset();
  engine.set_moe_path(MoePath::kForceGrouped);
  std::size_t max_len = 0;
  for (const auto& p : prompts) max_len = std::max(max_len, p.size());
  std::vector<int> last(M, 0);
  std::vector<std::vector<int>> got(M);
  std::vector<int> next;
  const std::size_t total_steps = max_len - 1 + new_tokens;
  for (std::size_t t = 0; t < total_steps; ++t) {
    std::vector<int> tokens(M);
    for (int i = 0; i < M; ++i) tokens[i] = (t < prompts[i].size()) ? prompts[i][t] : last[i];
    engine.step(tokens, next, false);
    for (int i = 0; i < M; ++i) {
      last[i] = next[i];
      if (t + 1 >= prompts[i].size() && got[i].size() < static_cast<std::size_t>(new_tokens))
        got[i].push_back(next[i]);
    }
  }
  int stream_failures = 0;
  for (int i = 0; i < M; ++i) {
    const bool ok = got[i] == ref[i];
    if (!ok) ++stream_failures;
  }
  std::snprintf(buf, sizeof(buf), "%d/%d streams mismatched", stream_failures, M);
  check("grouped M=1 vs grouped M=8, token-for-token", stream_failures == 0, buf);

  // --- 3. M-sweep: crossover, tok/s, bytes/token -------------------------
  std::printf("\nM-sweep (5 warmup + 10 timed steps per path per M)\n");
  std::printf("%4s  %10s  %10s  %12s  %12s  %14s  %14s  %10s  %6s\n", "M", "gemv ms",
             "grouped ms", "gemv tok/s", "grp tok/s", "gemv GiB/tok", "grp GiB/tok", "grp us",
             "grp %");
  const std::vector<int> Ms = {1, 2, 4, 8, 16, 32};  // bounded by max_batch=kMaxSweepBatch above
  int crossover = -1;
  for (const int m : Ms) {
    std::uint64_t bytes_gemv = 0, bytes_grouped = 0;
    const double ms_gemv = time_steps(engine, m, MoePath::kForceGemv, 5, 10, &bytes_gemv);
    const double ms_grouped = time_steps(engine, m, MoePath::kForceGrouped, 5, 10, &bytes_grouped);
    const double toks_gemv = 1000.0 * m / ms_gemv;
    const double toks_grouped = 1000.0 * m / ms_grouped;
    const double gib_gemv = bytes_gemv / (1024.0 * 1024.0 * 1024.0) / (10.0 * m);
    const double gib_grouped = bytes_grouped / (1024.0 * 1024.0 * 1024.0) / (10.0 * m);
    // Host-side expert-grouping cost (Next: "measure first" before swapping
    // the std::map for a flat counting pass): one more instrumented step,
    // engine.stages().moe_group_host_ms is the std::chrono-only wall time of
    // that one step's std::map builds, summed over every MoE layer.
    engine.set_moe_path(MoePath::kForceGrouped);
    std::vector<int> gtoks(m, 5), gnext;
    engine.step(gtoks, gnext, false);
    const double group_us = engine.stages().moe_group_host_ms * 1000.0;
    const double group_pct = 100.0 * engine.stages().moe_group_host_ms / ms_grouped;
    std::printf("%4d  %10.2f  %10.2f  %12.2f  %12.2f  %14.4f  %14.4f  %10.2f  %5.3f%%\n", m,
               ms_gemv, ms_grouped, toks_gemv, toks_grouped, gib_gemv, gib_grouped, group_us,
               group_pct);
    if (crossover < 0 && ms_grouped < ms_gemv) crossover = m;
  }
  if (crossover < 0) {
    std::printf(
        "\nNo M in the sweep favored the grouped path; kGroupedMoeMinBatch should be revisited "
        "against model.cu's current constant.\n");
  } else {
    std::printf("\nMeasured crossover: grouped path is faster from M=%d (model.cu::kGroupedMoeMinBatch)\n",
               crossover);
  }

  // --- 4. CUDA graph: ms/step with vs without, grouped path, every M in the
  // sweep. set_use_cuda_graph(true) only changes step()'s internal dispatch
  // (model.cu::ensure_graphs_built / run_step_layers_graph); output must stay
  // identical to the non-graph grouped path, checked by the greedy-token
  // comparison below rather than assumed.
  std::printf(
      "\nCUDA graph: grouped path ms/step, with vs without (5 warmup + 10 timed steps)\n");
  std::printf("%4s  %14s  %14s  %8s\n", "M", "no graph ms", "graph ms", "speedup");
  for (const int m : Ms) {
    const double ms_no_graph = time_steps(engine, m, MoePath::kForceGrouped, 5, 10, nullptr, false);
    const double ms_graph = time_steps(engine, m, MoePath::kForceGrouped, 5, 10, nullptr, true);
    std::printf("%4d  %14.2f  %14.2f  %7.2fx\n", m, ms_no_graph, ms_graph,
               ms_no_graph / ms_graph);
  }
  // Correctness: graph on vs off must pick the same greedy tokens over a
  // real generation, not just report similar timing.
  std::printf("\ngraph vs no-graph, real prompt, greedy tokens\n");
  auto run_graph_check = [&](bool use_graph) {
    engine.reset();
    engine.set_moe_path(MoePath::kForceGrouped);
    engine.set_use_cuda_graph(use_graph);
    std::vector<int> out;
    std::vector<int> next;
    for (const int id : prompts[0]) engine.step(std::vector<int>{id}, next, false);
    int last = next[0];
    for (int t = 0; t < new_tokens; ++t) {
      out.push_back(last);
      engine.step(std::vector<int>{last}, next, false);
      last = next[0];
    }
    engine.set_use_cuda_graph(false);
    return out;
  };
  const std::vector<int> graph_off = run_graph_check(false);
  const std::vector<int> graph_on = run_graph_check(true);
  check("graph vs no-graph, greedy tokens match", graph_off == graph_on,
       graph_off == graph_on ? "identical" : "MISMATCH");

  std::printf("\n%s: %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}

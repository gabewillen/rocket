// Static fire: both boosters lit, expert-parallel, and held against the
// single-booster result.
//
//   --mode single    one booster, all 288 experts, writes the reference tokens
//   --mode parallel  two ranks, 144 experts each, reads them back and compares
//
// The gate is token parity: the M=8 tokens the pair decodes must equal the M=8
// tokens one booster decodes from the same prompts. That is expected to hold
// exactly rather than approximately, because the merged routed-expert
// accumulator is bit-identical by construction
// (src/fabric/expert_parallel.h), so a single differing token is a defect and
// not tolerance.
//
// Both ranks also print a checksum of stream 0's logits row. The two ranks run
// the entire non-expert stack replicated, so those checksums have to match; if
// they drift, the router on the two ranks saw different inputs and the row
// split stopped meaning the same thing on both sides.
//
// Launch with scripts/fabric/static-fire.sh, which starts rank 1 over ssh.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "fabric/expert_balance.h"
#include "fabric/expert_parallel.h"
#include "model.h"
#include "model_config.h"
#include "tokenizer.h"

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

const char* arg_value(int argc, char** argv, const char* key, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return fallback;
}

const std::vector<std::string>& prompts() {
  // The batch-parity prompts (tests/test_batch_parity.cu), unchanged, so the
  // reference this compares against is the one the single-booster suite
  // already gates on.
  static const std::vector<std::string> p = {
      "The capital of France is",
      "def add(a, b):\n    return",
      "Water boils at a temperature of",
      "The opposite of hot is",
      "In 1969, humans first landed on the",
      "Two plus two equals",
      "The largest planet in the solar system is",
      "Roses are red, violets are",
  };
  return p;
}

std::vector<int> parse_ints(const std::string& csv) {
  std::vector<int> out;
  std::size_t i = 0;
  while (i < csv.size()) {
    std::size_t j = csv.find(',', i);
    if (j == std::string::npos) j = csv.size();
    out.push_back(std::atoi(csv.substr(i, j - i).c_str()));
    i = j + 1;
  }
  return out;
}

// Order-independent would hide a permutation, so this is order-dependent:
// a 64-bit FNV-1a over the raw float bits of the logits row.
std::uint64_t checksum(const std::vector<float>& v) {
  std::uint64_t h = 1469598103934665603ull;
  for (const float f : v) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    for (int b = 0; b < 4; ++b) {
      h ^= (bits >> (8 * b)) & 0xff;
      h *= 1099511628211ull;
    }
  }
  return h;
}

// Runs the prompts at batch M with staggered prefill and collects each
// stream's generated tokens, exactly as tests/test_batch_parity.cu does.
std::vector<std::vector<int>> decode_prompts(rocket::engine::DecodeEngine& engine,
                                             const std::vector<std::vector<int>>& prompt_ids,
                                             int new_tokens) {
  const int M = static_cast<int>(prompt_ids.size());
  engine.reset();
  std::size_t max_prompt_len = 0;
  for (const auto& p : prompt_ids) max_prompt_len = std::max(max_prompt_len, p.size());

  std::vector<int> last(M, 0), next_batch;
  std::vector<std::vector<int>> got(M);
  const std::size_t total_steps = max_prompt_len - 1 + static_cast<std::size_t>(new_tokens);
  for (std::size_t t = 0; t < total_steps; ++t) {
    std::vector<int> tokens(M);
    for (int i = 0; i < M; ++i)
      tokens[i] = (t < prompt_ids[i].size()) ? prompt_ids[i][t] : last[i];
    engine.step(tokens, next_batch, false);
    for (int i = 0; i < M; ++i) {
      last[i] = next_batch[i];
      if (t + 1 >= prompt_ids[i].size() &&
          got[i].size() < static_cast<std::size_t>(new_tokens))
        got[i].push_back(next_batch[i]);
    }
  }
  return got;
}

struct StepTiming {
  double median_ms = 0;
  double fabric_ms = 0;
  double bytes_per_step = 0;
  double streamed_gib = 0;  // expert bytes pulled from the checkpoint per step
  double hit_rate = 0;
};

// Steady-state decode timing at one batch size: the streams are already past
// their prompts, so every step is a decode step and none is a prefill.
StepTiming time_steps(rocket::engine::DecodeEngine& engine, int M, int warmup, int timed,
                      rocket::fabric::ExpertParallel* ep) {
  engine.reset();
  // Distinct tokens per stream, so the streams route to different experts the
  // way concurrent agent streams do. Feeding every slot the same token makes
  // the batch touch no more experts than M=1 does and turns the expert cache
  // into a hit-rate artifact.
  std::vector<int> tokens(M), next;
  for (int i = 0; i < M; ++i) tokens[i] = 100 + i * 997;
  for (int i = 0; i < warmup; ++i) {
    engine.step(tokens, next, false);
    tokens = next;
  }
  if (ep != nullptr) {
    ep->barrier();
    ep->reset_stats();
  }
  const std::uint64_t hits0 = engine.weights().expert_hits();
  const std::uint64_t miss0 = engine.weights().expert_misses();
  const std::size_t streamed0 = engine.weights().expert_bytes_streamed();
  std::vector<double> ms;
  double fabric_ms = 0;
  for (int i = 0; i < timed; ++i) {
    const auto t0 = Clock::now();
    // collect_stages off: StageTimer syncs the stream once per stage per
    // layer, which is most of a step. stages_ is reset every step regardless
    // (model.cu::step), so stages().fabric is still this step's fabric time.
    engine.step(tokens, next, false);
    ms.push_back(ms_since(t0));
    fabric_ms += engine.stages().fabric;
    tokens = next;
  }
  std::sort(ms.begin(), ms.end());
  StepTiming out;
  out.median_ms = ms[ms.size() / 2];
  out.fabric_ms = fabric_ms / timed;
  if (ep != nullptr)
    out.bytes_per_step = static_cast<double>(ep->stats().bytes_out) / timed;
  const double hits = static_cast<double>(engine.weights().expert_hits() - hits0);
  const double miss = static_cast<double>(engine.weights().expert_misses() - miss0);
  out.streamed_gib =
      static_cast<double>(engine.weights().expert_bytes_streamed() - streamed0) / 1073741824.0 / timed;
  out.hit_rate = (hits + miss) > 0 ? hits / (hits + miss) : 0.0;
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = arg_value(argc, argv, "--mode", "parallel");
  const int rank = std::atoi(arg_value(argc, argv, "--rank", "0"));
  const std::string tokens_file = arg_value(argc, argv, "--tokens-file", "/tmp/rocket-static-fire-reference.txt");
  const int new_tokens = std::atoi(arg_value(argc, argv, "--new-tokens", "20"));
  const double cache_gib = std::atof(arg_value(argc, argv, "--expert-cache-gib", "20"));
  const int max_tokens = std::atoi(arg_value(argc, argv, "--max-tokens", "512"));
  const std::vector<int> sweep = parse_ints(arg_value(argc, argv, "--sweep", "1,8"));
  const int M = static_cast<int>(prompts().size());
  const int max_batch = std::max(M, *std::max_element(sweep.begin(), sweep.end()));
  const bool parallel = (mode == "parallel");

  const std::filesystem::path snapshot = rocket::fuel::default_nvfp4_snapshot_dir();
  const std::filesystem::path attn = rocket::fuel::default_attention_yaml();
  if (snapshot.empty() || !std::filesystem::exists(snapshot / "config.json") || attn.empty() ||
      !std::filesystem::exists(attn)) {
    std::printf("no checkpoint or attention.yaml on this node\n");
    return 77;
  }
  const rocket::fuel::ModelConfig cfg = rocket::fuel::load_model_config(attn, snapshot);
  const rocket::fuel::Tokenizer tok(snapshot / "tokenizer.json");

  // The pair is connected before either rank starts its multi-minute weight
  // load, so a bootstrap failure is reported in seconds rather than after both
  // nodes have read tens of GiB.
  // The expert partition. Default is the id split, 144/144; with
  // --expert-histogram-in it is the equal-cardinality greedy bin pack over
  // measured firing counts (src/fabric/expert_balance.h). Both ranks are given
  // the same file and run the same deterministic pack, so neither has to be
  // told the other's set.
  const std::string hist_in = arg_value(argc, argv, "--expert-histogram-in", "");
  rocket::fabric::ExpertPartition part;
  if (hist_in.empty()) {
    part = rocket::fabric::contiguous_partition(cfg.n_routed_experts, {});
  } else {
    const std::vector<std::uint64_t> counts =
        rocket::fabric::load_expert_histogram(hist_in, cfg.n_routed_experts);
    part = rocket::fabric::balance_experts(counts);
    const rocket::fabric::ExpertPartition id_split =
        rocket::fabric::contiguous_partition(cfg.n_routed_experts, counts);
    std::printf("expert partition from %s: predicted load %.0f / %.0f rows, imbalance %.3fx "
                "(id split: %.0f / %.0f, %.3fx)\n",
                hist_in.c_str(), part.load[0], part.load[1], part.imbalance, id_split.load[0],
                id_split.load[1], id_split.imbalance);
  }

  std::unique_ptr<rocket::fabric::ExpertParallel> ep;
  if (parallel) {
    rocket::fabric::Config fc;
    fc.rank = rank;
    fc.bootstrap_host = arg_value(argc, argv, "--host", "192.168.100.10");
    fc.bootstrap_port = std::atoi(arg_value(argc, argv, "--port", "18779"));
    ep = std::make_unique<rocket::fabric::ExpertParallel>(
        fc, part.owner, max_batch * cfg.num_experts_per_tok, cfg.hidden_size);
    std::printf("rank %d: fabric up, %d experts, %s\n", ep->rank(), ep->expert_count(),
                ep->contiguous() ? "one contiguous id range" : "a balanced set");
    std::fflush(stdout);
  }

  const auto t_load = Clock::now();
  rocket::engine::DecodeEngine engine(cfg, snapshot,
                                      static_cast<std::size_t>(cache_gib * (1ull << 30)),
                                      max_tokens, max_batch);
  if (parallel) {
    engine.weights().set_expert_set(ep->owned_experts());
    engine.set_expert_parallel(ep.get());
    engine.set_expert_parallel_overlap(arg_value(argc, argv, "--overlap", "0")[0] == 0x31);
  }
  engine.set_use_cuda_graph(arg_value(argc, argv, "--cuda-graph", "0")[0] == 0x31);
  std::printf("rank %d: overlap %s, cuda graph %s\n", rank,
              engine.expert_parallel_overlap() ? "on" : "off",
              engine.use_cuda_graph() ? "on" : "off");
  std::printf("rank %d: engine loaded in %.1f s, resident %.2f GiB, %zu expert slots\n", rank,
              ms_since(t_load) / 1000.0, engine.weights().resident_bytes() / 1073741824.0,
              engine.weights().expert_slots());
  std::fflush(stdout);

  std::vector<std::vector<int>> prompt_ids(M);
  for (int i = 0; i < M; ++i) prompt_ids[i] = tok.encode(prompts()[i]);

  // The parity decode pays a cold expert cache once; --skip-parity is for
  // re-measuring the timing sweep alone against an already-proven build, and
  // must not be used to report a passing gate.
  const bool skip_parity = arg_value(argc, argv, "--skip-parity", "0")[0] == 0x31;
  if (ep) ep->barrier();
  const auto got = skip_parity ? std::vector<std::vector<int>>(M)
                               : decode_prompts(engine, prompt_ids, new_tokens);
  const std::uint64_t logit_sum = checksum(engine.last_logits(0));
  std::printf("rank %d: stream 0 logits checksum %016llx\n", rank,
              static_cast<unsigned long long>(logit_sum));

  // Expert-firing histogram (model.h::expert_fire_counts). Written after the
  // parity decode, which is eight real prompts of twenty tokens each, so the
  // counts describe routing on text rather than on the synthetic tokens the
  // timing sweep feeds. Run --mode single to see all 288 experts; a rank of
  // the pair only ever fires the experts it owns.
  const std::string hist_out = arg_value(argc, argv, "--expert-histogram-out", "");
  if (!hist_out.empty()) {
    const auto& counts = engine.expert_fire_counts();
    std::ofstream h(hist_out);
    if (!h) {
      std::fprintf(stderr, "cannot write %s\n", hist_out.c_str());
      return 1;
    }
    h << "# routed-expert firing counts, summed over " << cfg.text_layers
      << " text layers and the static-fire parity decode (M=" << M << ", " << new_tokens
      << " new tokens per stream)\n";
    for (std::size_t e = 0; e < counts.size(); ++e) h << e << " " << counts[e] << "\n";
    std::printf("rank %d: wrote expert histogram (%zu experts) to %s\n", rank, counts.size(),
                hist_out.c_str());
  }

  int failures = 0;
  if (skip_parity) {
    std::printf("rank %d: parity decode skipped (--skip-parity), timing only\n", rank);
  } else if (!parallel) {
    std::ofstream out(tokens_file);
    if (!out) {
      std::fprintf(stderr, "cannot write %s\n", tokens_file.c_str());
      return 1;
    }
    out << M << " " << new_tokens << "\n";
    for (int i = 0; i < M; ++i) {
      for (std::size_t t = 0; t < got[i].size(); ++t) out << (t ? " " : "") << got[i][t];
      out << "\n";
    }
    std::printf("wrote single-booster reference for M=%d to %s\n", M, tokens_file.c_str());
  } else {
    std::ifstream in(tokens_file);
    if (!in) {
      std::fprintf(stderr, "no reference tokens at %s; run --mode single first\n",
                   tokens_file.c_str());
      return 1;
    }
    int ref_m = 0, ref_n = 0;
    in >> ref_m >> ref_n;
    std::vector<std::vector<int>> ref(ref_m);
    for (int i = 0; i < ref_m; ++i)
      for (int t = 0; t < ref_n; ++t) {
        int v = 0;
        in >> v;
        ref[i].push_back(v);
      }
    if (ref_m != M) {
      std::fprintf(stderr, "reference has M=%d, this run has M=%d\n", ref_m, M);
      return 1;
    }
    for (int i = 0; i < M; ++i) {
      bool ok = got[i].size() == ref[i].size();
      for (std::size_t t = 0; ok && t < got[i].size(); ++t) ok = got[i][t] == ref[i][t];
      if (rank == 0) {
        std::string text = tok.decode(got[i]);
        for (char& c : text) if (c == '\n') c = ' ';
        std::printf("  stream %d 2-rank vs 1-booster: %-9s %s\n", i, ok ? "identical" : "MISMATCH",
                    text.c_str());
        if (!ok)
          for (std::size_t t = 0; t < got[i].size(); ++t)
            if (t >= ref[i].size() || got[i][t] != ref[i][t])
              std::printf("    token %zu: 1-booster=%d 2-rank=%d\n", t,
                          t < ref[i].size() ? ref[i][t] : -1, got[i][t]);
      }
      if (!ok) ++failures;
    }
  }

  std::printf("\nrank %d timing, %d warmup + %d timed steps per M\n", rank, 3, 10);
  std::printf("  %3s %10s %8s %10s %8s %10s %10s %8s\n", "M", "ms/step", "tok/s", "fabric ms",
              "fabric%", "KiB/token", "GiB/step", "cache hit");
  for (const int m : sweep) {
    if (m > max_batch) continue;
    if (ep) ep->barrier();
    const StepTiming t = time_steps(engine, m, 3, 10, ep.get());
    std::printf("  %3d %10.1f %8.2f %10.3f %7.2f%% %10.1f %10.3f %7.1f%%\n", m, t.median_ms,
                1000.0 * m / t.median_ms, t.fabric_ms, 100.0 * t.fabric_ms / t.median_ms,
                t.bytes_per_step / 1024.0 / m, t.streamed_gib, 100.0 * t.hit_rate);
    std::fflush(stdout);
  }

  if (parallel && rank == 0 && !skip_parity)
    std::printf("\n%s: %d/%d streams mismatched\n", failures == 0 ? "STATIC FIRE PASS" : "STATIC FIRE FAIL",
                failures, M);
  if (ep) ep->barrier();
  return failures == 0 ? 0 : 1;
}

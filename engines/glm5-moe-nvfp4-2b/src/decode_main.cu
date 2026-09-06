// Greedy decode on one booster: load the fuel, run the prompt through the
// step, then generate. Prints the text, the per-stage cost of one step, and
// the two validation signals that stand in for a reference engine here (the
// per-layer hidden RMS and the router entropy).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

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

std::string printable(const std::string& s) {
  std::string out;
  for (const char c : s) {
    if (c == '\n') out += "\\n";
    else if (c == '\t') out += "\\t";
    else out.push_back(c);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path snapshot = rocket::fuel::default_nvfp4_snapshot_dir();
  if (snapshot.empty() || !std::filesystem::exists(snapshot / "config.json")) {
    std::fprintf(stderr, "no NVFP4 snapshot; set $ROCKET_FUEL_NVFP4_DIR\n");
    return 77;
  }
  const std::filesystem::path attn = rocket::fuel::default_attention_yaml();
  if (attn.empty() || !std::filesystem::exists(attn)) {
    std::fprintf(stderr, "no attention.yaml; set $ROCKET_ATTENTION_YAML\n");
    return 77;
  }

  const std::string prompt = arg_value(argc, argv, "--prompt", "The capital of France is");
  const int n_new = std::atoi(arg_value(argc, argv, "--tokens", "20"));
  const double cache_gib = std::atof(arg_value(argc, argv, "--expert-cache-gib", "56"));
  const int max_tokens = std::atoi(arg_value(argc, argv, "--max-tokens", "4096"));

  const rocket::fuel::ModelConfig cfg = rocket::fuel::load_model_config(attn, snapshot);
  std::printf("fuel      glm-5.3-flash, %d text layers (%d KDA, %d sparse-MLA), %d experts top-%d\n",
              cfg.text_layers, cfg.kda_layer_count(), cfg.mla_layer_count(), cfg.n_routed_experts,
              cfg.num_experts_per_tok);

  auto t_tok = Clock::now();
  const rocket::fuel::Tokenizer tok(snapshot / "tokenizer.json");
  const double tok_ms = ms_since(t_tok);

  auto t_load = Clock::now();
  rocket::engine::DecodeEngine engine(cfg, snapshot,
                                      static_cast<std::size_t>(cache_gib * (1ull << 30)),
                                      max_tokens, /*max_batch=*/1);
  const double load_ms = ms_since(t_load);
  std::printf("resident  %.2f GiB   expert cache %zu slots x %.2f MiB = %.2f GiB\n",
              engine.weights().resident_bytes() / 1073741824.0, engine.weights().expert_slots(),
              engine.weights().expert_slot_bytes() / 1048576.0,
              engine.weights().expert_slots() * engine.weights().expert_slot_bytes() / 1073741824.0);
  std::printf("load      %.1f s weights, %.1f s tokenizer\n", load_ms / 1000.0, tok_ms / 1000.0);

  const std::vector<int> prompt_ids = tok.encode(prompt);
  std::printf("prompt    %zu tokens: \"%s\"\n", prompt_ids.size(), printable(prompt).c_str());

  engine.reset();

  // Prefill runs the decode path once per prompt token.
  auto t_prefill = Clock::now();
  int next = 0;
  std::vector<int> next_batch;
  for (const int id : prompt_ids) {
    engine.step(std::vector<int>{id}, next_batch, false);
    next = next_batch[0];
  }
  const double prefill_ms = ms_since(t_prefill);

  std::vector<int> generated;
  std::vector<double> token_ms;
  const int instrument_at = std::max(0, std::min(n_new - 2, 9));
  rocket::engine::StageMs stages;
  std::vector<float> rms;
  double entropy = 0.0;

  for (int i = 0; i < n_new; ++i) {
    generated.push_back(next);
    const bool last = (i == n_new - 1);
    if (last) break;
    const bool instrument = (i == instrument_at);
    const auto t0 = Clock::now();
    engine.step(std::vector<int>{next}, next_batch, instrument);
    next = next_batch[0];
    const double dt = ms_since(t0);
    if (instrument) {
      stages = engine.stages();
      rms = engine.layer_rms();
      entropy = engine.router_entropy();
    } else {
      token_ms.push_back(dt);
    }
  }

  std::printf("\n--- output ------------------------------------------------------\n");
  std::printf("%s%s\n", prompt.c_str(), tok.decode(generated).c_str());
  std::printf("--- token ids ---------------------------------------------------\n");
  for (std::size_t i = 0; i < generated.size(); ++i)
    std::printf("%d:%d |%s|%s", static_cast<int>(i), generated[i],
                printable(tok.decode_one(generated[i])).c_str(), (i % 4 == 3) ? "\n" : "  ");
  std::printf("\n");

  std::sort(token_ms.begin(), token_ms.end());
  const double median = token_ms.empty() ? 0.0 : token_ms[token_ms.size() / 2];
  std::printf("\n--- timing ------------------------------------------------------\n");
  std::printf("prefill              %8.1f ms for %zu tokens (%.1f ms/token)\n", prefill_ms,
              prompt_ids.size(), prefill_ms / static_cast<double>(prompt_ids.size()));
  std::printf("decode median        %8.1f ms/token  (%.2f tok/s)\n", median,
              median > 0 ? 1000.0 / median : 0.0);
  if (!token_ms.empty())
    std::printf("decode min/max       %8.1f / %.1f ms\n", token_ms.front(), token_ms.back());

  std::printf("\nper-stage, one instrumented step (a sync per stage, so the sum\n"
              "exceeds an uninstrumented step):\n");
  std::printf("  %-22s %10s\n", "stage", "ms");
  const struct {
    const char* name;
    double v;
  } rows[] = {
      {"embed", stages.embed},         {"hyper-connections", stages.hyper_connection},
      {"norms", stages.norms},         {"KDA (34 layers)", stages.kda},
      {"sparse MLA + indexer", stages.mla}, {"dense MLP (3 layers)", stages.dense_mlp},
      {"MoE incl. experts", stages.moe_experts}, {"  of which streaming", stages.expert_stream},
      {"lm_head + argmax", stages.lm_head},
  };
  for (const auto& r : rows) std::printf("  %-22s %10.2f\n", r.name, r.v);
  std::printf("  %-22s %10.2f\n", "sum", stages.sum());

  std::printf("\n--- validation --------------------------------------------------\n");
  if (rms.empty()) {
    std::printf("hidden RMS: not collected (no instrumented step ran)\n");
  } else {
    float lo = rms[0], hi = rms[0];
    bool finite = true;
    for (const float v : rms) {
      lo = std::min(lo, v);
      hi = std::max(hi, v);
      finite = finite && std::isfinite(v);
    }
    std::printf("hidden RMS per layer:");
    for (std::size_t i = 0; i < rms.size(); ++i)
      std::printf("%s%5.2f", (i % 15 == 0) ? "\n  " : " ", rms[i]);
    std::printf("\nhidden RMS min / max over %zu layers: %.3f / %.3f%s\n", rms.size(), lo, hi,
                (finite && hi < 1e4f && lo > 1e-4f) ? "  (finite, in band)" : "  OUT OF BAND");
  }
  std::printf("router entropy, mean over 42 MoE layers: %.4f nats (max ln 8 = %.4f)\n", entropy,
              std::log(8.0));
  std::printf("expert cache: %llu hits, %llu misses, %.2f GiB streamed\n",
              static_cast<unsigned long long>(engine.weights().expert_hits()),
              static_cast<unsigned long long>(engine.weights().expert_misses()),
              engine.weights().expert_bytes_streamed() / 1073741824.0);
  return 0;
}

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
#include <map>
#include <memory>
#include <fstream>
#include <string>
#include <vector>

#include "dflash2.h"
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
  std::filesystem::path snapshot = rocket::fuel::default_nvfp4_snapshot_dir();
  if (snapshot.empty() || !std::filesystem::exists(snapshot / "config.json")) {
    std::fprintf(stderr, "no snapshot; set $ROCKET_FUEL_NVFP4_DIR\n");
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
  const int batch = std::atoi(arg_value(argc, argv, "--batch", "1"));
  const int spec_k = std::atoi(arg_value(argc, argv, "--spec", "1"));
  const int rank = std::atoi(arg_value(argc, argv, "--rank", "-1"));
  const bool preload_owned = arg_value(argc, argv, "--preload-owned", "0")[0] == '1';
  const char* draft_file = arg_value(argc, argv, "--draft-file", nullptr);
  const bool telemetry = arg_value(argc, argv, "--telemetry", "0")[0] == '1';
  if (n_new <= 0 || max_tokens <= 0 || cache_gib < 0.0 || batch <= 0) {
    std::fprintf(stderr, "bad sizing: --tokens %d --max-tokens %d --expert-cache-gib %g "
                         "--batch %d (all > 0, cache >= 0)\n",
                 n_new, max_tokens, cache_gib, batch);
    return 2;
  }

  try {

  const rocket::fuel::ModelConfig cfg = rocket::fuel::load_model_config(attn, snapshot);
  std::printf("fuel      glm-5.3-flash, %d text layers (%d KDA, %d sparse-MLA), %d experts top-%d\n",
              cfg.text_layers, cfg.kda_layer_count(), cfg.mla_layer_count(), cfg.n_routed_experts,
              cfg.num_experts_per_tok);

  auto t_tok = Clock::now();
  const rocket::fuel::Tokenizer tok(snapshot / "tokenizer.json");
  const double tok_ms = ms_since(t_tok);

  auto t_load = Clock::now();
  std::unique_ptr<rocket::fabric::ExpertParallel> ep;
  if (rank >= 0) {
    rocket::fabric::Config fc;
    fc.rank = rank;
    fc.bootstrap_host = arg_value(argc, argv, "--host", "192.168.100.10");
    fc.bootstrap_port = std::atoi(arg_value(argc, argv, "--port", "18779"));
    const rocket::fabric::ExpertPartition part =
        rocket::fabric::contiguous_partition(cfg.n_routed_experts, {});
    // Sized for the spec verify width: step_spec routes batch*kSpecMax rows,
    // each contributing topk selections, into one exchange.
    ep = std::make_unique<rocket::fabric::ExpertParallel>(
        fc, part.owner, batch * rocket::engine::DecodeEngine::kSpecMax * cfg.num_experts_per_tok,
        cfg.hidden_size);
    std::printf("rank %d: fabric up, %d owned experts\n", ep->rank(), ep->expert_count());
  }
  rocket::engine::DecodeEngine engine(cfg, snapshot,
                                      static_cast<std::size_t>(cache_gib * (1ull << 30)),
                                      max_tokens, batch);
  engine.set_telemetry(telemetry);
  if (ep) {
    engine.weights().set_expert_set(ep->owned_experts());
    engine.set_expert_parallel(ep.get());
  }
  std::unique_ptr<rocket::engine::DFlash2DraftEngine> dflash;
  if (const char* draft_dir = std::getenv("ROCKET_DFLASH2_DIR")) {
    if (spec_k < 2 || spec_k > 8)
      throw std::runtime_error("DFlash2 requires --spec in [2,8]");
    const auto td = Clock::now();
    dflash = std::make_unique<rocket::engine::DFlash2DraftEngine>(
        draft_dir, engine.weights().embed(), engine.weights().lm_head(), batch, max_tokens, 7,
        ep.get());
    std::printf("dflash2   loaded in %.2f s\n", ms_since(td) / 1000.0);
  }
  if (ep && preload_owned) {
    const auto t_pre = Clock::now();
    const size_t pre0 = engine.weights().expert_bytes_streamed();
    const size_t loaded = engine.weights().preload_owned_experts(nullptr);
    std::printf("rank %d: preloaded %zu owned experts, %.2f GiB in %.1f s; streamed %.2f GiB so far\n",
                rank, loaded,
                (engine.weights().expert_bytes_streamed() - pre0) / 1073741824.0,
                ms_since(t_pre) / 1000.0,
                engine.weights().expert_bytes_streamed() / 1073741824.0);
    // A rank that finishes loading first reaches its first MoE exchange while
    // the peer can still be ~90 s into the preload; the timed-out spurious
    // doorbell kills the pair before a step runs. Hold both ranks here.
    ep->barrier();
  }
  if (arg_value(argc, argv, "--cuda-graph", "0")[0] == '1') engine.set_use_cuda_graph(true);
  const double load_ms = ms_since(t_load);
  std::printf("resident  %.2f GiB   expert cache %zu slots x %.2f MiB = %.2f GiB\n",
              engine.weights().resident_bytes() / 1073741824.0, engine.weights().expert_slots(),
              engine.weights().expert_slot_bytes() / 1048576.0,
              engine.weights().expert_slots() * engine.weights().expert_slot_bytes() / 1073741824.0);
  std::printf("load      %.1f s weights, %.1f s tokenizer\n", load_ms / 1000.0, tok_ms / 1000.0);

  const std::vector<int> prompt_ids = tok.encode(prompt);
  std::printf("prompt    %zu tokens: \"%s\"\n", prompt_ids.size(), printable(prompt).c_str());

  engine.reset();

  // Prefill runs the decode path once per prompt token; every stream gets
  // the same prompt so the batch is full from the first step.
  auto t_prefill = Clock::now();
  std::vector<int> next_batch;
  std::vector<int> last(batch, 0);
  for (const int id : prompt_ids) {
    std::vector<int> base(batch);
    for (int m = 0; m < batch; ++m) base[m] = engine.position(m);
    engine.step(std::vector<int>(batch, id), next_batch, false);
    if (dflash)
      dflash->append_context(engine.dflash_aux_hidden(), engine.dflash_aux_stride_rows(), 1, batch,
                             base, std::vector<int>(batch, 1), nullptr);
    for (int m = 0; m < batch; ++m) last[m] = next_batch[m];
  }
  const double prefill_ms = ms_since(t_prefill);

  std::vector<std::vector<int>> generated(batch);
  std::vector<double> token_ms;
  std::vector<double> draft_ms;
  std::vector<double> draft_context_ms;
  const int instrument_at = n_new > 12 ? n_new - 6 : std::max(0, n_new - 2);
  rocket::engine::StageMs stages;
  std::vector<float> rms;
  std::vector<int> spec_tokens;
  std::vector<int> verify_out;
  std::vector<int> accepted_cnt(batch);
  long long sl_accepted = 0, sl_rounds = 0;  // spec tallies hoisted for the timing print
  double entropy = 0.0;

  std::vector<int>* draft_ids = nullptr;
  std::vector<int> draft_storage;
  if (draft_file) {
    std::ifstream in(draft_file);
    if (!in) {
      std::fprintf(stderr, "cannot read --draft-file %s\n", draft_file);
      return 2;
    }
    int v;
    while (in >> v) draft_storage.push_back(v);
    draft_ids = &draft_storage;
    std::printf("draft     %zu reference token ids from %s\n", draft_storage.size(), draft_file);
  }

  if (spec_k <= 1) {
    // Plain decode: one token per stream per step.
    for (int i = 0; i < n_new; ++i) {
      for (int m = 0; m < batch; ++m) generated[m].push_back(last[m]);
      const bool last_tok = (i == n_new - 1);
      if (last_tok) break;
      const bool instrument = (i == instrument_at);
      const auto t0 = Clock::now();
      engine.step(last, next_batch, instrument);
      for (int m = 0; m < batch; ++m) last[m] = next_batch[m];
      const double dt = ms_since(t0);
      if (instrument) {
        stages = engine.stages();
        rms = engine.layer_rms();
        entropy = engine.router_entropy();
      } else {
        token_ms.push_back(dt);
      }
    }
  } else {
    // Spec-decode: each round drafts spec_k tokens per stream (position 0 is
    // the committed argmax, 1..spec_k-1 are n-gram continuations), verifies
    // them in one batched step_spec call, and accepts the longest prefix the
    // target agrees with. The bonus argmax after the last accepted position
    // becomes the next round's committed token.
    const int K = spec_k;
    long long drafted = 0;
    // Alias into the hoisted tallies so the loop body and the summary print
    // keep their original names while the timing section can read them.
    long long& accepted_total = sl_accepted;
    long long& rounds = sl_rounds;
    bool spec_instrumented = false;
  while (static_cast<int>(generated[0].size()) < n_new) {
      // Commit the token about to be verified, then build the draft.
      for (int m = 0; m < batch; ++m) generated[m].push_back(last[m]);
      bool any_left = false;
      for (int m = 0; m < batch; ++m)
        if (static_cast<int>(generated[m].size()) < n_new) any_left = true;
      if (!any_left) break;

      const auto round_t0 = Clock::now();
      spec_tokens.assign(batch * K, 0);
      std::vector<int> dflash_tokens;
      if (dflash && !draft_ids) {
        const auto draft_t0 = Clock::now();
        std::vector<int> draft_pos(batch);
        for (int m = 0; m < batch; ++m) draft_pos[m] = engine.position(m);
        dflash->propose(last, draft_pos, batch, K - 1, dflash_tokens, nullptr);
        draft_ms.push_back(ms_since(draft_t0));
      }
      for (int m = 0; m < batch; ++m) {
        spec_tokens[m] = last[m];
        // Draft sources: --draft-file draws the verify chain from a reference
        // generation (perfect-draft correctness probe); otherwise an n-gram
        // lookup proposes the continuation, padded with the committed token.
        const std::vector<int>& g = generated[m];
        std::vector<int> d;
        if (draft_ids) {
          for (int j = 1; j < K && g.size() + j - 1 < draft_ids->size(); ++j)
            d.push_back((*draft_ids)[g.size() + j - 1]);
        } else if (dflash) {
          for (int j = 0; j < K - 1; ++j)
            d.push_back(dflash_tokens[static_cast<std::size_t>(j) * batch + m]);
        } else if (g.size() >= 2) {
          const int a = g[g.size() - 2], b = g[g.size() - 1];
          for (int i = static_cast<int>(g.size()) - 3; i >= 0 && static_cast<int>(d.size()) < K - 1; --i) {
            if (g[i] == a && i + 1 < static_cast<int>(g.size()) && g[i + 1] == b) {
              for (int j = i + 2; j < static_cast<int>(g.size()) && static_cast<int>(d.size()) < K - 1; ++j)
                d.push_back(g[j]);
              break;
            }
          }
        }
        for (int j = 1; j < K; ++j) {
          const int t = (static_cast<int>(d.size()) >= j) ? d[j - 1] : last[m];
          spec_tokens[j * batch + m] = t;
        }
        drafted += K - 1;
      }
      if (ep && dflash) ep->sync_draft_tokens(spec_tokens);

      const bool instrument = !spec_instrumented &&
          static_cast<int>(generated[0].size()) + K >= instrument_at;
      engine.step_spec(spec_tokens, K, verify_out, instrument);
      spec_instrumented = spec_instrumented || instrument;
      if (std::getenv("ROCKET_SPEC_TRACE")) {
        std::fprintf(stderr, "[trace] r%lld:", rounds);
        for (int j = 0; j < K; ++j)
          std::fprintf(stderr, " d%d=%d a%d=%d", j, spec_tokens[static_cast<std::size_t>(j) * batch],
                       j, verify_out[static_cast<std::size_t>(j) * batch]);
        std::fprintf(stderr, "\n");
      }


      // Greedy acceptance per stream. In pair mode, rank 0's decision is
      // authoritative: tiny last-bit differences between otherwise equal
      // replicated logits must never let one rank enter an extra model round.
      for (int m = 0; m < batch; ++m) {
        int acc = 1;  // position 0 is the committed argmax, always correct
        for (int j = 0; j + 1 < K; ++j) {
          if (verify_out[j * batch + m] == spec_tokens[(j + 1) * batch + m]) ++acc;
          else break;
        }
        accepted_cnt[m] = acc;
        last[m] = verify_out[(acc - 1) * batch + m];
      }
      if (ep) ep->sync_round_decision(accepted_cnt, last);
      for (int m = 0; m < batch; ++m) {
        for (int j = 1; j < accepted_cnt[m]; ++j)
          if (static_cast<int>(generated[m].size()) < n_new)
            generated[m].push_back(spec_tokens[j * batch + m]);
        accepted_total += accepted_cnt[m];
      }
      if (dflash) {
        const auto context_t0 = Clock::now();
        std::vector<int> base(batch);
        for (int m = 0; m < batch; ++m) base[m] = engine.position(m);
        dflash->append_context(engine.dflash_aux_hidden(), engine.dflash_aux_stride_rows(), K, batch,
                               base, accepted_cnt, nullptr);
        draft_context_ms.push_back(ms_since(context_t0));
      }
      engine.commit_positions(accepted_cnt);
      const double dt = ms_since(round_t0);  // draft, verify, acceptance, state commit
      ++rounds;
      sl_accepted = accepted_total;
      sl_rounds = rounds;
      if (instrument) {
        stages = engine.stages();
        rms = engine.layer_rms();
        entropy = engine.router_entropy();
      } else {
        token_ms.push_back(dt);
      }
      if (static_cast<int>(generated[0].size()) >= n_new) break;
    }
    const long long accepted_drafts = accepted_total - rounds * batch;
    std::printf("spec       rounds=%lld drafted=%lld accepted=%lld + committed=%lld (%.1f%% draft acceptance)\n",
                rounds, drafted, accepted_drafts, rounds * batch,
                drafted > 0 ? 100.0 * accepted_drafts / drafted : 0.0);
  }

  std::printf("\n--- output ------------------------------------------------------\n");
  if (batch == 1)
    std::printf("%s%s\n", prompt.c_str(), tok.decode(generated[0]).c_str());
  else
    std::printf("batch %d, stream 0: %s%s\n", batch, prompt.c_str(),
                tok.decode(generated[0]).c_str());
  if (batch == 1 || std::getenv("ROCKET_TOKEN_IDS")) {
    std::printf("--- token ids ---------------------------------------------------\n");
    for (std::size_t i = 0; i < generated[0].size(); ++i)
      std::printf("%d:%d |%s|%s", static_cast<int>(i), generated[0][i],
                  printable(tok.decode_one(generated[0][i])).c_str(), (i % 4 == 3) ? "\n" : "  ");
    std::printf("\n");
  }

  std::sort(token_ms.begin(), token_ms.end());
  const double median = token_ms.empty() ? 0.0 : token_ms[token_ms.size() / 2];
  std::printf("\n--- timing ------------------------------------------------------\n");
  std::printf("prefill              %8.1f ms for %zu tokens (%.1f ms/token)\n", prefill_ms,
              prompt_ids.size(), prefill_ms / static_cast<double>(prompt_ids.size()));
  if (spec_k > 1 && sl_rounds > 0) {
    // Spec rounds absorb more than one token per stream (accepted prefix +
    // committed token), so batch/median undercounts by the acceptance length.
    const double tpr = static_cast<double>(sl_accepted) / static_cast<double>(sl_rounds);
  }
  std::printf("decode median        %8.1f ms/step  (%.2f agg tok/s at batch %d, %.2f tok/s/stream)\n",
              median,
              median > 0 ? (spec_k > 1 && sl_rounds > 0
                                ? 1000.0 * static_cast<double>(sl_accepted) / static_cast<double>(sl_rounds) / median
                                : 1000.0 * batch / median)
                         : 0.0,
              batch,
              median > 0 ? (spec_k > 1 && sl_rounds > 0
                                ? 1000.0 * static_cast<double>(sl_accepted) / static_cast<double>(sl_rounds) / median / static_cast<double>(batch)
                                : 1000.0 / median)
                         : 0.0);
  if (!token_ms.empty())
    std::printf("decode min/max       %8.1f / %.1f ms\n", token_ms.front(), token_ms.back());
  if (!draft_ms.empty()) {
    std::sort(draft_ms.begin(), draft_ms.end());
    std::sort(draft_context_ms.begin(), draft_context_ms.end());
    std::printf("dflash2 propose      %8.1f ms median\n", draft_ms[draft_ms.size() / 2]);
    std::printf("dflash2 context      %8.1f ms median\n",
                draft_context_ms[draft_context_ms.size() / 2]);
  }

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
  if (telemetry) {
    std::map<std::string, float> ordered(engine.telemetry_absmax().begin(),
                                         engine.telemetry_absmax().end());
    std::printf("\n--- quantization telemetry (fixed family vocabulary) -----------\n");
    for (const auto& [name, value] : ordered)
      std::printf("telemetry_absmax %-48s %.9g\n", name.c_str(), value);
  }
  return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
}

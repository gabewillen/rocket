// Real-weight cache namespace gate. The same full 128-token prefix is run at
// M=1, M=8, and M=16. Exact target MLA/indexer bytes, FP32 KDA state, and
// DFlash2 trailing-window bytes are compared. Cross-shape differences are
// expected because production cache keys include batch shape and measured
// eight-row slot group. Per-slot digests verify each group is internally exact.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "dflash2.h"
#include "model.h"
#include "model_config.h"
#include "tokenizer.h"

namespace {
struct Digests {
  rocket::engine::PrefixStateDigest target;
  std::uint64_t dflash = 0;
};
}

int main() {
  const auto snapshot = rocket::fuel::default_nvfp4_snapshot_dir();
  const auto attn = rocket::fuel::default_attention_yaml();
  const char* draft_dir = std::getenv("ROCKET_DFLASH2_DIR");
  if (snapshot.empty() || !std::filesystem::exists(snapshot / "config.json") ||
      attn.empty() || !std::filesystem::exists(attn) || !draft_dir ||
      !std::filesystem::exists(std::filesystem::path(draft_dir) / "model.safetensors")) {
    std::printf("checkpoint, attention.yaml, or DFlash2 model unavailable\n");
    return 77;
  }
  const auto cfg = rocket::fuel::load_model_config(attn, snapshot);
  const rocket::fuel::Tokenizer tok(snapshot / "tokenizer.json");
  std::string text;
  while (tok.encode(text).size() < 128)
    text += " Preserve cache state exactly across concurrent agent requests.";
  auto ids = tok.encode(text);
  ids.resize(128);
  auto run = [&](int batch) {
    // One shape per engine lifetime keeps the test below the production c16
    // memory footprint and avoids retaining M16 allocations during M1/M8.
    rocket::engine::DecodeEngine engine(cfg, snapshot, 70ull << 30, 128, batch,
                                        0, 128, 32);
    rocket::engine::DFlash2DraftEngine draft(
        draft_dir, engine.weights().embed(), engine.weights().lm_head(), batch, 128, 7);
    engine.reset();
    std::vector<int> out;
    for (int begin = 0; begin < 128; begin += 32) {
      std::vector<int> chunk(32 * batch);
      for (int j = 0; j < 32; ++j)
        for (int m = 0; m < batch; ++m) chunk[j * batch + m] = ids[begin + j];
      engine.step_spec(chunk, 32, out, false);
      engine.commit_positions(std::vector<int>(batch, 32));
      for (int off = 0; off < 32; off += rocket::engine::DecodeEngine::kSpecMax) {
        std::vector<int> base(batch, begin + off);
        const auto* aux = engine.dflash_aux_hidden() +
            static_cast<std::size_t>(off) * batch * cfg.hidden_size;
        draft.append_context(aux, engine.dflash_aux_stride_rows(),
                             rocket::engine::DecodeEngine::kSpecMax, batch, base,
                             std::vector<int>(batch, rocket::engine::DecodeEngine::kSpecMax),
                             nullptr);
      }
    }
    Digests first{engine.kv_state_digest(0), draft.prefix_state_digest(0, 128)};
    std::vector<Digests> group_reference{first};
    if (batch >= 16)
      group_reference.push_back(
          {engine.kv_state_digest(8), draft.prefix_state_digest(8, 128)});
    int within_shape_differences = 0;
    for (int m = 1; m < batch; ++m) {
      const Digests other{engine.kv_state_digest(m), draft.prefix_state_digest(m, 128)};
      const Digests& expected = group_reference[static_cast<std::size_t>(m / 8)];
      const int source = (m / 8) * 8;
      const bool exact = engine.kv_state_equal(source, m) &&
                         draft.prefix_state_equal(source, m, 128);
      if (!(other.target == expected.target) || other.dflash != expected.dflash || !exact) {
        std::printf("shape M=%d stream=%d target=%016llx/%016llx kda=%016llx/%016llx "
                    "dflash=%016llx/%016llx DIFFERENT\n", batch, m,
            static_cast<unsigned long long>(expected.target.target),
            static_cast<unsigned long long>(other.target.target),
            static_cast<unsigned long long>(expected.target.kda),
            static_cast<unsigned long long>(other.target.kda),
            static_cast<unsigned long long>(expected.dflash),
            static_cast<unsigned long long>(other.dflash));
        ++within_shape_differences;
      }
    }
    if (within_shape_differences) {
      std::printf("FAIL: %d stream(s) differ inside an eight-row state group\n",
                  within_shape_differences);
      std::exit(1);
    }
    std::printf("M=%d target=%016llx kda=%016llx dflash=%016llx\n", batch,
        static_cast<unsigned long long>(first.target.target),
        static_cast<unsigned long long>(first.target.kda),
        static_cast<unsigned long long>(first.dflash));
    if (batch >= 16)
      std::printf("M=%d uses two measured eight-row state namespaces\n", batch);
    return first;
  };

  if (const char* one = std::getenv("ROCKET_PREFIX_TEST_BATCH")) {
    const int batch = std::atoi(one);
    if (batch != 1 && batch != 8 && batch != 16) {
      std::fprintf(stderr, "ROCKET_PREFIX_TEST_BATCH must be 1, 8, or 16\n");
      return 2;
    }
    (void)run(batch);
    std::printf("PASS: exact digests recorded for M=%d; production namespace includes batch "
                "shape and eight-row slot group\n", batch);
    return 0;
  }
  const Digests m1 = run(1);
  const Digests m8 = run(8);
  const Digests m16 = run(16);
  const bool invariant = m1.target == m8.target && m1.target == m16.target &&
                         m1.dflash == m8.dflash && m1.dflash == m16.dflash;
  if (invariant) {
    std::printf("FAIL: execution shapes became byte-identical; revisit conservative namespace\n");
    return 1;
  }
  std::printf("PASS: exact bytes differ across shapes; production namespace includes "
              "batch shape and eight-row slot group\n");
  return 0;
}

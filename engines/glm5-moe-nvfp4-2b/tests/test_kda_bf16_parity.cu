// KDA state dtype parity: greedy decode, 8 prompts x 40 tokens each, must
// match a reference captured from this engine's pre-change build, which
// stored the KDA recurrent state as FP32 (model.h::kda_state_, before this
// change moved it to BF16 with FP32 compute; see fuels/glm-5.3-flash/
// attention.yaml kda.state_dtype). The recurrence itself runs FP32 in both
// builds -- kernels.cu::kda_recurrent_step launches the register-split
// kernel (bench/kda_step_dropin.cuh), which keeps the state tile in
// per-thread registers rather than shared memory and reduces partial sums
// through a small shared buffer, but every add is still FP32 either way.
// Only the value that survives between steps rounds to BF16 now, so 40
// steps is long enough for that per-step rounding to compound into a token
// flip if it is going to at all -- 20 (the usual batch-parity depth) is
// not, by design (this is the reason the task asked for a longer run).
//
// Reference tokens were captured by hand: `git stash push -- <the six
// tracked src/test files this change touches>` to get back to FP32 state
// and the pre-drop-in kernel, `git checkout stash@{0} -- tests/CMakeLists.txt`
// to keep this test registered against that reverted build, rebuild, run
// this binary once (kFp32Reference empty prints the rows to paste), then
// `git checkout -- tests/CMakeLists.txt && git stash pop` to restore. See
// blog/posts/cache/2026-09-08-kda-state-goes-bf16/ for the exact commands.
//
// kFp32Reference must be all 8 rows for this test to pass. A missing or
// short reference is a hard failure (return 1), not a skip: this test's
// entire job is comparing against that reference, so silently returning
// success without one would let the acceptance gate go unchecked.
// Returns 77 when the checkpoint is not on this node, matching the other
// real-weight tests.
#include <cstdio>
#include <string>
#include <vector>

#include "model.h"
#include "model_config.h"
#include "tokenizer.h"

namespace {

const int kNewTokens = 40;

const std::vector<std::string> kPrompts = {
    "The capital of France is",
    "def add(a, b):\n    return",
    "Water boils at a temperature of",
    "The opposite of hot is",
    "In 1969, humans first landed on the",
    "Two plus two equals",
    "The largest planet in the solar system is",
    "Roses are red, violets are",
};

// Captured once from the FP32-state, pre-drop-in-kernel build (see the file
// header for the exact git stash procedure), one row per prompt above,
// kNewTokens ids each.
const std::vector<std::vector<int>> kFp32Reference = {
    {12089,13,758,8584,11,12089,374,37322,330,13881,12,765,3263,8703,279,220,99419,339,9291,11,12089,702,1012,825,315,4505,594,3598,35008,315,16998,11,60809,11,35476,11,11148,11,8037,11,},  // prompt 0
    {264,488,293,271,750,32112,2877,11,293,982,262,470,264,481,293,271,750,30153,2877,11,293,982,262,470,264,353,293,271,750,21697,2877,11,293,982,262,421,293,621,220,15,},  // prompt 1
    {220,99457,30811,13,1096,374,264,1632,21260,2097,304,21271,323,29688,13,576,49518,1459,315,3015,374,279,9312,518,892,3015,4344,504,264,14463,311,264,6819,13,1096,374,264,15799,7286,304,},  // prompt 2
    {9252,13,11,576,13993,315,2409,374,2613,2572,576,13993,315,4937,374,6301,2572,576,13993,315,16203,374,2805,2572,576,13993,315,279,13993,315,4937,374,6301,2572,576,13993,315,279,13993,315,},  // prompt 3
    {17309,13,34976,220,98965,572,279,3550,38183,429,26039,279,1156,1378,1251,11,33973,32962,44605,323,56329,4688,17683,37754,30230,25210,11,389,279,56329,7329,13,44605,24256,8629,279,7329,11,10449,279,},  // prompt 4
    {4236,421,279,4236,374,3460,3322,13,481,17489,53690,271,40,614,264,56986,311,1281,13,358,1079,264,41055,6888,1319,30684,13,358,572,2581,1661,518,6888,13,358,572,2581,1661,518,6888,},  // prompt 5
    {49371,13,1084,374,279,17677,11575,504,279,8058,323,279,7772,304,279,12934,1849,13,49371,374,264,6819,14528,11,7290,432,374,23350,10003,315,6819,323,14463,13,1084,374,279,7772,11575,304,},  // prompt 6
    {6303,11,419,32641,374,3873,11,323,773,525,498,13,320,72,1327,631,15581,692,71907,553,508,26436,2533,58,4142,60,785,12762,51,16067,37,34734,39165,60,508,12332,60,508,64,60,},  // prompt 7
};

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

  const int M = static_cast<int>(kPrompts.size());
  if (static_cast<int>(kFp32Reference.size()) != M) {
    std::fprintf(stderr,
                 "kFp32Reference has %zu rows, expected %d: the FP32-state reference is not "
                 "populated (see the file header for the capture procedure). Failing rather than "
                 "skipping the acceptance gate.\n",
                 kFp32Reference.size(), M);
    return 1;
  }
  for (const auto& row : kFp32Reference) {
    if (static_cast<int>(row.size()) == kNewTokens) continue;
    std::fprintf(stderr,
                 "kFp32Reference row has %zu tokens, expected %d: the reference is malformed. "
                 "Failing rather than skipping the acceptance gate.\n",
                 row.size(), kNewTokens);
    return 1;
  }

  std::printf("loading engine...\n");
  const int max_tokens = 512;
  rocket::engine::DecodeEngine engine(cfg, snapshot, static_cast<std::size_t>(20) * (1ull << 30),
                                      max_tokens, /*max_batch=*/1);

  std::vector<std::vector<int>> prompt_ids(M);
  for (int i = 0; i < M; ++i) prompt_ids[i] = tok.encode(kPrompts[i]);

  int failures = 0;
  for (int i = 0; i < M; ++i) {
    engine.reset();
    std::vector<int> next_batch;
    int next = 0;
    for (const int id : prompt_ids[i]) {
      engine.step(std::vector<int>{id}, next_batch, false);
      next = next_batch[0];
    }
    std::vector<int> got;
    for (int t = 0; t < kNewTokens; ++t) {
      got.push_back(next);
      engine.step(std::vector<int>{next}, next_batch, false);
      next = next_batch[0];
    }

    const std::vector<int>& ref = kFp32Reference[i];
    bool ok = true;
    int divergence_step = -1;
    for (std::size_t t = 0; t < got.size(); ++t) {
      if (got[t] != ref[t]) {
        ok = false;
        divergence_step = static_cast<int>(t);
        break;
      }
    }
    std::printf("  prompt %d vs FP32-state reference: %s\n", i, ok ? "identical" : "MISMATCH");
    if (!ok) {
      ++failures;
      std::printf("    diverges at token %d: fp32=%d bf16=%d\n", divergence_step,
                  ref[divergence_step], got[divergence_step]);
      // Hidden-state RMS drift at the point of divergence: re-run to that
      // step with instrumentation on so the per-layer RMS is collected.
      engine.reset();
      int n = 0;
      for (const int id : prompt_ids[i]) {
        engine.step(std::vector<int>{id}, next_batch, false);
        n = next_batch[0];
      }
      for (int t = 0; t <= divergence_step; ++t) {
        const bool instrument = (t == divergence_step);
        engine.step(std::vector<int>{n}, next_batch, instrument);
        n = next_batch[0];
        if (instrument) {
          const auto& rms = engine.layer_rms();
          float lo = rms.empty() ? 0.0f : rms[0], hi = lo;
          for (const float v : rms) { lo = std::min(lo, v); hi = std::max(hi, v); }
          std::printf("    hidden RMS at divergence step, min/max over %zu layers: %.4f / %.4f\n",
                      rms.size(), lo, hi);
        }
      }
    }
  }

  std::printf("\n%s: %d/%d prompts diverged from the FP32-state reference over %d tokens\n",
              failures == 0 ? "PASS" : "FAIL", failures, M, kNewTokens);
  return failures == 0 ? 0 : 1;
}

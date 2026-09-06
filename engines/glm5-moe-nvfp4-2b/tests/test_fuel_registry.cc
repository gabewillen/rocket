// The layer registry and the tokenizer, both read off disk at serving time.
// Returns 77 when the checkpoint is not on this node.
#include <cstdio>
#include <string>
#include <vector>

#include "model_config.h"
#include "safetensors.h"
#include "tokenizer.h"

namespace {
int failures = 0;
void expect(bool ok, const std::string& what) {
  std::printf("  %-46s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}
}  // namespace

int main() {
  const std::filesystem::path snap = rocket::fuel::default_nvfp4_snapshot_dir();
  const std::filesystem::path attn = rocket::fuel::default_attention_yaml();
  if (snap.empty() || !std::filesystem::exists(snap / "config.json") || attn.empty() ||
      !std::filesystem::exists(attn)) {
    std::printf("no checkpoint or attention.yaml on this node\n");
    return 77;
  }

  std::printf("layer registry from attention.yaml, cross-checked against config.json\n");
  const rocket::fuel::ModelConfig c = rocket::fuel::load_model_config(attn, snap);
  expect(c.text_layers == 45, "45 text layers");
  expect(c.kda_layer_count() == 34, "34 KDA layers");
  expect(c.mla_layer_count() == 11, "11 sparse-MLA layers");
  expect(c.hidden_size == 4096, "hidden 4096");
  expect(c.hc_mult == 4 && c.hc_mix() == 24, "hyper-connections: 4 streams, 24 mix outputs");
  expect(c.n_routed_experts == 288 && c.num_experts_per_tok == 8, "288 experts, top-8");
  expect(c.qk_rope_head_dim == 0, "NoPE: no rope tail");
  expect(c.kv_lora_rank == 512, "MLA latent 512 wide");
  expect(c.index_topk == 2048 && c.index_select_pools() == 512, "indexer 2048 tokens = 512 pools");
  int dense = 0;
  for (const rocket::fuel::LayerSpec& l : c.layers)
    if (l.mlp == rocket::fuel::MlpKind::kDense) ++dense;
  expect(dense == 3, "3 dense MLP layers, 42 sparse");
  const std::vector<int> mla = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43};
  bool interleave = true;
  for (const int i : mla) interleave = interleave && c.layers[i].attn == rocket::fuel::AttnKind::kSparseMla;
  expect(interleave, "MLA at 3,7,...,43 (three KDA then one MLA)");

  std::printf("tokenizer\n");
  const rocket::fuel::Tokenizer tok(snap / "tokenizer.json");
  expect(tok.vocab_size() >= c.vocab_size - 64, "vocabulary covers the embedding table");
  const char* cases[] = {"The capital of France is",
                         "Hello, world! 12345  multiple   spaces\nand a newline.",
                         "def add(a, b):\n    return a + b\n", "it's don't I'll",
                         "<|user|>hi<|assistant|>"};
  bool round_trip = true;
  for (const char* s : cases) round_trip = round_trip && tok.decode(tok.encode(s)) == std::string(s);
  expect(round_trip, "byte-level round trip on 5 cases");
  const std::vector<int> ids = tok.encode("The capital of France is");
  expect(ids.size() == 5, "\"The capital of France is\" is 5 tokens");
  expect(tok.decode_one(ids[1]) == " capital", "second token decodes to \" capital\"");
  expect(tok.is_special(154820), "<|endoftext|> is a special token");

  std::printf("\n%s: %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}

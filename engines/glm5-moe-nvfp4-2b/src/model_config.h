// Model geometry for one chemistry: glm5-moe-288e-top8-kda-dsa.
//
// Two sources, both on disk, neither re-derived here:
//   fuels/glm-5.3-flash/attention.yaml  the layer interleave and attention dims
//   <snapshot>/config.json              the runtime scalars the yaml does not carry
//                                       (rms_norm_eps, vocab, MoE widths, mHC)
// The loader reads both and refuses to start when they disagree, so a fuel
// whose checkpoint drifted from the recorded geometry cannot be burned by
// accident.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace rocket::fuel {

enum class AttnKind { kKda, kSparseMla };
enum class MlpKind { kDense, kSparse };

struct LayerSpec {
  int index = 0;
  AttnKind attn = AttnKind::kKda;
  MlpKind mlp = MlpKind::kDense;
};

// Everything the decode step needs to know about the fuel's shape.
struct ModelConfig {
  // --- from attention.yaml -------------------------------------------------
  int hidden_size = 0;          // 4096
  int text_layers = 0;          // 45
  int mtp_layer_id = 0;         // 45, present on disk, not executed here

  int kda_heads = 0;            // 64
  int kda_head_dim = 0;         // 128
  int kda_conv_kernel = 0;      // 4
  float kda_gate_lower_bound = 0.0f;  // -5.0

  int mla_heads = 0;            // 64
  int q_lora_rank = 0;          // 1536
  int kv_lora_rank = 0;         // 512
  int qk_nope_head_dim = 0;     // 256
  int qk_rope_head_dim = 0;     // 0, NoPE
  int v_head_dim = 0;           // 256

  int index_n_heads = 0;        // 32
  int index_head_dim = 0;       // 128
  int index_topk = 0;           // 2048
  int index_kpool = 0;          // 4

  // --- from config.json ----------------------------------------------------
  int vocab_size = 0;              // 154880
  float rms_norm_eps = 0.0f;       // 1e-5
  int intermediate_size = 0;       // 12288, the dense layers
  int moe_intermediate_size = 0;   // 2048
  int n_routed_experts = 0;        // 288
  int num_experts_per_tok = 0;     // 8
  int n_shared_experts = 0;        // 1
  int n_group = 0;                 // 1
  int topk_group = 0;              // 1
  bool norm_topk_prob = true;
  float routed_scaling_factor = 0.0f;  // 2.5
  float swiglu_limit = 0.0f;           // 10.0
  bool index_kpool_always_select_tail = true;
  bool index_kpool_compress = true;

  // Manifold-constrained hyper-connections replace the residual stream.
  int hc_mult = 0;             // 4 parallel streams
  int hc_sinkhorn_iters = 0;   // 20
  float hc_eps = 0.0f;         // 1e-6

  int pad_token_id = 0;
  std::vector<int> eos_token_ids;

  std::vector<LayerSpec> layers;

  // Derived, all exact.
  int kda_qkv_dim() const { return kda_heads * kda_head_dim; }        // 8192
  int qk_head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; }  // 256
  int mla_kv_b_out() const { return mla_heads * (qk_nope_head_dim + v_head_dim); }  // 32768
  int hc_mix() const { return (2 + hc_mult) * hc_mult; }              // 24
  int conv_state_taps() const { return kda_conv_kernel - 1; }         // 3
  int index_select_pools() const { return index_topk / index_kpool; } // 512

  int kda_layer_count() const;
  int mla_layer_count() const;
};

// Reads attention.yaml, then config.json from the snapshot, and cross-checks.
// Throws std::runtime_error on a missing field or a disagreement.
ModelConfig load_model_config(const std::filesystem::path& attention_yaml,
                              const std::filesystem::path& snapshot_dir);

// Default location of fuels/glm-5.3-flash/attention.yaml, resolved by walking
// up from the executable's source tree. Override with $ROCKET_ATTENTION_YAML.
std::filesystem::path default_attention_yaml();

}  // namespace rocket::fuel

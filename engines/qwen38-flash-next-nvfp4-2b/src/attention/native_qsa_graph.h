// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string_view>

#include "attention/qsa_target_preprocess.h"
#include "decode/full_attention_layer.h"

namespace rocket::qwen38::attention {

inline constexpr std::string_view kTargetQsaIndexerSidecarKey =
    "bdbebd4f45c398f090a41ab98cd3881b969d958d8ae0bc42f3411844d3262edd";

struct TargetQsaProjectionWeights {
  const std::uint8_t* q_weight = nullptr;
  const std::uint8_t* q_scale = nullptr;
  float q_global = 0.0f;
  const std::uint8_t* k_weight = nullptr;
  const std::uint8_t* k_scale = nullptr;
  float k_global = 0.0f;
  const std::uint8_t* v_weight = nullptr;
  const std::uint8_t* v_scale = nullptr;
  float v_global = 0.0f;
  const std::uint8_t* o_weight = nullptr;
  const std::uint8_t* o_scale = nullptr;
  float o_global = 0.0f;
};

// Exact caller-owned c1 capture arena. Extents are fixed by the Qwen3.8
// topology and remain stable across generations.
struct TargetQsaGraphArena {
  std::uint8_t* qkv_packed = nullptr;          // [2560/2]
  std::uint8_t* qkv_sfa = nullptr;             // [128*40*4]
  __nv_bfloat16* raw_main_qkv = nullptr;        // [6656]
  __nv_bfloat16* index_projected_qk = nullptr; // [640]
  __nv_bfloat16* main_query = nullptr;          // [12,256]
  __nv_bfloat16* attention_gate = nullptr;      // [12,256]
  __nv_bfloat16* index_query = nullptr;         // [4,128]
  float* index_logits = nullptr;                // [65536]
  std::int32_t* visible_blocks = nullptr;       // [1]
  std::int32_t* selected_blocks = nullptr;      // [512]
  std::int32_t* selected_tokens = nullptr;      // [2051]
  float* attention_partial = nullptr;           // [32,1,12,256]
  float* attention_lse = nullptr;               // [32,1,12]
  __nv_bfloat16* attention_output = nullptr;    // [12,256]
  __nv_bfloat16* gated_attention = nullptr;     // [3072]
  std::uint8_t* output_packed = nullptr;         // [3072/2]
  std::uint8_t* output_sfa = nullptr;            // [128*48*4]
  __nv_bfloat16* projected_output = nullptr;     // [2560]
};

// Concrete rank-local production participant. Construction may initialize
// immutable CUTLASS plans. launch() only enqueues into caller-owned buffers.
class NativeQsaFullAttentionGraph final : public decode::FullAttentionGraph {
 public:
  NativeQsaFullAttentionGraph(
      int device, int rank, int layer, std::string_view sidecar_key,
      const TargetQsaProjectionWeights& projection_weights,
      const TargetQsaPreprocessWeights& preprocess_weights,
      const TargetQsaGraphArena& arena);
  ~NativeQsaFullAttentionGraph() override;

  NativeQsaFullAttentionGraph(const NativeQsaFullAttentionGraph&) = delete;
  NativeQsaFullAttentionGraph& operator=(const NativeQsaFullAttentionGraph&) = delete;

  int rank() const noexcept override { return rank_; }
  int layer() const noexcept override { return layer_; }
  std::string_view checkpoint_revision() const noexcept override {
    return decode::kFullAttentionCheckpointRevision;
  }
  std::string_view slab_key() const noexcept override {
    return rank_ == 0 ? "rank0-target" : "rank1-target";
  }
  void launch(const __nv_bfloat16* block_input,
              const TargetQsaStateView& state, std::uint64_t generation, int m,
              cudaStream_t stream) override;
  const __nv_bfloat16* projected_output() const noexcept override {
    return arena_.projected_output;
  }

 private:
  int rank_;
  int layer_;
  TargetQsaPreprocessWeights preprocess_weights_;
  TargetQsaGraphArena arena_;
  void* projection_plan_ = nullptr;
  bool faulted_ = false;
};

}  // namespace rocket::qwen38::attention

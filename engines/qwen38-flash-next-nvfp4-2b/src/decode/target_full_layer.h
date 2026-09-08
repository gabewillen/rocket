// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/full_attention_layer.h"

namespace rocket::qwen38::decode {

// Rank-local target router, routed experts, and shared expert. The projected
// BF16 partial is reduced across TP2 before the final hyperconnection combine.
class TargetMoeGraph {
 public:
  virtual ~TargetMoeGraph() = default;
  virtual int rank() const noexcept = 0;
  virtual int layer() const noexcept = 0;
  virtual std::string_view checkpoint_revision() const noexcept = 0;
  virtual std::string_view slab_key() const noexcept = 0;
  virtual void launch(const __nv_bfloat16* block_input,
                      std::uint64_t generation, int m,
                      cudaStream_t stream) = 0;
  // Called only after the enclosing owner has fenced the borrowed stream.
  // Deferred CUDA failures prevent this publication hook from running.
  virtual void publish_after_fence(std::uint64_t generation) = 0;
  virtual const __nv_bfloat16* projected_output() const noexcept = 0;
};

struct TargetFullLayerResult {
  std::uint64_t generation;
  int rank;
  int layer;
  const __nv_bfloat16* post_layer;
};

// Atomic production layer-3 transition used by the K0 oracle comparator.
// No intermediate is published. Any reachable launch/fence/reduction failure
// permanently faults the object, including a failure after the first TP2
// reduction has written remote state.
class TargetFullLayer final {
 public:
  TargetFullLayer(FullAttentionGraph& attention, TargetMoeGraph& moe,
                  HiddenPartialReducer& attention_reducer,
                  HiddenPartialReducer& moe_reducer,
                  FullAttentionHyperConnection& hyperconnection,
                  pair_reduce::OtelStageSink& telemetry);

  TargetFullLayerResult execute(
      std::uint64_t generation, const attention::TargetQsaStateView& qsa_state,
      const __nv_bfloat16* materialized_pre_layer,
      __nv_bfloat16* attention_input, __nv_bfloat16* attention_injection,
      float* reduced_attention, __nv_bfloat16* post_attention_hidden,
      __nv_bfloat16* moe_input, __nv_bfloat16* moe_injection,
      float* reduced_moe, __nv_bfloat16* post_layer,
      std::string_view trace_id, std::string_view request_id,
      cudaStream_t stream);

 private:
  void emit(std::string_view stage, pair_reduce::Outcome outcome,
            std::string_view trace_id, std::string_view request_id,
            std::uint64_t duration_ns, std::uint64_t bytes) noexcept;

  FullAttentionGraph& attention_;
  TargetMoeGraph& moe_;
  HiddenPartialReducer& attention_reducer_;
  HiddenPartialReducer& moe_reducer_;
  FullAttentionHyperConnection& hyperconnection_;
  pair_reduce::OtelStageSink& telemetry_;
  int rank_;
  std::uint64_t last_generation_ = 0;
  bool faulted_ = false;
};

}  // namespace rocket::qwen38::decode

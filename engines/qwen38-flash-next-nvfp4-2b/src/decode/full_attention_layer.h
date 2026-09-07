// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string_view>

#include "decode/execution.h"

namespace rocket::qwen38::decode {

// Borrowed adapter for the already captured rank-local layer-3 graph. The
// concrete adapter binds QKV, QSA score/radix/attention, and output projection.
class FullAttentionGraph {
 public:
  virtual ~FullAttentionGraph() = default;
  virtual int rank() const noexcept = 0;
  virtual int layer() const noexcept = 0;
  virtual void launch(const __nv_bfloat16* block_input, int m,
                      cudaStream_t stream) = 0;
  virtual const __nv_bfloat16* projected_output() const noexcept = 0;
};

// Exact Qwen HyperConnection boundaries around one full-attention block:
// attn_hc.mix before attention, then mlp_hc.combine_and_mix after TP reduction.
class FullAttentionHyperConnection {
 public:
  virtual ~FullAttentionHyperConnection() = default;
  virtual void mix(const __nv_bfloat16* hidden, __nv_bfloat16* block_input,
                   __nv_bfloat16* injection, int m,
                   cudaStream_t stream) = 0;
  virtual void combine_and_mix(
      const __nv_bfloat16* hidden, const float* block_output,
      const __nv_bfloat16* injection, __nv_bfloat16* updated_hidden,
      __nv_bfloat16* next_block_input, __nv_bfloat16* next_injection, int m,
      cudaStream_t stream) = 0;
  // Fence every operation previously enqueued on stream and surface deferred
  // CUDA failures before a stage or generation can be published.
  virtual void synchronize(cudaStream_t stream) = 0;
};

struct FullAttentionResult {
  std::uint64_t generation;
  int m_bucket;
};

// Single-owner exact layer-3 attention transition. A result is returned only
// after every ordered stage succeeds; borrowed output buffers are unpublished
// until then. Reconstruct after an exception because remote reduction writes
// or output buffers may have changed.
class FullAttentionLayer final {
 public:
  FullAttentionLayer(FullAttentionGraph& graph, HiddenPartialReducer& reducer,
                     FullAttentionHyperConnection& hyperconnection,
                     pair_reduce::OtelStageSink& telemetry);

  FullAttentionResult execute(
      std::uint64_t generation, int m, const __nv_bfloat16* hidden,
      __nv_bfloat16* block_input, __nv_bfloat16* injection,
      float* reduced_attention, __nv_bfloat16* updated_hidden,
      __nv_bfloat16* next_block_input, __nv_bfloat16* next_injection,
      std::string_view trace_id, std::string_view request_id,
      cudaStream_t stream);

 private:
  void emit(std::string_view stage, pair_reduce::Outcome outcome, int m,
            std::string_view trace_id, std::string_view request_id,
            std::uint64_t duration_ns, std::uint64_t bytes) noexcept;

  FullAttentionGraph& graph_;
  HiddenPartialReducer& reducer_;
  FullAttentionHyperConnection& hyperconnection_;
  pair_reduce::OtelStageSink& telemetry_;
  std::uint64_t last_generation_ = 0;
  bool faulted_ = false;
};

}  // namespace rocket::qwen38::decode

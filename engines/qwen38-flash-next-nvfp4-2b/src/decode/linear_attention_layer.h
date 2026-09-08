// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string_view>

#include "decode/execution.h"
#include "decode/target_k0_execution_progress.h"

namespace rocket::qwen38::decode {

inline constexpr std::string_view kQwen38CheckpointRevision =
    "fc694b54fb0174e0913e6adf86691ef85a4ead47";
inline constexpr int kLinearHidden = 2'560;
inline constexpr int kLinearLocalKeyHeads = 8;
inline constexpr int kLinearLocalValueHeads = 24;
inline constexpr int kLinearHeadDim = 128;
inline constexpr int kLinearConvWidth = 5'120;
inline constexpr int kLinearConvKernel = 4;

constexpr bool allowed_linear_m(int m) noexcept {
  return m == 1 || m == 2 || m == 4 || m == 8 || m == 16;
}

[[nodiscard]] constexpr bool is_linear_attention_layer(int layer) noexcept {
  return layer >= 0 && layer < 48 && layer % 4 != 3;
}

// Borrowed adapter for five immutable captured graphs. Each graph contains the
// fixed layer-local QKVZ/BA projection and requant, causal-convolution update,
// in-place FP32 GDN recurrence, gated RMSNorm, and rank-local output projection
// in the order used by pinned vLLM 8e685d198. The convolution and recurrent
// pointers are views of the two authenticated nine-family transaction extents.
class LinearAttentionGraph {
 public:
  virtual ~LinearAttentionGraph() = default;
  virtual int rank() const noexcept = 0;
  virtual int layer() const noexcept = 0;
  virtual std::string_view checkpoint_revision() const noexcept = 0;
  virtual std::string_view slab_key() const noexcept = 0;
  virtual std::string_view conv_state_family() const noexcept = 0;
  virtual std::string_view recurrent_state_family() const noexcept = 0;
  virtual bool has_captured_bucket(int m) const noexcept = 0;
  virtual std::uint64_t logical_bytes_per_row(int m) const noexcept = 0;
  virtual void launch(const __nv_bfloat16* block_input,
                      __nv_bfloat16* conv_state, float* recurrent_state,
                      const std::int32_t* state_indices, int m,
                      cudaStream_t stream,
                      TargetK0ExecutionProgress* progress = nullptr) = 0;
  virtual const __nv_bfloat16* projected_output() const noexcept = 0;
};

class LinearAttentionHyperConnection {
 public:
  virtual ~LinearAttentionHyperConnection() = default;
  virtual void mix(const __nv_bfloat16* hidden, __nv_bfloat16* block_input,
                   __nv_bfloat16* injection, int m,
                   cudaStream_t stream) = 0;
  virtual void combine_and_mix(
      const __nv_bfloat16* hidden, const float* block_output,
      const __nv_bfloat16* injection, __nv_bfloat16* updated_hidden,
      __nv_bfloat16* next_block_input, __nv_bfloat16* next_injection, int m,
      cudaStream_t stream) = 0;
  virtual void combine(const __nv_bfloat16* hidden, const float* block_output,
                       const __nv_bfloat16* injection,
                       __nv_bfloat16* updated_hidden, int m,
                       cudaStream_t stream) = 0;
  virtual void synchronize(cudaStream_t stream) = 0;
};

struct LinearAttentionResult {
  std::uint64_t generation;
  int m_bucket;
  int rank;
  int layer;
};

// Exact attention half of one GDN layer. The returned block input and injection are
// the pending MLP HC boundary. A downstream MoE/dense executor consumes them.
// GDN state mutates in-place, so any failure is terminal and requires restoring
// the accepted nine-family transaction before retrying.
class LinearAttentionLayer final {
 public:
  LinearAttentionLayer(LinearAttentionGraph& graph,
                       HiddenPartialReducer& reducer,
                       LinearAttentionHyperConnection& hyperconnection,
                       pair_reduce::OtelStageSink& telemetry);

  LinearAttentionResult execute(
      std::uint64_t generation, int m, const __nv_bfloat16* hidden,
      __nv_bfloat16* block_input, __nv_bfloat16* injection,
      __nv_bfloat16* conv_state, float* recurrent_state,
      const std::int32_t* state_indices, float* reduced_attention,
      __nv_bfloat16* updated_hidden, __nv_bfloat16* next_block_input,
      __nv_bfloat16* next_injection, std::string_view trace_id,
      std::string_view request_id, cudaStream_t stream,
      TargetK0ExecutionProgress* progress = nullptr);

 private:
  void emit(std::string_view stage, pair_reduce::Outcome outcome, int m,
            std::string_view trace_id, std::string_view request_id,
            std::uint64_t duration_ns, std::uint64_t bytes) noexcept;

  LinearAttentionGraph& graph_;
  HiddenPartialReducer& reducer_;
  LinearAttentionHyperConnection& hyperconnection_;
  pair_reduce::OtelStageSink& telemetry_;
  int rank_;
  int layer_;
  std::uint64_t last_generation_ = 0;
  bool faulted_ = false;
};

}  // namespace rocket::qwen38::decode

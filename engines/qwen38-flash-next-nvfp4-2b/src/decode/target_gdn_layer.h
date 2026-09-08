// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/linear_attention_layer.h"
#include "decode/target_full_layer.h"
#include "linear_attention/gdn_core.h"

#include <cstddef>

namespace rocket::qwen38::decode {

inline constexpr std::size_t kTargetGdnConvSlotElements =
    static_cast<std::size_t>(linear_attention::kConvStateRows) *
    linear_attention::kQkvWidth;
inline constexpr std::size_t kTargetGdnRecurrentSlotElements =
    static_cast<std::size_t>(linear_attention::kValueHeads) *
    linear_attention::kHeadDim * linear_attention::kHeadDim;
inline constexpr int kTargetGdnStateSlots = 2;

struct TargetGdnC1State {
  // Slot zero remains the kernel's null state. The sole K0 stream maps to
  // slot one. All fields are borrowed from one rank/layer device owner.
  __nv_bfloat16* convolution = nullptr;
  std::size_t convolution_elements = 0;
  float* recurrent = nullptr;
  std::size_t recurrent_elements = 0;
  const std::int32_t* state_index = nullptr;
  std::size_t state_index_elements = 0;
};

struct TargetGdnC1Buffers {
  __nv_bfloat16* attention_input = nullptr;
  __nv_bfloat16* attention_injection = nullptr;
  float* reduced_attention = nullptr;
  __nv_bfloat16* post_attention_hidden = nullptr;
  __nv_bfloat16* moe_input = nullptr;
  __nv_bfloat16* moe_injection = nullptr;
  float* reduced_moe = nullptr;
};

[[nodiscard]] constexpr bool valid_target_gdn_c1_state_extent(
    const TargetGdnC1State& state) noexcept {
  return state.convolution &&
         state.convolution_elements ==
             kTargetGdnStateSlots * kTargetGdnConvSlotElements &&
         state.recurrent &&
         state.recurrent_elements ==
             kTargetGdnStateSlots * kTargetGdnRecurrentSlotElements &&
         state.state_index && state.state_index_elements == 1;
}

[[nodiscard]] constexpr bool complete_target_gdn_c1_buffers(
    const TargetGdnC1Buffers& buffers) noexcept {
  return buffers.attention_input && buffers.attention_injection &&
         buffers.reduced_attention && buffers.post_attention_hidden &&
         buffers.moe_input && buffers.moe_injection && buffers.reduced_moe;
}

// Device generation publication used by the compact E10 MoE owner. The
// implementation must enqueue exactly one write on the borrowed stream and
// must not fence it.
class TargetGdnMoeGenerationPort {
 public:
  virtual ~TargetGdnMoeGenerationPort() = default;
  virtual int rank() const noexcept = 0;
  virtual int layer() const noexcept = 0;
  virtual bool authenticated() const noexcept = 0;
  virtual void enqueue(std::uint64_t generation, cudaStream_t stream) = 0;
};

// Atomic c1 GDN+MoE transition. Dependencies and device views are borrowed
// for this object's lifetime and are single-owner, single-stream. Construction
// performs no CUDA work. Any execution failure permanently faults the object;
// GDN state, remote reduction state, and output buffers may be partially
// mutated and are never replayed.
class TargetGdnLayer final {
 public:
  TargetGdnLayer(LinearAttentionGraph& attention, TargetMoeGraph& moe,
                 HiddenPartialReducer& attention_reducer,
                 HiddenPartialReducer& moe_reducer,
                 LinearAttentionHyperConnection& hyperconnection,
                 TargetGdnMoeGenerationPort& moe_generation,
                 pair_reduce::OtelStageSink& telemetry,
                 TargetGdnC1State state, TargetGdnC1Buffers buffers);

  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int layer() const noexcept { return layer_; }
  [[nodiscard]] bool faulted() const noexcept { return faulted_; }
  [[nodiscard]] const HiddenPartialReducer* attention_reducer_identity()
      const noexcept {
    return &attention_reducer_;
  }
  [[nodiscard]] const HiddenPartialReducer* moe_reducer_identity()
      const noexcept {
    return &moe_reducer_;
  }

  TargetFullLayerResult execute(
      std::uint64_t generation,
      const __nv_bfloat16* replicated_pre_layer,
      __nv_bfloat16* replicated_post_layer, std::string_view trace_id,
      std::string_view request_id, cudaStream_t stream);

 private:
  void emit(pair_reduce::Outcome outcome, std::string_view trace_id,
            std::string_view request_id) noexcept;

  LinearAttentionLayer attention_;
  TargetMoeGraph& moe_;
  HiddenPartialReducer& attention_reducer_;
  HiddenPartialReducer& moe_reducer_;
  LinearAttentionHyperConnection& hyperconnection_;
  TargetGdnMoeGenerationPort& moe_generation_;
  pair_reduce::OtelStageSink& telemetry_;
  TargetGdnC1State state_;
  TargetGdnC1Buffers buffers_;
  int rank_ = -1;
  int layer_ = -1;
  std::uint64_t last_generation_ = 0;
  bool faulted_ = false;
};

}  // namespace rocket::qwen38::decode

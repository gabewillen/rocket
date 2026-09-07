// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "linear_attention/gdn_cutlass.h"
#include "pair_reduce/otel.h"

namespace rocket::qwen38::linear_attention {

inline constexpr int kMaxVerifyRows = kMaxRows * kMaxVerifyWidth;

struct VerifierShape {
  int sequences;
  int verify_width;

  [[nodiscard]] constexpr int token_rows() const noexcept {
    return sequences * verify_width;
  }
};

[[nodiscard]] constexpr bool allowed_verifier_shape(
    VerifierShape shape) noexcept {
  return allowed_m(shape.sequences) && shape.verify_width >= 1 &&
         shape.verify_width <= kMaxVerifyWidth &&
         shape.token_rows() <= kMaxVerifyRows;
}

enum class VerifierState : std::uint8_t { kReady, kStaged, kFaulted };

// Position-major input and output use [verify_width, sequences, hidden]. Stage
// reads each authenticated accepted slot once, advances it in registers in
// causal draft order, and writes no recurrent snapshots. Accept replays only
// the selected prefix into inactive state. A zero prefix leaves that slot intact.
// One borrowed CUDA stream is bound for the verifier lifetime so reset and the
// next stage cannot race queued private-state work.
// accepted_state_indices is a device array authenticated by the state owner:
// active entries are unique and in [1, state_pool_slots); zero is the null slot.
class GdnVerifier final {
 public:
  GdnVerifier(int device, CutlassGdnGraph& graph, int state_pool_slots,
              pair_reduce::OtelStageSink& telemetry);
  ~GdnVerifier();
  GdnVerifier(const GdnVerifier&) = delete;
  GdnVerifier& operator=(const GdnVerifier&) = delete;

  void stage(const __nv_bfloat16* position_major_input,
             __nv_bfloat16* accepted_conv_state,
             float* accepted_recurrent_state,
             const std::int32_t* accepted_state_indices, VerifierShape shape,
             std::string_view trace_id, std::string_view request_id,
             cudaStream_t stream);
  // accepted_prefixes is a host array with one entry per staged sequence.
  void accept(const std::int32_t* accepted_prefixes,
              std::string_view trace_id, std::string_view request_id,
              cudaStream_t stream);
  void reset(std::string_view trace_id, std::string_view request_id) noexcept;

  [[nodiscard]] const __nv_bfloat16* staged_output() const noexcept;
  [[nodiscard]] VerifierState state() const noexcept;
  [[nodiscard]] VerifierShape staged_shape() const noexcept;
  [[nodiscard]] std::uint64_t logical_stage_bytes() const noexcept;
  [[nodiscard]] std::uint64_t logical_accept_bytes() const noexcept;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace rocket::qwen38::linear_attention

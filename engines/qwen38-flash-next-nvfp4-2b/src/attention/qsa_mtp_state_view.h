// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

#include "mtp/state_arena.h"

namespace rocket::qwen38::attention {

inline constexpr int kMtpQsaMainWidth = 256;
inline constexpr int kMtpQsaIndexerWidth = 128;
inline constexpr int kMtpQsaMropeAxes = 3;
static_assert(kMtpQsaMainWidth == mtp::kQsaMainKvWidth);
static_assert(kMtpQsaIndexerWidth == mtp::kQsaIndexerWidth);
static_assert(kMtpQsaMropeAxes == mtp::kMropeAxes);

// Borrowed row journal owned by the MTP StateArena. QSA writes each draft
// step directly into these pointers. The arena remains the allocation and
// accepted-prefix publication owner.
struct MtpQsaPrefixStateView {
  __nv_bfloat16* main_key = nullptr;
  __nv_bfloat16* main_value = nullptr;
  __nv_bfloat16* raw_key = nullptr;
  __nv_bfloat16* compressed_key = nullptr;
  std::int64_t* rope_positions = nullptr;
  std::int32_t* main_slots = nullptr;
  std::int32_t* raw_slots = nullptr;
  std::int32_t* compressed_slots = nullptr;
  std::int32_t* compressed_valid = nullptr;
  int rows = 0;
  bool uses_mrope = false;
};

// The subset passed to QSA preprocessing and row formatting. These pointers
// alias MtpQsaPrefixStateView exactly. No causal-state copy or allocation is
// performed by the adapter.
struct MtpQsaWriteView {
  __nv_bfloat16* key;
  __nv_bfloat16* value;
  __nv_bfloat16* raw_key;
  __nv_bfloat16* compressed_key;
  std::int64_t* rope_positions;
  std::int32_t* main_slots;
  std::int32_t* raw_slots;
  std::int32_t* compressed_slots;
  std::int32_t* compressed_valid;
  int rows;
};

class MtpQsaStateViewError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

// Identity supplied by NativeExecutor at its typed MtpMiddleStagePort boundary.
// It carries no pointers and cannot extend the StateArena allocation lifetime.
struct MtpQsaArenaIdentity {
  int sequences;
  int depth;
  int query_tokens;
  int step;
  bool uses_mrope;
  std::uint64_t generation;
  std::uint64_t expected_generation;
};

[[nodiscard]] constexpr bool allowed_mtp_qsa_rows(int rows) noexcept {
  return rows == 1 || rows == 2 || rows == 4 || rows == 8 || rows == 16;
}

[[nodiscard]] constexpr bool allowed_mtp_qsa_query_tokens(
    int query_tokens) noexcept {
  return query_tokens == 300 || query_tokens == 8192;
}

[[nodiscard]] inline MtpQsaWriteView bind_mtp_qsa_write_view(
    MtpQsaPrefixStateView state, int expected_rows, bool require_mrope) {
  if (!allowed_mtp_qsa_rows(expected_rows) || state.rows != expected_rows ||
      !state.main_key || !state.main_value || !state.raw_key ||
      !state.compressed_key || !state.main_slots || !state.raw_slots ||
      !state.compressed_slots || !state.compressed_valid ||
      state.uses_mrope != require_mrope ||
      (state.uses_mrope && !state.rope_positions) ||
      (!state.uses_mrope && state.rope_positions)) {
    throw MtpQsaStateViewError("exact MTP QSA prefix-state view is required");
  }
  return {state.main_key,
          state.main_value,
          state.raw_key,
          state.compressed_key,
          state.rope_positions,
          state.main_slots,
          state.raw_slots,
          state.compressed_slots,
          state.compressed_valid,
          state.rows};
}

// Adapt the pushed MTP StateArena ABI without copying causal state. Every
// returned pointer aliases PrefixStateView and remains owned by StateArena.
[[nodiscard]] inline MtpQsaWriteView bind_mtp_qsa_write_view(
    mtp::PrefixStateView state, MtpQsaArenaIdentity identity) {
  if (!allowed_mtp_qsa_rows(identity.sequences) || identity.depth < 1 ||
      identity.depth > mtp::kStateMaxDepth ||
      !allowed_mtp_qsa_query_tokens(identity.query_tokens) ||
      identity.step < 0 || identity.step >= identity.depth ||
      identity.generation == 0 ||
      identity.generation != identity.expected_generation) {
    throw MtpQsaStateViewError(
        "exact MTP QSA arena identity and generation are required");
  }
  return bind_mtp_qsa_write_view(
      {state.main_key,
       state.main_value,
       state.raw_key,
       state.compressed_key,
       state.rope_positions,
       state.main_slots,
       state.raw_slots,
       state.compressed_slots,
       state.compressed_valid,
       identity.sequences,
       identity.uses_mrope},
      identity.sequences, identity.uses_mrope);
}

}  // namespace rocket::qwen38::attention

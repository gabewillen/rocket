// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace rocket::qwen38::attention {

inline constexpr int kMtpQsaMainWidth = 256;
inline constexpr int kMtpQsaIndexerWidth = 128;
inline constexpr int kMtpQsaMropeAxes = 3;

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

[[nodiscard]] constexpr bool allowed_mtp_qsa_rows(int rows) noexcept {
  return rows == 1 || rows == 2 || rows == 4 || rows == 8 || rows == 16;
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

}  // namespace rocket::qwen38::attention

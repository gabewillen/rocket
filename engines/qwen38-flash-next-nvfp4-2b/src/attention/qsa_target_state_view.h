// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace rocket::qwen38::attention {

inline constexpr int kTargetQsaContextTokens = 262144;
inline constexpr int kTargetQsaMainWidth = 256;
inline constexpr int kTargetQsaIndexerWidth = 128;
inline constexpr int kTargetQsaRawStateRows = 8;
inline constexpr int kTargetQsaCompressedRows = 65536;
inline constexpr int kTargetQsaCompressRatio = 4;
inline constexpr int kTargetQsaSelectedTokens = 2051;

enum class TargetQsaServingDtype : std::uint8_t {
  kBfloat16,
};

// One rank-local, layer-local view borrowed from the target StateArena. The
// production K0 graph writes the three causal-state families in place and
// borrows all scheduling metadata. It never owns, allocates, copies, or
// publishes these buffers.
struct TargetQsaStateView {
  __nv_bfloat16* main_key_cache = nullptr;
  __nv_bfloat16* main_value_cache = nullptr;
  __nv_bfloat16* raw_key_cache = nullptr;
  __nv_bfloat16* compressed_key_cache = nullptr;

  const std::int64_t* positions = nullptr;
  const std::int32_t* main_slot_mapping = nullptr;
  const std::int32_t* main_block_table = nullptr;
  const std::int32_t* raw_slot_mapping = nullptr;
  const std::int32_t* raw_block_table = nullptr;
  const std::int32_t* compressed_slot_mapping = nullptr;
  const std::int32_t* compressed_block_table = nullptr;
  const std::int32_t* query_start_locations = nullptr;
  const std::int64_t* logical_positions = nullptr;
  const std::int32_t* sequence_lengths = nullptr;
  const std::int32_t* token_to_request = nullptr;
  const std::int32_t* compression_work = nullptr;

  int main_blocks = 0;
  int compressed_blocks = 0;
  int compression_work_items = 0;
  int rows = 0;
  int rank = -1;
  int layer = -1;
  bool uses_mrope = false;
  TargetQsaServingDtype main_kv_dtype = TargetQsaServingDtype::kBfloat16;
  TargetQsaServingDtype side_cache_dtype = TargetQsaServingDtype::kBfloat16;
  std::uint64_t generation = 0;
  std::uint64_t expected_generation = 0;
};

class TargetQsaStateViewError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

[[nodiscard]] constexpr bool is_target_qsa_layer(int layer) noexcept {
  return layer >= 0 && layer < 48 && layer % 4 == 3;
}

// K0 is a single real target token. K1-K7 must enter through a later contract
// rather than widening this graph implicitly.
inline void validate_target_qsa_state_view(const TargetQsaStateView& state,
                                           int expected_rank,
                                           int expected_layer,
                                           std::uint64_t generation) {
  const bool identity_ok =
      (expected_rank == 0 || expected_rank == 1) &&
      is_target_qsa_layer(expected_layer) && state.rank == expected_rank &&
      state.layer == expected_layer && state.rows == 1 && generation != 0 &&
      state.generation == generation &&
      state.expected_generation == generation;
  const bool dtype_ok =
      state.main_kv_dtype == TargetQsaServingDtype::kBfloat16 &&
      state.side_cache_dtype == TargetQsaServingDtype::kBfloat16;
  const bool state_ok =
      state.main_key_cache && state.main_value_cache && state.raw_key_cache &&
      state.compressed_key_cache && state.positions &&
      state.main_slot_mapping && state.main_block_table &&
      state.raw_slot_mapping && state.raw_block_table &&
      state.compressed_slot_mapping && state.compressed_block_table &&
      state.query_start_locations && state.logical_positions &&
      state.sequence_lengths && state.token_to_request &&
      state.compression_work && state.main_blocks > 0 &&
      state.compressed_blocks > 0 && state.compression_work_items > 0;
  if (!identity_ok || !dtype_ok || !state_ok) {
    throw TargetQsaStateViewError(
        "exact caller-owned target QSA BF16 state view is required");
  }
}

}  // namespace rocket::qwen38::attention

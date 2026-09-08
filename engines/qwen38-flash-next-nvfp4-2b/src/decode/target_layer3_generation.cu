// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_generation.h"

#include <stdexcept>
#include <string>

namespace rocket::qwen38::decode {
namespace {

__global__ void prepare_layer3_generation(
    int row, std::uint64_t generation, std::int64_t* positions,
    std::int32_t* main_slot_mapping, std::int32_t* raw_slot_mapping,
    std::int32_t* compressed_slot_mapping,
    std::int32_t* query_start_locations, std::int64_t* logical_positions,
    std::int32_t* sequence_lengths, std::int32_t* token_to_request,
    std::int32_t* compression_work,
    std::uint64_t* target_moe_requested_generation) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *positions = row;
  *main_slot_mapping = row;
  *raw_slot_mapping = row % 8;
  *compressed_slot_mapping = row % 4 == 3 ? row / 4 : -1;
  *query_start_locations = 0;
  *logical_positions = row;
  *sequence_lengths = row + 1;
  *token_to_request = 0;
  *compression_work = row % 4 == 3 ? 1 : 0;
  *target_moe_requested_generation = generation;
}

bool complete_storage(const attention::TargetQsaStateView& state) noexcept {
  return state.main_key_cache && state.main_value_cache && state.raw_key_cache &&
         state.compressed_key_cache && state.positions &&
         state.main_slot_mapping && state.main_block_table &&
         state.raw_slot_mapping && state.raw_block_table &&
         state.compressed_slot_mapping && state.compressed_block_table &&
         state.query_start_locations && state.logical_positions &&
         state.sequence_lengths && state.token_to_request &&
         state.compression_work && state.main_blocks > 0 &&
         state.compressed_blocks > 0 && state.compression_work_items > 0;
}

}  // namespace

NativeTargetLayer3GenerationOwner::NativeTargetLayer3GenerationOwner(
    int rank, const attention::TargetQsaStateView& storage,
    std::uint64_t* target_moe_requested_generation)
    : rank_(rank),
      target_moe_requested_generation_(target_moe_requested_generation) {
  if ((rank != 0 && rank != 1) || storage.rank != rank ||
      storage.layer != 3 || storage.rows != 1 || storage.uses_mrope ||
      storage.main_kv_dtype != attention::TargetQsaServingDtype::kBfloat16 ||
      storage.side_cache_dtype != attention::TargetQsaServingDtype::kBfloat16 ||
      !complete_storage(storage) || !target_moe_requested_generation)
    throw std::invalid_argument("layer-3 generation storage changed");
  for (int row = 0; row < kTargetLayer3OracleRows; ++row) {
    views_[row] = storage;
    views_[row].generation = static_cast<std::uint64_t>(row + 1);
    views_[row].expected_generation = static_cast<std::uint64_t>(row + 1);
  }
  authenticated_ = true;
}

const attention::TargetQsaStateView&
NativeTargetLayer3GenerationOwner::view(
    int row, std::uint64_t generation) const {
  if (!authenticated_ || faulted_ || row < 0 ||
      row >= kTargetLayer3OracleRows ||
      generation != static_cast<std::uint64_t>(row + 1))
    throw std::logic_error("layer-3 generation view changed");
  return views_[row];
}

void NativeTargetLayer3GenerationOwner::enqueue_prepare(
    int row, std::uint64_t generation, cudaStream_t stream) {
  if (!authenticated_ || faulted_ || row < 0 ||
      row >= kTargetLayer3OracleRows || row != next_row_ || !stream ||
      generation != static_cast<std::uint64_t>(row + 1)) {
    faulted_ = true;
    throw std::logic_error("layer-3 generation enqueue changed");
  }
  prepare_layer3_generation<<<1, 1, 0, stream>>>(
      row, generation, const_cast<std::int64_t*>(views_[row].positions),
      const_cast<std::int32_t*>(views_[row].main_slot_mapping),
      const_cast<std::int32_t*>(views_[row].raw_slot_mapping),
      const_cast<std::int32_t*>(views_[row].compressed_slot_mapping),
      const_cast<std::int32_t*>(views_[row].query_start_locations),
      const_cast<std::int64_t*>(views_[row].logical_positions),
      const_cast<std::int32_t*>(views_[row].sequence_lengths),
      const_cast<std::int32_t*>(views_[row].token_to_request),
      const_cast<std::int32_t*>(views_[row].compression_work),
      target_moe_requested_generation_);
  const cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    faulted_ = true;
    throw std::runtime_error(
        std::string("layer-3 generation CUDA launch: ") +
        cudaGetErrorString(status));
  }
  ++next_row_;
}

}  // namespace rocket::qwen38::decode

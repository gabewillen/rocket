// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_generation.h"

#include <stdexcept>
#include <string>

namespace rocket::qwen38::decode {
namespace {

__global__ void prepare_target_qsa_generation(
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

NativeTargetQsaGenerationOwner::NativeTargetQsaGenerationOwner(
    int rank, int layer, const attention::TargetQsaStateView& storage,
    std::uint64_t* target_moe_requested_generation, int max_rows)
    : rank_(rank), layer_(layer), max_rows_(max_rows), storage_(storage),
      target_moe_requested_generation_(target_moe_requested_generation) {
  if ((rank != 0 && rank != 1) || storage.rank != rank ||
      !is_full_attention_layer(layer) || storage.layer != layer ||
      storage.rows != 1 || storage.uses_mrope || max_rows < 1 || max_rows > 87 ||
      storage.main_kv_dtype != attention::TargetQsaServingDtype::kBfloat16 ||
      storage.side_cache_dtype != attention::TargetQsaServingDtype::kBfloat16 ||
      !complete_storage(storage) || !target_moe_requested_generation)
    throw std::invalid_argument("target QSA generation storage changed");
  authenticated_ = true;
}

const attention::TargetQsaStateView&
NativeTargetQsaGenerationOwner::view(
    int row, std::uint64_t generation) const {
  if (!authenticated_ || faulted_ || row < 0 || row >= max_rows_ ||
      generation == 0 ||
      (last_generation_ != 0 && generation != last_generation_ + 1))
    throw std::logic_error("target QSA generation view changed");
  view_ = storage_;
  view_.generation = generation;
  view_.expected_generation = generation;
  view_pending_ = true;
  pending_row_ = row;
  pending_generation_ = generation;
  return view_;
}

void NativeTargetQsaGenerationOwner::enqueue_prepare(
    int row, std::uint64_t generation, cudaStream_t stream) {
  if (!authenticated_ || faulted_ || !view_pending_ || row < 0 ||
      row >= max_rows_ || row != next_row_ || row != pending_row_ || !stream ||
      generation != pending_generation_) {
    faulted_ = true;
    throw std::logic_error("target QSA generation enqueue changed");
  }
  prepare_target_qsa_generation<<<1, 1, 0, stream>>>(
      row, generation, const_cast<std::int64_t*>(view_.positions),
      const_cast<std::int32_t*>(view_.main_slot_mapping),
      const_cast<std::int32_t*>(view_.raw_slot_mapping),
      const_cast<std::int32_t*>(view_.compressed_slot_mapping),
      const_cast<std::int32_t*>(view_.query_start_locations),
      const_cast<std::int64_t*>(view_.logical_positions),
      const_cast<std::int32_t*>(view_.sequence_lengths),
      const_cast<std::int32_t*>(view_.token_to_request),
      const_cast<std::int32_t*>(view_.compression_work),
      target_moe_requested_generation_);
  const cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    faulted_ = true;
    throw std::runtime_error(
        std::string("target QSA generation CUDA launch: ") +
        cudaGetErrorString(status));
  }
  view_pending_ = false;
  last_generation_ = generation;
  ++next_row_;
}

}  // namespace rocket::qwen38::decode

// SPDX-License-Identifier: Apache-2.0
#include "decode/full_attention_graph.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

namespace rocket::qwen38::decode {
namespace {

constexpr std::array<int, 5> kSequenceBuckets = {1, 2, 4, 8, 16};

bool is_hex(std::string_view value, std::size_t length) noexcept {
  return value.size() == length &&
         std::all_of(value.begin(), value.end(), [](char item) {
           return (item >= '0' && item <= '9') ||
                  (item >= 'a' && item <= 'f');
         });
}

bool extent(const FullAttentionDeviceExtent& value,
            std::uint64_t bytes) noexcept {
  return value.pointer != nullptr && value.bytes == bytes;
}

bool valid_shape(FullAttentionLaunchShape shape) noexcept {
  return std::find(kSequenceBuckets.begin(), kSequenceBuckets.end(),
                   shape.sequences) != kSequenceBuckets.end() &&
         shape.verify_width >= 1 &&
         shape.verify_width <= kFullAttentionMaxVerifyWidth &&
         shape.token_rows == shape.sequences * shape.verify_width &&
         shape.token_rows <= kFullAttentionMaxTokenRows;
}

void require(bool condition, const char* message) {
  if (!condition) throw FullAttentionGraphContractError(message);
}

bool valid_bindings(const FullAttentionDeviceBindings& b) noexcept {
  return
      // ModelOpt NVFP4 W4 payloads and SFB block scales.
      extent(b.q_weight, 7864320) && extent(b.q_scale, 983040) &&
      extent(b.k_weight, 327680) && extent(b.k_scale, 40960) &&
      extent(b.v_weight, 327680) && extent(b.v_scale, 40960) &&
      extent(b.o_weight, 3932160) && extent(b.o_scale, 491520) &&
      extent(b.q_norm, 512) && extent(b.k_norm, 512) &&
      // The published artifact accidentally split this replicated [640,2560]
      // BF16 tensor. Both authenticated 320-row halves are required.
      extent(b.index_qk_weight_first, 1638400) &&
      extent(b.index_qk_weight_second, 1638400) &&
      extent(b.index_q_norm, 256) && extent(b.index_k_norm, 256) &&
      extent(b.main_state, kFullAttentionMainStateBytes) &&
      extent(b.raw_state, kFullAttentionRawStateBytes) &&
      extent(b.compressed_state, kFullAttentionCompressedStateBytes) &&
      extent(b.projected_output, 655360) &&
      extent(b.rope_positions, 3072) &&
      extent(b.logical_positions, 1024) &&
      extent(b.sequence_lengths, 64) &&
      extent(b.token_to_request, 512) &&
      extent(b.query_start_offsets, 68) &&
      b.active_state_generation != nullptr && b.program.context != nullptr &&
      b.program.stage != nullptr && b.program.accept != nullptr &&
      b.program.reset != nullptr && b.program.projected_output != nullptr &&
      b.program.last_error != nullptr;
}

}  // namespace

bool is_full_attention_layer(int layer) noexcept {
  return std::find(kFullAttentionLayers.begin(), kFullAttentionLayers.end(),
                   layer) != kFullAttentionLayers.end();
}

RankLocalFullAttentionGraph::RankLocalFullAttentionGraph(
    FullAttentionIdentity identity, FullAttentionDeviceBindings bindings,
    FullAttentionGraphOtelSink* telemetry)
    : identity_(std::move(identity)), bindings_(bindings), telemetry_(telemetry) {
  require(valid_bindings(bindings_), "exact full-attention device bindings are required");
  require(telemetry_ != nullptr, "bounded full-attention telemetry is required");
  require(is_hex(identity_.revision, 40), "checkpoint revision identity changed");
  require(is_hex(identity_.artifact_key, 64), "rank slab artifact identity changed");
  require(is_hex(identity_.weight_chunk_sha256, 64) &&
              is_hex(identity_.peer_index_chunk_sha256, 64),
          "both authenticated indexer source chunks are required");
  require(identity_.weight_chunk_sha256 != identity_.peer_index_chunk_sha256,
          "indexer reconstruction requires distinct rank chunks");
  require(is_hex(identity_.state_commit_sha256, 64),
          "state commit identity changed");
  require(identity_.rank == 0 || identity_.rank == 1,
          "TP rank must be 0 or 1");
  require(is_full_attention_layer(identity_.layer),
          "layer is not in the fixed Qwen full-attention set");
  require(identity_.state_generation != 0 &&
              *bindings_.active_state_generation == identity_.state_generation,
          "active state generation does not match the binding");
  require(bindings_.program.projected_output(bindings_.program.context) ==
              bindings_.projected_output.pointer,
          "native projected output identity changed");
  state_generation_ = identity_.state_generation;
  emit(FullAttentionGraphStage::kValidate, FullAttentionGraphOutcome::kOk, 0);
}

void RankLocalFullAttentionGraph::launch(
    const __nv_bfloat16* block_input, FullAttentionLaunchShape shape,
    cudaStream_t stream) {
  if (faulted_) throw FullAttentionGraphContractError("full-attention graph is faulted");
  if (staged_) throw FullAttentionGraphContractError("a speculative attention fork is already staged");
  if (!block_input || !stream || !valid_shape(shape) ||
      *bindings_.active_state_generation != state_generation_) {
    emit(FullAttentionGraphStage::kStage, FullAttentionGraphOutcome::kError,
         valid_shape(shape) ? shape.sequences : 0);
    throw FullAttentionGraphContractError("full-attention launch contract changed");
  }
  if (bindings_.program.stage(bindings_.program.context, block_input, shape,
                              stream) != 0) {
    faulted_ = true;
    emit(FullAttentionGraphStage::kStage, FullAttentionGraphOutcome::kError,
         shape.sequences);
    fail_callback("stage");
  }
  if (*bindings_.active_state_generation != state_generation_) {
    faulted_ = true;
    emit(FullAttentionGraphStage::kStage, FullAttentionGraphOutcome::kError,
         shape.sequences);
    throw FullAttentionGraphContractError(
        "full-attention CUDA stage changed active state generation");
  }
  staged_ = true;
  staged_shape_ = shape;
  staged_stream_ = stream;
  emit(FullAttentionGraphStage::kStage, FullAttentionGraphOutcome::kOk,
       shape.sequences);
}

const __nv_bfloat16* RankLocalFullAttentionGraph::projected_output() const noexcept {
  if (faulted_ || !staged_) return nullptr;
  return bindings_.program.projected_output(bindings_.program.context);
}

void RankLocalFullAttentionGraph::accept(
    const std::int32_t* accepted_lengths, int count,
    std::uint64_t next_state_generation, cudaStream_t stream) {
  if (faulted_) throw FullAttentionGraphContractError("full-attention graph is faulted");
  require(staged_ && accepted_lengths && stream &&
              count == staged_shape_.sequences &&
              next_state_generation == state_generation_ + 1 &&
              stream == staged_stream_ &&
              *bindings_.active_state_generation == state_generation_,
          "accepted attention prefix does not match the staged fork");
  for (int index = 0; index < count; ++index) {
    require(accepted_lengths[index] >= 0 &&
                accepted_lengths[index] <= staged_shape_.verify_width,
            "accepted attention prefix exceeds the verify width");
  }
  if (bindings_.program.accept(bindings_.program.context, accepted_lengths,
                               count, next_state_generation, stream) != 0) {
    faulted_ = true;
    emit(FullAttentionGraphStage::kAccept, FullAttentionGraphOutcome::kError,
         staged_shape_.sequences);
    fail_callback("accept");
  }
  if (*bindings_.active_state_generation != next_state_generation) {
    faulted_ = true;
    emit(FullAttentionGraphStage::kAccept, FullAttentionGraphOutcome::kError,
         staged_shape_.sequences);
    throw FullAttentionGraphContractError(
        "full-attention CUDA accept did not publish its state generation");
  }
  state_generation_ = next_state_generation;
  staged_ = false;
  staged_stream_ = nullptr;
  emit(FullAttentionGraphStage::kAccept, FullAttentionGraphOutcome::kOk,
       staged_shape_.sequences);
}

void RankLocalFullAttentionGraph::reset(cudaStream_t stream) {
  if (faulted_) throw FullAttentionGraphContractError("full-attention graph is faulted");
  require(staged_ && stream && stream == staged_stream_ &&
              *bindings_.active_state_generation == state_generation_,
          "attention reset does not match the staged fork");
  if (bindings_.program.reset(bindings_.program.context, stream) != 0) {
    faulted_ = true;
    emit(FullAttentionGraphStage::kReset, FullAttentionGraphOutcome::kError,
         staged_shape_.sequences);
    fail_callback("reset");
  }
  if (*bindings_.active_state_generation != state_generation_) {
    faulted_ = true;
    emit(FullAttentionGraphStage::kReset, FullAttentionGraphOutcome::kError,
         staged_shape_.sequences);
    throw FullAttentionGraphContractError(
        "full-attention CUDA reset changed active state generation");
  }
  staged_ = false;
  staged_stream_ = nullptr;
  emit(FullAttentionGraphStage::kReset, FullAttentionGraphOutcome::kOk,
       staged_shape_.sequences);
}

void RankLocalFullAttentionGraph::emit(
    FullAttentionGraphStage stage, FullAttentionGraphOutcome outcome,
    int m) noexcept {
  telemetry_->emit({identity_.rank, identity_.layer, m, stage, outcome});
}

[[noreturn]] void RankLocalFullAttentionGraph::fail_callback(
    const char* operation) {
  const char* detail = bindings_.program.last_error(bindings_.program.context);
  throw FullAttentionGraphContractError(
      std::string("full-attention CUDA ") + operation + " failed: " +
      (detail && *detail ? detail : "unknown failure"));
}

}  // namespace rocket::qwen38::decode

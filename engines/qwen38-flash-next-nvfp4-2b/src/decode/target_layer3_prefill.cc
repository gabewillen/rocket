// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_prefill.h"

#include <stdexcept>

namespace rocket::qwen38::decode {
namespace {
constexpr std::size_t kHcWidth =
    kTargetLayer3HiddenStreams * pair_reduce::kHidden;
}

TargetFullLayerResult NativeTargetLayer3RowPort::execute_row(
    std::uint64_t generation, const attention::TargetQsaStateView& state,
    const __nv_bfloat16* replicated_pre_layer,
    const TargetLayer3RowBuffers& b, __nv_bfloat16* replicated_post_layer,
    cudaStream_t stream) {
  return layer_.execute(
      generation, state, replicated_pre_layer, b.attention_input,
      b.attention_injection, b.reduced_attention, b.post_attention_hidden,
      b.moe_input, b.moe_injection, b.reduced_moe, replicated_post_layer,
      "layer3-prefill", "oracle-05ea3af", stream);
}

TargetLayer3Prefill::TargetLayer3Prefill(
    int rank, TargetLayer3RowPort& rows,
    TargetLayer3GenerationOwner& generations,
    TargetLayer3Comparator& comparator,
    pair_reduce::OtelStageSink& telemetry)
    : rank_(rank), rows_(rows), generations_(generations),
      comparator_(comparator), telemetry_(telemetry) {
  if ((rank != 0 && rank != 1) || rows.rank() != rank || rows.layer() != 3 ||
      !rows.authenticated() || generations.rank() != rank ||
      generations.layer() != 3 || !generations.authenticated() ||
      !comparator.authenticated()) {
    const int diagnostic_rank = (rank_ == 0 || rank_ == 1) ? rank_ : -1;
    telemetry_.emit_span_and_log({
        "rocket.qwen38.layer3_prefill.lifecycle", "layer3-prefill",
        "oracle-05ea3af", diagnostic_rank, 1, pair_reduce::kDtype,
        pair_reduce::Outcome::kContractError, 0, 0});
    if (diagnostic_rank >= 0)
      telemetry_.record_duration({diagnostic_rank, 1, pair_reduce::kDtype,
                                  pair_reduce::Outcome::kContractError, 0});
    throw std::invalid_argument("layer-3 prefill rank changed");
  }
}

const __nv_bfloat16* TargetLayer3Prefill::execute(
    const __nv_bfloat16* replicated_layer02,
    __nv_bfloat16* replicated_layer03,
    const TargetLayer3RowBuffers& b, cudaStream_t stream) {
  const auto emit = [&](pair_reduce::Outcome outcome, std::uint64_t bytes) {
    telemetry_.emit_span_and_log({
        "rocket.qwen38.layer3_prefill.lifecycle", "layer3-prefill",
        "oracle-05ea3af", rank_, 1, pair_reduce::kDtype, outcome, 0, bytes});
    telemetry_.record_duration({rank_, 1, pair_reduce::kDtype, outcome, 0});
  };
  try {
    if (faulted_ || completed_ || !replicated_layer02 || !replicated_layer03 ||
        !b.attention_input || !b.attention_injection ||
        !b.reduced_attention || !b.post_attention_hidden || !b.moe_input ||
        !b.moe_injection || !b.reduced_moe || !stream)
      throw std::invalid_argument("layer-3 prefill buffers changed");
    for (int row = 0; row < kTargetLayer3OracleRows; ++row) {
      const std::uint64_t generation = static_cast<std::uint64_t>(row + 1);
      const auto& state = generations_.view(row, generation);
      if (state.rank != rank_ || state.layer != 3 || state.rows != 1 ||
          state.generation != generation ||
          state.expected_generation != generation)
        throw std::logic_error("layer-3 generation owner changed");
      generations_.enqueue_prepare(row, generation, stream);
      const auto result = rows_.execute_row(
          generation, state, replicated_layer02 + row * kHcWidth, b,
          replicated_layer03 + row * kHcWidth, stream);
      if (result.generation != generation || result.rank != rank_ ||
          result.layer != 3 ||
          result.post_layer != replicated_layer03 + row * kHcWidth)
        throw std::logic_error("layer-3 row publication changed");
    }
    const auto* last =
        replicated_layer03 + (kTargetLayer3OracleRows - 1) * kHcWidth;
    if (!comparator_.compare_row34(last, stream))
      throw std::logic_error("layer-3 row-34 oracle mismatch");
  } catch (const std::invalid_argument&) {
    faulted_ = true;
    emit(pair_reduce::Outcome::kContractError, 0);
    throw;
  } catch (const std::logic_error&) {
    faulted_ = true;
    emit(pair_reduce::Outcome::kContractError, 0);
    throw;
  } catch (...) {
    faulted_ = true;
    emit(pair_reduce::Outcome::kCudaError, 0);
    throw;
  }
  completed_ = true;
  emit(pair_reduce::Outcome::kOk,
       2 * kTargetLayer3OracleRows * kHcWidth * sizeof(__nv_bfloat16));
  return replicated_layer03 + (kTargetLayer3OracleRows - 1) * kHcWidth;
}

}  // namespace rocket::qwen38::decode

// SPDX-License-Identifier: Apache-2.0
#include "decode/linear_attention_layer.h"

#include <array>
#include <chrono>
#include <stdexcept>
#include <string>

namespace rocket::qwen38::decode {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::array<int, 5> kBuckets = {1, 2, 4, 8, 16};
constexpr std::uint64_t kHiddenBytesPerRow = 2 * kLinearHidden;
constexpr std::uint64_t kHyperBytesPerRow = 4 * kHiddenBytesPerRow;

std::uint64_t elapsed_ns(Clock::time_point start) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

void require(bool condition, int rank, int layer, const char* message) {
  if (!condition) {
    throw DecodeExecutionContractError(
        "qwen38 rank " + std::to_string(rank) + " layer " +
        std::to_string(layer) + " linear attention: " + message);
  }
}

std::string stage_name(int layer, std::string_view stage) {
  return "rocket.qwen38.layer" + std::to_string(layer) + ".linear." +
         std::string(stage);
}

}  // namespace

LinearAttentionLayer::LinearAttentionLayer(
    LinearAttentionGraph& graph, HiddenPartialReducer& reducer,
    LinearAttentionHyperConnection& hyperconnection,
    pair_reduce::OtelStageSink& telemetry)
    : graph_(graph), reducer_(reducer), hyperconnection_(hyperconnection),
      telemetry_(telemetry), rank_(graph.rank()), layer_(graph.layer()) {
  require((rank_ == 0 || rank_ == 1) && is_linear_attention_layer(layer_),
          rank_, layer_, "graph must bind a TP2 GDN layer");
  const std::string expected_slab =
      rank_ == 0 ? "rank0-target" : "rank1-target";
  require(graph_.checkpoint_revision() == kQwen38CheckpointRevision &&
              graph_.slab_key() == expected_slab,
          rank_, layer_, "graph checkpoint or slab identity changed");
  require(graph_.conv_state_family() == "target_gdn_conv" &&
              graph_.recurrent_state_family() == "target_gdn_recurrent",
          rank_, layer_, "graph state-family identity changed");
  for (const int m : kBuckets) {
    require(graph_.has_captured_bucket(m),
            rank_, layer_, "all immutable M buckets must be captured");
    require(graph_.logical_bytes_per_row(m) > 0,
            rank_, layer_, "captured bucket must report bounded traffic");
  }
  require(reducer_.rank() == rank_ && reducer_.world_size() == 2,
          rank_, layer_, "reducer must bind the same TP2 rank");
}

LinearAttentionResult LinearAttentionLayer::execute(
    std::uint64_t generation, int m, const __nv_bfloat16* hidden,
    __nv_bfloat16* block_input, __nv_bfloat16* injection,
    __nv_bfloat16* conv_state, float* recurrent_state,
    const std::int32_t* state_indices, float* reduced_attention,
    __nv_bfloat16* updated_hidden, __nv_bfloat16* next_block_input,
    __nv_bfloat16* next_injection, std::string_view trace_id,
    std::string_view request_id, cudaStream_t stream,
    TargetK0ExecutionProgress* progress) {
  const auto lifecycle_start = Clock::now();
  try {
    require(!faulted_, rank_, layer_, "faulted transition cannot be retried");
    require(generation != 0 && generation == last_generation_ + 1,
            rank_, layer_, "generation must increase by one");
    require(allowed_linear_m(m), rank_, layer_,
            "M must be one of 1,2,4,8,16");
    require(hidden && block_input && injection && conv_state &&
                recurrent_state && state_indices && reduced_attention &&
                updated_hidden && next_block_input && next_injection && stream,
            rank_, layer_, "all borrowed buffers and stream are required");

    auto start = Clock::now();
    target_k0_enter_layer(progress,
                          TargetK0LayerExecutionStage::kAttentionHyperconnection);
    hyperconnection_.mix(hidden, block_input, injection, m, stream);
    hyperconnection_.synchronize(stream);
    emit(stage_name(layer_, "attn_hc_mix"),
         pair_reduce::Outcome::kOk, m, trace_id, request_id,
         elapsed_ns(start), kHyperBytesPerRow * m);

    start = Clock::now();
    target_k0_enter_layer(progress, TargetK0LayerExecutionStage::kAttention);
    graph_.launch(block_input, conv_state, recurrent_state, state_indices, m,
                  stream);
    hyperconnection_.synchronize(stream);
    const __nv_bfloat16* partial = graph_.projected_output();
    require(partial != nullptr, rank_, layer_,
            "GDN graph returned no projected output");
    emit(stage_name(layer_, "gdn_graph"),
         pair_reduce::Outcome::kOk, m, trace_id, request_id,
         elapsed_ns(start), graph_.logical_bytes_per_row(m) * m);

    start = Clock::now();
    target_k0_enter_layer(progress,
                          TargetK0LayerExecutionStage::kAttentionReduction);
    reducer_.reduce(partial, reduced_attention, m, trace_id, request_id,
                    stream);
    emit(stage_name(layer_, "pair_reduce"),
         pair_reduce::Outcome::kOk, m, trace_id, request_id,
         elapsed_ns(start), kHiddenBytesPerRow * m);

    start = Clock::now();
    target_k0_enter_layer(progress,
                          TargetK0LayerExecutionStage::kMlpHyperconnection);
    hyperconnection_.combine_and_mix(
        hidden, reduced_attention, injection, updated_hidden,
        next_block_input, next_injection, m, stream);
    hyperconnection_.synchronize(stream);
    emit(stage_name(layer_, "mlp_hc_combine_mix"),
         pair_reduce::Outcome::kOk, m, trace_id, request_id,
         elapsed_ns(start), kHyperBytesPerRow * m);
  } catch (...) {
    faulted_ = true;
    emit(stage_name(layer_, "lifecycle"),
         pair_reduce::Outcome::kTransportError, m, trace_id, request_id,
         elapsed_ns(lifecycle_start), 0);
    throw;
  }
  last_generation_ = generation;
  emit(stage_name(layer_, "lifecycle"), pair_reduce::Outcome::kOk, m,
       trace_id, request_id, elapsed_ns(lifecycle_start), 0);
  return {generation, m, rank_, layer_};
}

void LinearAttentionLayer::emit(
    std::string_view stage, pair_reduce::Outcome outcome, int m,
    std::string_view trace_id, std::string_view request_id,
    std::uint64_t duration_ns, std::uint64_t bytes) noexcept {
  telemetry_.emit_span_and_log({stage, trace_id, request_id, rank_,
                                allowed_linear_m(m) ? m : 0,
                                pair_reduce::kDtype, outcome, duration_ns,
                                bytes});
}

}  // namespace rocket::qwen38::decode

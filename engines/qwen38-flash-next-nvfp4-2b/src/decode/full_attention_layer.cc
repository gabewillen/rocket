// SPDX-License-Identifier: Apache-2.0
#include "decode/full_attention_layer.h"

#include <chrono>
#include <stdexcept>
#include <string>

namespace rocket::qwen38::decode {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kHiddenBytesPerRow = 2 * pair_reduce::kHidden;
constexpr std::uint64_t kHyperBytesPerRow = 4 * kHiddenBytesPerRow;

std::uint64_t elapsed_ns(Clock::time_point start) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

void require(bool condition, const char* message) {
  if (!condition) throw DecodeExecutionContractError(
      std::string("qwen38 layer-3 full attention: ") + message);
}

}  // namespace

FullAttentionLayer::FullAttentionLayer(
    FullAttentionGraph& graph, HiddenPartialReducer& reducer,
    FullAttentionHyperConnection& hyperconnection,
    pair_reduce::OtelStageSink& telemetry)
    : graph_(graph), reducer_(reducer), hyperconnection_(hyperconnection),
      telemetry_(telemetry) {
  require(graph_.rank() == 0 && graph_.layer() == 3,
          "graph must bind authenticated rank 0 layer 3");
  require(reducer_.rank() == 0 && reducer_.world_size() == 2,
          "reducer must bind rank 0 of TP2");
}

FullAttentionResult FullAttentionLayer::execute(
    std::uint64_t generation, int m, const __nv_bfloat16* hidden,
    __nv_bfloat16* block_input, __nv_bfloat16* injection,
    float* reduced_attention, __nv_bfloat16* updated_hidden,
    __nv_bfloat16* next_block_input, __nv_bfloat16* next_injection,
    std::string_view trace_id, std::string_view request_id,
    cudaStream_t stream) {
  const auto lifecycle_start = Clock::now();
  try {
    require(!faulted_, "faulted transition cannot be retried");
    require(generation != 0 && generation == last_generation_ + 1,
            "generation must increase by one");
    require(pair_reduce::allowed_m(m), "M must be one of 1,2,4,8,16");
    require(hidden && block_input && injection && reduced_attention &&
                updated_hidden && next_block_input && next_injection && stream,
            "all borrowed buffers and stream are required");

    auto start = Clock::now();
    hyperconnection_.mix(hidden, block_input, injection, m, stream);
    hyperconnection_.synchronize(stream);
    emit("rocket.qwen38.layer3.attn_hc_mix", pair_reduce::Outcome::kOk, m,
         trace_id, request_id, elapsed_ns(start), kHyperBytesPerRow * m);

    start = Clock::now();
    graph_.launch(block_input, m, stream);
    hyperconnection_.synchronize(stream);
    const __nv_bfloat16* partial = graph_.projected_output();
    require(partial != nullptr, "attention graph returned no projected output");
    emit("rocket.qwen38.layer3.qsa_attention", pair_reduce::Outcome::kOk, m,
         trace_id, request_id, elapsed_ns(start), kHiddenBytesPerRow * m);

    start = Clock::now();
    reducer_.reduce(partial, reduced_attention, m, trace_id, request_id, stream);
    emit("rocket.qwen38.layer3.pair_reduce", pair_reduce::Outcome::kOk, m,
         trace_id, request_id, elapsed_ns(start), kHiddenBytesPerRow * m);

    start = Clock::now();
    hyperconnection_.combine_and_mix(
        hidden, reduced_attention, injection, updated_hidden, next_block_input,
        next_injection, m, stream);
    hyperconnection_.synchronize(stream);
    emit("rocket.qwen38.layer3.mlp_hc_combine_mix", pair_reduce::Outcome::kOk,
         m, trace_id, request_id, elapsed_ns(start), kHyperBytesPerRow * m);
  } catch (...) {
    faulted_ = true;
    emit("rocket.qwen38.layer3.lifecycle",
         pair_reduce::Outcome::kTransportError, m, trace_id, request_id,
         elapsed_ns(lifecycle_start), 0);
    throw;
  }
  last_generation_ = generation;
  emit("rocket.qwen38.layer3.lifecycle", pair_reduce::Outcome::kOk, m,
       trace_id, request_id, elapsed_ns(lifecycle_start), 0);
  return {generation, m};
}

void FullAttentionLayer::emit(
    std::string_view stage, pair_reduce::Outcome outcome, int m,
    std::string_view trace_id, std::string_view request_id,
    std::uint64_t duration_ns, std::uint64_t bytes) noexcept {
  telemetry_.emit_span_and_log({stage, trace_id, request_id, 0,
                                pair_reduce::allowed_m(m) ? m : 0,
                                pair_reduce::kDtype, outcome, duration_ns,
                                bytes});
}

}  // namespace rocket::qwen38::decode

// SPDX-License-Identifier: Apache-2.0
#include "decode/target_full_layer.h"

#include <chrono>
#include <stdexcept>
#include <string>

namespace rocket::qwen38::decode {
namespace {
using Clock = std::chrono::steady_clock;
constexpr int kLayer = 3;
constexpr std::uint64_t kHiddenBytes = 2 * pair_reduce::kHidden;
constexpr std::uint64_t kHyperBytes = 4 * kHiddenBytes;

std::uint64_t elapsed_ns(Clock::time_point start) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

void require(bool condition, int rank, const char* message) {
  if (!condition)
    throw DecodeExecutionContractError(
        "qwen38 rank " + std::to_string(rank) +
        " layer 3 K0 comparator: " + message);
}

std::string stage(std::string_view name) {
  return "rocket.qwen38.layer3.composite." + std::string(name);
}
}  // namespace

TargetFullLayer::TargetFullLayer(
    FullAttentionGraph& attention, TargetMoeGraph& moe,
    HiddenPartialReducer& attention_reducer, HiddenPartialReducer& moe_reducer,
    FullAttentionHyperConnection& hyperconnection,
    pair_reduce::OtelStageSink& telemetry)
    : attention_(attention), moe_(moe),
      attention_reducer_(attention_reducer), moe_reducer_(moe_reducer),
      hyperconnection_(hyperconnection), telemetry_(telemetry),
      rank_(attention.rank()) {
  const std::string expected = rank_ == 0 ? "rank0-target" : "rank1-target";
  require((rank_ == 0 || rank_ == 1) && attention.layer() == kLayer &&
              moe.rank() == rank_ && moe.layer() == kLayer,
          rank_, "rank/layer participants changed");
  require(attention.checkpoint_revision() == kFullAttentionCheckpointRevision &&
              moe.checkpoint_revision() == kFullAttentionCheckpointRevision &&
              attention.slab_key() == expected && moe.slab_key() == expected,
          rank_, "checkpoint or target slab identity changed");
  require(attention_reducer.rank() == rank_ &&
              attention_reducer.world_size() == 2 &&
              moe_reducer.rank() == rank_ && moe_reducer.world_size() == 2,
          rank_, "both PairReduce participants must bind the same TP2 rank");
}

TargetFullLayerResult TargetFullLayer::execute(
    std::uint64_t generation, const attention::TargetQsaStateView& qsa_state,
    const __nv_bfloat16* materialized_pre_layer,
    __nv_bfloat16* attention_input, __nv_bfloat16* attention_injection,
    float* reduced_attention, __nv_bfloat16* post_attention_hidden,
    __nv_bfloat16* moe_input, __nv_bfloat16* moe_injection,
    float* reduced_moe, __nv_bfloat16* post_layer,
    std::string_view trace_id, std::string_view request_id,
    cudaStream_t stream) {
  const auto lifecycle = Clock::now();
  try {
    require(!faulted_, rank_, "faulted transition cannot replay");
    require(generation != 0 && generation == last_generation_ + 1, rank_,
            "generation must increase by one");
    attention::validate_target_qsa_state_view(qsa_state, rank_, kLayer,
                                               generation);
    require(materialized_pre_layer && attention_input && attention_injection &&
                reduced_attention && post_attention_hidden && moe_input &&
                moe_injection && reduced_moe && post_layer && stream,
            rank_, "all caller-owned c1 buffers and stream are required");

    auto start = Clock::now();
    hyperconnection_.mix(materialized_pre_layer, attention_input,
                         attention_injection, 1, stream);
    hyperconnection_.synchronize(stream);
    emit("attn_hc_mix", pair_reduce::Outcome::kOk, trace_id, request_id,
         elapsed_ns(start), kHyperBytes);

    start = Clock::now();
    attention_.launch(attention_input, qsa_state, generation, 1, stream);
    hyperconnection_.synchronize(stream);
    require(attention_.projected_output(), rank_,
            "QSA participant published no rank-local partial");
    emit("qsa", pair_reduce::Outcome::kOk, trace_id, request_id,
         elapsed_ns(start), kHiddenBytes);

    start = Clock::now();
    attention_reducer_.reduce(attention_.projected_output(), reduced_attention,
                              1, trace_id, request_id, stream);
    emit("attention_pair_reduce", pair_reduce::Outcome::kOk, trace_id,
         request_id, elapsed_ns(start), kHiddenBytes);

    start = Clock::now();
    hyperconnection_.combine_and_mix(
        materialized_pre_layer, reduced_attention, attention_injection,
        post_attention_hidden, moe_input, moe_injection, 1, stream);
    hyperconnection_.synchronize(stream);
    emit("mlp_hc_mix", pair_reduce::Outcome::kOk, trace_id, request_id,
         elapsed_ns(start), kHyperBytes);

    start = Clock::now();
    moe_.launch(moe_input, generation, 1, stream);
    hyperconnection_.synchronize(stream);
    require(moe_.projected_output(), rank_,
            "target MoE participant published no rank-local partial");
    emit("target_moe", pair_reduce::Outcome::kOk, trace_id, request_id,
         elapsed_ns(start), kHiddenBytes);

    start = Clock::now();
    moe_reducer_.reduce(moe_.projected_output(), reduced_moe, 1, trace_id,
                        request_id, stream);
    emit("moe_pair_reduce", pair_reduce::Outcome::kOk, trace_id, request_id,
         elapsed_ns(start), kHiddenBytes);

    start = Clock::now();
    hyperconnection_.combine(post_attention_hidden, reduced_moe,
                             moe_injection, post_layer, 1, stream);
    hyperconnection_.synchronize(stream);
    emit("post_layer", pair_reduce::Outcome::kOk, trace_id, request_id,
         elapsed_ns(start), kHyperBytes);
  } catch (...) {
    faulted_ = true;
    emit("lifecycle", pair_reduce::Outcome::kCudaError, trace_id, request_id,
         elapsed_ns(lifecycle), 0);
    throw;
  }
  last_generation_ = generation;
  emit("lifecycle", pair_reduce::Outcome::kOk, trace_id, request_id,
       elapsed_ns(lifecycle), 0);
  return {generation, rank_, kLayer, post_layer};
}

void TargetFullLayer::emit(
    std::string_view name, pair_reduce::Outcome outcome,
    std::string_view trace_id, std::string_view request_id,
    std::uint64_t duration_ns, std::uint64_t bytes) noexcept {
  telemetry_.emit_span_and_log({stage(name), trace_id, request_id, rank_, 1,
                                pair_reduce::kDtype, outcome, duration_ns,
                                bytes});
}

}  // namespace rocket::qwen38::decode

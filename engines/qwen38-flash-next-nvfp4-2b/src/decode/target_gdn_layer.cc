// SPDX-License-Identifier: Apache-2.0
#include "decode/target_gdn_layer.h"

#include <stdexcept>
#include <string>

namespace rocket::qwen38::decode {
namespace {

void require(bool condition, int rank, int layer, const char* message) {
  if (!condition)
    throw DecodeExecutionContractError(
        "qwen38 rank " + std::to_string(rank) + " layer " +
        std::to_string(layer) + " K0 GDN: " + message);
}

}  // namespace

TargetGdnLayer::TargetGdnLayer(
    LinearAttentionGraph& attention, TargetMoeGraph& moe,
    HiddenPartialReducer& attention_reducer, HiddenPartialReducer& moe_reducer,
    LinearAttentionHyperConnection& hyperconnection,
    TargetGdnMoeGenerationPort& moe_generation,
    pair_reduce::OtelStageSink& telemetry, TargetGdnC1State state,
    TargetGdnC1Buffers buffers)
    : attention_(attention, attention_reducer, hyperconnection, telemetry),
      moe_(moe), attention_reducer_(attention_reducer),
      moe_reducer_(moe_reducer), hyperconnection_(hyperconnection),
      moe_generation_(moe_generation), telemetry_(telemetry), state_(state),
      buffers_(buffers), rank_(attention.rank()), layer_(attention.layer()) {
  const std::string_view expected_slab =
      rank_ == 0 ? "rank0-target" : "rank1-target";
  require((rank_ == 0 || rank_ == 1) && is_linear_attention_layer(layer_) &&
              moe_.rank() == rank_ && moe_.layer() == layer_ &&
              moe_.checkpoint_revision() == kQwen38CheckpointRevision &&
              moe_.slab_key() == expected_slab,
          rank_, layer_, "rank, layer, or slab participant changed");
  require(attention_reducer_.rank() == rank_ &&
              attention_reducer_.world_size() == 2 &&
              moe_reducer_.rank() == rank_ && moe_reducer_.world_size() == 2 &&
              &attention_reducer_ != &moe_reducer_,
          rank_, layer_, "PairReduce identities changed");
  require(moe_generation_.rank() == rank_ &&
              moe_generation_.layer() == layer_ &&
              moe_generation_.authenticated(),
          rank_, layer_, "MoE generation owner changed");
  require(valid_target_gdn_c1_state_extent(state_) &&
              complete_target_gdn_c1_buffers(buffers_),
          rank_, layer_, "state or scratch extent changed");
}

TargetFullLayerResult TargetGdnLayer::execute(
    std::uint64_t generation, const __nv_bfloat16* replicated_pre_layer,
    __nv_bfloat16* replicated_post_layer, std::string_view trace_id,
    std::string_view request_id, cudaStream_t stream,
    TargetK0ExecutionProgress* progress) {
  bool moe_needs_failure_fence = false;
  const auto fail = [&](pair_reduce::Outcome outcome) noexcept {
    if (moe_needs_failure_fence) {
      try {
        hyperconnection_.synchronize(stream);
      } catch (...) {
        outcome = pair_reduce::Outcome::kCudaError;
      }
      moe_.fault_after_fence(generation);
    }
    faulted_ = true;
    emit(outcome, trace_id, request_id);
  };
  try {
    require(!faulted_, rank_, layer_, "faulted transition cannot replay");
    require(generation != 0 && generation == last_generation_ + 1 &&
                replicated_pre_layer && replicated_post_layer && stream &&
                trace_id.size() <= 128 && request_id.size() <= 128,
            rank_, layer_, "generation, buffers, or stream changed");
    attention_.execute(
        generation, 1, replicated_pre_layer, buffers_.attention_input,
        buffers_.attention_injection, state_.convolution, state_.recurrent,
        state_.state_index, buffers_.reduced_attention,
        buffers_.post_attention_hidden, buffers_.moe_input,
        buffers_.moe_injection, trace_id, request_id, stream, progress);

    target_k0_enter_layer(progress, TargetK0LayerExecutionStage::kMoe);
    moe_generation_.enqueue(generation, stream);
    moe_needs_failure_fence = true;
    moe_.launch(buffers_.moe_input, generation, 1, stream);
    hyperconnection_.synchronize(stream);
    moe_needs_failure_fence = false;
    moe_.terminal_fence_succeeded(generation);
    moe_.publish_after_fence(generation);
    require(moe_.projected_output(), rank_, layer_,
            "target MoE published no rank-local partial");
    target_k0_observe_layer0(
        progress, TargetK0LayerBoundary::kMoeOutput, moe_.projected_output(),
        kLinearHidden, TargetK0DiagnosticDtype::kBfloat16, stream);
    target_k0_enter_layer(progress, TargetK0LayerExecutionStage::kMoeReduction);
    moe_reducer_.reduce(moe_.projected_output(), buffers_.reduced_moe, 1,
                        trace_id, request_id, stream);
    target_k0_observe_layer0(
        progress, TargetK0LayerBoundary::kMoeReduction, buffers_.reduced_moe,
        kLinearHidden, TargetK0DiagnosticDtype::kFloat32, stream);
    target_k0_enter_layer(progress,
                          TargetK0LayerExecutionStage::kFinalHyperconnection);
    hyperconnection_.combine(
        buffers_.post_attention_hidden, buffers_.reduced_moe,
        buffers_.moe_injection, replicated_post_layer, 1, stream);
    hyperconnection_.synchronize(stream);
    target_k0_observe_layer0(
        progress, TargetK0LayerBoundary::kFinalHyperconnection,
        replicated_post_layer, 4 * kLinearHidden,
        TargetK0DiagnosticDtype::kBfloat16, stream);
  } catch (const pair_reduce::PairReduceContractError&) {
    fail(pair_reduce::Outcome::kContractError);
    throw;
  } catch (const DecodeExecutionContractError&) {
    fail(pair_reduce::Outcome::kContractError);
    throw;
  } catch (const std::invalid_argument&) {
    fail(pair_reduce::Outcome::kContractError);
    throw;
  } catch (const std::logic_error&) {
    fail(pair_reduce::Outcome::kContractError);
    throw;
  } catch (const pair_reduce::PairReduceTransportError&) {
    fail(pair_reduce::Outcome::kTransportError);
    throw;
  } catch (const DecodeExecutionTransportError&) {
    fail(pair_reduce::Outcome::kTransportError);
    throw;
  } catch (const pair_reduce::PairReduceCudaError&) {
    fail(pair_reduce::Outcome::kCudaError);
    throw;
  } catch (const DecodeExecutionCudaError&) {
    fail(pair_reduce::Outcome::kCudaError);
    throw;
  } catch (...) {
    fail(pair_reduce::Outcome::kCudaError);
    throw;
  }
  last_generation_ = generation;
  emit(pair_reduce::Outcome::kOk, trace_id, request_id);
  return {generation, rank_, layer_, replicated_post_layer};
}

void TargetGdnLayer::emit(pair_reduce::Outcome outcome,
                          std::string_view trace_id,
                          std::string_view request_id) noexcept {
  telemetry_.emit_span_and_log({
      "rocket.qwen38.k0.gdn_layer.lifecycle", trace_id, request_id, rank_, 1,
      pair_reduce::kDtype, outcome, 0, 0});
  telemetry_.record_duration(
      {rank_, 1, pair_reduce::kDtype, outcome, 0});
}

}  // namespace rocket::qwen38::decode

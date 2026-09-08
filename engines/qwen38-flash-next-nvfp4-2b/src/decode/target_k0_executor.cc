// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_executor.h"

#include <stdexcept>

namespace rocket::qwen38::decode {
namespace {

TargetK0AttentionKind expected_attention(int layer) noexcept {
  return is_qsa_layer(layer) ? TargetK0AttentionKind::kQsa
                             : TargetK0AttentionKind::kGdn;
}

}  // namespace

TargetK0Executor::TargetK0Executor(
    int rank, std::array<TargetK0LayerPort*, kDecoderLayers> layers,
    TargetK0TokenIoPort& token_io, TargetK0PairReduceSchedule& reductions,
    TargetK0OracleComparator& comparator,
    pair_reduce::OtelStageSink& telemetry, TargetK0ExecutorArena arena,
    cudaStream_t stream)
    : rank_(rank), layers_(layers), token_io_(token_io),
      reductions_(reductions), comparator_(comparator), telemetry_(telemetry),
      arena_(arena), stream_(stream) {
  if ((rank_ != 0 && rank_ != 1) || token_io_.rank() != rank_ ||
      !token_io_.authenticated() || reductions_.rank() != rank_ ||
      comparator_.rank() != rank_ || !comparator_.authenticated() ||
      !accepted_target_k0_oracle(comparator_.manifest_sha256(),
                                 comparator_.rows()) ||
      !arena_.hidden_a || !arena_.hidden_b ||
      arena_.hidden_a == arena_.hidden_b || !stream_)
    throw std::invalid_argument("K0 executor ownership changed");
  for (int layer = 0; layer < kDecoderLayers; ++layer) {
    const auto* port = layers_[layer];
    if (!port || port->rank() != rank_ || port->layer() != layer ||
        port->attention_kind() != expected_attention(layer) ||
        !port->authenticated() ||
        port->attention_reducer_identity() !=
            &reductions_.attention_port(layer) ||
        port->moe_reducer_identity() != &reductions_.moe_port(layer))
      throw std::invalid_argument("K0 layer ownership changed");
  }
}

TargetK0ExecutionResult TargetK0Executor::execute_prefill(
    std::uint64_t first_generation,
    std::span<const std::int32_t> prompt_tokens,
    std::string_view trace_id, std::string_view request_id) {
  if (phase_ != TargetK0ExecutorPhase::kReady || first_generation == 0 ||
      prompt_tokens.size() != static_cast<std::size_t>(comparator_.rows()) ||
      trace_id.size() > 128 || request_id.size() > 128)
    throw std::invalid_argument("K0 prefill request changed");
  for (const auto token : prompt_tokens)
    if (token < 0 || token >= 248'320)
      throw std::invalid_argument("K0 prompt token changed");

  phase_ = TargetK0ExecutorPhase::kActive;
  std::uint64_t bytes = 0;
  try {
    token_io_.wait_source(stream_);
    for (auto* layer : layers_) layer->wait_source(stream_);
    reductions_.begin_sequence(first_generation,
                               static_cast<int>(prompt_tokens.size()));
    for (std::size_t row = 0; row < prompt_tokens.size(); ++row) {
      const auto generation =
          first_generation + static_cast<std::uint64_t>(row);
      reductions_.begin_row(static_cast<int>(row), generation);
      token_io_.embed_row(prompt_tokens[row], generation, arena_.hidden_a,
                          stream_);
      comparator_.compare(TargetK0Boundary::kEmbedding,
                          static_cast<int>(row), -1, arena_.hidden_a,
                          kTargetK0Hidden, stream_);
      bytes += kTargetK0Hidden * sizeof(__nv_bfloat16);

      const __nv_bfloat16* input = arena_.hidden_a;
      __nv_bfloat16* output = arena_.hidden_b;
      for (int layer = 0; layer < kDecoderLayers; ++layer) {
        layers_[layer]->execute_row(generation, input, output, stream_);
        comparator_.compare(TargetK0Boundary::kLayer,
                            static_cast<int>(row), layer, output,
                            kTargetK0HyperHidden, stream_);
        bytes += kTargetK0HyperHidden * sizeof(__nv_bfloat16);
        input = output;
        output = output == arena_.hidden_a ? arena_.hidden_b : arena_.hidden_a;
      }
      // All 48 layers is even, so the last publication must be hidden_a.
      if (input != arena_.hidden_a)
        throw std::logic_error("K0 layer ping-pong changed");
    }
    if (!reductions_.complete())
      throw std::logic_error("K0 reduction sequence incomplete");

    const auto final_generation =
        first_generation + prompt_tokens.size() - 1;
    const auto output =
        token_io_.finish_prefill(arena_.hidden_a, final_generation, stream_);
    if (!output.final_hidden_bf16 || !output.local_logits ||
        output.global_token < 0 ||
        output.global_token >= 248'320)
      throw std::logic_error("K0 token output changed");
    comparator_.compare(TargetK0Boundary::kFinalNorm,
                        static_cast<int>(prompt_tokens.size() - 1), -1,
                        output.final_hidden_bf16, kTargetK0Hidden, stream_);
    comparator_.compare(TargetK0Boundary::kLocalLogits,
                        static_cast<int>(prompt_tokens.size() - 1), -1,
                        output.local_logits, kTargetK0LocalVocab, stream_);
    comparator_.compare_token(output.global_token);
    bytes += kTargetK0Hidden * sizeof(__nv_bfloat16) +
             kTargetK0LocalVocab * sizeof(float);
    phase_ = TargetK0ExecutorPhase::kCompleted;
    emit(pair_reduce::Outcome::kOk, trace_id, request_id, bytes);
    return {output.global_token, static_cast<int>(prompt_tokens.size()),
            final_generation};
  } catch (const std::invalid_argument&) {
    phase_ = TargetK0ExecutorPhase::kFaulted;
    emit(pair_reduce::Outcome::kContractError, trace_id, request_id, bytes);
    throw;
  } catch (const pair_reduce::PairReduceTransportError&) {
    phase_ = TargetK0ExecutorPhase::kFaulted;
    emit(pair_reduce::Outcome::kTransportError, trace_id, request_id, bytes);
    throw;
  } catch (...) {
    phase_ = TargetK0ExecutorPhase::kFaulted;
    emit(pair_reduce::Outcome::kCudaError, trace_id, request_id, bytes);
    throw;
  }
}

void TargetK0Executor::emit(pair_reduce::Outcome outcome,
                            std::string_view trace_id,
                            std::string_view request_id,
                            std::uint64_t bytes) noexcept {
  telemetry_.emit_span_and_log({"rocket.qwen38.k0_executor.lifecycle", trace_id,
                                request_id, rank_, 1, pair_reduce::kDtype,
                                outcome, 0, bytes});
  telemetry_.record_duration({rank_, 1, pair_reduce::kDtype, outcome, 0});
}

}  // namespace rocket::qwen38::decode

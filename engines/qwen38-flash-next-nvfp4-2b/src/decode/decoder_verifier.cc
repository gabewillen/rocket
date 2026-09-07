// SPDX-License-Identifier: Apache-2.0
#include "decode/decoder_verifier.h"

#include <chrono>
#include <cstddef>
#include <exception>
#include <stdexcept>

namespace rocket::qwen38::decode {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point begin) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin)
          .count());
}

bool allowed_shape(DecoderVerifierShape shape) noexcept {
  const linear_attention::VerifierShape native{shape.sequences,
                                                shape.verify_width};
  return linear_attention::allowed_verifier_shape(native) &&
         (shape.verify_width <= 4 || shape.sequences <= 4);
}

}  // namespace

void NativeGdnVerifierPort::stage(
    const __nv_bfloat16* input, GdnInactiveState inactive,
    DecoderVerifierShape shape, std::string_view trace_id,
    std::string_view request_id, cudaStream_t stream) {
  verifier_.stage(input, inactive.convolution, inactive.recurrent,
                  inactive.authenticated_slots,
                  {shape.sequences, shape.verify_width}, trace_id, request_id,
                  stream);
}

const __nv_bfloat16* NativeGdnVerifierPort::staged_output() const noexcept {
  return verifier_.staged_output();
}

void NativeGdnVerifierPort::accept(const std::int32_t* prefixes,
                                   std::string_view trace_id,
                                   std::string_view request_id,
                                   cudaStream_t stream) {
  verifier_.accept(prefixes, trace_id, request_id, stream);
}

void NativeGdnVerifierPort::reset(std::string_view trace_id,
                                  std::string_view request_id) noexcept {
  verifier_.reset(trace_id, request_id);
}

DecoderVerifier::DecoderVerifier(
    std::array<GdnVerifierPort*, kDecoderLayers> gdn_by_layer,
    DecoderStepRuntime& runtime, DecoderStateTransaction& state,
    pair_reduce::OtelStageSink& telemetry, cudaStream_t stream)
    : gdn_(gdn_by_layer), runtime_(runtime), state_(state),
      telemetry_(telemetry), stream_(stream) {
  if (!stream_) throw DecoderVerifierError("decoder verifier requires one stream");
  int gdn_count = 0;
  for (int layer = 0; layer < kDecoderLayers; ++layer) {
    const bool present = gdn_[layer] != nullptr;
    if (present == is_qsa_layer(layer))
      throw DecoderVerifierError("GDN/QSA verifier topology changed");
    gdn_count += present;
  }
  if (gdn_count != kDecoderGdnLayers)
    throw DecoderVerifierError("decoder requires exactly 36 GDN verifiers");
}

VerificationOutput DecoderVerifier::step(
    std::uint64_t generation, const std::int32_t* token_ids,
    DecoderVerifierShape shape, std::string_view trace_id,
    std::string_view request_id) {
  const auto begin = Clock::now();
  if (phase_ != DecoderVerifierPhase::kReady || generation == 0 || !token_ids ||
      !allowed_shape(shape) || trace_id.size() > 128 || request_id.size() > 128 ||
      generation != state_.active_generation() + 1) {
    emit("rocket.qwen38.decoder_verifier.validate",
         pair_reduce::Outcome::kContractError, shape, trace_id, request_id,
         elapsed_ns(begin));
    throw DecoderVerifierError("decoder verifier step contract changed");
  }

  void* inactive = nullptr;
  std::array<bool, kDecoderLayers> staged{};
  phase_ = DecoderVerifierPhase::kActive;
  try {
    inactive = state_.begin(generation, shape);
    if (!inactive)
      throw DecoderVerifierError("state transaction returned no inactive token");
    const __nv_bfloat16* hidden = runtime_.embed(token_ids, shape, stream_);
    if (!hidden) throw DecoderVerifierError("embedding returned no hidden state");

    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      const __nv_bfloat16* attention = nullptr;
      if (is_qsa_layer(layer)) {
        attention = runtime_.stage_qsa(layer, hidden, inactive, shape, stream_);
      } else {
        const __nv_bfloat16* input =
            runtime_.gdn_input(layer, hidden, shape, stream_);
        if (!input) throw DecoderVerifierError("GDN input staging failed");
        GdnInactiveState state = state_.gdn_state(inactive, layer);
        if (!state.convolution || !state.recurrent ||
            !state.authenticated_slots)
          throw DecoderVerifierError("inactive GDN state is incomplete");
        gdn_[layer]->stage(input, state, shape, trace_id, request_id, stream_);
        staged[layer] = true;
        attention = gdn_[layer]->staged_output();
      }
      if (!attention) throw DecoderVerifierError("attention produced no output");
      const __nv_bfloat16* partial = runtime_.consume_attention(
          layer, attention, shape, stream_);
      hidden = runtime_.pair_reduce(layer, ReductionKind::kAttentionOutput,
                                    partial, shape, stream_);
      if (!hidden) throw DecoderVerifierError("attention PairReduce failed");
      partial = runtime_.stage_moe(layer, hidden, shape, stream_);
      if (!partial) throw DecoderVerifierError("MoE produced no partial");
      hidden = runtime_.pair_reduce(layer, ReductionKind::kMoeOutput, partial,
                                    shape, stream_);
      if (!hidden) throw DecoderVerifierError("MoE PairReduce failed");
    }

    runtime_.produce_logits(hidden, shape, stream_);
    VerificationOutput output = runtime_.sample_and_verify(shape, stream_);
    if (output.sequences != shape.sequences ||
        !output.accepted_prefixes_device)
      throw DecoderVerifierError("verification output sequence count changed");
    for (int sequence = 0; sequence < shape.sequences; ++sequence) {
      if (output.tokens[sequence] < 0 || output.tokens[sequence] >= 248'320 ||
          output.accepted_prefixes[sequence] < 0 ||
          output.accepted_prefixes[sequence] > shape.verify_width)
        throw DecoderVerifierError("sampled token or accepted prefix is invalid");
    }

    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      if (!is_qsa_layer(layer)) {
        gdn_[layer]->accept(output.accepted_prefixes.data(), trace_id,
                            request_id, stream_);
        staged[layer] = false;
      }
    }
    runtime_.accept_qsa(inactive, output.accepted_prefixes.data(), shape,
                        stream_);
    runtime_.synchronize(stream_);
    state_.publish(inactive);
    inactive = nullptr;
    if (state_.active_generation() != generation) {
      phase_ = DecoderVerifierPhase::kFaulted;
      throw DecoderVerifierError("state publisher violated atomic commit contract");
    }
    phase_ = DecoderVerifierPhase::kReady;
    emit("rocket.qwen38.decoder_verifier.step", pair_reduce::Outcome::kOk,
         shape, trace_id, request_id, elapsed_ns(begin));
    return output;
  } catch (...) {
    for (int layer = 0; layer < kDecoderLayers; ++layer)
      if (staged[layer]) gdn_[layer]->reset(trace_id, request_id);
    if (inactive) {
      runtime_.reset_qsa(inactive);
      state_.discard(inactive);
    }
    phase_ = DecoderVerifierPhase::kFaulted;
    emit("rocket.qwen38.decoder_verifier.step",
         pair_reduce::Outcome::kCudaError, shape, trace_id, request_id,
         elapsed_ns(begin));
    throw;
  }
}

void DecoderVerifier::emit(std::string_view stage,
                           pair_reduce::Outcome outcome,
                           DecoderVerifierShape shape,
                           std::string_view trace_id,
                           std::string_view request_id,
                           std::uint64_t duration_ns) noexcept {
  const int bucket = allowed_shape(shape) ? shape.sequences : 0;
  telemetry_.emit_span_and_log({stage, trace_id, request_id, -1, bucket,
                                "bf16_fp32", outcome, duration_ns, 0});
  telemetry_.record_duration(
      {-1, bucket, "bf16_fp32", outcome, duration_ns});
}

}  // namespace rocket::qwen38::decode

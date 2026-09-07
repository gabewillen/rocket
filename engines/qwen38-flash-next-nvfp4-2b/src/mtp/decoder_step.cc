// SPDX-License-Identifier: Apache-2.0
#include "mtp/decoder_step.h"

namespace rocket::qwen38::mtp {

decode::VerificationOutput DecoderStep::step(std::uint64_t generation,
                                             std::string_view trace_id,
                                             std::string_view request_id) {
  if (phase_ != DecoderStepPhase::kReady)
    throw DecoderStepError("faulted MTP decoder step cannot be reused");
  const DeviceDraftView draft = drafter_.draft(generation);
  try {
    decode::VerificationOutput output = verifier_.step(
        generation, draft.verification_tokens,
        {draft.sequences, draft.depth + 1}, trace_id, request_id);
    // DecoderVerifier::step has completed the sole stream fence here.
    drafter_.export_telemetry_after_fence(generation);
    drafter_.publish(generation, output.accepted_prefixes_device);
    return output;
  } catch (...) {
    drafter_.discard(generation);
    phase_ = DecoderStepPhase::kFaulted;
    throw;
  }
}

}  // namespace rocket::qwen38::mtp

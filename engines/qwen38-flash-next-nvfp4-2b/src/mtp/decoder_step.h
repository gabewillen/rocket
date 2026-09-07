// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "decode/decoder_verifier.h"
#include "mtp/native_executor.h"

namespace rocket::qwen38::mtp {

enum class DecoderStepPhase : std::uint8_t { kReady, kFaulted };

class DecoderStepError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Coarse native engine operation. One call enqueues every MTP proposal step,
// verifies target tokens, exports bounded telemetry after the verifier's sole
// terminal fence, then enqueues accepted MTP snapshot publication.
class DecoderStep final {
 public:
  DecoderStep(NativeExecutor& drafter,
              decode::DecoderVerifier& verifier) noexcept
      : drafter_(drafter), verifier_(verifier) {}

  decode::VerificationOutput step(std::uint64_t generation,
                                  std::string_view trace_id,
                                  std::string_view request_id);
  [[nodiscard]] DecoderStepPhase phase() const noexcept { return phase_; }

 private:
  NativeExecutor& drafter_;
  decode::DecoderVerifier& verifier_;
  DecoderStepPhase phase_ = DecoderStepPhase::kReady;
};

}  // namespace rocket::qwen38::mtp

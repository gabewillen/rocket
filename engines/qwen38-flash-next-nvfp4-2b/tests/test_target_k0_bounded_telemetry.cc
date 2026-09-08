// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_bounded_telemetry.h"

#include <stdexcept>

using namespace rocket::qwen38;

int main() {
  decode::TargetK0BoundedTelemetry sink;
  sink.emit_span_and_log({"stage", "ignored-trace", "ignored-request", 0, 1,
                          "bf16_fp32",
                          pair_reduce::Outcome::kTransportError, 9, 17});
  sink.record_duration(
      {0, 1, "bf16_fp32", pair_reduce::Outcome::kTransportError, 9});
  sink.emit({moe::TargetFullMoeComponent::kStaging,
             moe::TargetDenseOutcome::kOk, 0, 3});
  sink.add_counter({moe::TargetMoeStageCounter::kScratchBytes,
                    moe::TargetMoeOutcome::kOk, 0, 3, 99});
  sink.emit({0, 35, attention::TargetK0OracleQsaStateOutcome::kOk, 23});
  const auto result = sink.snapshot();
  if (result.lifecycle_outcomes[2] != 1 || result.moe_components[2] != 1 ||
      result.stage_counters[3] != 1 || result.state_outcomes[0] != 1 ||
      result.duration_samples != 1 || result.total_bytes != 40)
    throw std::runtime_error("bounded K0 telemetry changed");
}

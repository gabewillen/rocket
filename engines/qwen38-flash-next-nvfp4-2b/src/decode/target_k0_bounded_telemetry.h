// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "attention/target_k0_qsa_state_owner.h"
#include "moe/target_full_moe_c1.h"
#include "moe/target_moe_n640_device_stage.h"
#include "pair_reduce/otel.h"

#include <array>
#include <cstdint>

namespace rocket::qwen38::decode {

struct TargetK0BoundedTelemetrySnapshot {
  std::array<std::uint64_t, 4> lifecycle_outcomes{};
  std::array<std::uint64_t, 5> moe_components{};
  std::array<std::uint64_t, 4> stage_counters{};
  std::array<std::uint64_t, 4> state_outcomes{};
  std::uint64_t duration_samples = 0;
  std::uint64_t total_bytes = 0;
};

// Allocation-free bounded native sink. It keeps only counters indexed by
// closed enums. The process-local launcher converts one terminal snapshot to
// OpenTelemetry after the native call returns; trace/request strings and paths
// are never retained here.
class TargetK0BoundedTelemetry final
    : public pair_reduce::OtelStageSink,
      public moe::TargetFullMoeOtelSink,
      public moe::TargetMoeStageOtelSink,
      public attention::TargetK0OracleQsaStateOtelSink {
 public:
  void emit_span_and_log(const pair_reduce::SpanRecord&) noexcept override;
  void record_duration(const pair_reduce::MetricPoint&) noexcept override;
  void emit(const moe::TargetFullMoeOtelPoint&) noexcept override;
  void add_counter(const moe::TargetMoeStageOtelPoint&) noexcept override;
  void emit(const attention::TargetK0OracleQsaStateOtelPoint&) noexcept override;
  [[nodiscard]] TargetK0BoundedTelemetrySnapshot snapshot() const noexcept;

 private:
  TargetK0BoundedTelemetrySnapshot counters_{};
};

}  // namespace rocket::qwen38::decode

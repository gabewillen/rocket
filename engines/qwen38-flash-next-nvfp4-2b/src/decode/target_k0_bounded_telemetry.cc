// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_bounded_telemetry.h"

#include <cstddef>

namespace rocket::qwen38::decode {
namespace {
template <class Enum, std::size_t N>
void increment(std::array<std::uint64_t, N>& values, Enum value) noexcept {
  const auto index = static_cast<std::size_t>(value);
  if (index < values.size()) ++values[index];
}
}  // namespace

void TargetK0BoundedTelemetry::emit_span_and_log(
    const pair_reduce::SpanRecord& record) noexcept {
  increment(counters_.lifecycle_outcomes, record.outcome);
  counters_.total_bytes += record.bytes;
}

void TargetK0BoundedTelemetry::record_duration(
    const pair_reduce::MetricPoint&) noexcept {
  ++counters_.duration_samples;
}

void TargetK0BoundedTelemetry::emit(
    const moe::TargetFullMoeOtelPoint& point) noexcept {
  increment(counters_.moe_components, point.component);
}

void TargetK0BoundedTelemetry::emit_span_and_log(
    const mtp::NcclBootstrapTelemetryRecord& record) noexcept {
  increment(counters_.nccl_stages, record.stage);
  increment(counters_.nccl_outcomes, record.outcome);
}

void TargetK0BoundedTelemetry::record_duration(
    const mtp::NcclBootstrapTelemetryRecord&) noexcept {
  ++counters_.duration_samples;
}

void TargetK0BoundedTelemetry::add_counter(
    const moe::TargetMoeStageOtelPoint& point) noexcept {
  increment(counters_.stage_counters, point.counter);
}

void TargetK0BoundedTelemetry::emit(
    const attention::TargetK0OracleQsaStateOtelPoint& point) noexcept {
  increment(counters_.state_outcomes, point.outcome);
  counters_.total_bytes += point.bytes;
}

TargetK0BoundedTelemetrySnapshot TargetK0BoundedTelemetry::snapshot()
    const noexcept {
  return counters_;
}

}  // namespace rocket::qwen38::decode

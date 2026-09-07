#pragma once

#include <cstdint>
#include <string_view>

namespace rocket::qwen38::pair_reduce {

enum class Outcome : std::uint8_t { kOk, kContractError, kTransportError, kCudaError };

constexpr std::string_view outcome_name(Outcome outcome) noexcept {
  switch (outcome) {
    case Outcome::kOk: return "ok";
    case Outcome::kContractError: return "contract_error";
    case Outcome::kTransportError: return "transport_error";
    case Outcome::kCudaError: return "cuda_error";
  }
  return "contract_error";
}

// OpenTelemetry adapter boundary owned by the engine embedder. Metric labels
// are exactly the four bounded fields in MetricPoint: rank {0,1}, M bucket
// {invalid,1,2,4,8,16}, dtype {bf16_fp32}, and outcome {four enum values}. Trace and
// request IDs are unbounded correlation fields and therefore appear only on
// spans/logs, never as metric dimensions. Calls are synchronous; the sink must
// copy borrowed string_views before returning and must not throw.
struct SpanRecord {
  std::string_view stage;
  std::string_view trace_id;
  std::string_view request_id;
  int rank;
  int m_bucket;
  std::string_view dtype;
  Outcome outcome;
  std::uint64_t duration_ns;
  std::uint64_t bytes;
};

struct MetricPoint {
  int rank;
  int m_bucket;
  std::string_view dtype;
  Outcome outcome;
  std::uint64_t duration_ns;
};

class OtelStageSink {
 public:
  virtual ~OtelStageSink() = default;
  virtual void emit_span_and_log(const SpanRecord& record) noexcept = 0;
  virtual void record_duration(const MetricPoint& point) noexcept = 0;
};

}  // namespace rocket::qwen38::pair_reduce

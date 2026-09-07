#include "decode/execution.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace decode = rocket::qwen38::decode;
namespace pr = rocket::qwen38::pair_reduce;

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void check(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

class CaptureOtel final : public pr::OtelStageSink {
 public:
  void emit_span_and_log(const pr::SpanRecord& record) noexcept override {
    spans.push_back({std::string(record.stage), record.rank, record.m_bucket,
                     std::string(record.dtype), record.outcome,
                     std::string(record.trace_id), std::string(record.request_id)});
  }
  void record_duration(const pr::MetricPoint&) noexcept override { ++metrics; }

  struct Span {
    std::string stage;
    int rank;
    int m;
    std::string dtype;
    pr::Outcome outcome;
    std::string trace_id;
    std::string request_id;
  };
  std::vector<Span> spans;
  int metrics = 0;
};

class SyntheticReducer final : public decode::HiddenPartialReducer {
 public:
  int rank() const noexcept override { return rank_; }
  int world_size() const noexcept override { return world_size_; }
  void reduce(const __nv_bfloat16* input, float* output, int m,
              std::string_view trace_id, std::string_view request_id,
              cudaStream_t stream) override {
    check(input != nullptr && output != nullptr, "reducer received null buffers");
    check(m == expected_m_, "reducer M drift");
    check(trace_id == "trace-1" && request_id == "request-1",
          "reducer correlation drift");
    check(stream == expected_stream_, "reducer stream drift");
    if (calls == fail_at_) throw std::runtime_error("injected transport failure");
    ++calls;
  }
  void complete(cudaStream_t stream, std::uint32_t reduction_count,
                std::string_view trace_id,
                std::string_view request_id) override {
    check(stream == expected_stream_ && reduction_count == 96 &&
              trace_id == "trace-1" && request_id == "request-1",
          "completion fence contract drift");
    ++completions;
  }

  int rank_ = 0;
  int world_size_ = 2;
  int expected_m_ = 4;
  cudaStream_t expected_stream_ = reinterpret_cast<cudaStream_t>(0x3000);
  int calls = 0;
  int completions = 0;
  int fail_at_ = -1;
};

void test_complete_step_invokes_all_exact_points() {
  SyntheticReducer reducer;
  CaptureOtel telemetry;
  decode::Tp2DecodeExecution execution(reducer, telemetry);
  __nv_bfloat16 partial{};
  float reduced = 0.0f;
  execution.begin_step(1, 4, "trace-1", "request-1");
  for (int ordinal = 0; ordinal < decode::kReductionPoints; ++ordinal) {
    execution.reduce_at(1, execution.point_for_ordinal(ordinal), &partial, &reduced,
                        "trace-1", "request-1", reducer.expected_stream_);
  }
  execution.finish_step(1, "trace-1", "request-1");

  check(reducer.calls == 96 && reducer.completions == 1,
        "complete step did not invoke 96 reductions and one fence");
  check(execution.phase() == decode::StepPhase::kIdle &&
            execution.last_completed_generation() == 1 &&
            execution.next_ordinal() == 0,
        "complete step did not return to idle");
  int attention = 0, moe = 0, lifecycle = 0;
  for (const auto& span : telemetry.spans) {
    check(span.rank == 0 && span.m == 4 &&
              span.dtype == "bf16_fp32" && span.outcome == pr::Outcome::kOk &&
              span.trace_id == "trace-1" && span.request_id == "request-1",
          "bounded success observation drift");
    attention += span.stage == "rocket.qwen38.decode.tp2.attention";
    moe += span.stage == "rocket.qwen38.decode.tp2.moe";
    lifecycle += span.stage == "rocket.qwen38.decode.tp2.lifecycle";
  }
  check(attention == 48 && moe == 48 && lifecycle == 2 && telemetry.metrics == 0,
        "decode OTEL stage cardinality drift");
}

void test_sequence_and_point_drift_do_not_advance() {
  SyntheticReducer reducer;
  CaptureOtel telemetry;
  decode::Tp2DecodeExecution execution(reducer, telemetry);
  __nv_bfloat16 partial{};
  float reduced = 0.0f;

  bool step_threw = false;
  try { execution.begin_step(2, 4, "trace-1", "request-1"); }
  catch (const decode::DecodeExecutionContractError&) { step_threw = true; }
  check(step_threw && execution.phase() == decode::StepPhase::kIdle,
        "step sequence drift mutated idle state");

  execution.begin_step(1, 4, "trace-1", "request-1");
  bool point_threw = false;
  try {
    execution.reduce_at(1, {0, decode::ReductionKind::kMoeOutput}, &partial, &reduced,
                        "trace-1", "request-1", reducer.expected_stream_);
  } catch (const decode::DecodeExecutionContractError&) { point_threw = true; }
  check(point_threw && reducer.calls == 0 && execution.next_ordinal() == 0 &&
            execution.phase() == decode::StepPhase::kActive,
        "point order drift reached reducer or advanced state");

  bool finish_threw = false;
  try { execution.finish_step(1, "trace-1", "request-1"); }
  catch (const decode::DecodeExecutionContractError&) { finish_threw = true; }
  check(finish_threw && execution.phase() == decode::StepPhase::kActive,
        "early finish mutated active state");
}

void test_reducer_failure_faults_terminally() {
  SyntheticReducer reducer;
  reducer.fail_at_ = 0;
  CaptureOtel telemetry;
  decode::Tp2DecodeExecution execution(reducer, telemetry);
  __nv_bfloat16 partial{};
  float reduced = 0.0f;
  execution.begin_step(1, 4, "trace-1", "request-1");
  bool threw = false;
  try {
    execution.reduce_at(1, {0, decode::ReductionKind::kAttentionOutput}, &partial,
                        &reduced, "trace-1", "request-1", reducer.expected_stream_);
  } catch (const decode::DecodeExecutionTransportError&) { threw = true; }
  check(threw && execution.phase() == decode::StepPhase::kFaulted &&
            execution.next_ordinal() == 0 &&
            telemetry.spans.back().outcome == pr::Outcome::kTransportError,
        "reducer failure did not fault without advancing");
  bool retry_threw = false;
  try { execution.begin_step(1, 4, "trace-1", "request-1"); }
  catch (const decode::DecodeExecutionContractError&) { retry_threw = true; }
  check(retry_threw && execution.phase() == decode::StepPhase::kFaulted,
        "faulted execution accepted retry");
}

void test_topology_and_shape_fail_closed() {
  SyntheticReducer reducer;
  reducer.rank_ = 7;
  reducer.world_size_ = 8;
  CaptureOtel telemetry;
  bool topology_threw = false;
  try { decode::Tp2DecodeExecution execution(reducer, telemetry); }
  catch (const decode::DecodeExecutionContractError&) { topology_threw = true; }
  check(topology_threw && telemetry.spans.size() == 1 &&
            telemetry.spans[0].rank == -1 && telemetry.spans[0].m == 0 &&
            telemetry.spans[0].outcome == pr::Outcome::kContractError,
        "invalid topology was not safely observed");

  reducer.rank_ = 0;
  reducer.world_size_ = 2;
  bool generation_threw = false;
  try {
    decode::Tp2DecodeExecution exhausted(
        reducer, telemetry, std::numeric_limits<std::uint64_t>::max());
  } catch (const decode::DecodeExecutionContractError&) {
    generation_threw = true;
  }
  check(generation_threw, "exhausted device generation was accepted");

  bool ordinal_threw = false;
  try { (void)decode::Tp2DecodeExecution::point_for_ordinal(96); }
  catch (const decode::DecodeExecutionContractError&) { ordinal_threw = true; }
  check(ordinal_threw, "out-of-range reduction ordinal was accepted");

  reducer.rank_ = 1;
  reducer.world_size_ = 2;
  decode::Tp2DecodeExecution execution(reducer, telemetry);
  bool shape_threw = false;
  try { execution.begin_step(1, 9, "trace-1", "request-1"); }
  catch (const decode::DecodeExecutionContractError&) { shape_threw = true; }
  check(shape_threw && execution.phase() == decode::StepPhase::kIdle,
        "invalid M mutated execution state");

  decode::Tp2DecodeExecution resumed(reducer, telemetry, 41);
  resumed.begin_step(42, 4, "trace-1", "request-1");
  check(resumed.phase() == decode::StepPhase::kActive,
        "explicit completed generation did not support reconstruction");
}

}  // namespace

int main() {
  try {
    test_complete_step_invokes_all_exact_points();
    test_sequence_and_point_drift_do_not_advance();
    test_reducer_failure_faults_terminally();
    test_topology_and_shape_fail_closed();
    std::puts("qwen38 decode PairReduce: 96 ordered points and terminal fault contract passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_pair_reduce.h"

#include <cstdio>
#include <stdexcept>
#include <string_view>

namespace decode = rocket::qwen38::decode;
namespace pr = rocket::qwen38::pair_reduce;

namespace {

struct Reducer final : decode::HiddenPartialReducer {
  int rank() const noexcept override { return 0; }
  int world_size() const noexcept override { return 2; }
  void reduce(const __nv_bfloat16*, float*, int m, std::string_view,
              std::string_view, cudaStream_t) override {
    if (m != 1) throw std::runtime_error("M changed");
    if (calls++ == fail_at) throw std::runtime_error("injected failure");
  }
  int calls = 0;
  int fail_at = -1;
};

struct Telemetry final : pr::OtelStageSink {
  void emit_span_and_log(const pr::SpanRecord& record) noexcept override {
    ++spans;
    last = record.outcome;
    bytes = record.bytes;
  }
  void record_duration(const pr::MetricPoint&) noexcept override { ++metrics; }
  int spans = 0;
  int metrics = 0;
  std::uint64_t bytes = 0;
  pr::Outcome last = pr::Outcome::kContractError;
};

void check(bool value) {
  if (!value) throw std::runtime_error("layer-3 PairReduce proof failed");
}

}  // namespace

int main() {
  try {
    Reducer reducer;
    Telemetry telemetry;
    decode::TargetLayer3PairReduceSchedule schedule(reducer, telemetry);
    __nv_bfloat16 input{};
    float output{};
    auto stream = reinterpret_cast<cudaStream_t>(0x1);
    check(schedule.phase() ==
          decode::TargetLayer3PairReducePhase::kAwaitRow);
    for (int row = 0; row < 35; ++row) {
      check(schedule.next_generation() == static_cast<std::uint64_t>(row + 1));
      schedule.begin_row(row, static_cast<std::uint64_t>(row + 1));
      check(schedule.phase() ==
            decode::TargetLayer3PairReducePhase::kAwaitAttention);
      schedule.attention_port().reduce(&input, &output, 1, "trace", "request",
                                       stream);
      check(schedule.phase() ==
            decode::TargetLayer3PairReducePhase::kAwaitMoe);
      schedule.moe_port().reduce(&input, &output, 1, "trace", "request",
                                 stream);
      check(schedule.phase() ==
            (row == 34 ? decode::TargetLayer3PairReducePhase::kCompleted
                       : decode::TargetLayer3PairReducePhase::kAwaitRow));
    }
    check(reducer.calls == 70 && schedule.complete() && !schedule.faulted() &&
          telemetry.spans == 1 && telemetry.metrics == 1 &&
          telemetry.last == pr::Outcome::kOk &&
          telemetry.bytes == 70ULL * 2560ULL * 2ULL);
    bool rejected = false;
    try { schedule.begin_row(35, 36); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && schedule.complete() && !schedule.faulted());
    rejected = false;
    try {
      schedule.attention_port().reduce(&input, &output, 1, "trace", "request",
                                       stream);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && schedule.complete() && !schedule.faulted());

    Reducer drift_reducer;
    Telemetry drift_telemetry;
    decode::TargetLayer3PairReduceSchedule drift(drift_reducer,
                                                  drift_telemetry);
    rejected = false;
    try {
      drift.begin_row(0, 1);
      drift.moe_port().reduce(&input, &output, 1, "trace", "request", stream);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && drift.faulted() && drift_reducer.calls == 0 &&
          drift_telemetry.last == pr::Outcome::kContractError);

    Reducer stale_reducer;
    Telemetry stale_telemetry;
    decode::TargetLayer3PairReduceSchedule stale(stale_reducer,
                                                  stale_telemetry);
    rejected = false;
    try { stale.begin_row(0, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && stale.faulted() && stale_reducer.calls == 0);

    Reducer row_reducer;
    Telemetry row_telemetry;
    decode::TargetLayer3PairReduceSchedule wrong_row(row_reducer,
                                                      row_telemetry);
    rejected = false;
    try { wrong_row.begin_row(1, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && wrong_row.faulted() && row_reducer.calls == 0);

    Reducer duplicate_reducer;
    Telemetry duplicate_telemetry;
    decode::TargetLayer3PairReduceSchedule duplicate(duplicate_reducer,
                                                      duplicate_telemetry);
    duplicate.begin_row(0, 1);
    duplicate.attention_port().reduce(&input, &output, 1, "trace", "request",
                                      stream);
    rejected = false;
    try {
      duplicate.attention_port().reduce(&input, &output, 1, "trace", "request",
                                        stream);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && duplicate.faulted() && duplicate_reducer.calls == 1);

    Reducer failure_reducer;
    failure_reducer.fail_at = 0;
    Telemetry failure_telemetry;
    decode::TargetLayer3PairReduceSchedule failure(failure_reducer,
                                                    failure_telemetry);
    failure.begin_row(0, 1);
    rejected = false;
    try {
      failure.attention_port().reduce(&input, &output, 1, "trace", "request",
                                      stream);
    } catch (const std::runtime_error&) { rejected = true; }
    check(rejected && failure.faulted() && !failure.complete() &&
          failure.next_ordinal() == 0 && failure_reducer.calls == 1 &&
          failure_telemetry.last == pr::Outcome::kTransportError);
    for (auto* port : {&failure.attention_port(), &failure.moe_port()}) {
      rejected = false;
      try { port->reduce(&input, &output, 1, "trace", "request", stream); }
      catch (const std::invalid_argument&) { rejected = true; }
      check(rejected && failure_reducer.calls == 1 && failure.faulted());
    }

    decode::TargetLayer3PairReduceBootstrap bootstrap;
    bootstrap.rank = 0;
    bootstrap.peer_rank = 1;
    bootstrap.bootstrap_host = "192.168.100.10";
    bootstrap.bootstrap_port = 18839;
    bootstrap.timeout_ms = 120000;
    bootstrap.session_sha256 =
        decode::TargetLayer3PairReduceOwner::oracle_session_sha256();
    decode::TargetLayer3PairReduceOwner::validate_bootstrap(bootstrap);
    bootstrap.peer_rank = 0;
    rejected = false;
    try { decode::TargetLayer3PairReduceOwner::validate_bootstrap(bootstrap); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected);
    Telemetry bootstrap_telemetry;
    rejected = false;
    try {
      decode::TargetLayer3PairReduceOwner owner(bootstrap,
                                                bootstrap_telemetry);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && bootstrap_telemetry.spans == 1 &&
          bootstrap_telemetry.metrics == 1 &&
          bootstrap_telemetry.last == pr::Outcome::kContractError);
    std::puts("qwen38 layer3 PairReduce: 35 attention/MoE M1 pairs passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

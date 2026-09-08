// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_pair_reduce.h"

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
    if (calls++ == fail_at) throw std::runtime_error("injected transport");
  }
  int calls = 0;
  int fail_at = -1;
};

struct Sink final : pr::OtelStageSink {
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
  if (!value) throw std::runtime_error("K0 PairReduce proof failed");
}

void run_row(decode::TargetK0PairReduceSchedule& schedule, int row,
             std::uint64_t generation, __nv_bfloat16* input, float* output,
             cudaStream_t stream) {
  schedule.begin_row(row, generation);
  for (int layer = 0; layer < decode::kLayers; ++layer) {
    schedule.attention_port(layer).reduce(input, output, 1, "trace", "request",
                                           stream);
    schedule.moe_port(layer).reduce(input, output, 1, "trace", "request",
                                     stream);
  }
}
}  // namespace

int main() {
  try {
    __nv_bfloat16 input{};
    float output{};
    auto stream = reinterpret_cast<cudaStream_t>(0x1);
    Reducer reducer;
    Sink sink;
    decode::TargetK0PairReduceSchedule schedule(reducer, sink);
    schedule.begin_sequence(41, 35);
    for (int row = 0; row < 35; ++row)
      run_row(schedule, row, 41 + static_cast<std::uint64_t>(row), &input,
              &output, stream);
    check(schedule.complete() && !schedule.faulted() &&
          reducer.calls == 35 * 96 && schedule.completed_calls() == 35 * 96 &&
          sink.last == pr::Outcome::kOk && sink.spans == 1 && sink.metrics == 1 &&
          sink.bytes == 35ULL * 96ULL * 2'560ULL * 2ULL);

    bool rejected = false;
    try { schedule.begin_row(35, 76); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && schedule.complete() && !schedule.faulted());

    Reducer order_reducer;
    Sink order_sink;
    decode::TargetK0PairReduceSchedule order(order_reducer, order_sink);
    order.begin_sequence(1, 1);
    order.begin_row(0, 1);
    rejected = false;
    try {
      order.moe_port(0).reduce(&input, &output, 1, "trace", "request", stream);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && order.faulted() && order_reducer.calls == 0);

    Reducer layer_reducer;
    Sink layer_sink;
    decode::TargetK0PairReduceSchedule layer(layer_reducer, layer_sink);
    layer.begin_sequence(1, 1);
    layer.begin_row(0, 1);
    rejected = false;
    try {
      layer.attention_port(1).reduce(&input, &output, 1, "trace", "request",
                                     stream);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && layer.faulted() && layer_reducer.calls == 0);

    Reducer stale_reducer;
    Sink stale_sink;
    decode::TargetK0PairReduceSchedule stale(stale_reducer, stale_sink);
    stale.begin_sequence(9, 2);
    rejected = false;
    try { stale.begin_row(0, 8); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && stale.faulted() && stale_reducer.calls == 0);

    Reducer failure_reducer;
    failure_reducer.fail_at = 17;
    Sink failure_sink;
    decode::TargetK0PairReduceSchedule failure(failure_reducer, failure_sink);
    failure.begin_sequence(1, 1);
    failure.begin_row(0, 1);
    for (int call = 0; call < 17; ++call) {
      const int selected_layer = call / 2;
      auto& port = call % 2 == 0 ? failure.attention_port(selected_layer)
                                  : failure.moe_port(selected_layer);
      port.reduce(&input, &output, 1, "trace", "request", stream);
    }
    rejected = false;
    try {
      failure.moe_port(8).reduce(&input, &output, 1, "trace", "request",
                                 stream);
    } catch (const std::runtime_error&) { rejected = true; }
    check(rejected && failure.faulted() && failure.completed_calls() == 17 &&
          failure_reducer.calls == 18 &&
          failure_sink.last == pr::Outcome::kTransportError);

    decode::TargetK0PairReduceBootstrap bootstrap;
    bootstrap.rank = 0;
    bootstrap.peer_rank = 1;
    bootstrap.bootstrap_host = "192.0.2.1";
    bootstrap.bootstrap_port = 18840;
    bootstrap.timeout_ms = 120'000;
    bootstrap.session_sha256[0] = 1;
    decode::TargetK0PairReduceOwner::validate_bootstrap(bootstrap);
    bootstrap.session_sha256.fill(0);
    rejected = false;
    try { decode::TargetK0PairReduceOwner::validate_bootstrap(bootstrap); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected);

    std::puts("qwen38 K0 PairReduce: 35 rows x 96 ordered c1 calls passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

#include "decode/full_attention_layer.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace decode = rocket::qwen38::decode;
namespace pr = rocket::qwen38::pair_reduce;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class Trace final : public pr::OtelStageSink {
 public:
  void emit_span_and_log(const pr::SpanRecord& record) noexcept override {
    stages.emplace_back(record.stage);
    outcomes.push_back(record.outcome);
  }
  void record_duration(const pr::MetricPoint&) noexcept override {}
  std::vector<std::string_view> stages;
  std::vector<pr::Outcome> outcomes;
};

class Graph final : public decode::FullAttentionGraph {
 public:
  int rank() const noexcept override { return 0; }
  int layer() const noexcept override { return 3; }
  void launch(const __nv_bfloat16* block_input, int m,
              cudaStream_t stream) override {
    check(block_input != nullptr && m == expected_m && stream == expected_stream,
          "graph launch contract drift");
    calls.push_back("attention");
  }
  const __nv_bfloat16* projected_output() const noexcept override {
    return &partial;
  }
  int expected_m = 4;
  cudaStream_t expected_stream = reinterpret_cast<cudaStream_t>(0x1230);
  __nv_bfloat16 partial{};
  std::vector<std::string_view> calls;
};

class Reducer final : public decode::HiddenPartialReducer {
 public:
  int rank() const noexcept override { return 0; }
  int world_size() const noexcept override { return 2; }
  void reduce(const __nv_bfloat16* input, float* output, int m,
              std::string_view, std::string_view,
              cudaStream_t stream) override {
    check(input != nullptr && output != nullptr && m == 4 &&
              stream == reinterpret_cast<cudaStream_t>(0x1230),
          "reducer contract drift");
    calls.push_back("reduce");
  }
  std::vector<std::string_view> calls;
};

class HyperConnection final : public decode::FullAttentionHyperConnection {
 public:
  void mix(const __nv_bfloat16* hidden, __nv_bfloat16* block_input,
           __nv_bfloat16* injection, int m, cudaStream_t stream) override {
    check(hidden && block_input && injection && m == 4 && stream,
          "mix contract drift");
    calls.push_back("mix");
  }
  void combine_and_mix(const __nv_bfloat16* hidden, const float* block_output,
                       const __nv_bfloat16* injection,
                       __nv_bfloat16* updated_hidden,
                       __nv_bfloat16* next_block_input,
                       __nv_bfloat16* next_injection, int m,
                       cudaStream_t stream) override {
    check(hidden && block_output && injection && updated_hidden &&
              next_block_input && next_injection && m == 4 && stream,
          "combine-and-mix contract drift");
    calls.push_back("combine_and_mix");
  }
  void synchronize(cudaStream_t stream) override {
    check(stream == reinterpret_cast<cudaStream_t>(0x1230),
          "completion stream drift");
    calls.push_back("synchronize");
    if (fail_next_completion) {
      fail_next_completion = false;
      throw std::runtime_error("injected deferred CUDA failure");
    }
  }
  bool fail_next_completion = false;
  std::vector<std::string_view> calls;
};

}  // namespace

int main() {
  try {
    Graph graph;
    Reducer reducer;
    HyperConnection hc;
    Trace trace;
    decode::FullAttentionLayer layer(graph, reducer, hc, trace);
    __nv_bfloat16 hidden{}, block_input{}, injection{}, updated{}, moe_input{},
        next_injection{};
    float reduced{};
    const auto result = layer.execute(
        1, 4, &hidden, &block_input, &injection, &reduced, &updated,
        &moe_input, &next_injection, "trace", "request",
        graph.expected_stream);
    check(result.generation == 1 && result.m_bucket == 4,
          "publication identity drift");
    check(hc.calls == std::vector<std::string_view>{
                          "mix", "synchronize", "synchronize",
                          "combine_and_mix", "synchronize"} &&
              graph.calls == std::vector<std::string_view>{"attention"} &&
              reducer.calls == std::vector<std::string_view>{"reduce"},
          "full-attention stage order drift");
    check(trace.stages.size() == 5, "bounded stage count drift");

    HyperConnection failing_hc;
    failing_hc.fail_next_completion = true;
    Trace failing_trace;
    decode::FullAttentionLayer failing_layer(graph, reducer, failing_hc,
                                             failing_trace);
    bool deferred_failure = false;
    try {
      failing_layer.execute(
          1, 4, &hidden, &block_input, &injection, &reduced, &updated,
          &moe_input, &next_injection, "trace-fail", "request-fail",
          graph.expected_stream);
    } catch (const std::runtime_error& error) {
      deferred_failure = std::string_view(error.what()).find("deferred CUDA") !=
                         std::string_view::npos;
    }
    check(deferred_failure && failing_trace.stages.size() == 1 &&
              failing_trace.stages[0] == "rocket.qwen38.layer3.lifecycle" &&
              failing_trace.outcomes[0] != pr::Outcome::kOk,
          "deferred completion failure was published");
    bool retried_fault = false;
    try {
      failing_layer.execute(
          1, 4, &hidden, &block_input, &injection, &reduced, &updated,
          &moe_input, &next_injection, "trace-retry", "request-retry",
          graph.expected_stream);
    } catch (const decode::DecodeExecutionContractError&) {
      retried_fault = true;
    }
    check(retried_fault, "deferred completion failure did not fault transition");
    std::puts("qwen38 full-attention layer order and publication contract passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

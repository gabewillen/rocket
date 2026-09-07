// SPDX-License-Identifier: Apache-2.0
#include "decode/linear_attention_layer.h"

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

class Graph final : public decode::LinearAttentionGraph {
 public:
  explicit Graph(std::vector<std::string_view>& ordered) : ordered(ordered) {}
  int rank() const noexcept override { return 0; }
  int layer() const noexcept override { return 0; }
  std::string_view checkpoint_revision() const noexcept override {
    return decode::kQwen38CheckpointRevision;
  }
  std::string_view slab_key() const noexcept override { return "rank0-target"; }
  std::string_view conv_state_family() const noexcept override {
    return "target_gdn_conv";
  }
  std::string_view recurrent_state_family() const noexcept override {
    return "target_gdn_recurrent";
  }
  bool has_captured_bucket(int m) const noexcept override {
    return decode::allowed_linear_m(m);
  }
  std::uint64_t logical_bytes_per_row(int m) const noexcept override {
    return decode::allowed_linear_m(m) ? 3'200'000 : 0;
  }
  void launch(const __nv_bfloat16* block_input,
              __nv_bfloat16* conv_state, float* recurrent_state,
              const std::int32_t* state_indices, int m,
              cudaStream_t stream) override {
    check(block_input && conv_state && recurrent_state && state_indices &&
              m == 4 && stream == expected_stream,
          "linear graph launch contract drift");
    calls.push_back("gdn_graph");
    ordered.push_back("gdn_graph");
  }
  const __nv_bfloat16* projected_output() const noexcept override {
    return &partial;
  }
  __nv_bfloat16 partial{};
  cudaStream_t expected_stream = reinterpret_cast<cudaStream_t>(0x1230);
  std::vector<std::string_view> calls;
  std::vector<std::string_view>& ordered;
};

class Reducer final : public decode::HiddenPartialReducer {
 public:
  explicit Reducer(std::vector<std::string_view>& ordered) : ordered(ordered) {}
  int rank() const noexcept override { return 0; }
  int world_size() const noexcept override { return 2; }
  void reduce(const __nv_bfloat16* input, float* output, int m,
              std::string_view, std::string_view,
              cudaStream_t stream) override {
    check(input && output && m == 4 && stream,
          "linear reducer contract drift");
    calls.push_back("reduce");
    ordered.push_back("reduce");
  }
  void complete(cudaStream_t, std::uint32_t, std::string_view,
                std::string_view) override {}
  std::vector<std::string_view> calls;
  std::vector<std::string_view>& ordered;
};

class HyperConnection final : public decode::LinearAttentionHyperConnection {
 public:
  explicit HyperConnection(std::vector<std::string_view>& ordered)
      : ordered(ordered) {}
  void mix(const __nv_bfloat16* hidden, __nv_bfloat16* block_input,
           __nv_bfloat16* injection, int m, cudaStream_t stream) override {
    check(hidden && block_input && injection && m == 4 && stream,
          "linear HC mix drift");
    calls.push_back("mix");
    ordered.push_back("mix");
  }
  void combine_and_mix(const __nv_bfloat16* hidden, const float* block_output,
                       const __nv_bfloat16* injection,
                       __nv_bfloat16* updated_hidden,
                       __nv_bfloat16* next_block_input,
                       __nv_bfloat16* next_injection, int m,
                       cudaStream_t stream) override {
    check(hidden && block_output && injection && updated_hidden &&
              next_block_input && next_injection && m == 4 && stream,
          "linear HC combine drift");
    calls.push_back("combine_and_mix");
    ordered.push_back("combine_and_mix");
  }
  void synchronize(cudaStream_t stream) override {
    check(stream != nullptr, "linear HC completion stream drift");
    calls.push_back("synchronize");
    ordered.push_back("synchronize");
    ++synchronizations;
    if (synchronizations == fail_on_sync) {
      throw std::runtime_error("deferred CUDA failure");
    }
  }
  int fail_on_sync = -1;
  int synchronizations = 0;
  std::vector<std::string_view> calls;
  std::vector<std::string_view>& ordered;
};

}  // namespace

int main() {
  try {
    std::vector<std::string_view> ordered;
    Graph graph(ordered);
    Reducer reducer(ordered);
    HyperConnection hc(ordered);
    Trace trace;
    decode::LinearAttentionLayer layer(graph, reducer, hc, trace);
    __nv_bfloat16 hidden{}, block_input{}, injection{}, conv{}, updated{},
        moe_input{}, next_injection{};
    float recurrent{}, reduced{};
    std::int32_t state_index = 1;
    const auto result = layer.execute(
        1, 4, &hidden, &block_input, &injection, &conv, &recurrent,
        &state_index, &reduced, &updated, &moe_input, &next_injection,
        "trace", "request", graph.expected_stream);
    check(result.generation == 1 && result.m_bucket == 4,
          "linear publication drift");
    check(hc.calls == std::vector<std::string_view>{
                          "mix", "synchronize", "synchronize",
                          "combine_and_mix", "synchronize"} &&
              graph.calls == std::vector<std::string_view>{"gdn_graph"} &&
              reducer.calls == std::vector<std::string_view>{"reduce"},
          "linear layer order drift");
    check(ordered == std::vector<std::string_view>{
                         "mix", "synchronize", "gdn_graph", "synchronize",
                         "reduce", "combine_and_mix", "synchronize"},
          "linear cross-component order drift");
    check(trace.stages.size() == 5, "linear OTEL stage count drift");

    std::vector<std::string_view> failing_order;
    Graph failing_graph(failing_order);
    Reducer failing_reducer(failing_order);
    HyperConnection failing_hc(failing_order);
    // The second fence occurs after the graph may have mutated GDN state.
    failing_hc.fail_on_sync = 2;
    Trace failing_trace;
    decode::LinearAttentionLayer failing(
        failing_graph, failing_reducer, failing_hc, failing_trace);
    bool failed = false;
    try {
      failing.execute(
          1, 4, &hidden, &block_input, &injection, &conv, &recurrent,
          &state_index, &reduced, &updated, &moe_input, &next_injection,
          "trace-fail", "request-fail", failing_graph.expected_stream);
    } catch (const std::runtime_error&) {
      failed = true;
    }
    check(failed && failing_trace.stages.size() == 2 &&
              failing_trace.stages[0] ==
                  "rocket.qwen38.layer0.linear.attn_hc_mix" &&
              failing_trace.stages[1] ==
                  "rocket.qwen38.layer0.linear.lifecycle",
          "linear deferred failure published");
    check(failing_order == std::vector<std::string_view>{
                               "mix", "synchronize", "gdn_graph",
                               "synchronize"},
          "post-mutation completion failure order drift");
    try {
      failing.execute(
          1, 4, &hidden, &block_input, &injection, &conv, &recurrent,
          &state_index, &reduced, &updated, &moe_input, &next_injection,
          "trace-retry", "request-retry", failing_graph.expected_stream);
      check(false, "faulted GDN transition retried");
    } catch (const decode::DecodeExecutionContractError&) {
    }
    std::puts("qwen38 linear-attention layer contract passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

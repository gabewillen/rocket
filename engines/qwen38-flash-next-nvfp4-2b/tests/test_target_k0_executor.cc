// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_executor.h"

#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace decode = rocket::qwen38::decode;
namespace pr = rocket::qwen38::pair_reduce;

namespace {

void check(bool value) {
  if (!value) throw std::runtime_error("K0 executor proof failed");
}

struct Sink final : pr::OtelStageSink {
  void emit_span_and_log(const pr::SpanRecord& value) noexcept override {
    ++spans;
    last = value.outcome;
  }
  void record_duration(const pr::MetricPoint&) noexcept override { ++metrics; }
  int spans = 0;
  int metrics = 0;
  pr::Outcome last = pr::Outcome::kContractError;
};

struct PhysicalReducer final : decode::HiddenPartialReducer {
  int rank() const noexcept override { return 0; }
  int world_size() const noexcept override { return 2; }
  void reduce(const __nv_bfloat16*, float*, int m, std::string_view,
              std::string_view, cudaStream_t) override {
    if (m != 1) throw std::logic_error("reduction M changed");
    ++calls;
  }
  int calls = 0;
};

struct Layer final : decode::TargetK0LayerPort {
  Layer(int layer, decode::HiddenPartialReducer& attention,
        decode::HiddenPartialReducer& moe, bool fail = false)
      : layer_(layer), attention_(attention), moe_(moe), fail_(fail) {}
  int rank() const noexcept override { return 0; }
  int layer() const noexcept override { return layer_; }
  decode::TargetK0AttentionKind attention_kind() const noexcept override {
    return decode::is_qsa_layer(layer_)
               ? decode::TargetK0AttentionKind::kQsa
               : decode::TargetK0AttentionKind::kGdn;
  }
  bool authenticated() const noexcept override { return true; }
  const decode::HiddenPartialReducer* attention_reducer_identity()
      const noexcept override { return &attention_; }
  const decode::HiddenPartialReducer* moe_reducer_identity()
      const noexcept override { return &moe_; }
  void wait_source(cudaStream_t) override { ++waits; }
  void execute_row(std::uint64_t, const __nv_bfloat16*, __nv_bfloat16*,
                   cudaStream_t stream,
                   decode::TargetK0ExecutionProgress* progress) override {
    decode::target_k0_enter_layer(
        progress, decode::TargetK0LayerExecutionStage::kAttentionReduction);
    if (fail_) throw std::runtime_error("injected layer failure");
    __nv_bfloat16 partial{};
    float reduced{};
    attention_.reduce(&partial, &reduced, 1, "trace", "request", stream);
    moe_.reduce(&partial, &reduced, 1, "trace", "request", stream);
    ++rows;
  }
  int layer_;
  decode::HiddenPartialReducer& attention_;
  decode::HiddenPartialReducer& moe_;
  int waits = 0;
  int rows = 0;
  bool fail_ = false;
};

struct TokenIo final : decode::TargetK0TokenIoPort {
  int rank() const noexcept override { return 0; }
  bool authenticated() const noexcept override { return true; }
  void wait_source(cudaStream_t) override { ++waits; }
  void embed_row(std::int32_t, std::uint64_t, __nv_bfloat16*,
                 cudaStream_t) override { ++embeds; }
  decode::TargetK0TokenOutput finish_prefill(
      const __nv_bfloat16*, std::uint64_t, cudaStream_t,
      decode::TargetK0ExecutionProgress* progress) override {
    decode::target_k0_enter_stage(
        progress, decode::TargetK0ExecutionStage::kTerminalFence);
    ++finishes;
    return {&final_hidden, logits.data(), 248'046};
  }
  int waits = 0;
  int embeds = 0;
  int finishes = 0;
  __nv_bfloat16 final_hidden{};
  std::array<float, decode::kTargetK0LocalVocab> logits{};
};

struct Comparator final : decode::TargetK0OracleComparator {
  int rank() const noexcept override { return 0; }
  int rows() const noexcept override { return 35; }
  std::string_view manifest_sha256() const noexcept override {
    return decode::kTargetK0OracleManifestSha256;
  }
  bool authenticated() const noexcept override { return true; }
  std::int32_t expected_input_token(int) const override { return 13; }
  void compare(decode::TargetK0Boundary boundary, int row, int layer,
               const void* values, std::size_t elements,
               cudaStream_t stream) override {
    check(values && stream && row >= 0 && row < rows());
    if (boundary == decode::TargetK0Boundary::kLayer)
      check(layer >= 0 && layer < 48 &&
            elements == decode::kTargetK0HyperHidden);
    ++boundaries;
  }
  void compare_token(std::int32_t token) override {
    check(token == 248'046);
    ++tokens;
  }
  int boundaries = 0;
  int tokens = 0;
};

}  // namespace

int main() {
  try {
    PhysicalReducer reducer;
    Sink sink;
    decode::TargetK0PairReduceSchedule reductions(reducer, sink);
    std::array<std::unique_ptr<Layer>, decode::kDecoderLayers> owners;
    std::array<decode::TargetK0LayerPort*, decode::kDecoderLayers> layers{};
    for (int layer = 0; layer < decode::kDecoderLayers; ++layer) {
      owners[layer] = std::make_unique<Layer>(
          layer, reductions.attention_port(layer), reductions.moe_port(layer));
      layers[layer] = owners[layer].get();
    }
    TokenIo token_io;
    Comparator comparator;
    std::array<__nv_bfloat16, decode::kTargetK0HyperHidden> hidden_a{};
    std::array<__nv_bfloat16, decode::kTargetK0HyperHidden> hidden_b{};
    auto stream = reinterpret_cast<cudaStream_t>(0x1);
    decode::TargetK0Executor executor(
        0, layers, token_io, reductions, comparator, sink,
        {hidden_a.data(), hidden_b.data()}, stream);
    std::array<std::int32_t, 35> prompt{};
    prompt.fill(13);
    decode::TargetK0ExecutionProgress progress;
    const auto result = executor.execute_prefill(1, prompt, "trace", "request",
                                                 &progress);
    check(result.token == 248'046 && result.rows == 35 &&
          result.final_generation == 35 &&
          executor.phase() == decode::TargetK0ExecutorPhase::kCompleted &&
          reducer.calls == 35 * 96 && token_io.waits == 1 &&
          token_io.embeds == 35 && token_io.finishes == 1 &&
          comparator.boundaries == 35 * 49 + 2 && comparator.tokens == 1 &&
          sink.last == pr::Outcome::kOk);
    check(progress.stage == decode::TargetK0ExecutionStage::kComplete &&
          progress.row == 34 && progress.layer == -1 &&
          progress.layer_stage ==
              decode::TargetK0LayerExecutionStage::kNone);
    for (const auto& layer : owners)
      check(layer->waits == 1 && layer->rows == 35);

    PhysicalReducer failed_reducer;
    Sink failed_sink;
    decode::TargetK0PairReduceSchedule failed_schedule(failed_reducer,
                                                       failed_sink);
    std::array<std::unique_ptr<Layer>, decode::kDecoderLayers> failed_owners;
    std::array<decode::TargetK0LayerPort*, decode::kDecoderLayers>
        failed_layers{};
    for (int layer = 0; layer < decode::kDecoderLayers; ++layer) {
      failed_owners[layer] = std::make_unique<Layer>(
          layer, failed_schedule.attention_port(layer),
          failed_schedule.moe_port(layer), layer == 7);
      failed_layers[layer] = failed_owners[layer].get();
    }
    TokenIo failed_token_io;
    Comparator failed_comparator;
    decode::TargetK0Executor failed_executor(
        0, failed_layers, failed_token_io, failed_schedule, failed_comparator,
        failed_sink, {hidden_a.data(), hidden_b.data()}, stream);
    decode::TargetK0ExecutionProgress failed_progress;
    bool execution_failed = false;
    try {
      (void)failed_executor.execute_prefill(1, prompt, "trace", "request",
                                            &failed_progress);
    } catch (const std::runtime_error&) {
      execution_failed = true;
    }
    check(execution_failed &&
          failed_executor.phase() == decode::TargetK0ExecutorPhase::kFaulted &&
          failed_progress.stage ==
              decode::TargetK0ExecutionStage::kLayerExecution &&
          failed_progress.row == 0 && failed_progress.layer == 7 &&
          failed_progress.layer_stage ==
              decode::TargetK0LayerExecutionStage::kAttentionReduction &&
          failed_sink.spans == 1 &&
          failed_sink.last == pr::Outcome::kCudaError);

    PhysicalReducer rejected_reducer;
    Sink rejected_sink;
    decode::TargetK0PairReduceSchedule rejected_schedule(rejected_reducer,
                                                          rejected_sink);
    std::array<std::unique_ptr<Layer>, decode::kDecoderLayers> rejected_owners;
    std::array<decode::TargetK0LayerPort*, decode::kDecoderLayers>
        rejected_layers{};
    for (int layer = 0; layer < decode::kDecoderLayers; ++layer) {
      rejected_owners[layer] = std::make_unique<Layer>(
          layer, rejected_schedule.attention_port(layer),
          rejected_schedule.moe_port(layer));
      rejected_layers[layer] = rejected_owners[layer].get();
    }
    rejected_owners[7] = std::make_unique<Layer>(
        7, rejected_schedule.attention_port(8),
        rejected_schedule.moe_port(7));
    rejected_layers[7] = rejected_owners[7].get();
    bool rejected = false;
    try {
      decode::TargetK0Executor invalid(
          0, rejected_layers, token_io, rejected_schedule, comparator,
          rejected_sink, {hidden_a.data(), hidden_b.data()}, stream);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && rejected_reducer.calls == 0);

    std::puts("qwen38 K0 executor: authenticated 35x48 composition order passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

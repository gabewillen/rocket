// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_prefill.h"

#include <array>
#include <stdexcept>

namespace attention = rocket::qwen38::attention;
namespace decode = rocket::qwen38::decode;
namespace pr = rocket::qwen38::pair_reduce;

namespace {
attention::TargetQsaStateView state{};

struct Generations final : decode::TargetLayer3GenerationOwner {
  int rank() const noexcept override { return 1; }
  int layer() const noexcept override { return 3; }
  bool authenticated() const noexcept override { return true; }
  const attention::TargetQsaStateView& view(
      int row, std::uint64_t generation) const override {
    if (row != views++) throw std::runtime_error("row order changed");
    state.rank = 1;
    state.layer = 3;
    state.rows = 1;
    state.generation = generation;
    state.expected_generation = generation;
    return state;
  }
  void enqueue_prepare(int row, std::uint64_t generation,
                       cudaStream_t) override {
    if (row != calls || generation != static_cast<std::uint64_t>(row + 1))
      throw std::runtime_error("generation enqueue order changed");
    ++calls;
  }
  mutable int views = 0;
  int calls = 0;
};

struct Rows final : decode::TargetLayer3RowPort {
  int rank() const noexcept override { return 1; }
  int layer() const noexcept override { return 3; }
  bool authenticated() const noexcept override { return true; }
  decode::TargetFullLayerResult execute_row(
      std::uint64_t generation, const attention::TargetQsaStateView&,
      const __nv_bfloat16*, const decode::TargetLayer3RowBuffers&,
      __nv_bfloat16* output, cudaStream_t) override {
    ++calls;
    return {generation, 1, 3, output};
  }
  int calls = 0;
};

struct Comparator final : decode::TargetLayer3Comparator {
  bool authenticated() const noexcept override { return true; }
  bool compare_row34(const __nv_bfloat16* output,
                     cudaStream_t) override {
    observed = output;
    return accept;
  }
  const __nv_bfloat16* observed = nullptr;
  bool accept = true;
};

struct Telemetry final : pr::OtelStageSink {
  void emit_span_and_log(const pr::SpanRecord& point) noexcept override {
    ++calls;
    last = point.outcome;
    last_rank = point.rank;
  }
  void record_duration(const pr::MetricPoint& point) noexcept override {
    ++metrics;
    last_metric = point.outcome;
  }
  int calls = 0;
  int metrics = 0;
  int last_rank = -2;
  pr::Outcome last = pr::Outcome::kContractError;
  pr::Outcome last_metric = pr::Outcome::kContractError;
};
}  // namespace

int main() {
  Generations generations;
  Rows rows;
  Comparator comparator;
  Telemetry telemetry;
  decode::TargetLayer3Prefill prefill(
      1, rows, generations, comparator, telemetry);
  std::array<__nv_bfloat16, 35 * 4 * 2560> before{};
  std::array<__nv_bfloat16, 35 * 4 * 2560> after{};
  __nv_bfloat16 bf16{};
  float f32{};
  decode::TargetLayer3RowBuffers buffers{
      &bf16, &bf16, &f32, &bf16, &bf16, &bf16, &f32};
  const auto* result = prefill.execute(
      before.data(), after.data(), buffers,
      reinterpret_cast<cudaStream_t>(0x10));
  if (generations.views != 35 || generations.calls != 35 || rows.calls != 35 ||
      result != after.data() + 34 * 4 * 2560 || comparator.observed != result ||
      telemetry.calls != 1 || telemetry.metrics != 1 ||
      telemetry.last != pr::Outcome::kOk ||
      telemetry.last_metric != pr::Outcome::kOk)
    return 1;
  try {
    prefill.execute(before.data(), after.data(), buffers,
                    reinterpret_cast<cudaStream_t>(0x10));
    return 2;
  } catch (const std::invalid_argument&) {
  }
  if (telemetry.calls != 2 || telemetry.metrics != 2 ||
      telemetry.last == pr::Outcome::kOk) return 3;

  Generations mismatch_generations;
  Rows mismatch_rows;
  Comparator mismatch;
  mismatch.accept = false;
  Telemetry mismatch_telemetry;
  decode::TargetLayer3Prefill rejected(
      1, mismatch_rows, mismatch_generations, mismatch, mismatch_telemetry);
  try {
    rejected.execute(before.data(), after.data(), buffers,
                     reinterpret_cast<cudaStream_t>(0x10));
    return 4;
  } catch (const std::logic_error&) {
  }
  if (mismatch_telemetry.calls != 1 ||
      mismatch_telemetry.last == pr::Outcome::kOk)
    return 5;
  Generations invalid_generations;
  Rows invalid_rows;
  Comparator invalid_comparator;
  Telemetry invalid_telemetry;
  try {
    decode::TargetLayer3Prefill invalid(
        123456, invalid_rows, invalid_generations,
        invalid_comparator, invalid_telemetry);
    return 6;
  } catch (const std::invalid_argument&) {
  }
  return invalid_telemetry.calls == 1 && invalid_telemetry.metrics == 0 &&
                 invalid_telemetry.last_rank == -1
             ? 0
             : 7;
}

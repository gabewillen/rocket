// SPDX-License-Identifier: Apache-2.0
#include "moe/native_target_moe_graph.h"

#include <type_traits>
#include <stdexcept>

namespace decode = rocket::qwen38::decode;
namespace moe = rocket::qwen38::moe;

static_assert(std::is_final_v<moe::NativeTargetMoeGraph>);
static_assert(std::is_base_of_v<decode::TargetMoeGraph,
                                moe::NativeTargetMoeGraph>);
static_assert(std::is_constructible_v<
              moe::NativeTargetMoeGraph, moe::TargetFullMoeC1Port&,
              const moe::TargetFullMoeC1Workspace&, __nv_bfloat16*,
              moe::TargetFullMoeOtelSink&>);

namespace {
struct Port final : moe::TargetFullMoeC1Port {
  moe::TargetDenseIdentity value{};
  mutable int calls = 0;
  moe::TargetDenseOutcome outcome = moe::TargetDenseOutcome::kOk;

  moe::TargetDenseOutcome enqueue(
      const moe::TargetFullMoeC1Launch& launch) const noexcept override {
    if (!launch.hidden_bf16 || !launch.rank_local_partial_bf16 ||
        !launch.stream)
      return moe::TargetDenseOutcome::kContractError;
    ++calls;
    return outcome;
  }
  moe::TargetDenseOutcome enqueue_with_telemetry(
      const moe::TargetFullMoeC1Launch& launch,
      moe::TargetFullMoeOtelSink& telemetry) const noexcept override {
    const auto result = enqueue(launch);
    for (auto component : {moe::TargetFullMoeComponent::kRouter,
                           moe::TargetFullMoeComponent::kLocalization,
                           moe::TargetFullMoeComponent::kRoutedExperts,
                           moe::TargetFullMoeComponent::kSharedExpert})
      telemetry.emit({component, result, value.rank, value.layer});
    return result;
  }
  const moe::TargetDenseIdentity& identity() const noexcept override {
    return value;
  }
};

struct Telemetry final : moe::TargetFullMoeOtelSink {
  void emit(const moe::TargetFullMoeOtelPoint& point) noexcept override {
    ++count;
    last = point;
  }
  int count = 0;
  moe::TargetFullMoeOtelPoint last{};
};
}  // namespace

int main() {
  Port port;
  port.value.rank = 1;
  port.value.layer = 3;
  moe::TargetFullMoeC1Workspace workspace{};
  auto* output = reinterpret_cast<__nv_bfloat16*>(0x10);
  auto* input = reinterpret_cast<const __nv_bfloat16*>(0x20);
  auto stream = reinterpret_cast<cudaStream_t>(0x30);
  Telemetry telemetry;
  moe::NativeTargetMoeGraph graph(port, workspace, output, telemetry);
  if (graph.rank() != 1 || graph.layer() != 3 || graph.projected_output())
    return 1;
  graph.launch(input, 1, 1, stream);
  if (port.calls != 1 || graph.projected_output() || telemetry.count != 4)
    return 2;
  graph.publish_after_fence(1);
  if (graph.projected_output() != output) return 3;
  const auto routed = graph.component_identity(
      moe::TargetFullMoeComponent::kRoutedExperts);
  const auto shared = graph.component_identity(
      moe::TargetFullMoeComponent::kSharedExpert);
  if (routed.identity.rank != 1 ||
      routed.serving_dtype != moe::TargetMoeServingDtype::kNvfp4 ||
      shared.serving_dtype != moe::TargetMoeServingDtype::kBfloat16)
    return 4;
  try {
    graph.launch(input, 1, 1, stream);
    return 5;
  } catch (const std::invalid_argument&) {
  }
  if (graph.projected_output()) return 6;

  Port failed;
  failed.value.layer = 3;
  failed.outcome = moe::TargetDenseOutcome::kCudaError;
  Telemetry failed_telemetry;
  moe::NativeTargetMoeGraph failed_graph(
      failed, workspace, output, failed_telemetry);
  try {
    failed_graph.launch(input, 1, 1, stream);
    return 7;
  } catch (const std::runtime_error&) {
  }
  return failed_graph.projected_output() ? 8 : 0;
}

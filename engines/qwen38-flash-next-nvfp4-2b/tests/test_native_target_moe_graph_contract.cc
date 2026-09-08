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
  mutable int publish_calls = 0;
  moe::TargetDenseOutcome outcome = moe::TargetDenseOutcome::kOk;
  moe::TargetDenseOutcome publish_outcome = moe::TargetDenseOutcome::kOk;
  bool waited = false;

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
                           moe::TargetFullMoeComponent::kStaging,
                           moe::TargetFullMoeComponent::kRoutedExperts,
                           moe::TargetFullMoeComponent::kSharedExpert})
      telemetry.emit({component, result, value.rank, value.layer});
    return result;
  }
  const moe::TargetDenseIdentity& identity() const noexcept override {
    return value;
  }
  void wait_source(cudaStream_t stream) override {
    if (!stream || waited) throw std::invalid_argument("source wait");
    waited = true;
  }
  moe::TargetDenseOutcome publish_after_fence(
      std::uint64_t generation,
      moe::TargetFullMoeOtelSink&) noexcept override {
    ++publish_calls;
    return waited && generation != 0 ? publish_outcome
                                      : moe::TargetDenseOutcome::kContractError;
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
struct CudaApi final : moe::NativeTargetMoeCudaApi {
  cudaError_t stream_synchronize(cudaStream_t stream) noexcept override {
    if (!stream) return cudaErrorInvalidValue;
    ++syncs;
    return stream_outcome;
  }
  cudaError_t device_synchronize() noexcept override {
    ++device_syncs;
    return device_outcome;
  }
  int syncs = 0;
  int device_syncs = 0;
  cudaError_t stream_outcome = cudaSuccess;
  cudaError_t device_outcome = cudaSuccess;
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
  CudaApi cuda_api;
  for (const int layer : {0, 47}) {
    port.value.layer = layer;
    moe::NativeTargetMoeGraph accepted(
        port, workspace, output, telemetry, &cuda_api);
    if (accepted.layer() != layer) return 1;
  }
  for (const int layer : {-1, 48}) {
    port.value.layer = layer;
    try {
      moe::NativeTargetMoeGraph rejected(
          port, workspace, output, telemetry, &cuda_api);
      return 1;
    } catch (const std::invalid_argument&) {
    }
  }
  port.value.layer = 3;
  moe::NativeTargetMoeGraph graph(
      port, workspace, output, telemetry, &cuda_api);
  if (graph.rank() != 1 || graph.layer() != 3 || graph.projected_output())
    return 1;
  graph.wait_source(stream);
  graph.launch(input, 1, 1, stream);
  if (port.calls != 1 || graph.projected_output() || telemetry.count != 5)
    return 2;
  graph.terminal_fence_succeeded(1);
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

  Port late_failed;
  late_failed.value.layer = 3;
  late_failed.publish_outcome = moe::TargetDenseOutcome::kContractError;
  Telemetry late_telemetry;
  moe::NativeTargetMoeGraph late_graph(
      late_failed, workspace, output, late_telemetry, &cuda_api);
  late_graph.wait_source(stream);
  late_graph.launch(input, 1, 1, stream);
  late_graph.terminal_fence_succeeded(1);
  try {
    late_graph.publish_after_fence(1);
    return 7;
  } catch (const decode::DecodeExecutionContractError&) {
  }
  if (late_graph.projected_output()) return 8;
  if (late_failed.publish_calls != 1) return 9;

  Port unfenced;
  unfenced.value.layer = 3;
  Telemetry unfenced_telemetry;
  moe::NativeTargetMoeGraph unfenced_graph(
      unfenced, workspace, output, unfenced_telemetry, &cuda_api);
  unfenced_graph.wait_source(stream);
  unfenced_graph.launch(input, 1, 1, stream);
  try {
    unfenced_graph.publish_after_fence(1);
    return 10;
  } catch (const std::invalid_argument&) {
  }
  if (unfenced.publish_calls != 0 || unfenced_graph.projected_output())
    return 11;

  Port fenced_failed;
  fenced_failed.value.layer = 3;
  Telemetry fenced_telemetry;
  moe::NativeTargetMoeGraph fenced_graph(
      fenced_failed, workspace, output, fenced_telemetry, &cuda_api);
  fenced_graph.wait_source(stream);
  fenced_graph.launch(input, 1, 1, stream);
  fenced_graph.fault_after_fence(1);
  if (fenced_failed.publish_calls != 1 || fenced_graph.projected_output())
    return 12;

  Port no_wait;
  no_wait.value.layer = 3;
  Telemetry no_wait_telemetry;
  moe::NativeTargetMoeGraph no_wait_graph(
      no_wait, workspace, output, no_wait_telemetry, &cuda_api);
  try {
    no_wait_graph.launch(input, 1, 1, stream);
    return 13;
  } catch (const std::invalid_argument&) {
  }
  if (no_wait_graph.projected_output()) return 14;

  Port failed;
  failed.value.layer = 3;
  failed.outcome = moe::TargetDenseOutcome::kCudaError;
  Telemetry failed_telemetry;
  {
    moe::NativeTargetMoeGraph failed_graph(
        failed, workspace, output, failed_telemetry, &cuda_api);
    failed_graph.wait_source(stream);
    try {
      failed_graph.launch(input, 1, 1, stream);
      return 15;
    } catch (const decode::DecodeExecutionCudaError&) {
    }
    if (failed_graph.projected_output()) return 16;
    try {
      failed_graph.launch(input, 1, 1, stream);
      return 17;
    } catch (const std::invalid_argument&) {
    }
  }
  if (failed.publish_calls != 1) return 18;

  Port contract_failed;
  contract_failed.value.layer = 3;
  contract_failed.outcome = moe::TargetDenseOutcome::kContractError;
  Telemetry contract_telemetry;
  {
    moe::NativeTargetMoeGraph contract_graph(
        contract_failed, workspace, output, contract_telemetry, &cuda_api);
    contract_graph.wait_source(stream);
    try {
      contract_graph.launch(input, 1, 1, stream);
      return 19;
    } catch (const decode::DecodeExecutionContractError&) {
    }
  }
  if (contract_failed.publish_calls != 1) return 20;

  Port abandoned_fenced;
  abandoned_fenced.value.layer = 3;
  Telemetry abandoned_telemetry;
  {
    moe::NativeTargetMoeGraph abandoned_graph(
        abandoned_fenced, workspace, output, abandoned_telemetry, &cuda_api);
    abandoned_graph.wait_source(stream);
    abandoned_graph.launch(input, 1, 1, stream);
    abandoned_graph.terminal_fence_succeeded(1);
    try {
      abandoned_graph.terminal_fence_succeeded(1);
      return 21;
    } catch (const std::invalid_argument&) {
    }
  }
  if (abandoned_fenced.publish_calls != 1) return 22;

  Port invalid_fault;
  invalid_fault.value.layer = 3;
  Telemetry invalid_fault_telemetry;
  moe::NativeTargetMoeGraph invalid_fault_graph(
      invalid_fault, workspace, output, invalid_fault_telemetry, &cuda_api);
  invalid_fault_graph.wait_source(stream);
  invalid_fault_graph.fault_after_fence(0);
  try {
    invalid_fault_graph.launch(input, 1, 1, stream);
    return 23;
  } catch (const std::invalid_argument&) {
  }

  Port ready_fault;
  ready_fault.value.layer = 3;
  Telemetry ready_fault_telemetry;
  moe::NativeTargetMoeGraph ready_fault_graph(
      ready_fault, workspace, output, ready_fault_telemetry, &cuda_api);
  ready_fault_graph.fault_after_fence(0);
  try {
    ready_fault_graph.wait_source(stream);
    return 24;
  } catch (const std::invalid_argument&) {
  }

  Port published_fault;
  published_fault.value.layer = 3;
  Telemetry published_fault_telemetry;
  moe::NativeTargetMoeGraph published_fault_graph(
      published_fault, workspace, output, published_fault_telemetry, &cuda_api);
  published_fault_graph.wait_source(stream);
  published_fault_graph.launch(input, 1, 1, stream);
  published_fault_graph.terminal_fence_succeeded(1);
  published_fault_graph.publish_after_fence(1);
  published_fault_graph.fault_after_fence(1);
  if (published_fault_graph.projected_output()) return 25;
  try {
    published_fault_graph.launch(input, 2, 1, stream);
    return 26;
  } catch (const std::invalid_argument&) {
  }

  Port fallback_port;
  fallback_port.value.layer = 3;
  fallback_port.outcome = moe::TargetDenseOutcome::kCudaError;
  Telemetry fallback_telemetry;
  CudaApi fallback_api;
  fallback_api.stream_outcome = cudaErrorInvalidResourceHandle;
  moe::NativeTargetMoeGraph fallback_graph(
      fallback_port, workspace, output, fallback_telemetry, &fallback_api);
  fallback_graph.wait_source(stream);
  try {
    fallback_graph.launch(input, 1, 1, stream);
    return 27;
  } catch (const decode::DecodeExecutionCudaError&) {
  }
  if (!fallback_graph.drain_for_destruction() ||
      fallback_api.syncs != 1 || fallback_api.device_syncs != 1 ||
      fallback_port.publish_calls != 1)
    return 28;

  Port quarantine_port;
  quarantine_port.value.layer = 3;
  quarantine_port.outcome = moe::TargetDenseOutcome::kCudaError;
  Telemetry quarantine_telemetry;
  CudaApi quarantine_api;
  quarantine_api.stream_outcome = cudaErrorInvalidResourceHandle;
  quarantine_api.device_outcome = cudaErrorUnknown;
  auto* quarantine_graph = new moe::NativeTargetMoeGraph(
      quarantine_port, workspace, output, quarantine_telemetry,
      &quarantine_api);
  quarantine_graph->wait_source(stream);
  try {
    quarantine_graph->launch(input, 1, 1, stream);
    return 29;
  } catch (const decode::DecodeExecutionCudaError&) {
  }
  if (quarantine_graph->drain_for_destruction() ||
      quarantine_port.publish_calls != 0 || quarantine_telemetry.count != 6)
    return 30;
  if (quarantine_graph->drain_for_destruction() ||
      quarantine_api.syncs != 1 || quarantine_api.device_syncs != 1 ||
      quarantine_port.publish_calls != 0 || quarantine_telemetry.count != 6)
    return 31;
  // Mirrors the production owner's process-lifetime quarantine. Deleting this
  // object would discard a still-live borrowed participant/resource graph.
  (void)quarantine_graph;
  return 0;
}

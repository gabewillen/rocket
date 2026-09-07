// SPDX-License-Identifier: Apache-2.0
#include "mtp/native_executor.h"

#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#undef assert
#define assert(condition)            \
  do {                               \
    if (!(condition)) std::abort();  \
  } while (false)

namespace {

using rocket::qwen38::mtp::BoundGraph;
using rocket::qwen38::mtp::ExpertUsageMetric;
using rocket::qwen38::mtp::ImmutableSlabs;
using rocket::qwen38::mtp::NativeExecutor;
using rocket::qwen38::mtp::Outcome;
using rocket::qwen38::mtp::PhaseMetric;

class Sink final : public rocket::qwen38::mtp::TelemetrySink {
 public:
  void record_phase(const PhaseMetric& metric) noexcept override {
    phases.push_back(metric);
  }
  void record_expert_usage(const ExpertUsageMetric& metric) noexcept override {
    experts.push_back(metric);
  }
  std::vector<PhaseMetric> phases;
  std::vector<ExpertUsageMetric> experts;
};

cudaGraphExec_t empty_graph() {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  assert(cudaGraphCreate(&graph, 0) == cudaSuccess);
  cudaGraphNode_t node = nullptr;
  assert(cudaGraphAddEmptyNode(&node, graph, nullptr, 0) == cudaSuccess);
  assert(cudaGraphInstantiate(&executable, graph, 0) == cudaSuccess);
  assert(cudaGraphDestroy(graph) == cudaSuccess);
  return executable;
}

}  // namespace

int main() {
  static_assert(rocket::qwen38::mtp::allowed_graph_key({1, 16}));
  static_assert(rocket::qwen38::mtp::allowed_graph_key({2, 16}));
  static_assert(rocket::qwen38::mtp::allowed_graph_key({3, 16}));
  static_assert(rocket::qwen38::mtp::allowed_graph_key({4, 16}));
  static_assert(!rocket::qwen38::mtp::allowed_graph_key({5, 16}));
  static_assert(!rocket::qwen38::mtp::allowed_graph_key({6, 8}));
  static_assert(rocket::qwen38::mtp::allowed_graph_key({7, 4}));
  constexpr int kDepth = 2;
  constexpr int kSequences = 2;
  constexpr std::size_t kStateBytes = 4;
  cudaStream_t stream = nullptr;
  assert(cudaStreamCreate(&stream) == cudaSuccess);

  BoundGraph graph;
  graph.key = {kDepth, kSequences};
  std::array<std::array<std::int32_t, 16>, kDepth> router{{
      {0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 0, 1, 2, 3, 3, 3},
      {7, 7, 8, 8, 9, 9, 10, 10, 7, 8, 9, 10, 10, 10, 10, 10},
  }};
  std::array<std::int32_t, kSequences * (kDepth + 1)> tokens{
      11, 12, 17, 19, 18, 20};
  std::array<std::array<std::byte, kSequences * kStateBytes>, kDepth>
      snapshots{};
  snapshots[0].fill(std::byte{0x11});
  snapshots[1].fill(std::byte{0x22});
  std::array<std::byte, kSequences * kStateBytes> active{};

  std::array<std::int32_t*, kDepth> router_device{};
  std::int32_t* token_device = nullptr;
  std::array<std::byte*, kDepth> snapshot_device{};
  std::byte* active_device = nullptr;
  for (int step = 0; step < kDepth; ++step) {
    assert(cudaMalloc(&router_device[step], sizeof(router[step])) == cudaSuccess);
    assert(cudaMemcpy(router_device[step], router[step].data(),
                      sizeof(router[step]), cudaMemcpyHostToDevice) ==
           cudaSuccess);
    assert(cudaMalloc(&snapshot_device[step], snapshots[step].size()) ==
           cudaSuccess);
    assert(cudaMemcpy(snapshot_device[step], snapshots[step].data(),
                      snapshots[step].size(), cudaMemcpyHostToDevice) ==
           cudaSuccess);
    graph.router_expert_ids[step] = router_device[step];
    graph.causal_snapshots[step] = snapshot_device[step];
    for (auto& executable : graph.phase_graphs[step]) executable = empty_graph();
  }
  assert(cudaMalloc(&token_device, sizeof(tokens)) == cudaSuccess);
  assert(cudaMemcpy(token_device, tokens.data(), sizeof(tokens),
                    cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMalloc(&active_device, active.size()) == cudaSuccess);
  graph.verification_tokens = token_device;
  graph.active_causal_state = active_device;
  graph.state_bytes_per_sequence = kStateBytes;

  std::array<std::uint8_t, 32> digest{};
  digest[0] = 1;
  Sink sink;
  {
    NativeExecutor executor(
        ImmutableSlabs{reinterpret_cast<void*>(1), 1,
                       reinterpret_cast<void*>(2),
                       rocket::qwen38::mtp::kNativeRankSlabBytes, digest},
        graph, sink, stream);
    const auto result = executor.draft(1);
    assert(result.verification_tokens == token_device);
    assert(result.depth == kDepth && result.sequences == kSequences);
    // This is the one terminal fence owned by the target verifier.
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    executor.export_telemetry_after_fence(1);
    assert(sink.experts.size() == kDepth);
    assert(sink.experts[0].unique_local_experts == 4);
    assert(sink.experts[1].unique_local_experts == 4);
    assert(sink.experts[0].resident_expert_bytes ==
           4 * rocket::qwen38::mtp::kNvidiaFp8BytesPerLocalExpert);
    for (const auto& phase : sink.phases)
      std::cout << "phase=" << static_cast<int>(phase.phase)
                << " duration_ns=" << phase.duration_ns << '\n';
    for (const auto& usage : sink.experts)
      std::cout << "draft_step=" << usage.draft_step
                << " unique_local_experts=" << usage.unique_local_experts
                << " resident_expert_bytes=" << usage.resident_expert_bytes
                << '\n';
    const std::array<std::int32_t, kSequences> accepted{1, 3};
    std::int32_t* accepted_device = nullptr;
    assert(cudaMalloc(&accepted_device, sizeof(accepted)) == cudaSuccess);
    assert(cudaMemcpy(accepted_device, accepted.data(), sizeof(accepted),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    executor.publish(1, accepted_device);
    // Publication is asynchronous and ordered before the next engine step.
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    assert(cudaMemcpy(active.data(), active_device, active.size(),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    for (std::size_t index = 0; index < kStateBytes; ++index)
      assert(active[index] == std::byte{0x11});
    for (std::size_t index = kStateBytes; index < active.size(); ++index)
      assert(active[index] == std::byte{0x22});
    std::cout << "accepted_widths=1,3 published_snapshots=0,1 exact=1\n";
    assert(executor.phase() == rocket::qwen38::mtp::ExecutorPhase::kReady);
    assert(cudaFree(accepted_device) == cudaSuccess);
  }

  for (int step = 0; step < kDepth; ++step) {
    for (auto executable : graph.phase_graphs[step])
      assert(cudaGraphExecDestroy(executable) == cudaSuccess);
    assert(cudaFree(router_device[step]) == cudaSuccess);
    assert(cudaFree(snapshot_device[step]) == cudaSuccess);
  }
  assert(cudaFree(token_device) == cudaSuccess);
  assert(cudaFree(active_device) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  return 0;
}

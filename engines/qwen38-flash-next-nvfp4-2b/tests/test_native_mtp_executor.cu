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

  std::array<std::int32_t*, kDepth> router_device{};
  std::int32_t* token_device = nullptr;
  std::byte* inactive_device = nullptr;
  for (int step = 0; step < kDepth; ++step) {
    assert(cudaMalloc(&router_device[step], sizeof(router[step])) == cudaSuccess);
    assert(cudaMemcpy(router_device[step], router[step].data(),
                      sizeof(router[step]), cudaMemcpyHostToDevice) ==
           cudaSuccess);
    graph.router_expert_ids[step] = router_device[step];
    for (auto& executable : graph.phase_graphs[step]) executable = empty_graph();
  }
  assert(cudaMalloc(&token_device, sizeof(tokens)) == cudaSuccess);
  assert(cudaMemcpy(token_device, tokens.data(), sizeof(tokens),
                    cudaMemcpyHostToDevice) == cudaSuccess);
  graph.verification_tokens = token_device;
  rocket::qwen38::mtp::StateArena state(kSequences, kDepth, false);
  assert(cudaMalloc(&inactive_device, state.transaction_bytes()) == cudaSuccess);
  assert(cudaMemset(state.prefix(0).multi_hidden, 0x11,
                    kSequences * rocket::qwen38::mtp::kMtpMultiHidden * 2) == cudaSuccess);
  assert(cudaMemset(state.prefix(1).multi_hidden, 0x22,
                    kSequences * rocket::qwen38::mtp::kMtpMultiHidden * 2) == cudaSuccess);

  std::array<std::uint8_t, 32> digest{};
  digest[0] = 1;
  Sink sink;
  {
    const std::array<std::int32_t, kSequences> accepted{1, 3};
    std::int32_t* accepted_device = nullptr;
    assert(cudaMalloc(&accepted_device, sizeof(accepted)) == cudaSuccess);
    assert(cudaMemcpy(accepted_device, accepted.data(), sizeof(accepted),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    NativeExecutor executor(
        ImmutableSlabs{reinterpret_cast<void*>(1), 1,
                       reinterpret_cast<void*>(2),
                       rocket::qwen38::mtp::kNativeRankSlabBytes, digest},
        graph, state, sink, stream);
    const auto result = executor.draft(1);
    assert(result.verification_tokens == token_device);
    assert(result.depth == kDepth && result.sequences == kSequences);
    executor.stage_accept(1, inactive_device, accepted_device,
                          {kSequences, kDepth + 1}, stream);
    // This is the one terminal fence owned by the target verifier.
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    executor.commit(1);
    std::array<std::byte, 2> selected{};
    assert(cudaMemcpy(&selected[0], inactive_device, 1,
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(&selected[1], inactive_device +
                          rocket::qwen38::mtp::kMtpMultiHidden * 2,
                      1, cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(selected[0] == std::byte{0x11} && selected[1] == std::byte{0x22});
    std::cout << "accepted_widths=1,3 published_snapshots=0,1 exact=1\n";
    executor.export_telemetry_after_fence(1);
    assert(sink.experts.size() == kDepth);
    assert(sink.experts[0].unique_local_experts == 4);
    assert(sink.experts[1].unique_local_experts == 4);
    assert(sink.experts[0].resident_expert_bytes ==
           4 * rocket::qwen38::mtp::kNvidiaFp8BytesPerLocalExpert);
    assert(executor.phase() == rocket::qwen38::mtp::ExecutorPhase::kReady);
    assert(cudaFree(accepted_device) == cudaSuccess);
  }

  for (int step = 0; step < kDepth; ++step) {
    for (auto executable : graph.phase_graphs[step])
      assert(cudaGraphExecDestroy(executable) == cudaSuccess);
    assert(cudaFree(router_device[step]) == cudaSuccess);
  }
  assert(cudaFree(token_device) == cudaSuccess);
  assert(cudaFree(inactive_device) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  return 0;
}

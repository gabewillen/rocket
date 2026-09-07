#include "pair_reduce/pair_reduce.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace pr = rocket::qwen38::pair_reduce;

namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}
void check(bool condition, const std::string& message) {
  if (!condition) fail(message);
}
void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    fail(std::string(operation) + ": " + cudaGetErrorString(status));
}

__global__ void pair_sum(const __nv_bfloat16* local,
                         const __nv_bfloat16* peer,
                         __nv_bfloat16* output, std::size_t elements) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < elements)
    output[index] = __float2bfloat16(__bfloat162float(local[index]) +
                                     __bfloat162float(peer[index]));
}

__global__ void bounded_stall(std::uint64_t clocks) {
  const std::uint64_t start = clock64();
  while (clock64() - start < clocks) {}
}

class FakeCollective final : public pr::DeviceCollective {
 public:
  explicit FakeCollective(const std::vector<__nv_bfloat16>& peer) {
    cuda_check(cudaMalloc(&peer_, peer.size() * sizeof(*peer_)), "allocate peer");
    cuda_check(cudaMemcpy(peer_, peer.data(), peer.size() * sizeof(*peer_),
                          cudaMemcpyHostToDevice), "copy peer");
  }
  ~FakeCollective() override { cudaFree(peer_); }
  int rank() const noexcept override { return 0; }
  int world_size() const noexcept override { return 2; }
  void enqueue_sum(const __nv_bfloat16* input, __nv_bfloat16* output,
                   std::size_t elements, cudaStream_t stream) override {
    if (stall_) {
      bounded_stall<<<1, 1, 0, stream>>>(600'000'000);
      return;
    }
    pair_sum<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
        input, peer_, output, elements);
  }
  bool healthy() noexcept override { return healthy_; }
  void abort() noexcept override { aborted_ = true; }
  bool stall_ = false;
  bool healthy_ = true;
  bool aborted_ = false;

 private:
  __nv_bfloat16* peer_ = nullptr;
};

class CaptureOtel final : public pr::OtelStageSink {
 public:
  void emit_span_and_log(const pr::SpanRecord& value) noexcept override {
    stages.emplace_back(value.stage);
    outcomes.push_back(value.outcome);
  }
  void record_duration(const pr::MetricPoint& value) noexcept override {
    buckets.push_back(value.m_bucket);
  }
  std::vector<std::string> stages;
  std::vector<pr::Outcome> outcomes;
  std::vector<int> buckets;
};

std::vector<__nv_bfloat16> partial(int rank, int m) {
  std::vector<__nv_bfloat16> result(static_cast<std::size_t>(m) * pr::kHidden);
  for (std::size_t i = 0; i < result.size(); ++i) {
    const int centered = static_cast<int>(i % 97) - 48;
    result[i] = __float2bfloat16(
        static_cast<float>(rank == 0 ? centered : 3 * centered + 1) / 64.0f);
  }
  return result;
}

void test_all_verifier_shapes() {
  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreate(&stream), "create stream");
  for (const int m : pr::kAllowedM) {
    const auto local = partial(0, m);
    const auto peer = partial(1, m);
    FakeCollective collective(peer);
    CaptureOtel otel;
    pr::PairReduce reduction(collective, otel, 1000);
    __nv_bfloat16* input = nullptr;
    float* output = nullptr;
    cuda_check(cudaMalloc(&input, local.size() * sizeof(*input)), "allocate input");
    cuda_check(cudaMalloc(&output, local.size() * sizeof(*output)), "allocate output");
    cuda_check(cudaMemcpyAsync(input, local.data(), local.size() * sizeof(*input),
                               cudaMemcpyHostToDevice, stream), "copy input");
    reduction.enqueue(input, output, m, "trace", "request", stream);
    reduction.complete(stream, 1, "trace", "request");
    std::vector<float> observed(local.size());
    cuda_check(cudaMemcpy(observed.data(), output, observed.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy output");
    for (std::size_t i = 0; i < observed.size(); ++i) {
      const float expected = __bfloat162float(__float2bfloat16(
          __bfloat162float(local[i]) + __bfloat162float(peer[i])));
      check(std::bit_cast<std::uint32_t>(observed[i]) ==
                std::bit_cast<std::uint32_t>(expected),
            "BF16 sum mismatch at M=" + std::to_string(m));
    }
    check(otel.buckets.size() == 2 && otel.buckets[0] == m &&
              otel.buckets[1] == 0,
          "bounded OTEL buckets changed");
    cudaFree(output);
    cudaFree(input);
  }
  cudaStreamDestroy(stream);
}

void test_cuda_graph_replay() {
  constexpr int m = 128;
  const auto local = partial(0, m);
  const auto peer = partial(1, m);
  FakeCollective collective(peer);
  CaptureOtel otel;
  pr::PairReduce reduction(collective, otel, 1000);
  __nv_bfloat16* input = nullptr;
  float* output = nullptr;
  cudaStream_t stream = nullptr;
  cuda_check(cudaMalloc(&input, local.size() * sizeof(*input)), "allocate graph input");
  cuda_check(cudaMalloc(&output, local.size() * sizeof(*output)), "allocate graph output");
  cuda_check(cudaMemcpy(input, local.data(), local.size() * sizeof(*input),
                        cudaMemcpyHostToDevice), "copy graph input");
  cuda_check(cudaStreamCreate(&stream), "create graph stream");
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
             "begin capture");
  reduction.enqueue(input, output, m, "capture", "capture", stream);
  cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
  cuda_check(cudaGraphInstantiate(&executable, graph, 0), "instantiate graph");
  for (int replay = 0; replay < 2; ++replay) {
    cuda_check(cudaGraphLaunch(executable, stream), "launch graph");
    reduction.complete(stream, 1, "replay", "replay");
  }
  check(reduction.phase() == pr::Phase::kReady,
        "graph replay changed ready phase");
  cudaGraphExecDestroy(executable);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(output);
  cudaFree(input);
}

void test_fault_is_terminal() {
  const auto data = partial(0, 1);
  FakeCollective collective(data);
  collective.healthy_ = false;
  collective.stall_ = true;
  CaptureOtel otel;
  pr::PairReduce reduction(collective, otel, 100);
  __nv_bfloat16* input = nullptr;
  float* output = nullptr;
  cudaStream_t stream = nullptr;
  cuda_check(cudaMalloc(&input, data.size() * sizeof(*input)), "allocate fault input");
  cuda_check(cudaMalloc(&output, data.size() * sizeof(*output)), "allocate fault output");
  cuda_check(cudaStreamCreate(&stream), "create fault stream");
  reduction.enqueue(input, output, 1, "fault", "fault", stream);
  bool failed = false;
  try {
    reduction.complete(stream, 1, "fault", "fault");
  } catch (const pr::PairReduceTransportError&) {
    failed = true;
  }
  check(failed && collective.aborted_ && reduction.phase() == pr::Phase::kFaulted,
        "collective failure was not terminal");
  bool retried = false;
  try {
    reduction.enqueue(input, output, 1, "retry", "retry", stream);
  } catch (const pr::PairReduceContractError&) {
    retried = true;
  }
  check(retried, "faulted reduction accepted retry");
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  cudaFree(output);
  cudaFree(input);
}

}  // namespace

int main() {
  try {
    test_all_verifier_shapes();
    test_cuda_graph_replay();
    test_fault_is_terminal();
    std::puts("qwen38 PairReduce: 24 verifier shapes, graph replay, and terminal fault passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

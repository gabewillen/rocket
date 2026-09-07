#include "pair_reduce/nccl_collective.h"
#include "pair_reduce/pair_reduce.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace pr = rocket::qwen38::pair_reduce;

namespace {

using Clock = std::chrono::steady_clock;
[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    fail(std::string(operation) + ": " + cudaGetErrorString(status));
}
std::string_view argument(int argc, char** argv, std::string_view name,
                          std::string_view fallback) {
  for (int index = 1; index + 1 < argc; ++index)
    if (argv[index] == name) return argv[index + 1];
  return fallback;
}
int integer_argument(int argc, char** argv, std::string_view name, int fallback) {
  const std::string_view text = argument(argc, argv, name, {});
  if (text.empty()) return fallback;
  int value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    fail("invalid integer for " + std::string(name));
  return value;
}

class BoundedOtel final : public pr::OtelStageSink {
 public:
  void emit_span_and_log(const pr::SpanRecord&) noexcept override { ++spans; }
  void record_duration(const pr::MetricPoint&) noexcept override { ++metrics; }
  std::uint64_t spans = 0;
  std::uint64_t metrics = 0;
};

std::uint64_t output_hash(const std::vector<float>& values) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const float value : values) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= (bits >> (byte * 8)) & 0xffu;
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    pr::NcclConfig config;
    config.rank = integer_argument(argc, argv, "--rank", -1);
    config.bootstrap_host = std::string(argument(argc, argv, "--host", "192.168.100.10"));
    config.bootstrap_port = integer_argument(argc, argv, "--port", 18838);
    config.operation_timeout_ms = static_cast<std::uint32_t>(
        integer_argument(argc, argv, "--timeout-ms", 120'000));
    const int warmup = integer_argument(argc, argv, "--warmup", 100);
    const int iterations = integer_argument(argc, argv, "--iterations", 1000);
    const int fault_stall_rank = integer_argument(argc, argv, "--fault-stall-rank", -1);
    const int fault_stall_ms = integer_argument(argc, argv, "--fault-stall-ms", 0);
    if ((config.rank != 0 && config.rank != 1) || warmup < 0 ||
        iterations < 1 || iterations > 100'000 ||
        config.operation_timeout_ms < 100 || config.operation_timeout_ms > 120'000 ||
        (fault_stall_rank != -1 && fault_stall_rank != 0 && fault_stall_rank != 1))
      fail("benchmark argument contract rejected");

    cuda_check(cudaSetDevice(0), "set GB10 device");
    pr::NcclCollective collective(config);
    BoundedOtel telemetry;
    pr::PairReduce reduction(collective, telemetry, config.operation_timeout_ms);
    constexpr std::size_t max_elements = pr::kMaxRows * pr::kHidden;
    __nv_bfloat16* device_input = nullptr;
    float* device_output = nullptr;
    cudaStream_t stream = nullptr;
    cuda_check(cudaMalloc(&device_input, max_elements * sizeof(*device_input)), "allocate input");
    cuda_check(cudaMalloc(&device_output, max_elements * sizeof(*device_output)), "allocate output");
    cuda_check(cudaStreamCreate(&stream), "create transaction stream");

    for (const int m : {32, 64, 80, 128}) {
      const std::size_t elements = static_cast<std::size_t>(m) * pr::kHidden;
      std::vector<__nv_bfloat16> input(elements);
      std::vector<float> expected(elements);
      for (std::size_t i = 0; i < elements; ++i) {
        const int centered = static_cast<int>(i % 97) - 48;
        input[i] = __float2bfloat16(
            static_cast<float>(config.rank == 0 ? centered : 3 * centered + 1) / 64.0f);
        const float rank0 = __bfloat162float(__float2bfloat16(static_cast<float>(centered) / 64.0f));
        const float rank1 = __bfloat162float(__float2bfloat16(static_cast<float>(3 * centered + 1) / 64.0f));
        expected[i] = __bfloat162float(__float2bfloat16(rank0 + rank1));
      }
      cuda_check(cudaMemcpy(device_input, input.data(), elements * sizeof(*device_input),
                            cudaMemcpyHostToDevice), "copy input");
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t executable = nullptr;
      cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
      for (int point = 0; point < 96; ++point)
        reduction.enqueue(device_input, device_output, m, "capture", "capture", stream);
      cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
      cuda_check(cudaGraphInstantiate(&executable, graph, 0), "instantiate graph");
      if (fault_stall_rank != -1) {
        if (config.rank == fault_stall_rank) {
          std::this_thread::sleep_for(std::chrono::milliseconds(fault_stall_ms));
          collective.abort();
          std::_Exit(2);
        }
        cuda_check(cudaGraphLaunch(executable, stream), "fault replay");
        reduction.complete(stream, 96, "fault", "fault");
        fail("fault injection unexpectedly completed");
      }
      for (int i = 0; i < warmup; ++i) {
        cuda_check(cudaGraphLaunch(executable, stream), "warmup replay");
        reduction.complete(stream, 96, "warmup", "warmup");
      }
      std::vector<std::uint64_t> samples;
      samples.reserve(static_cast<std::size_t>(iterations));
      for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        cuda_check(cudaGraphLaunch(executable, stream), "measured replay");
        reduction.complete(stream, 96, "measure", "measure");
        samples.push_back(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()));
      }
      std::sort(samples.begin(), samples.end());
      std::vector<float> output(elements);
      cuda_check(cudaMemcpy(output.data(), device_output, elements * sizeof(float),
                            cudaMemcpyDeviceToHost), "copy output");
      float max_abs_error = 0.0f;
      for (std::size_t i = 0; i < elements; ++i)
        max_abs_error = std::max(max_abs_error, std::abs(output[i] - expected[i]));
      std::printf(
          "{\"result\":\"qwen38_pair_reduce_nccl_graph\",\"rank\":%d,"
          "\"m\":%d,\"reductions_per_graph\":96,\"input_bytes_per_rank\":%zu,"
          "\"median_step_us\":%.3f,\"p95_step_us\":%.3f,"
          "\"median_per_reduce_us\":%.3f,\"output_fnv1a64\":\"%016llx\","
          "\"max_abs_error\":%.9g,\"otel_spans\":%llu,\"otel_metrics\":%llu}\n",
          config.rank, m, elements * sizeof(__nv_bfloat16),
          samples[samples.size() / 2] / 1000.0,
          samples[(samples.size() * 95) / 100] / 1000.0,
          samples[samples.size() / 2] / 96'000.0,
          static_cast<unsigned long long>(output_hash(output)), max_abs_error,
          static_cast<unsigned long long>(telemetry.spans),
          static_cast<unsigned long long>(telemetry.metrics));
      cudaGraphExecDestroy(executable);
      cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(stream);
    cudaFree(device_output);
    cudaFree(device_input);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "qwen38-pair-reduce-bench: %s\n", error.what());
    std::fflush(stderr);
    std::_Exit(1);
  }
}

#include "pair_reduce/pair_reduce.h"
#include "pair_reduce/rdma.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pr = rocket::qwen38::pair_reduce;

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) fail(std::string(operation) + ": " + cudaGetErrorString(status));
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

class JsonOtel final : public pr::OtelStageSink {
 public:
  void emit_span_and_log(const pr::SpanRecord& record) noexcept override {
    std::printf(
        "{\"otel_signal\":\"span_log\",\"name\":\"%.*s\",\"trace_id\":\"%.*s\","
        "\"request_id\":\"%.*s\",\"attributes\":{\"rank\":%d,\"m_bucket\":%d,"
        "\"dtype\":\"%.*s\",\"outcome\":\"%.*s\"},\"duration_ns\":%llu,"
        "\"bytes\":%llu}\n",
        static_cast<int>(record.stage.size()), record.stage.data(),
        static_cast<int>(record.trace_id.size()), record.trace_id.data(),
        static_cast<int>(record.request_id.size()), record.request_id.data(), record.rank,
        record.m_bucket, static_cast<int>(record.dtype.size()), record.dtype.data(),
        static_cast<int>(pr::outcome_name(record.outcome).size()),
        pr::outcome_name(record.outcome).data(),
        static_cast<unsigned long long>(record.duration_ns),
        static_cast<unsigned long long>(record.bytes));
  }
  void record_duration(const pr::MetricPoint& point) noexcept override {
    if (point.outcome == pr::Outcome::kOk && point.m_bucket != 0)
      durations[static_cast<std::size_t>(std::countr_zero(
          static_cast<unsigned>(point.m_bucket)))]
          .push_back(point.duration_ns);
  }
  std::vector<std::uint64_t>& for_m(int m) {
    return durations[static_cast<std::size_t>(std::countr_zero(static_cast<unsigned>(m)))];
  }

 private:
  std::vector<std::uint64_t> durations[5];
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
    pr::RdmaConfig config;
    config.rank = integer_argument(argc, argv, "--rank", -1);
    config.bootstrap_host = std::string(argument(argc, argv, "--host", "192.168.100.10"));
    config.bootstrap_port = integer_argument(argc, argv, "--port", 18838);
    const int warmup = integer_argument(argc, argv, "--warmup", 100);
    const int iterations = integer_argument(argc, argv, "--iterations", 1000);
    if ((config.rank != 0 && config.rank != 1) || warmup < 0 || warmup > 10'000 ||
        iterations <= 0 || iterations > 100'000)
      fail("rank, warmup, or iteration contract rejected");

    cuda_check(cudaSetDevice(0), "set GB10 device");
    pr::RdmaTransport transport(config);
    JsonOtel telemetry;
    pr::PairReduce reduction(transport, telemetry);

    constexpr std::size_t max_elements = 16 * pr::kHidden;
    __nv_bfloat16* device_input = nullptr;
    float* device_output = nullptr;
    cuda_check(cudaMalloc(&device_input, max_elements * sizeof(*device_input)), "allocate input");
    cuda_check(cudaMalloc(&device_output, max_elements * sizeof(*device_output)), "allocate output");

    for (const int m : pr::kAllowedM) {
      const std::size_t elements = static_cast<std::size_t>(m) * pr::kHidden;
      std::vector<__nv_bfloat16> input(elements);
      for (std::size_t index = 0; index < elements; ++index) {
        const int centered = static_cast<int>(index % 97) - 48;
        input[index] = __float2bfloat16(
            static_cast<float>(config.rank == 0 ? centered : 3 * centered + 1) / 64.0f);
      }
      cuda_check(cudaMemcpy(device_input, input.data(), elements * sizeof(*device_input),
                            cudaMemcpyHostToDevice), "copy input");
      for (int iteration = 0; iteration < warmup; ++iteration)
        reduction.reduce(device_input, device_output, m, "qwen38-pair-reduce-live",
                         "warmup");

      auto& samples = telemetry.for_m(m);
      samples.clear();
      for (int iteration = 0; iteration < iterations; ++iteration) {
        const std::string request = "m" + std::to_string(m) + "-i" +
                                    std::to_string(iteration);
        reduction.reduce(device_input, device_output, m, "qwen38-pair-reduce-live", request);
      }
      std::sort(samples.begin(), samples.end());
      std::vector<float> output(elements);
      cuda_check(cudaMemcpy(output.data(), device_output, elements * sizeof(float),
                            cudaMemcpyDeviceToHost), "copy output");
      const std::uint64_t median = samples[samples.size() / 2];
      const std::uint64_t p95 = samples[(samples.size() * 95) / 100];
      std::printf(
          "{\"result\":\"qwen38_pair_reduce\",\"rank\":%d,\"m\":%d,"
          "\"input_bytes_per_rank\":%zu,\"median_us\":%.3f,\"p95_us\":%.3f,"
          "\"output_fnv1a64\":\"%016llx\"}\n",
          config.rank, m, elements * sizeof(__nv_bfloat16), median / 1000.0,
          p95 / 1000.0, static_cast<unsigned long long>(output_hash(output)));
    }
    cudaFree(device_output);
    cudaFree(device_input);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "qwen38-pair-reduce-bench: %s\n", error.what());
    return 1;
  }
}

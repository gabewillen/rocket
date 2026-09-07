#include "pair_reduce/pair_reduce.h"
#include "pair_reduce/rdma.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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

std::uint64_t unix_ns() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}

void nvml_check(nvmlReturn_t status, const char* operation) {
  if (status != NVML_SUCCESS)
    fail(std::string(operation) + ": " + nvmlErrorString(status));
}

struct HardwareWindow {
  std::uint64_t start_unix_ns = 0;
  std::uint64_t end_unix_ns = 0;
  std::uint64_t samples = 0;
  std::uint64_t sm_clock_sum_mhz = 0;
  std::uint64_t gpu_util_sum_percent = 0;
  unsigned sm_clock_min_mhz = std::numeric_limits<unsigned>::max();
  unsigned sm_clock_max_mhz = 0;
  unsigned gpu_util_min_percent = std::numeric_limits<unsigned>::max();
  unsigned gpu_util_max_percent = 0;
};

// NVML is the only clock/utilization source. Sampling is observational and runs
// on one owned thread; values are published only after join(). JSON field names
// are fixed, so no sampled value becomes a telemetry label.
class HardwareSampler final {
 public:
  HardwareSampler() {
    nvml_check(nvmlInit_v2(), "initialize NVML");
    try {
      nvml_check(nvmlDeviceGetHandleByIndex_v2(0, &device_), "open NVML device 0");
    } catch (...) {
      nvmlShutdown();
      throw;
    }
  }
  ~HardwareSampler() {
    cancel();
    nvmlShutdown();
  }
  HardwareSampler(const HardwareSampler&) = delete;
  HardwareSampler& operator=(const HardwareSampler&) = delete;

  void start() {
    if (thread_.joinable()) fail("hardware sampler already active");
    window_ = HardwareWindow{};
    stop_.store(false, std::memory_order_relaxed);
    failed_ = false;
    window_.start_unix_ns = unix_ns();
    thread_ = std::thread([this] { sample_loop(); });
  }

  HardwareWindow stop() {
    stop_.store(true, std::memory_order_relaxed);
    thread_.join();
    window_.end_unix_ns = unix_ns();
    if (failed_) fail("NVML hardware sample failed");
    if (window_.samples == 0) fail("NVML hardware window has no samples");
    return window_;
  }

 private:
  void sample_loop() noexcept {
    while (!stop_.load(std::memory_order_relaxed)) {
      unsigned clock_mhz = 0;
      nvmlUtilization_t utilization{};
      if (nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &clock_mhz) != NVML_SUCCESS ||
          nvmlDeviceGetUtilizationRates(device_, &utilization) != NVML_SUCCESS) {
        failed_ = true;
        return;
      }
      ++window_.samples;
      window_.sm_clock_sum_mhz += clock_mhz;
      window_.gpu_util_sum_percent += utilization.gpu;
      window_.sm_clock_min_mhz = std::min(window_.sm_clock_min_mhz, clock_mhz);
      window_.sm_clock_max_mhz = std::max(window_.sm_clock_max_mhz, clock_mhz);
      window_.gpu_util_min_percent = std::min(window_.gpu_util_min_percent, utilization.gpu);
      window_.gpu_util_max_percent = std::max(window_.gpu_util_max_percent, utilization.gpu);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  void cancel() noexcept {
    if (!thread_.joinable()) return;
    stop_.store(true, std::memory_order_relaxed);
    thread_.join();
  }

  nvmlDevice_t device_{};
  std::thread thread_;
  std::atomic_bool stop_{false};
  bool failed_ = false;
  HardwareWindow window_;
};

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
    const int timeout_ms = integer_argument(argc, argv, "--timeout-ms", 120'000);
    const int warmup = integer_argument(argc, argv, "--warmup", 100);
    const int iterations = integer_argument(argc, argv, "--iterations", 1000);
    const int fault_stall_rank = integer_argument(argc, argv, "--fault-stall-rank", -1);
    const int fault_stall_ms = integer_argument(argc, argv, "--fault-stall-ms", 0);
    if ((config.rank != 0 && config.rank != 1) || warmup < 0 || warmup > 10'000 ||
        iterations <= 0 || iterations > 100'000 || timeout_ms < 0 ||
        !pr::valid_operation_timeout_ms(static_cast<std::uint32_t>(timeout_ms)) ||
        (fault_stall_rank != -1 && fault_stall_rank != 0 && fault_stall_rank != 1) ||
        ((fault_stall_rank == -1) != (fault_stall_ms == 0)) ||
        (fault_stall_rank != -1 && fault_stall_ms <= timeout_ms))
      fail("rank, warmup, iteration, timeout, or fault-stall contract rejected");
    config.operation_timeout_ms = static_cast<std::uint32_t>(timeout_ms);

    cuda_check(cudaSetDevice(0), "set GB10 device");
    pr::RdmaTransport transport(config);
    JsonOtel telemetry;
    pr::PairReduce reduction(transport, telemetry);
    if (config.rank == fault_stall_rank) {
      std::printf(
          "{\"result\":\"qwen38_pair_reduce_fault\",\"rank\":%d,"
          "\"fault\":\"peer_stall\",\"stall_ms\":%d,\"timeout_ms\":%d}\n",
          config.rank, fault_stall_ms, timeout_ms);
      std::fflush(stdout);
      std::this_thread::sleep_for(std::chrono::milliseconds(fault_stall_ms));
      return 2;
    }
    HardwareSampler hardware;

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
      hardware.start();
      for (int iteration = 0; iteration < iterations; ++iteration) {
        const std::string request = "m" + std::to_string(m) + "-i" +
                                    std::to_string(iteration);
        reduction.reduce(device_input, device_output, m, "qwen38-pair-reduce-live", request);
      }
      const HardwareWindow window = hardware.stop();
      std::sort(samples.begin(), samples.end());
      std::vector<float> output(elements);
      cuda_check(cudaMemcpy(output.data(), device_output, elements * sizeof(float),
                            cudaMemcpyDeviceToHost), "copy output");
      const std::uint64_t median = samples[samples.size() / 2];
      const std::uint64_t p95 = samples[(samples.size() * 95) / 100];
      std::printf(
          "{\"result\":\"qwen38_pair_reduce\",\"rank\":%d,\"m\":%d,"
          "\"input_bytes_per_rank\":%zu,\"median_us\":%.3f,\"p95_us\":%.3f,"
          "\"output_fnv1a64\":\"%016llx\",\"window_start_unix_ns\":%llu,"
          "\"window_end_unix_ns\":%llu,\"nvml_samples\":%llu,"
          "\"sm_clock_mhz_min\":%u,\"sm_clock_mhz_max\":%u,"
          "\"sm_clock_mhz_mean\":%.3f,\"gpu_util_percent_min\":%u,"
          "\"gpu_util_percent_max\":%u,\"gpu_util_percent_mean\":%.3f}\n",
          config.rank, m, elements * sizeof(__nv_bfloat16), median / 1000.0,
          p95 / 1000.0, static_cast<unsigned long long>(output_hash(output)),
          static_cast<unsigned long long>(window.start_unix_ns),
          static_cast<unsigned long long>(window.end_unix_ns),
          static_cast<unsigned long long>(window.samples), window.sm_clock_min_mhz,
          window.sm_clock_max_mhz,
          static_cast<double>(window.sm_clock_sum_mhz) / window.samples,
          window.gpu_util_min_percent, window.gpu_util_max_percent,
          static_cast<double>(window.gpu_util_sum_percent) / window.samples);
    }
    cudaFree(device_output);
    cudaFree(device_input);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "qwen38-pair-reduce-bench: %s\n", error.what());
    return 1;
  }
}

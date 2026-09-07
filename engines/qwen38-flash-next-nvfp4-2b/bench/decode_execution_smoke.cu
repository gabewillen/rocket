#include "decode/execution.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace decode = rocket::qwen38::decode;
namespace pr = rocket::qwen38::pair_reduce;

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void check(bool condition, const std::string& message) {
  if (!condition) fail(message);
}
void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) fail(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename T>
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t elements) {
    cuda_check(cudaMalloc(&pointer_, elements * sizeof(T)), "allocate device buffer");
  }
  ~DeviceBuffer() { cudaFree(pointer_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() noexcept { return pointer_; }

 private:
  T* pointer_ = nullptr;
};

class SmokeOtel final : public pr::OtelStageSink {
 public:
  void emit_span_and_log(const pr::SpanRecord& record) noexcept override {
    ++spans;
    failures += record.outcome != pr::Outcome::kOk;
  }
  void record_duration(const pr::MetricPoint& point) noexcept override {
    ++metrics;
    failures += point.outcome != pr::Outcome::kOk;
  }
  int spans = 0;
  int metrics = 0;
  int failures = 0;
};

class InProcessPeer final : public pr::Transport {
 public:
  explicit InProcessPeer(std::vector<__nv_bfloat16> peer) : peer_(std::move(peer)) {}
  int rank() const noexcept override { return 0; }
  int world_size() const noexcept override { return 2; }
  int register_region(void* address, std::size_t bytes) override {
    check(address != nullptr && bytes == pr::PairReduce::region_bytes(),
          "smoke region contract");
    region_ = static_cast<std::byte*>(address);
    return 0;
  }
  void unregister_region(int handle) noexcept override {
    if (handle == 0) region_ = nullptr;
  }
  std::uint64_t next_sequence() override { return ++sequence_; }
  void post_unsignaled_write(int handle, std::size_t source_offset,
                             std::size_t peer_offset, std::size_t bytes) override {
    check(handle == 0 && source_offset == 0 && peer_offset == pr::PairReduce::peer_offset(),
          "smoke write extent");
    pr::WireHeader header{};
    std::memcpy(&header, region_, sizeof(header));
    header.rank = 1;
    check(bytes == sizeof(header) + header.payload_bytes &&
              header.payload_bytes == peer_.size() * sizeof(__nv_bfloat16),
          "smoke wire shape");
    std::memcpy(region_ + peer_offset, &header, sizeof(header));
    std::memcpy(region_ + peer_offset + sizeof(header), peer_.data(),
                header.payload_bytes);
  }
  void signal_sequence(std::uint64_t sequence) override {
    check(sequence == sequence_, "smoke ready sequence");
  }
  void wait_peer(std::uint64_t sequence) override {
    check(sequence == sequence_, "smoke peer-ready sequence");
  }
  void flush_signaled() override {}
  void acknowledge_consumed(std::uint64_t sequence) override {
    check(sequence == sequence_, "smoke consumed sequence");
  }
  void wait_peer_consumed(std::uint64_t sequence) override {
    check(sequence == sequence_, "smoke peer-consumed sequence");
  }

 private:
  std::vector<__nv_bfloat16> peer_;
  std::byte* region_ = nullptr;
  std::uint64_t sequence_ = 0;
};

}  // namespace

int main() {
  try {
    constexpr int m = 1;
    std::vector<__nv_bfloat16> local(pr::kHidden), peer(pr::kHidden);
    for (int index = 0; index < pr::kHidden; ++index) {
      local[static_cast<std::size_t>(index)] = __float2bfloat16(index / 256.0f);
      peer[static_cast<std::size_t>(index)] = __float2bfloat16((index + 1) / 512.0f);
    }
    InProcessPeer transport(peer);
    SmokeOtel telemetry;
    pr::PairReduce reduction(transport, telemetry);
    decode::PairReduceAdapter adapter(reduction);
    decode::Tp2DecodeExecution execution(adapter, telemetry);

    DeviceBuffer<__nv_bfloat16> device_input(local.size());
    DeviceBuffer<float> device_output(local.size());
    cuda_check(cudaMemcpy(device_input.get(), local.data(), local.size() * sizeof(local[0]),
                          cudaMemcpyHostToDevice), "copy smoke input");

    execution.begin_step(1, m, "qwen38-decode-smoke", "generation-1");
    for (int ordinal = 0; ordinal < decode::kReductionPoints; ++ordinal)
      execution.reduce_at(1, execution.point_for_ordinal(ordinal), device_input.get(),
                          device_output.get(), "qwen38-decode-smoke", "generation-1");
    execution.finish_step(1, "qwen38-decode-smoke", "generation-1");

    std::vector<float> output(local.size());
    cuda_check(cudaMemcpy(output.data(), device_output.get(), output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy smoke output");
    for (std::size_t index = 0; index < output.size(); ++index) {
      const float expected = __bfloat162float(local[index]) + __bfloat162float(peer[index]);
      check(std::bit_cast<std::uint32_t>(output[index]) ==
                std::bit_cast<std::uint32_t>(expected),
            "smoke rank-order output mismatch");
    }
    check(telemetry.spans == 194 && telemetry.metrics == 96 && telemetry.failures == 0,
          "smoke OTEL count or outcome drift");
    std::printf(
        "{\"result\":\"qwen38_decode_pair_reduce_smoke\","
        "\"device_generation\":1,\"reduction_points\":96,"
        "\"elements_verified\":2560,\"otel_spans\":194,"
        "\"otel_metrics\":96,\"failures\":0}\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "qwen38-decode-pair-reduce-smoke: %s\n", error.what());
    return 1;
  }
}

#include "pair_reduce/pair_reduce.h"
#include "pair_reduce/rdma.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cstddef>
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
void check(bool condition, const std::string& message) {
  if (!condition) fail(message);
}
void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) fail(std::string(operation) + ": " + cudaGetErrorString(status));
}

class CaptureOtel final : public pr::OtelStageSink {
 public:
  void emit_span_and_log(const pr::SpanRecord& record) noexcept override {
    spans.push_back({std::string(record.stage), std::string(record.trace_id),
                     std::string(record.request_id), record.rank, record.m_bucket,
                     std::string(record.dtype), record.outcome, record.duration_ns,
                     record.bytes});
  }
  void record_duration(const pr::MetricPoint& point) noexcept override {
    metrics.push_back({point.rank, point.m_bucket, std::string(point.dtype),
                       point.outcome, point.duration_ns});
  }
  struct Span {
    std::string stage, trace_id, request_id;
    int rank, m;
    std::string dtype;
    pr::Outcome outcome;
    std::uint64_t duration, bytes;
  };
  struct Metric {
    int rank, m;
    std::string dtype;
    pr::Outcome outcome;
    std::uint64_t duration;
  };
  std::vector<Span> spans;
  std::vector<Metric> metrics;
};

class FakeTransport final : public pr::Transport {
 public:
  FakeTransport(int rank, std::vector<__nv_bfloat16> peer)
      : rank_(rank), peer_(std::move(peer)) {}
  int rank() const noexcept override { return rank_; }
  int world_size() const noexcept override { return world_size_; }
  int register_region(void* address, std::size_t bytes) override {
    check(address != nullptr && bytes == pr::PairReduce::region_bytes(),
          "registered region contract");
    address_ = static_cast<std::byte*>(address);
    operations.push_back("register");
    return 7;
  }
  void unregister_region(int region) noexcept override {
    if (region == 7) operations.push_back("unregister");
    address_ = nullptr;
  }
  std::uint64_t next_sequence() override {
    operations.push_back("next_sequence");
    return fixed_sequence_ == 0 ? ++sequence_ : fixed_sequence_;
  }
  void post_unsignaled_write(int region, std::size_t source_offset,
                             std::size_t peer_offset, std::size_t bytes) override {
    check(region == 7 && source_offset == 0 && peer_offset == pr::PairReduce::peer_offset(),
          "payload extent contract");
    operations.push_back("unsignaled_payload");
    if (throw_on_post_) throw std::runtime_error("injected post failure");
    auto header = *reinterpret_cast<const pr::WireHeader*>(address_);
    header.rank = static_cast<std::uint32_t>(1 - rank_);
    if (peer_m_delta_ != 0) header.m += peer_m_delta_;
    check(bytes == sizeof(header) + header.payload_bytes,
          "wire contains header plus BF16 payload");
    std::memcpy(address_ + peer_offset, &header, sizeof(header));
    std::memcpy(address_ + peer_offset + sizeof(header), peer_.data(),
                std::min<std::size_t>(header.payload_bytes,
                                      peer_.size() * sizeof(__nv_bfloat16)));
  }
  void signal_sequence(std::uint64_t sequence) override {
    check(sequence != 0, "signaled sequence is nonzero");
    operations.push_back("signaled_doorbell");
  }
  void wait_peer(std::uint64_t) override { operations.push_back("wait_peer"); }
  void flush_signaled() override { operations.push_back("flush_signaled"); }
  void acknowledge_consumed(std::uint64_t sequence) override {
    check(sequence != 0, "consumed sequence is nonzero");
    operations.push_back("acknowledge_consumed");
  }
  void wait_peer_consumed(std::uint64_t) override {
    operations.push_back("wait_peer_consumed");
  }

  int rank_;
  int world_size_ = 2;
  std::vector<__nv_bfloat16> peer_;
  std::byte* address_ = nullptr;
  std::uint64_t sequence_ = 0;
  std::uint64_t fixed_sequence_ = 0;
  std::uint32_t peer_m_delta_ = 0;
  bool throw_on_post_ = false;
  std::vector<std::string> operations;
};

std::vector<__nv_bfloat16> partial(int rank, int m) {
  std::vector<__nv_bfloat16> result(static_cast<std::size_t>(m) * pr::kHidden);
  for (std::size_t index = 0; index < result.size(); ++index) {
    const int centered = static_cast<int>(index % 97) - 48;
    const float value = static_cast<float>((rank == 0 ? centered : 3 * centered + 1)) / 64.0f;
    result[index] = __float2bfloat16(value);
  }
  return result;
}

std::vector<float> run_rank(int rank, int m, const std::vector<__nv_bfloat16>& local,
                            const std::vector<__nv_bfloat16>& peer,
                            FakeTransport* observed_transport = nullptr,
                            CaptureOtel* observed_otel = nullptr) {
  FakeTransport owned_transport(rank, peer);
  CaptureOtel owned_otel;
  FakeTransport& transport = observed_transport == nullptr ? owned_transport : *observed_transport;
  CaptureOtel& otel = observed_otel == nullptr ? owned_otel : *observed_otel;
  __nv_bfloat16* device_input = nullptr;
  float* device_output = nullptr;
  cuda_check(cudaMalloc(&device_input, local.size() * sizeof(*device_input)), "cudaMalloc input");
  cuda_check(cudaMalloc(&device_output, local.size() * sizeof(*device_output)), "cudaMalloc output");
  cuda_check(cudaMemcpy(device_input, local.data(), local.size() * sizeof(*device_input),
                        cudaMemcpyHostToDevice), "copy input");
  std::vector<float> output(local.size());
  {
    pr::PairReduce reduction(transport, otel);
    reduction.reduce(device_input, device_output, m, "trace-fixed", "request-42");
    cuda_check(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy output");
  }
  cudaFree(device_output);
  cudaFree(device_input);
  return output;
}

void test_all_shapes_are_bit_identical() {
  for (const int m : pr::kAllowedM) {
    const auto rank0 = partial(0, m);
    const auto rank1 = partial(1, m);
    FakeTransport transport0(0, rank1);
    FakeTransport transport1(1, rank0);
    CaptureOtel otel0, otel1;
    const auto output0 = run_rank(0, m, rank0, rank1, &transport0, &otel0);
    const auto output1 = run_rank(1, m, rank1, rank0, &transport1, &otel1);
    check(std::memcmp(output0.data(), output1.data(), output0.size() * sizeof(float)) == 0,
          "rank outputs are not bit-identical at M=" + std::to_string(m));
    for (std::size_t index = 0; index < output0.size(); ++index) {
      const float expected = __bfloat162float(rank0[index]) + __bfloat162float(rank1[index]);
      check(std::bit_cast<std::uint32_t>(output0[index]) == std::bit_cast<std::uint32_t>(expected),
            "rank-order FP32 reference mismatch");
    }
    const std::vector<std::string> expected_order{
        "register", "next_sequence", "unsignaled_payload", "signaled_doorbell",
        "wait_peer", "flush_signaled", "acknowledge_consumed", "wait_peer_consumed",
        "flush_signaled", "unregister"};
    check(transport0.operations == expected_order && transport1.operations == expected_order,
          "payload/doorbell protocol order drift");
    check(otel0.spans.size() == 1 && otel0.metrics.size() == 1 &&
              otel0.metrics[0].rank == 0 && otel0.metrics[0].m == m &&
              otel0.metrics[0].dtype == "bf16_fp32" &&
              otel0.metrics[0].outcome == pr::Outcome::kOk &&
              otel0.spans[0].trace_id == "trace-fixed" &&
              otel0.spans[0].request_id == "request-42",
          "OTEL stage contract drift");
  }
}

void test_invalid_m_and_peer_drift_fail_closed() {
  const auto rank0 = partial(0, 1);
  const auto rank1 = partial(1, 1);
  FakeTransport invalid_transport(0, rank1);
  CaptureOtel invalid_otel;
  __nv_bfloat16* input = nullptr;
  float* output = nullptr;
  cuda_check(cudaMalloc(&input, rank0.size() * sizeof(*input)), "cudaMalloc invalid input");
  cuda_check(cudaMalloc(&output, rank0.size() * sizeof(*output)), "cudaMalloc invalid output");
  {
    pr::PairReduce reduction(invalid_transport, invalid_otel);
    bool threw = false;
    try { reduction.reduce(input, output, 3, "trace-invalid", "request-invalid"); }
    catch (const pr::PairReduceContractError&) { threw = true; }
    check(threw, "invalid M was accepted");
  }
  check(invalid_transport.operations == std::vector<std::string>({"register", "unregister"}),
        "invalid M reached transport");
  check(invalid_otel.metrics.size() == 1 && invalid_otel.metrics[0].m == 0 &&
            invalid_otel.metrics[0].outcome == pr::Outcome::kContractError,
        "invalid M did not use bounded OTEL failure labels");

  FakeTransport drift_transport(0, rank1);
  drift_transport.peer_m_delta_ = 1;
  CaptureOtel drift_otel;
  cuda_check(cudaMemcpy(input, rank0.data(), rank0.size() * sizeof(*input), cudaMemcpyHostToDevice),
             "copy drift input");
  {
    pr::PairReduce reduction(drift_transport, drift_otel);
    bool threw = false;
    try { reduction.reduce(input, output, 1, "trace-drift", "request-drift"); }
    catch (const pr::PairReduceContractError& error) {
      threw = std::string_view(error.what()).find("message drift") != std::string_view::npos;
    }
    check(threw, "peer message drift was accepted");
  }
  check(drift_otel.metrics.size() == 1 &&
            drift_otel.metrics[0].outcome == pr::Outcome::kContractError,
        "peer drift OTEL outcome is wrong");
  cudaFree(output);
  cudaFree(input);
}

void test_topology_and_sequence_drift_fail_closed() {
  const auto rank0 = partial(0, 1);
  const auto rank1 = partial(1, 1);
  CaptureOtel otel;
  FakeTransport topology(0, rank1);
  topology.world_size_ = 4;
  bool topology_threw = false;
  try { pr::PairReduce reduction(topology, otel); }
  catch (const pr::PairReduceContractError&) { topology_threw = true; }
  check(topology_threw && topology.operations.empty(), "topology drift was accepted");

  FakeTransport sequence(0, rank1);
  sequence.fixed_sequence_ = 9;
  __nv_bfloat16* input = nullptr;
  float* output = nullptr;
  cuda_check(cudaMalloc(&input, rank0.size() * sizeof(*input)), "cudaMalloc sequence input");
  cuda_check(cudaMalloc(&output, rank0.size() * sizeof(*output)), "cudaMalloc sequence output");
  cuda_check(cudaMemcpy(input, rank0.data(), rank0.size() * sizeof(*input), cudaMemcpyHostToDevice),
             "copy sequence input");
  {
    pr::PairReduce reduction(sequence, otel);
    reduction.reduce(input, output, 1, "trace-seq", "request-1");
    bool threw = false;
    try { reduction.reduce(input, output, 1, "trace-seq", "request-2"); }
    catch (const pr::PairReduceContractError&) { threw = true; }
    check(threw, "repeated sequence was accepted");
  }
  cudaFree(output);
  cudaFree(input);
}

void test_transport_failure_is_typed_and_observed() {
  const auto rank0 = partial(0, 1);
  const auto rank1 = partial(1, 1);
  FakeTransport transport(0, rank1);
  transport.throw_on_post_ = true;
  CaptureOtel otel;
  __nv_bfloat16* input = nullptr;
  float* output = nullptr;
  cuda_check(cudaMalloc(&input, rank0.size() * sizeof(*input)), "cudaMalloc transport input");
  cuda_check(cudaMalloc(&output, rank0.size() * sizeof(*output)), "cudaMalloc transport output");
  cuda_check(cudaMemcpy(input, rank0.data(), rank0.size() * sizeof(*input), cudaMemcpyHostToDevice),
             "copy transport input");
  {
    pr::PairReduce reduction(transport, otel);
    bool threw = false;
    try { reduction.reduce(input, output, 1, "trace-transport", "request-transport"); }
    catch (const pr::PairReduceTransportError&) { threw = true; }
    check(threw, "transport failure was not typed");
  }
  check(otel.metrics.size() == 1 &&
            otel.metrics[0].outcome == pr::Outcome::kTransportError,
        "transport failure OTEL outcome is wrong");
  cudaFree(output);
  cudaFree(input);
}

void test_operation_timeout_bounds() {
  check(!pr::valid_operation_timeout_ms(0), "zero operation timeout was accepted");
  check(!pr::valid_operation_timeout_ms(99), "sub-floor operation timeout was accepted");
  check(pr::valid_operation_timeout_ms(100), "minimum operation timeout was rejected");
  check(pr::valid_operation_timeout_ms(120'000), "maximum operation timeout was rejected");
  check(!pr::valid_operation_timeout_ms(120'001), "over-ceiling timeout was accepted");
}

}  // namespace

int main() {
  try {
    test_all_shapes_are_bit_identical();
    test_invalid_m_and_peer_drift_fail_closed();
    test_topology_and_sequence_drift_fail_closed();
    test_transport_failure_is_typed_and_observed();
    test_operation_timeout_bounds();
    std::puts("qwen38 PairReduce: 5 shapes bit-identical; protocol, drift, and timeout contracts passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

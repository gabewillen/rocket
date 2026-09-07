// SPDX-License-Identifier: Apache-2.0
#include "pair_reduce/pair_reduce.h"
#include "pair_reduce/rdma.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <new>
#include <string>

namespace pr = rocket::qwen38::pair_reduce;

namespace {
thread_local std::string last_error;

class BoundedSink final : public pr::OtelStageSink {
 public:
  void emit_span_and_log(const pr::SpanRecord& record) noexcept override {
    ++spans;
    if (record.outcome != pr::Outcome::kOk) ++failures;
  }
  void record_duration(const pr::MetricPoint& point) noexcept override {
    ++metrics;
    if (point.outcome != pr::Outcome::kOk) ++failures;
  }
  std::uint64_t spans = 0;
  std::uint64_t metrics = 0;
  std::uint64_t failures = 0;
};

struct Plan {
  BoundedSink sink;
  std::unique_ptr<pr::RdmaTransport> transport;
  std::unique_ptr<pr::PairReduce> reduction;
};

template <typename Operation>
int invoke(Operation operation) noexcept {
  last_error.clear();
  try {
    operation();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
  } catch (...) {
    last_error = "unknown Qwen layer PairReduce failure";
  }
  return 1;
}
}  // namespace

extern "C" int qwen38_layer_pair_reduce_create(
    int rank, const char* bootstrap_host, int bootstrap_port,
    std::uint32_t timeout_ms, void** result) {
  if (!result || !bootstrap_host) return 1;
  *result = nullptr;
  return invoke([&] {
    pr::RdmaConfig config;
    config.rank = rank;
    config.bootstrap_host = bootstrap_host;
    config.bootstrap_port = bootstrap_port;
    config.operation_timeout_ms = timeout_ms;
    auto plan = std::make_unique<Plan>();
    plan->transport = std::make_unique<pr::RdmaTransport>(config);
    plan->reduction = std::make_unique<pr::PairReduce>(*plan->transport,
                                                       plan->sink);
    *result = plan.release();
  });
}

extern "C" int qwen38_layer_pair_reduce_launch(
    void* opaque, const __nv_bfloat16* input, float* output, int m,
    cudaStream_t stream) {
  if (!opaque) return 1;
  return invoke([&] {
    static_cast<Plan*>(opaque)->reduction->reduce(
        input, output, m, "qwen38-layer3-full-attention", "fixed-bucket",
        stream);
  });
}

extern "C" int qwen38_layer_pair_reduce_counts(
    void* opaque, std::uint64_t* spans, std::uint64_t* metrics,
    std::uint64_t* failures) {
  if (!opaque || !spans || !metrics || !failures) return 1;
  const auto& sink = static_cast<Plan*>(opaque)->sink;
  *spans = sink.spans;
  *metrics = sink.metrics;
  *failures = sink.failures;
  return 0;
}

extern "C" int qwen38_layer_pair_reduce_destroy(void* opaque) {
  delete static_cast<Plan*>(opaque);
  return 0;
}

extern "C" const char* qwen38_layer_pair_reduce_last_error() {
  return last_error.c_str();
}

#include "pair_reduce/pair_reduce.h"

#include <cuda_runtime.h>

#include <chrono>
#include <string>
#include <thread>

namespace rocket::qwen38::pair_reduce {
namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void contract_fail(const std::string& reason) {
  throw PairReduceContractError("qwen38 PairReduce contract: " + reason);
}

std::uint64_t elapsed_ns(Clock::time_point start) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

__global__ void bf16_to_fp32(const __nv_bfloat16* input, float* output,
                             std::size_t elements) {
  const std::size_t index =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < elements) output[index] = __bfloat162float(input[index]);
}

}  // namespace

PairReduce::PairReduce(DeviceCollective& collective, OtelStageSink& telemetry,
                       std::uint32_t timeout_ms)
    : collective_(collective), telemetry_(telemetry), timeout_ms_(timeout_ms) {
  if (collective_.world_size() != kWorldSize ||
      (collective_.rank() != 0 && collective_.rank() != 1))
    contract_fail("topology must be exactly ranks 0 and 1");
  if (timeout_ms_ < 100 || timeout_ms_ > 120'000)
    contract_fail("timeout must be within 100..120000 milliseconds");
  cudaError_t status = cudaMalloc(&reduced_,
                                  static_cast<std::size_t>(kMaxRows) * kHidden *
                                      sizeof(*reduced_));
  if (status != cudaSuccess)
    throw PairReduceCudaError("qwen38 PairReduce CUDA allocate scratch: " +
                              std::string(cudaGetErrorString(status)));
  status = cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming);
  if (status != cudaSuccess) {
    cudaFree(reduced_);
    reduced_ = nullptr;
    throw PairReduceCudaError("qwen38 PairReduce CUDA create completion: " +
                              std::string(cudaGetErrorString(status)));
  }
}

PairReduce::~PairReduce() {
  // A terminal collective fault can leave graph work owning both resources.
  // Process replacement is the recovery contract, so destruction must not
  // block while the detached communicator abort releases that work.
  if (phase_ == Phase::kFaulted) return;
  if (completion_) cudaEventDestroy(completion_);
  if (reduced_) cudaFree(reduced_);
}

void PairReduce::enqueue(const __nv_bfloat16* input, float* output, int m,
                         std::string_view trace_id,
                         std::string_view request_id, cudaStream_t stream) {
  const auto start = Clock::now();
  if (phase_ != Phase::kReady) contract_fail("faulted instance cannot enqueue");
  if (!input || !output || !stream)
    contract_fail("device buffers and stream are required");
  if (!allowed_m(m))
    contract_fail("M must be a sequences*(K+1) verifier row bucket");
  const std::size_t elements = static_cast<std::size_t>(m) * kHidden;
  try {
    collective_.enqueue_sum(input, reduced_, elements, stream);
    bf16_to_fp32<<<static_cast<unsigned>((elements + 255) / 256), 256, 0,
                   stream>>>(reduced_, output, elements);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
      throw PairReduceCudaError("qwen38 PairReduce CUDA launch conversion: " +
                                std::string(cudaGetErrorString(status)));
    ++enqueued_;
  } catch (const PairReduceCudaError&) {
    fault(Outcome::kCudaError, m, trace_id, request_id, elapsed_ns(start));
    throw;
  } catch (...) {
    fault(Outcome::kTransportError, m, trace_id, request_id,
          elapsed_ns(start));
    throw;
  }
  const auto duration = elapsed_ns(start);
  const auto bytes = static_cast<std::uint64_t>(elements * sizeof(*input));
  telemetry_.emit_span_and_log({"rocket.qwen38.pair_reduce.enqueue", trace_id,
                                request_id, rank(), m, kDtype, Outcome::kOk,
                                duration, bytes});
  telemetry_.record_duration({rank(), m, kDtype, Outcome::kOk, duration});
}

void PairReduce::complete(cudaStream_t stream, std::uint32_t reduction_count,
                          std::string_view trace_id,
                          std::string_view request_id) {
  const auto start = Clock::now();
  if (phase_ != Phase::kReady) contract_fail("faulted instance cannot complete");
  if (!stream) contract_fail("completion stream is required");
  if (reduction_count == 0 || reduction_count > 96)
    contract_fail("completion count must be within 1..96");
  cudaError_t status = cudaEventRecord(completion_, stream);
  if (status != cudaSuccess) {
    fault(Outcome::kCudaError, 0, trace_id, request_id, elapsed_ns(start));
    throw PairReduceCudaError("qwen38 PairReduce CUDA record completion: " +
                              std::string(cudaGetErrorString(status)));
  }
  for (;;) {
    status = cudaEventQuery(completion_);
    if (status == cudaSuccess) break;
    if (status != cudaErrorNotReady) {
      fault(Outcome::kCudaError, 0, trace_id, request_id, elapsed_ns(start));
      throw PairReduceCudaError("qwen38 PairReduce CUDA query completion: " +
                                std::string(cudaGetErrorString(status)));
    }
    if (!collective_.healthy()) {
      fault(Outcome::kTransportError, 0, trace_id, request_id,
            elapsed_ns(start));
      throw PairReduceTransportError("qwen38 PairReduce collective fault");
    }
    if (elapsed_ns(start) > static_cast<std::uint64_t>(timeout_ms_) * 1'000'000) {
      const auto duration = elapsed_ns(start);
      fault(Outcome::kTransportError, 0, trace_id, request_id, duration);
      throw PairReduceTransportError(
          "qwen38 PairReduce completion timed out after " +
          std::to_string(duration) + " ns");
    }
    std::this_thread::yield();
  }
  const auto duration = elapsed_ns(start);
  telemetry_.emit_span_and_log({"rocket.qwen38.pair_reduce.complete", trace_id,
                                request_id, rank(), 0, kDtype, Outcome::kOk,
                                duration, reduction_count});
  telemetry_.record_duration({rank(), 0, kDtype, Outcome::kOk, duration});
}

void PairReduce::fault(Outcome outcome, int m, std::string_view trace_id,
                       std::string_view request_id,
                       std::uint64_t duration_ns) {
  phase_ = Phase::kFaulted;
  collective_.abort();
  telemetry_.emit_span_and_log({"rocket.qwen38.pair_reduce.fault", trace_id,
                                request_id, rank(), allowed_m(m) ? m : 0,
                                kDtype, outcome, duration_ns, enqueued_});
  telemetry_.record_duration(
      {rank(), allowed_m(m) ? m : 0, kDtype, outcome, duration_ns});
}

}  // namespace rocket::qwen38::pair_reduce

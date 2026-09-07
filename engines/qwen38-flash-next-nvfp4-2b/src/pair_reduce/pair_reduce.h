#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "pair_reduce/otel.h"

namespace rocket::qwen38::pair_reduce {

inline constexpr int kWorldSize = 2;
inline constexpr int kHidden = 2'560;
inline constexpr int kMaxRows = 128;
inline constexpr int kAllowedM[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16,
    20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128};
inline constexpr std::string_view kDtype = "bf16_fp32";

constexpr bool allowed_m(int m) noexcept {
  for (const int allowed : kAllowedM)
    if (m == allowed) return true;
  return false;
}

class PairReduceError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
class PairReduceContractError final : public PairReduceError {
 public:
  using PairReduceError::PairReduceError;
};
class PairReduceCudaError final : public PairReduceError {
 public:
  using PairReduceError::PairReduceError;
};
class PairReduceTransportError final : public PairReduceError {
 public:
  using PairReduceError::PairReduceError;
};

// enqueue_sum() enqueues an out-of-place BF16 sum on stream without allocating
// or synchronizing and must be CUDA Graph capturable. health() and abort() are
// host operations called only after capture at the transaction completion
// boundary. Implementations are single-owner and non-reentrant.
class DeviceCollective {
 public:
  virtual ~DeviceCollective() = default;
  virtual int rank() const noexcept = 0;
  virtual int world_size() const noexcept = 0;
  virtual void enqueue_sum(const __nv_bfloat16* input,
                           __nv_bfloat16* output, std::size_t elements,
                           cudaStream_t stream) = 0;
  virtual bool healthy() noexcept = 0;
  virtual void abort() noexcept = 0;
};

enum class Phase : std::uint8_t { kReady, kFaulted };

// Owns one fixed device scratch tensor and completion event. enqueue() borrows
// input/output until its stream reaches the enqueued conversion. Success leaves
// FP32 [M,2560]. enqueue() performs no allocation, host payload copy, or stream
// synchronization and can execute inside CUDA capture. complete() is the sole
// transaction fence. It waits at most timeout_ms and terminally faults and
// aborts on CUDA, transport, or timeout failure. Reconstruct before retry.
class PairReduce final {
 public:
  PairReduce(DeviceCollective& collective, OtelStageSink& telemetry,
             std::uint32_t timeout_ms = 120'000);
  ~PairReduce();
  PairReduce(const PairReduce&) = delete;
  PairReduce& operator=(const PairReduce&) = delete;

  void enqueue(const __nv_bfloat16* input, float* output, int m,
               std::string_view trace_id, std::string_view request_id,
               cudaStream_t stream);
  void complete(cudaStream_t stream, std::uint32_t reduction_count,
                std::string_view trace_id, std::string_view request_id);

  int rank() const noexcept { return collective_.rank(); }
  int world_size() const noexcept { return collective_.world_size(); }
  Phase phase() const noexcept { return phase_; }
  std::uint64_t enqueued() const noexcept { return enqueued_; }

 private:
  void fault(Outcome outcome, int m, std::string_view trace_id,
             std::string_view request_id, std::uint64_t duration_ns);

  DeviceCollective& collective_;
  OtelStageSink& telemetry_;
  std::uint32_t timeout_ms_;
  __nv_bfloat16* reduced_ = nullptr;
  cudaEvent_t completion_ = nullptr;
  Phase phase_ = Phase::kReady;
  std::uint64_t enqueued_ = 0;
};

}  // namespace rocket::qwen38::pair_reduce

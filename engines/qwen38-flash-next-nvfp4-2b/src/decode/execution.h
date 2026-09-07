#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "pair_reduce/otel.h"
#include "pair_reduce/pair_reduce.h"

namespace rocket::qwen38::decode {

inline constexpr int kLayers = 48;
inline constexpr int kReductionsPerLayer = 2;
inline constexpr int kReductionPoints = kLayers * kReductionsPerLayer;

enum class ReductionKind : std::uint8_t { kAttentionOutput, kMoeOutput };

// Qwen3.8 has one TP2 reduction after each rank-local attention output
// projection and one after each rank-local routed/shared expert aggregation.
// The reduced FP32 value is consumed before residual or next-layer work.
struct ReductionPoint {
  int layer;
  ReductionKind kind;
};

enum class StepPhase : std::uint8_t { kIdle, kActive, kFaulted };

class DecodeExecutionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
class DecodeExecutionContractError final : public DecodeExecutionError {
 public:
  using DecodeExecutionError::DecodeExecutionError;
};
class DecodeExecutionTransportError final : public DecodeExecutionError {
 public:
  using DecodeExecutionError::DecodeExecutionError;
};
class DecodeExecutionCudaError final : public DecodeExecutionError {
 public:
  using DecodeExecutionError::DecodeExecutionError;
};

// Synchronous, single-owner boundary for one hidden-partial reducer. Input and
// output are borrowed device buffers for the duration of reduce(). Success
// leaves FP32 [M,2560] in output. Failure makes output unspecified.
class HiddenPartialReducer {
 public:
  virtual ~HiddenPartialReducer() = default;
  virtual int rank() const noexcept = 0;
  virtual int world_size() const noexcept = 0;
  virtual void reduce(const __nv_bfloat16* input, float* output, int m,
                      std::string_view trace_id, std::string_view request_id,
                      cudaStream_t stream) = 0;
  virtual void complete(cudaStream_t stream, std::uint32_t reduction_count,
                        std::string_view trace_id,
                        std::string_view request_id) = 0;
};

class PairReduceAdapter final : public HiddenPartialReducer {
 public:
  // Borrows reduction for the adapter lifetime; the caller owns both objects.
  explicit PairReduceAdapter(pair_reduce::PairReduce& reduction) noexcept
      : reduction_(reduction) {}
  int rank() const noexcept override { return reduction_.rank(); }
  int world_size() const noexcept override { return reduction_.world_size(); }
  void reduce(const __nv_bfloat16* input, float* output, int m,
              std::string_view trace_id, std::string_view request_id,
              cudaStream_t stream) override {
    reduction_.enqueue(input, output, m, trace_id, request_id, stream);
  }
  void complete(cudaStream_t stream, std::uint32_t reduction_count,
                std::string_view trace_id,
                std::string_view request_id) override {
    reduction_.complete(stream, reduction_count, trace_id, request_id);
  }

 private:
  pair_reduce::PairReduce& reduction_;
};

// Explicit step machine. Idle accepts begin_step; Active accepts only the next
// layer-major attention/MoE point and finish_step after point 96. A reducer
// failure enters terminal Faulted because remote writes may have committed.
// Contract rejection does not mutate state. Reconstruct the object to retry a
// faulted step. Every call is synchronous, externally serialized, and bounded.
class Tp2DecodeExecution final {
 public:
  // Borrows reducer and telemetry for this object's lifetime.
  Tp2DecodeExecution(HiddenPartialReducer& reducer,
                     pair_reduce::OtelStageSink& telemetry,
                     std::uint64_t initial_completed_generation = 0);

  StepPhase phase() const noexcept { return phase_; }
  int next_ordinal() const noexcept { return next_ordinal_; }
  std::uint64_t last_completed_generation() const noexcept {
    return last_completed_generation_;
  }

  void begin_step(std::uint64_t device_generation, int m,
                  std::string_view trace_id, std::string_view request_id);
  void reduce_at(std::uint64_t device_generation, ReductionPoint point,
                 const __nv_bfloat16* local_partial, float* reduced_hidden,
                 std::string_view trace_id, std::string_view request_id,
                 cudaStream_t stream = nullptr);
  void finish_step(std::uint64_t device_generation,
                   std::string_view trace_id, std::string_view request_id);

  static ReductionPoint point_for_ordinal(int ordinal);

 private:
  void emit(std::string_view stage, pair_reduce::Outcome outcome, int m,
            std::string_view trace_id, std::string_view request_id,
            std::uint64_t duration_ns) noexcept;
  static bool valid_point(ReductionPoint point) noexcept;
  static std::string_view stage_for(ReductionPoint point) noexcept;

  HiddenPartialReducer& reducer_;
  pair_reduce::OtelStageSink& telemetry_;
  StepPhase phase_ = StepPhase::kIdle;
  int m_ = 0;
  int next_ordinal_ = 0;
  std::uint64_t active_generation_ = 0;
  std::uint64_t last_completed_generation_ = 0;
  cudaStream_t active_stream_ = nullptr;
};

}  // namespace rocket::qwen38::decode

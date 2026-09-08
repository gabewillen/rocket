// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "decode/execution.h"
#include "pair_reduce/rdma.h"

namespace rocket::qwen38::decode {

inline constexpr int kTargetK0ReductionsPerRow = 2 * kLayers;
inline constexpr int kTargetK0MaxRows = 262'144;

enum class TargetK0ReductionStage : std::uint8_t { kAttention, kMoe };
enum class TargetK0PairReducePhase : std::uint8_t {
  kAwaitSequence,
  kAwaitRow,
  kAwaitReduction,
  kCompleted,
  kFaulted,
};

// One process-local schedule over one physical TP2 PairReduce. A sequence is a
// series of c1 rows. Every row has exactly 96 calls in layer-major order:
// attention then MoE for layers 0..47. Ports are fixed to one layer/stage and
// cannot be cross-wired by a layer owner.
class TargetK0PairReduceSchedule final {
 public:
  TargetK0PairReduceSchedule(HiddenPartialReducer& reducer,
                             pair_reduce::OtelStageSink& telemetry);

  HiddenPartialReducer& attention_port(int layer);
  HiddenPartialReducer& moe_port(int layer);
  void begin_sequence(std::uint64_t first_generation, int rows);
  void begin_row(int row, std::uint64_t generation);

  [[nodiscard]] int rank() const noexcept { return reducer_.rank(); }
  [[nodiscard]] TargetK0PairReducePhase phase() const noexcept { return phase_; }
  [[nodiscard]] int completed_calls() const noexcept { return completed_calls_; }
  [[nodiscard]] int current_row() const noexcept { return current_row_; }
  [[nodiscard]] bool complete() const noexcept {
    return phase_ == TargetK0PairReducePhase::kCompleted;
  }
  [[nodiscard]] bool faulted() const noexcept {
    return phase_ == TargetK0PairReducePhase::kFaulted;
  }

 private:
  class StagePort final : public HiddenPartialReducer {
   public:
    StagePort(TargetK0PairReduceSchedule& owner, int layer,
              TargetK0ReductionStage stage) noexcept
        : owner_(owner), layer_(layer), stage_(stage) {}
    int rank() const noexcept override { return owner_.reducer_.rank(); }
    int world_size() const noexcept override {
      return owner_.reducer_.world_size();
    }
    void reduce(const __nv_bfloat16* input, float* output, int m,
                std::string_view trace_id, std::string_view request_id,
                cudaStream_t stream) override;

   private:
    TargetK0PairReduceSchedule& owner_;
    int layer_;
    TargetK0ReductionStage stage_;
  };

  void reduce(int layer, TargetK0ReductionStage stage,
              const __nv_bfloat16* input, float* output, int m,
              std::string_view trace_id, std::string_view request_id,
              cudaStream_t stream);
  void emit(pair_reduce::Outcome outcome, std::string_view trace_id,
            std::string_view request_id) noexcept;
  [[nodiscard]] int expected_layer() const noexcept;
  [[nodiscard]] TargetK0ReductionStage expected_stage() const noexcept;

  HiddenPartialReducer& reducer_;
  pair_reduce::OtelStageSink& telemetry_;
  std::array<std::unique_ptr<StagePort>, kLayers> attention_{};
  std::array<std::unique_ptr<StagePort>, kLayers> moe_{};
  TargetK0PairReducePhase phase_ = TargetK0PairReducePhase::kAwaitSequence;
  std::uint64_t first_generation_ = 0;
  int rows_ = 0;
  int current_row_ = -1;
  int row_calls_ = 0;
  int completed_calls_ = 0;
};

struct TargetK0PairReduceBootstrap {
  int rank = -1;
  int peer_rank = -1;
  std::string bootstrap_host;
  int bootstrap_port = 0;
  std::uint32_t timeout_ms = 0;
  std::array<std::uint8_t, 32> session_sha256{};
};

// Owns the existing RDMA transport, PairReduce, and the sole 96-call schedule.
// Construction is synchronous on both ranks. The session digest is supplied by
// the authenticated request/oracle root and is checked by the transport peer
// exchange; an all-zero or mismatched digest is rejected.
class TargetK0PairReduceOwner final {
 public:
  TargetK0PairReduceOwner(const TargetK0PairReduceBootstrap& bootstrap,
                          pair_reduce::OtelStageSink& telemetry);
  TargetK0PairReduceSchedule& schedule() noexcept { return *schedule_; }
  static void validate_bootstrap(const TargetK0PairReduceBootstrap& bootstrap);

 private:
  std::unique_ptr<pair_reduce::RdmaTransport> transport_;
  std::unique_ptr<pair_reduce::PairReduce> reduction_;
  std::unique_ptr<PairReduceAdapter> adapter_;
  std::unique_ptr<TargetK0PairReduceSchedule> schedule_;
};

}  // namespace rocket::qwen38::decode

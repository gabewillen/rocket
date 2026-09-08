// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "decode/target_layer3_prefill.h"
#include "pair_reduce/rdma.h"

namespace rocket::qwen38::decode {

inline constexpr int kTargetLayer3PairReduceRows = 35;
inline constexpr int kTargetLayer3PairReduceCalls =
    2 * kTargetLayer3PairReduceRows;

enum class TargetLayer3ReductionStage : std::uint8_t { kAttention, kMoe };
enum class TargetLayer3PairReducePhase : std::uint8_t {
  kAwaitRow,
  kAwaitAttention,
  kAwaitMoe,
  kCompleted,
  kFaulted,
};

struct TargetLayer3RowEvent {
  int row;
  std::uint64_t generation;
};

struct TargetLayer3ReductionEvent {
  TargetLayer3ReductionStage stage;
};

// Two distinct reducer ports share one synchronous PairReduce instance. The
// schedule is fixed to attention then MoE for each of rows 0..34. Any drift or
// underlying failure faults both ports terminally, because remote writes may
// already be visible. The 70th successful return is the peer-consumed terminal
// publication fence.
class TargetLayer3PairReduceSchedule final
    : public TargetLayer3ReductionGeneration {
 public:
  TargetLayer3PairReduceSchedule(HiddenPartialReducer& reducer,
                                 pair_reduce::OtelStageSink& telemetry);
  HiddenPartialReducer& attention_port() noexcept { return attention_; }
  HiddenPartialReducer& moe_port() noexcept { return moe_; }
  int rank() const noexcept override { return reducer_.rank(); }
  bool authenticated() const noexcept override { return true; }
  void begin_row(int row, std::uint64_t generation) override;
  int next_ordinal() const noexcept { return next_ordinal_; }
  std::uint64_t next_generation() const noexcept {
    return static_cast<std::uint64_t>(next_ordinal_ / 2 + 1);
  }
  bool complete() const noexcept {
    return phase_ == TargetLayer3PairReducePhase::kCompleted;
  }
  bool faulted() const noexcept {
    return phase_ == TargetLayer3PairReducePhase::kFaulted;
  }
  TargetLayer3PairReducePhase phase() const noexcept { return phase_; }

 private:
  class StagePort final : public HiddenPartialReducer {
   public:
    StagePort(TargetLayer3PairReduceSchedule& owner,
              TargetLayer3ReductionStage stage) noexcept
        : owner_(owner), stage_(stage) {}
    int rank() const noexcept override { return owner_.reducer_.rank(); }
    int world_size() const noexcept override {
      return owner_.reducer_.world_size();
    }
    void reduce(const __nv_bfloat16* input, float* output, int m,
                std::string_view trace_id, std::string_view request_id,
                cudaStream_t stream) override;
   private:
    TargetLayer3PairReduceSchedule& owner_;
    TargetLayer3ReductionStage stage_;
  };

  void reduce(TargetLayer3ReductionStage stage, const __nv_bfloat16* input,
              float* output, int m, std::string_view trace_id,
              std::string_view request_id, cudaStream_t stream);
  void transition(TargetLayer3RowEvent event);
  void guard(TargetLayer3ReductionEvent event) const;
  void transition(TargetLayer3ReductionEvent event);
  void emit(pair_reduce::Outcome outcome, std::string_view trace_id,
            std::string_view request_id) noexcept;

  HiddenPartialReducer& reducer_;
  pair_reduce::OtelStageSink& telemetry_;
  StagePort attention_;
  StagePort moe_;
  int next_ordinal_ = 0;
  TargetLayer3PairReducePhase phase_ =
      TargetLayer3PairReducePhase::kAwaitRow;
};

struct TargetLayer3PairReduceBootstrap {
  int rank = -1;
  int peer_rank = -1;
  std::string bootstrap_host;
  int bootstrap_port = 0;
  std::uint32_t timeout_ms = 0;
  std::array<std::uint8_t, 32> session_sha256{};
};

// Physical owner. Construction performs the existing synchronous RDMA
// bootstrap and PairReduce region registration. Call only after all slab,
// sidecar, rank, peer, and session preflight checks have succeeded.
class TargetLayer3PairReduceOwner final {
 public:
  TargetLayer3PairReduceOwner(const TargetLayer3PairReduceBootstrap& bootstrap,
                              pair_reduce::OtelStageSink& telemetry);
  HiddenPartialReducer& attention_port() noexcept {
    return schedule_->attention_port();
  }
  HiddenPartialReducer& moe_port() noexcept { return schedule_->moe_port(); }
  TargetLayer3PairReduceSchedule& schedule() noexcept { return *schedule_; }

  static std::array<std::uint8_t, 32> oracle_session_sha256() noexcept;
  static void validate_bootstrap(
      const TargetLayer3PairReduceBootstrap& bootstrap);

 private:
  std::unique_ptr<pair_reduce::RdmaTransport> transport_;
  std::unique_ptr<pair_reduce::PairReduce> reduction_;
  std::unique_ptr<PairReduceAdapter> adapter_;
  std::unique_ptr<TargetLayer3PairReduceSchedule> schedule_;
};

}  // namespace rocket::qwen38::decode

// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "attention/qsa_target_state_view.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rocket::qwen38::attention {

enum class TargetK0OracleQsaStateOutcome : std::uint8_t {
  kOk,
  kContractError,
  kCudaError,
  kCleanupIncomplete,
};

struct TargetK0OracleQsaStateOtelPoint {
  int rank;
  int max_rows_bucket;
  TargetK0OracleQsaStateOutcome outcome;
  std::uint64_t bytes;
};

// Bounded dimensions are rank {0,1}, max_rows_bucket {35}, and outcome.
class TargetK0OracleQsaStateOtelSink {
 public:
  virtual ~TargetK0OracleQsaStateOtelSink() = default;
  virtual void emit(const TargetK0OracleQsaStateOtelPoint&) noexcept = 0;
};

inline constexpr int kTargetK0QsaLayersPerRank = 12;

std::size_t target_k0_qsa_state_layer_bytes() noexcept;
std::size_t target_k0_qsa_state_bytes() noexcept;

struct TargetK0OracleQsaStateStorageBinding {
  std::array<TargetQsaStateView, kTargetK0QsaLayersPerRank> views;
  std::size_t used_bytes;
};

// Allocation-free layout proof used by the production owner and CPU tests.
TargetK0OracleQsaStateStorageBinding bind_target_k0_qsa_state_storage(
    void* storage, std::size_t bytes, int rank, int max_rows);

// Oracle-only c1 prefill arena. One allocation and one readiness event own all
// 12 rank-local QSA views for exactly 35 rows. This one-page integration arena
// cannot satisfy the 87-row short-oracle or 262144-context serving contracts.
class TargetK0OracleQsaStateOwner final {
 public:
  static std::unique_ptr<TargetK0OracleQsaStateOwner> create(
      int device, int rank, int max_rows,
      std::shared_ptr<TargetK0OracleQsaStateOtelSink> telemetry);
  ~TargetK0OracleQsaStateOwner();
  TargetK0OracleQsaStateOwner(const TargetK0OracleQsaStateOwner&) = delete;
  TargetK0OracleQsaStateOwner& operator=(const TargetK0OracleQsaStateOwner&) = delete;

  const TargetQsaStateView& view(int layer) const;
  void wait(cudaStream_t stream);
  cudaEvent_t ready_event() const noexcept { return ready_event_; }
  int rank() const noexcept { return rank_; }
  bool authenticated() const noexcept { return authenticated_; }

 private:
  TargetK0OracleQsaStateOwner(
      int device, int rank, int max_rows,
      std::shared_ptr<TargetK0OracleQsaStateOtelSink> telemetry);
  int device_ = -1;
  int rank_ = -1;
  int max_rows_ = 0;
  bool authenticated_ = false;
  bool waited_ = false;
  void* storage_ = nullptr;
  cudaStream_t init_stream_ = nullptr;
  cudaEvent_t ready_event_ = nullptr;
  TargetK0OracleQsaStateStorageBinding binding_{};
  std::shared_ptr<TargetK0OracleQsaStateOtelSink> telemetry_;
};

}  // namespace rocket::qwen38::attention

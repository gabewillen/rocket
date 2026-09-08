// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
#include <string_view>
#include "mtp/graph_runtime.h"
#include "mtp/nccl_communicator_owner.h"

namespace rocket::qwen38::mtp {
// Owns the communicator owner it consumes. The telemetry sink borrowed by that
// owner must outlive this exchange. Construction is initialization-only;
// enqueue and validate_after_fence are single-owner runtime operations.
class NcclWinnerExchange final : public WinnerExchangePort {
 public:
  explicit NcclWinnerExchange(
      std::unique_ptr<NcclCommunicatorOwner> communicator_owner);
  ~NcclWinnerExchange();
  NcclWinnerExchange(const NcclWinnerExchange&) = delete;
  NcclWinnerExchange& operator=(const NcclWinnerExchange&) = delete;
  void enqueue(const output::Winner*, output::Winner*, int, int,
               cudaStream_t) override;
  void validate_after_fence() override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Synchronously authenticates the peer and initializes a TP2 communicator on
// CUDA0. No collective or CUDA kernel is launched during construction. The
// returned WinnerExchangePort owns all communicator resources.
std::unique_ptr<WinnerExchangePort> make_nccl_winner_exchange(
    const NcclCommunicatorConfig& config, NcclBootstrapOtelSink& telemetry);
}  // namespace rocket::qwen38::mtp

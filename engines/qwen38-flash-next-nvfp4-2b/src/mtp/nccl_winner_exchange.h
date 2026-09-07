// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string_view>
#include "mtp/graph_runtime.h"

namespace rocket::qwen38::mtp {
// Dynamically binds the pinned NCCL ABI supplied by the physical runtime.
// The communicator is borrowed, already initialized for exactly two ranks,
// and remains owned by the launcher.
class NcclWinnerExchange final : public WinnerExchangePort {
 public:
  NcclWinnerExchange(void* communicator, int rank,
                     std::string_view library = "libnccl.so.2");
  ~NcclWinnerExchange();
  NcclWinnerExchange(const NcclWinnerExchange&) = delete;
  NcclWinnerExchange& operator=(const NcclWinnerExchange&) = delete;
  void enqueue(const output::Winner*, output::Winner*, int, int,
               cudaStream_t) override;
  void validate_after_fence() override;
 private:
  struct Impl;
  Impl* impl_;
};
}  // namespace rocket::qwen38::mtp

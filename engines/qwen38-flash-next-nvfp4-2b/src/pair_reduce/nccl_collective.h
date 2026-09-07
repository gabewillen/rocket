#pragma once

#include <nccl.h>

#include <cstdint>
#include <string>

#include "pair_reduce/pair_reduce.h"

namespace rocket::qwen38::pair_reduce {

struct NcclConfig {
  int rank = 0;
  std::string bootstrap_host = "192.168.100.10";
  int bootstrap_port = 18838;
  std::uint32_t operation_timeout_ms = 120'000;
};

// NCCL 2.30.7 TP2 boundary. Construction performs the TCP unique-id exchange
// and communicator initialization. enqueue_sum() is allocation-free and uses
// only the caller stream. Destruction and abort are host-only lifecycle calls
// and must occur outside CUDA Graph capture.
class NcclCollective final : public DeviceCollective {
 public:
  explicit NcclCollective(const NcclConfig& config);
  ~NcclCollective() override;
  NcclCollective(const NcclCollective&) = delete;
  NcclCollective& operator=(const NcclCollective&) = delete;

  int rank() const noexcept override { return rank_; }
  int world_size() const noexcept override { return kWorldSize; }
  void enqueue_sum(const __nv_bfloat16* input, __nv_bfloat16* output,
                   std::size_t elements, cudaStream_t stream) override;
  bool healthy() noexcept override;
  void abort() noexcept override;

 private:
  int rank_ = -1;
  ncclComm_t communicator_ = nullptr;
  bool aborted_ = false;
};

}  // namespace rocket::qwen38::pair_reduce

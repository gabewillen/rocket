#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pair_reduce/transport.h"

namespace rocket::qwen38::pair_reduce {

class RdmaError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct RdmaConfig {
  int rank = 0;
  std::string bootstrap_host = "192.168.100.10";
  int bootstrap_port = 18838;
  std::vector<std::string> devices{"rocep1s0f1", "roceP2p1s0f1"};
  int gid_index = 3;
  std::size_t rail_split_bytes = 65'536;
  std::uint32_t operation_timeout_ms = 120'000;
};

constexpr bool valid_operation_timeout_ms(std::uint32_t timeout_ms) noexcept {
  return timeout_ms >= 100 && timeout_ms <= 120'000;
}

// Fixed two-rank, two-rail RC transport. Registered regions must match in size
// and registration order across ranks. Payload writes are unsignaled. A
// signaled inline sequence write on every rail is the ordered publication
// boundary. All operations are single-owner and synchronous. Peer doorbell and
// send-completion waits either finish or throw RdmaError within
// operation_timeout_ms. A timeout can leave peer-visible writes committed; the
// caller must discard the failed PairReduce instance rather than retry it.
class RdmaTransport final : public Transport {
 public:
  explicit RdmaTransport(const RdmaConfig& config);
  ~RdmaTransport() override;
  RdmaTransport(const RdmaTransport&) = delete;
  RdmaTransport& operator=(const RdmaTransport&) = delete;

  int rank() const noexcept override;
  int world_size() const noexcept override { return 2; }
  int register_region(void* address, std::size_t bytes) override;
  void unregister_region(int region) noexcept override;
  std::uint64_t next_sequence() override;
  void post_unsignaled_write(int region, std::size_t source_offset,
                             std::size_t peer_offset, std::size_t bytes) override;
  void signal_sequence(std::uint64_t sequence) override;
  void wait_peer(std::uint64_t sequence) override;
  void flush_signaled() override;
  void acknowledge_consumed(std::uint64_t sequence) override;
  void wait_peer_consumed(std::uint64_t sequence) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rocket::qwen38::pair_reduce

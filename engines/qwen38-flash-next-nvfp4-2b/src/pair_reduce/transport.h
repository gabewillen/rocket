#pragma once

#include <cstddef>
#include <cstdint>

namespace rocket::qwen38::pair_reduce {

// Synchronous fixed-pair transport boundary. Implementations borrow every
// registered region until their destructor and are single-threaded: one owner
// must serialize registration and operations on both ranks in the same order.
class Transport {
 public:
  virtual ~Transport() = default;
  virtual int rank() const noexcept = 0;
  virtual int world_size() const noexcept = 0;
  virtual int register_region(void* address, std::size_t bytes) = 0;
  virtual void unregister_region(int region) noexcept = 0;
  virtual std::uint64_t next_sequence() = 0;
  virtual void post_unsignaled_write(int region, std::size_t source_offset,
                                     std::size_t peer_offset, std::size_t bytes) = 0;
  virtual void signal_sequence(std::uint64_t sequence) = 0;
  virtual void wait_peer(std::uint64_t sequence) = 0;
  virtual void flush_signaled() = 0;
};

}  // namespace rocket::qwen38::pair_reduce

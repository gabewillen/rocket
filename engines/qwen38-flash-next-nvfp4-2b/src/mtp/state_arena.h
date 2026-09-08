// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace rocket::qwen38::mtp {
inline constexpr int kStateMaxDepth = 7;
inline constexpr int kMtpMultiHidden = 10'240;
inline constexpr int kQsaMainKvWidth = 256;
inline constexpr int kQsaIndexerWidth = 128;
inline constexpr int kMropeAxes = 3;

struct PrefixStateView {
  __nv_bfloat16* multi_hidden;
  __nv_bfloat16* main_key;
  __nv_bfloat16* main_value;
  __nv_bfloat16* raw_key;
  __nv_bfloat16* compressed_key;
  std::int64_t* rope_positions;
  std::int32_t* main_slots;
  std::int32_t* raw_slots;
  std::int32_t* compressed_slots;
  std::int32_t* compressed_valid;
};
struct InactiveStateView { PrefixStateView selected; int sequences; std::uint64_t generation; };
class StateArenaError : public std::runtime_error { public: using std::runtime_error::runtime_error; };

class StateArena final {
 public:
  StateArena(int sequences, int depth, bool uses_mrope);
  ~StateArena();
  StateArena(const StateArena&) = delete;
  StateArena& operator=(const StateArena&) = delete;
  [[nodiscard]] PrefixStateView prefix(int step) const;
  [[nodiscard]] InactiveStateView inactive(std::uint64_t generation) const;
  void select(const std::int32_t* accepted_widths_device,
              std::uint64_t generation, cudaStream_t stream,
              std::byte* transaction_storage = nullptr);
  void commit(std::uint64_t generation);
  void discard(std::uint64_t generation) noexcept;
  [[nodiscard]] int sequences() const noexcept { return sequences_; }
  [[nodiscard]] int depth() const noexcept { return depth_; }
  [[nodiscard]] bool uses_mrope() const noexcept { return uses_mrope_; }
  [[nodiscard]] std::size_t allocated_bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::size_t transaction_bytes() const noexcept;
 private:
  int sequences_; int depth_; bool uses_mrope_; std::size_t bytes_ = 0;
  PrefixStateView prefixes_{}; PrefixStateView selected_{};
  void* allocation_ = nullptr; std::uint64_t pending_generation_ = 0;
  PrefixStateView bind_view(std::byte* base, std::size_t rows) const noexcept;
};
}  // namespace rocket::qwen38::mtp

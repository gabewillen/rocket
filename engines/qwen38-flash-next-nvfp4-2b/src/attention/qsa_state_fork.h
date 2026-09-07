// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace rocket::qwen38::attention {

constexpr int kQsaStateSequences = 16;
constexpr int kQsaStateContext = 262144;
constexpr int kQsaRawWindow = 8;
constexpr int kQsaCompressRatio = 4;
constexpr int kQsaMainBytesPerToken = 512;
constexpr int kQsaRawBytesPerToken = 280;
constexpr int kQsaCompressedBytesPerBlock = 256;
constexpr int kQsaMaxForkRows = 128;

struct QsaStateExtents {
  std::uint8_t* main_state;
  std::size_t main_bytes;
  std::uint8_t* raw_state;
  std::size_t raw_bytes;
  std::uint8_t* compressed_state;
  std::size_t compressed_bytes;
};

// Owns one private speculative fork. The caller supplies already-formatted
// FP8 main-cache, BF16 raw-cache, and BF16 compressed-cache rows produced by
// the native attention preprocessor. stage() never writes active state.
class QsaStateFork final {
 public:
  QsaStateFork(int device, QsaStateExtents active);
  ~QsaStateFork();
  QsaStateFork(const QsaStateFork&) = delete;
  QsaStateFork& operator=(const QsaStateFork&) = delete;

  void stage(const std::uint8_t* main_rows,
             const std::uint8_t* raw_rows,
             const std::uint8_t* compressed_rows,
             const std::int64_t* logical_positions,
             const std::int32_t* token_to_request,
             int sequences, int verify_width, int token_rows,
             cudaStream_t stream);
  void accept(const std::int32_t* accepted_lengths, int count,
              cudaStream_t stream);
  void reset(cudaStream_t stream);

  bool staged() const noexcept { return staged_; }
  bool faulted() const noexcept { return faulted_; }
  const char* last_error() const noexcept { return error_; }

 private:
  void fail(const char* message) noexcept;
  struct Impl;
  Impl* impl_;
  const char* error_ = "";
  int sequences_ = 0;
  int verify_width_ = 0;
  int token_rows_ = 0;
  cudaStream_t staged_stream_ = nullptr;
  bool staged_ = false;
  bool faulted_ = false;
};

}  // namespace rocket::qwen38::attention

// SPDX-License-Identifier: Apache-2.0
#include "attention/qsa_state_fork.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace rocket::qwen38::attention {
namespace {

constexpr std::size_t kMainBytes =
    static_cast<std::size_t>(kQsaStateSequences) * kQsaStateContext *
    kQsaMainBytesPerToken;
constexpr std::size_t kRawBytes =
    static_cast<std::size_t>(kQsaStateSequences) * kQsaRawWindow *
    kQsaRawBytesPerToken;
constexpr std::size_t kCompressedBytes =
    static_cast<std::size_t>(kQsaStateSequences) *
    (kQsaStateContext / kQsaCompressRatio) * kQsaCompressedBytesPerBlock;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

__global__ void publish_prefixes(
    std::uint8_t* main_state, std::uint8_t* raw_state,
    std::uint8_t* compressed_state, const std::uint8_t* main_rows,
    const std::uint8_t* raw_rows, const std::uint8_t* compressed_rows,
    const std::int64_t* logical_positions,
    const std::int32_t* token_to_request,
    const std::int32_t* accepted_lengths, int sequences, int verify_width,
    int token_rows) {
  const int row = blockIdx.x;
  const int byte = threadIdx.x;
  if (row >= token_rows) return;
  const int sequence = row / verify_width;
  const int speculative_index = row % verify_width;
  if (sequence >= sequences || speculative_index >= accepted_lengths[sequence])
    return;
  const int request = token_to_request[row];
  const std::int64_t position = logical_positions[row];
  if (request < 0 || request >= kQsaStateSequences || position < 0 ||
      position >= kQsaStateContext)
    return;
  if (byte < kQsaMainBytesPerToken) {
    const std::size_t destination =
        (static_cast<std::size_t>(request) * kQsaStateContext + position) *
            kQsaMainBytesPerToken +
        byte;
    main_state[destination] =
        main_rows[static_cast<std::size_t>(row) * kQsaMainBytesPerToken + byte];
  }
  if (byte < kQsaRawBytesPerToken) {
    const std::size_t destination =
        (static_cast<std::size_t>(request) * kQsaRawWindow +
         position % kQsaRawWindow) *
            kQsaRawBytesPerToken +
        byte;
    raw_state[destination] =
        raw_rows[static_cast<std::size_t>(row) * kQsaRawBytesPerToken + byte];
  }
  if (position % kQsaCompressRatio == kQsaCompressRatio - 1 &&
      byte < kQsaCompressedBytesPerBlock) {
    const std::size_t destination =
        (static_cast<std::size_t>(request) *
             (kQsaStateContext / kQsaCompressRatio) +
         position / kQsaCompressRatio) *
            kQsaCompressedBytesPerBlock +
        byte;
    compressed_state[destination] =
        compressed_rows[static_cast<std::size_t>(row) *
                            kQsaCompressedBytesPerBlock +
                        byte];
  }
}

__global__ void validate_prefixes(
    const std::int64_t* logical_positions,
    const std::int32_t* token_to_request,
    const std::int32_t* accepted_lengths, int sequences, int verify_width,
    int token_rows, int* invalid) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= token_rows) return;
  const int sequence = row / verify_width;
  const int speculative_index = row % verify_width;
  if (sequence >= sequences || speculative_index >= accepted_lengths[sequence])
    return;
  if (token_to_request[row] < 0 ||
      token_to_request[row] >= kQsaStateSequences ||
      logical_positions[row] < 0 ||
      logical_positions[row] >= kQsaStateContext) {
    atomicExch(invalid, 1);
  }
}

}  // namespace

struct QsaStateFork::Impl {
  int device;
  QsaStateExtents active;
  std::uint8_t* main_rows = nullptr;
  std::uint8_t* raw_rows = nullptr;
  std::uint8_t* compressed_rows = nullptr;
  std::int64_t* logical_positions = nullptr;
  std::int32_t* token_to_request = nullptr;
  std::int32_t* accepted_lengths = nullptr;
  int* invalid = nullptr;

  ~Impl() {
    cudaSetDevice(device);
    cudaFree(invalid);
    cudaFree(accepted_lengths);
    cudaFree(token_to_request);
    cudaFree(logical_positions);
    cudaFree(compressed_rows);
    cudaFree(raw_rows);
    cudaFree(main_rows);
  }
};

QsaStateFork::QsaStateFork(int device, QsaStateExtents active)
    : impl_(new Impl{device, active}) {
  if (device < 0 || !active.main_state || active.main_bytes != kMainBytes ||
      !active.raw_state || active.raw_bytes != kRawBytes ||
      !active.compressed_state || active.compressed_bytes != kCompressedBytes) {
    delete impl_;
    impl_ = nullptr;
    throw std::invalid_argument("exact QSA active-state extents are required");
  }
  try {
    check(cudaSetDevice(device), "cudaSetDevice");
    check(cudaMalloc(&impl_->main_rows,
                     kQsaMaxForkRows * kQsaMainBytesPerToken),
          "allocate private main rows");
    check(cudaMalloc(&impl_->raw_rows,
                     kQsaMaxForkRows * kQsaRawBytesPerToken),
          "allocate private raw rows");
    check(cudaMalloc(&impl_->compressed_rows,
                     kQsaMaxForkRows * kQsaCompressedBytesPerBlock),
          "allocate private compressed rows");
    check(cudaMalloc(&impl_->logical_positions,
                     kQsaMaxForkRows * sizeof(std::int64_t)),
          "allocate private logical positions");
    check(cudaMalloc(&impl_->token_to_request,
                     kQsaMaxForkRows * sizeof(std::int32_t)),
          "allocate private request map");
    check(cudaMalloc(&impl_->accepted_lengths,
                     kQsaStateSequences * sizeof(std::int32_t)),
          "allocate accepted lengths");
    check(cudaMalloc(&impl_->invalid, sizeof(int)),
          "allocate publication validator");
  } catch (...) {
    delete impl_;
    impl_ = nullptr;
    throw;
  }
}

QsaStateFork::~QsaStateFork() { delete impl_; }

void QsaStateFork::stage(
    const std::uint8_t* main_rows, const std::uint8_t* raw_rows,
    const std::uint8_t* compressed_rows,
    const std::int64_t* logical_positions,
    const std::int32_t* token_to_request, int sequences, int verify_width,
    int token_rows, cudaStream_t stream) {
  if (faulted_ || staged_ || !main_rows || !raw_rows || !compressed_rows ||
      !logical_positions || !token_to_request || !stream || sequences < 1 ||
      sequences > kQsaStateSequences || verify_width < 1 || verify_width > 8 ||
      token_rows != sequences * verify_width ||
      token_rows > kQsaMaxForkRows) {
    throw std::invalid_argument("QSA speculative state stage contract changed");
  }
  try {
    check(cudaMemcpyAsync(impl_->main_rows, main_rows,
                          token_rows * kQsaMainBytesPerToken,
                          cudaMemcpyDeviceToDevice, stream),
          "stage private main rows");
    check(cudaMemcpyAsync(impl_->raw_rows, raw_rows,
                          token_rows * kQsaRawBytesPerToken,
                          cudaMemcpyDeviceToDevice, stream),
          "stage private raw rows");
    check(cudaMemcpyAsync(impl_->compressed_rows, compressed_rows,
                          token_rows * kQsaCompressedBytesPerBlock,
                          cudaMemcpyDeviceToDevice, stream),
          "stage private compressed rows");
    check(cudaMemcpyAsync(impl_->logical_positions, logical_positions,
                          token_rows * sizeof(std::int64_t),
                          cudaMemcpyDeviceToDevice, stream),
          "stage private logical positions");
    check(cudaMemcpyAsync(impl_->token_to_request, token_to_request,
                          token_rows * sizeof(std::int32_t),
                          cudaMemcpyDeviceToDevice, stream),
          "stage private request map");
  } catch (...) {
    fail("CUDA state staging failed");
    throw;
  }
  sequences_ = sequences;
  verify_width_ = verify_width;
  token_rows_ = token_rows;
  staged_stream_ = stream;
  staged_ = true;
}

void QsaStateFork::accept(const std::int32_t* accepted_lengths, int count,
                          cudaStream_t stream) {
  if (faulted_ || !staged_ || !accepted_lengths || count != sequences_ ||
      !stream || stream != staged_stream_) {
    throw std::invalid_argument("QSA accepted-prefix contract changed");
  }
  for (int sequence = 0; sequence < count; ++sequence) {
    if (accepted_lengths[sequence] < 0 ||
        accepted_lengths[sequence] > verify_width_) {
      throw std::invalid_argument("QSA accepted prefix exceeds verify width");
    }
  }
  if (cudaMemcpyAsync(impl_->accepted_lengths, accepted_lengths,
                      count * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    fail("CUDA accepted-prefix upload failed");
    throw std::runtime_error(error_);
  }
  int invalid = 0;
  if (cudaMemsetAsync(impl_->invalid, 0, sizeof(int), stream) != cudaSuccess) {
    fail("CUDA publication validation setup failed");
    throw std::runtime_error(error_);
  }
  validate_prefixes<<<(token_rows_ + 127) / 128, 128, 0, stream>>>(
      impl_->logical_positions, impl_->token_to_request,
      impl_->accepted_lengths, sequences_, verify_width_, token_rows_,
      impl_->invalid);
  cudaError_t validation = cudaGetLastError();
  if (validation == cudaSuccess)
    validation = cudaMemcpyAsync(&invalid, impl_->invalid, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream);
  if (validation == cudaSuccess) validation = cudaStreamSynchronize(stream);
  if (validation != cudaSuccess) {
    fail("CUDA publication validation failed");
    throw std::runtime_error(error_);
  }
  if (invalid != 0)
    throw std::invalid_argument("QSA accepted metadata exceeds active state");
  publish_prefixes<<<token_rows_, kQsaMainBytesPerToken, 0, stream>>>(
      impl_->active.main_state, impl_->active.raw_state,
      impl_->active.compressed_state, impl_->main_rows, impl_->raw_rows,
      impl_->compressed_rows, impl_->logical_positions,
      impl_->token_to_request, impl_->accepted_lengths, sequences_, verify_width_,
      token_rows_);
  const cudaError_t launch = cudaGetLastError();
  const cudaError_t fence =
      launch == cudaSuccess ? cudaStreamSynchronize(stream) : launch;
  if (fence != cudaSuccess) {
    fail("CUDA accepted-prefix publication failed");
    throw std::runtime_error(error_);
  }
  staged_ = false;
  staged_stream_ = nullptr;
}

void QsaStateFork::reset(cudaStream_t stream) {
  if (faulted_ || !staged_ || !stream || stream != staged_stream_)
    throw std::invalid_argument("QSA state reset contract changed");
  // All speculative bytes live in graph-owned row buffers. No active-state
  // write is needed to reject a fork, but the private D2D copies must finish
  // before a later stage is allowed to reuse the row buffers.
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    fail("CUDA state reset fence failed");
    throw std::runtime_error(error_);
  }
  staged_ = false;
  staged_stream_ = nullptr;
}

void QsaStateFork::fail(const char* message) noexcept {
  error_ = message;
  faulted_ = true;
}

}  // namespace rocket::qwen38::attention

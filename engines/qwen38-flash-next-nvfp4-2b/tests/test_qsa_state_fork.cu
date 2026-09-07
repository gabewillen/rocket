// SPDX-License-Identifier: Apache-2.0
#include "attention/qsa_state_fork.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace attention = rocket::qwen38::attention;

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct DeviceBytes {
  std::uint8_t* pointer = nullptr;
  std::size_t bytes;
  explicit DeviceBytes(std::size_t size) : bytes(size) {
    check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
  }
  ~DeviceBytes() { cudaFree(pointer); }
};

std::vector<std::uint8_t> rows(int count, int width) {
  std::vector<std::uint8_t> result(static_cast<std::size_t>(count) * width);
  for (int row = 0; row < count; ++row)
    for (int byte = 0; byte < width; ++byte)
      result[static_cast<std::size_t>(row) * width + byte] =
          static_cast<std::uint8_t>(row + 1);
  return result;
}

void upload(DeviceBytes& target, const void* source) {
  check_cuda(cudaMemcpy(target.pointer, source, target.bytes,
                        cudaMemcpyHostToDevice),
             "upload");
}

std::uint8_t byte_at(const DeviceBytes& source, std::size_t offset) {
  std::uint8_t value = 0;
  check_cuda(cudaMemcpy(&value, source.pointer + offset, 1,
                        cudaMemcpyDeviceToHost),
             "read sentinel");
  return value;
}

}  // namespace

int main() try {
  constexpr std::size_t main_bytes =
      static_cast<std::size_t>(attention::kQsaStateSequences) *
      attention::kQsaStateContext * attention::kQsaMainBytesPerToken;
  constexpr std::size_t raw_bytes =
      static_cast<std::size_t>(attention::kQsaStateSequences) *
      attention::kQsaRawWindow * attention::kQsaRawBytesPerToken;
  constexpr std::size_t compressed_bytes =
      static_cast<std::size_t>(attention::kQsaStateSequences) *
      (attention::kQsaStateContext / attention::kQsaCompressRatio) *
      attention::kQsaCompressedBytesPerBlock;
  DeviceBytes main_state(main_bytes), raw_state(raw_bytes),
      compressed_state(compressed_bytes);
  check_cuda(cudaMemset(main_state.pointer, 0, main_state.bytes), "clear main");
  check_cuda(cudaMemset(raw_state.pointer, 0, raw_state.bytes), "clear raw");
  check_cuda(cudaMemset(compressed_state.pointer, 0, compressed_state.bytes),
             "clear compressed");

  attention::QsaStateFork fork(
      0, {main_state.pointer, main_state.bytes, raw_state.pointer,
          raw_state.bytes, compressed_state.pointer, compressed_state.bytes});
  constexpr int row_count = 8;
  auto main_rows = rows(row_count, attention::kQsaMainBytesPerToken);
  auto raw_rows = rows(row_count, attention::kQsaRawBytesPerToken);
  auto compressed_rows = rows(row_count,
                              attention::kQsaCompressedBytesPerBlock);
  const std::array<std::int64_t, row_count> positions = {0, 1, 2, 3, 8, 9, 10, 11};
  const std::array<std::int32_t, row_count> requests = {0, 0, 0, 0, 1, 1, 1, 1};
  DeviceBytes device_main(main_rows.size()), device_raw(raw_rows.size()),
      device_compressed(compressed_rows.size()),
      device_positions(sizeof(positions)), device_requests(sizeof(requests));
  upload(device_main, main_rows.data());
  upload(device_raw, raw_rows.data());
  upload(device_compressed, compressed_rows.data());
  upload(device_positions, positions.data());
  upload(device_requests, requests.data());
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreate(&stream), "create stream");

  fork.stage(device_main.pointer, device_raw.pointer, device_compressed.pointer,
             reinterpret_cast<const std::int64_t*>(device_positions.pointer),
             reinterpret_cast<const std::int32_t*>(device_requests.pointer),
             2, 4, row_count, stream);
  // stage() owns copies only. Active state remains unchanged before accept.
  check_cuda(cudaStreamSynchronize(stream), "fence private stage");
  check(byte_at(main_state, 0) == 0,
        "private stage changed active main state");
  const std::array<std::int32_t, 2> accepted = {2, 4};
  fork.accept(accepted.data(), accepted.size(), stream);
  check(!fork.staged() && !fork.faulted(), "accepted fork did not close");

  const auto main_offset = [](int request, int position) {
    return (static_cast<std::size_t>(request) * attention::kQsaStateContext +
            position) * attention::kQsaMainBytesPerToken;
  };
  check(byte_at(main_state, main_offset(0, 0)) == 1 &&
            byte_at(main_state, main_offset(0, 1)) == 2 &&
            byte_at(main_state, main_offset(0, 2)) == 0 &&
            byte_at(main_state, main_offset(1, 8)) == 5 &&
            byte_at(main_state, main_offset(1, 11)) == 8,
        "accepted-prefix main-state scatter changed");
  const std::size_t accepted_compressed =
      (static_cast<std::size_t>(1) *
           (attention::kQsaStateContext / attention::kQsaCompressRatio) +
       11 / attention::kQsaCompressRatio) *
      attention::kQsaCompressedBytesPerBlock;
  check(byte_at(compressed_state, accepted_compressed) == 8 &&
            byte_at(compressed_state, 0) == 0,
        "compressed state published a rejected or incomplete group");

  std::array<double, 5> publication_us{};
  for (double& sample : publication_us) {
    const auto start = std::chrono::steady_clock::now();
    fork.stage(device_main.pointer, device_raw.pointer,
               device_compressed.pointer,
               reinterpret_cast<const std::int64_t*>(device_positions.pointer),
               reinterpret_cast<const std::int32_t*>(device_requests.pointer),
               2, 4, row_count, stream);
    fork.accept(accepted.data(), accepted.size(), stream);
    sample = std::chrono::duration<double, std::micro>(
                 std::chrono::steady_clock::now() - start)
                 .count();
  }

  fork.stage(device_main.pointer, device_raw.pointer, device_compressed.pointer,
             reinterpret_cast<const std::int64_t*>(device_positions.pointer),
             reinterpret_cast<const std::int32_t*>(device_requests.pointer),
             2, 4, row_count, stream);
  fork.reset(stream);
  check(byte_at(main_state, main_offset(0, 0)) == 1 &&
            byte_at(main_state, main_offset(0, 2)) == 0,
        "reset changed active state");

  auto invalid_requests = requests;
  invalid_requests[0] = attention::kQsaStateSequences;
  upload(device_requests, invalid_requests.data());
  fork.stage(device_main.pointer, device_raw.pointer, device_compressed.pointer,
             reinterpret_cast<const std::int64_t*>(device_positions.pointer),
             reinterpret_cast<const std::int32_t*>(device_requests.pointer),
             2, 4, row_count, stream);
  bool invalid_rejected = false;
  try {
    fork.accept(accepted.data(), accepted.size(), stream);
  } catch (const std::invalid_argument&) {
    invalid_rejected = true;
  }
  check(invalid_rejected && byte_at(main_state, main_offset(0, 0)) == 1,
        "invalid accepted metadata changed active state");
  fork.reset(stream);
  check_cuda(cudaStreamDestroy(stream), "destroy stream");
  double total = 0.0, minimum = publication_us[0], maximum = publication_us[0];
  for (double sample : publication_us) {
    total += sample;
    minimum = sample < minimum ? sample : minimum;
    maximum = sample > maximum ? sample : maximum;
  }
  std::printf(
      "qwen38 QSA accepted-prefix native state fork passed: "
      "rows=8 mean_us=%.3f min_us=%.3f max_us=%.3f\n",
      total / publication_us.size(), minimum, maximum);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "FAIL: %s\n", error.what());
  return 1;
}

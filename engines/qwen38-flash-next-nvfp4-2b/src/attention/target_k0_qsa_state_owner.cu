// SPDX-License-Identifier: Apache-2.0
#include "attention/target_k0_qsa_state_owner.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace rocket::qwen38::attention {
namespace {

constexpr std::size_t align_up(std::size_t value,
                               std::size_t alignment) noexcept {
  return (value + alignment - 1) & ~(alignment - 1);
}

constexpr std::size_t append(std::size_t offset, std::size_t bytes,
                             std::size_t alignment) noexcept {
  return align_up(offset, alignment) + bytes;
}

constexpr std::size_t layer_size() noexcept {
  std::size_t o = 0;
  o = append(o, 1600ULL * 256 * 2, 256);  // main K
  o = append(o, 1600ULL * 256 * 2, 256);  // main V
  o = append(o, 8ULL * 140 * 2, 256);     // raw K ring
  o = append(o, 400ULL * 128 * 2, 256);   // compressed K
  o = append(o, 8, 8);                    // positions
  o = append(o, 4, 4);                    // main slot
  o = append(o, 164ULL * 4, 4);           // main table
  o = append(o, 4, 4);                    // raw slot
  o = append(o, 4, 4);                    // raw table
  o = append(o, 4, 4);                    // compressed slot
  o = append(o, 164ULL * 4, 4);           // compressed table
  o = append(o, 4, 4);                    // query start
  o = append(o, 8, 8);                    // logical position
  o = append(o, 4, 4);                    // sequence length
  o = append(o, 4, 4);                    // token to request
  o = append(o, 4, 4);                    // compression work
  return align_up(o, 256);
}

struct Cursor {
  std::uint8_t* base;
  std::size_t capacity;
  std::size_t offset;

  template <class T>
  T* take(std::size_t count, std::size_t alignment = alignof(T)) {
    offset = align_up(offset, alignment);
    if (count > (capacity - offset) / sizeof(T))
      throw std::invalid_argument("K0 QSA state storage is short");
    auto* result = reinterpret_cast<T*>(base + offset);
    offset += count * sizeof(T);
    return result;
  }
};

[[noreturn]] void cuda_fail(const char* operation, cudaError_t status) {
  throw std::runtime_error(std::string("K0 QSA state ") + operation + ": " +
                           cudaGetErrorString(status));
}

}  // namespace

std::size_t target_k0_qsa_state_layer_bytes() noexcept {
  return layer_size();
}

std::size_t target_k0_qsa_state_bytes() noexcept {
  return kTargetK0QsaLayersPerRank * layer_size();
}

TargetK0OracleQsaStateStorageBinding bind_target_k0_qsa_state_storage(
    void* storage, std::size_t bytes, int rank, int max_rows) {
  if (!storage || bytes != target_k0_qsa_state_bytes() ||
      (rank != 0 && rank != 1) || max_rows != 35)
    throw std::invalid_argument("K0 QSA state layout identity changed");
  Cursor cursor{static_cast<std::uint8_t*>(storage), bytes, 0};
  TargetK0OracleQsaStateStorageBinding result{};
  for (int index = 0; index < kTargetK0QsaLayersPerRank; ++index) {
    const std::size_t begin = cursor.offset;
    TargetQsaStateView view{};
    view.main_key_cache = cursor.take<__nv_bfloat16>(1600ULL * 256, 256);
    view.main_value_cache = cursor.take<__nv_bfloat16>(1600ULL * 256, 256);
    view.raw_key_cache = cursor.take<__nv_bfloat16>(8ULL * 140, 256);
    view.compressed_key_cache = cursor.take<__nv_bfloat16>(400ULL * 128, 256);
    view.positions = cursor.take<std::int64_t>(1);
    view.main_slot_mapping = cursor.take<std::int32_t>(1);
    view.main_block_table = cursor.take<std::int32_t>(164);
    view.raw_slot_mapping = cursor.take<std::int32_t>(1);
    view.raw_block_table = cursor.take<std::int32_t>(1);
    view.compressed_slot_mapping = cursor.take<std::int32_t>(1);
    view.compressed_block_table = cursor.take<std::int32_t>(164);
    view.query_start_locations = cursor.take<std::int32_t>(1);
    view.logical_positions = cursor.take<std::int64_t>(1);
    view.sequence_lengths = cursor.take<std::int32_t>(1);
    view.token_to_request = cursor.take<std::int32_t>(1);
    view.compression_work = cursor.take<std::int32_t>(1);
    cursor.offset = align_up(cursor.offset, 256);
    if (cursor.offset - begin != layer_size())
      throw std::logic_error("K0 QSA state layer stride changed");
    view.main_blocks = 1;
    view.compressed_blocks = 1;
    view.compression_work_items = 1;
    view.rows = 1;
    view.rank = rank;
    view.layer = 4 * index + 3;
    view.uses_mrope = false;
    result.views[index] = view;
  }
  result.used_bytes = cursor.offset;
  if (result.used_bytes != bytes)
    throw std::logic_error("K0 QSA state cursor changed");
  return result;
}

std::unique_ptr<TargetK0OracleQsaStateOwner> TargetK0OracleQsaStateOwner::create(
    int device, int rank, int max_rows,
    std::shared_ptr<TargetK0OracleQsaStateOtelSink> telemetry) {
  return std::unique_ptr<TargetK0OracleQsaStateOwner>(
      new TargetK0OracleQsaStateOwner(device, rank, max_rows,
                                      std::move(telemetry)));
}

TargetK0OracleQsaStateOwner::TargetK0OracleQsaStateOwner(
    int device, int rank, int max_rows,
    std::shared_ptr<TargetK0OracleQsaStateOtelSink> telemetry)
    : device_(device), rank_(rank), max_rows_(max_rows),
      telemetry_(std::move(telemetry)) {
  if (device < 0 || (rank != 0 && rank != 1) ||
      max_rows != 35 || !telemetry_) {
    if (telemetry_)
      telemetry_->emit({rank_, max_rows_,
                        TargetK0OracleQsaStateOutcome::kContractError, 0});
    throw std::invalid_argument("K0 QSA state owner identity changed");
  }
  try {
    cudaError_t status = cudaSetDevice(device_);
    if (status != cudaSuccess) cuda_fail("device selection", status);
    status = cudaStreamCreateWithFlags(&init_stream_, cudaStreamNonBlocking);
    if (status != cudaSuccess) cuda_fail("stream creation", status);
    status = cudaEventCreateWithFlags(&ready_event_, cudaEventDisableTiming);
    if (status != cudaSuccess) cuda_fail("event creation", status);
    status = cudaMalloc(&storage_, target_k0_qsa_state_bytes());
    if (status != cudaSuccess) cuda_fail("allocation", status);
    binding_ = bind_target_k0_qsa_state_storage(
        storage_, target_k0_qsa_state_bytes(), rank_, max_rows_);
    status = cudaMemsetAsync(storage_, 0, target_k0_qsa_state_bytes(),
                             init_stream_);
    if (status != cudaSuccess) cuda_fail("initialization", status);
    status = cudaEventRecord(ready_event_, init_stream_);
    if (status != cudaSuccess) cuda_fail("ready event", status);
  } catch (...) {
    telemetry_->emit({rank_, max_rows_,
                      TargetK0OracleQsaStateOutcome::kCudaError, 0});
    const bool settled = cudaSetDevice(device_) == cudaSuccess &&
                         cudaDeviceSynchronize() == cudaSuccess;
    if (settled) {
      if (storage_) (void)cudaFree(storage_);
      if (ready_event_) (void)cudaEventDestroy(ready_event_);
      if (init_stream_) (void)cudaStreamDestroy(init_stream_);
    } else {
      telemetry_->emit({rank_, max_rows_,
                        TargetK0OracleQsaStateOutcome::kCleanupIncomplete,
                        target_k0_qsa_state_bytes()});
    }
    storage_ = nullptr;
    ready_event_ = nullptr;
    init_stream_ = nullptr;
    throw;
  }
  authenticated_ = true;
  telemetry_->emit({rank_, max_rows_, TargetK0OracleQsaStateOutcome::kOk,
                    target_k0_qsa_state_bytes()});
}

TargetK0OracleQsaStateOwner::~TargetK0OracleQsaStateOwner() {
  if (device_ >= 0) (void)cudaSetDevice(device_);
  const cudaError_t settled = storage_ ? cudaDeviceSynchronize() : cudaSuccess;
  if (settled != cudaSuccess) {
    if (telemetry_)
      telemetry_->emit({rank_, max_rows_,
                        TargetK0OracleQsaStateOutcome::kCleanupIncomplete,
                        target_k0_qsa_state_bytes()});
    return;
  }
  if (storage_) (void)cudaFree(storage_);
  if (ready_event_) (void)cudaEventDestroy(ready_event_);
  if (init_stream_) (void)cudaStreamDestroy(init_stream_);
}

const TargetQsaStateView& TargetK0OracleQsaStateOwner::view(int layer) const {
  if (!authenticated_ || !is_target_qsa_layer(layer))
    throw std::out_of_range("K0 QSA state layer changed");
  return binding_.views[static_cast<std::size_t>((layer - 3) / 4)];
}

void TargetK0OracleQsaStateOwner::wait(cudaStream_t stream) {
  if (!authenticated_ || waited_ || !stream)
    throw std::logic_error("K0 QSA state wait changed");
  const cudaError_t status = cudaStreamWaitEvent(stream, ready_event_, 0);
  if (status != cudaSuccess) cuda_fail("source wait", status);
  waited_ = true;
}

}  // namespace rocket::qwen38::attention

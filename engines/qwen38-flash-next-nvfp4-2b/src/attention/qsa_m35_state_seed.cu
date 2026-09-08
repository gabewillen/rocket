// SPDX-License-Identifier: Apache-2.0
#include "attention/qsa_m35_state_seed.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace rocket::qwen38::attention {
namespace {
constexpr int kMainWidth = 256;
constexpr int kRawWidth = 140;
constexpr int kCompressedWidth = 128;
constexpr int kRawRows = 4;
constexpr int kCompressedRows = 8;

__device__ float e4m3_to_float(std::uint8_t bits) {
  __nv_fp8_e4m3 value;
  reinterpret_cast<std::uint8_t&>(value) = bits;
  return static_cast<float>(value);
}

__global__ void seed_main_cache(const std::uint8_t* source, float scale,
                                __nv_bfloat16* target) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kTargetQsaSeedRows * kMainWidth) return;
  const int row = index / kMainWidth;
  const int column = index % kMainWidth;
  // Captured slots name the reference process's physical block. The owner
  // authenticates their aligned, contiguous relative identity on the host,
  // then rebases that one request into its caller-owned block zero.
  target[static_cast<std::size_t>(row) * kMainWidth + column] =
      __float2bfloat16(e4m3_to_float(source[index]) * scale);
}

__global__ void seed_raw_cache(const __nv_bfloat16* source,
                               __nv_bfloat16* target) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kRawRows * kRawWidth) return;
  const int row = index / kRawWidth;
  const int column = index % kRawWidth;
  target[static_cast<std::size_t>(row) * kRawWidth + column] =
      source[index];
}

__global__ void seed_compressed_cache(const __nv_bfloat16* source,
                                      __nv_bfloat16* target) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kCompressedRows * kCompressedWidth) return;
  const int row = index / kCompressedWidth;
  const int column = index % kCompressedWidth;
  target[static_cast<std::size_t>(row) * kCompressedWidth + column] =
      source[index];
}

__global__ void publish_row35_metadata(TargetQsaStateView state) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const int row = kTargetQsaSeedRows;
  const_cast<std::int64_t*>(state.positions)[0] = row;
  const_cast<std::int32_t*>(state.main_slot_mapping)[0] = row;
  const_cast<std::int32_t*>(state.main_block_table)[0] = 0;
  const_cast<std::int32_t*>(state.raw_slot_mapping)[0] = row % 8;
  const_cast<std::int32_t*>(state.raw_block_table)[0] = 0;
  const_cast<std::int32_t*>(state.compressed_slot_mapping)[0] = row / 4;
  const_cast<std::int32_t*>(state.compressed_block_table)[0] = 0;
  const_cast<std::int32_t*>(state.query_start_locations)[0] = 0;
  const_cast<std::int64_t*>(state.logical_positions)[0] = row;
  const_cast<std::int32_t*>(state.sequence_lengths)[0] = row + 1;
  const_cast<std::int32_t*>(state.token_to_request)[0] = 0;
  const_cast<std::int32_t*>(state.compression_work)[0] = 1;
}

void require_device_pointer(const void* pointer, int device) {
  cudaPointerAttributes attributes{};
  if (!pointer || cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess ||
      attributes.type != cudaMemoryTypeDevice || attributes.device != device)
    throw std::invalid_argument("QSA M35 seed device pointer changed");
}
}  // namespace

bool validate_target_qsa_m35_seed_slots(const std::int64_t* main_slots,
                                        const std::int64_t* raw_slots,
                                        const std::int64_t* compressed_slots)
    noexcept {
  if (!main_slots || !raw_slots || !compressed_slots) return false;
  // Physical block numbers are allocator-owned and may differ between the
  // reference and Rocket processes. Only aligned block bases plus the exact
  // within-block sequence are portable and authenticated by this contract.
  if (main_slots[0] < 0 || main_slots[0] % 1600 != 0 || raw_slots[0] < 0 ||
      raw_slots[0] % kRawRows != 0 || compressed_slots[0] < 0 ||
      compressed_slots[0] % 400 != 0)
    return false;
  for (int row = 0; row < kTargetQsaSeedRows; ++row)
    if (main_slots[row] != main_slots[0] + row) return false;
  for (int row = 0; row < kRawRows; ++row)
    if (raw_slots[row] != raw_slots[0] + row) return false;
  for (int row = 0; row < kCompressedRows; ++row)
    if (compressed_slots[row] != compressed_slots[0] + row) return false;
  return true;
}

TargetQsaM35StateSeedOwner::TargetQsaM35StateSeedOwner(
    const TargetQsaSeedConfig& config, const TargetQsaStateView& state)
    : device_(config.device), rank_(config.rank), layer_(config.layer),
      state_(state), publish_(config.publish),
      publish_context_(config.publish_context) {
  if (device_ != 0 || rank_ != 0 || layer_ != 3 ||
      config.rows != kTargetQsaSeedRows || !publish_ || state_.rank != rank_ ||
      state_.layer != layer_ || state_.rows != 1 || state_.uses_mrope ||
      state_.main_blocks != 1 || state_.compressed_blocks != 1 ||
      state_.main_kv_dtype != TargetQsaServingDtype::kBfloat16 ||
      state_.side_cache_dtype != TargetQsaServingDtype::kBfloat16) {
    if (publish_)
      publish_(publish_context_,
               {TargetQsaSeedOperation::kValidation,
                TargetQsaSeedOutcome::kIdentity,
                static_cast<std::int8_t>(rank_),
                static_cast<std::int8_t>(layer_),
                static_cast<std::uint8_t>(
                    config.rows == kTargetQsaSeedRows ? config.rows : 0),
                false});
    throw std::invalid_argument("QSA M35 seed owner identity changed");
  }
  emit(TargetQsaSeedOperation::kValidation, TargetQsaSeedOutcome::kSuccess,
       true);
}

void TargetQsaM35StateSeedOwner::emit(TargetQsaSeedOperation operation,
                                      TargetQsaSeedOutcome outcome,
                                      bool success) const noexcept {
  publish_(publish_context_,
           {operation, outcome, static_cast<std::int8_t>(rank_),
            static_cast<std::int8_t>(layer_), kTargetQsaSeedRows, success});
}

void TargetQsaM35StateSeedOwner::seed(const TargetQsaSeedBundle& bundle,
                                     cudaStream_t stream) {
  if (seeded_ || !stream || !std::isfinite(bundle.key_scale) ||
      !std::isfinite(bundle.value_scale) || bundle.key_scale <= 0.0F ||
      bundle.value_scale <= 0.0F) {
    emit(TargetQsaSeedOperation::kValidation, TargetQsaSeedOutcome::kIdentity,
         false);
    throw std::invalid_argument("QSA M35 seed request changed");
  }
  const void* pointers[] = {
      bundle.main_key_fp8, bundle.main_value_fp8, bundle.main_slots,
      bundle.raw_state, bundle.raw_state_slots, bundle.compressed_state,
      bundle.compressed_state_slots, state_.main_key_cache,
      state_.main_value_cache, state_.raw_key_cache,
      state_.compressed_key_cache, state_.positions, state_.main_slot_mapping,
      state_.main_block_table, state_.raw_slot_mapping,
      state_.raw_block_table, state_.compressed_slot_mapping,
      state_.compressed_block_table, state_.query_start_locations,
      state_.logical_positions, state_.sequence_lengths,
      state_.token_to_request, state_.compression_work};
  try {
    for (const void* pointer : pointers) require_device_pointer(pointer, device_);
  } catch (...) {
    emit(TargetQsaSeedOperation::kValidation, TargetQsaSeedOutcome::kPointer,
         false);
    throw;
  }
  std::int64_t main_slots[kTargetQsaSeedRows];
  std::int64_t raw_slots[kRawRows];
  std::int64_t compressed_slots[kCompressedRows];
  if (cudaMemcpyAsync(main_slots, bundle.main_slots, sizeof(main_slots),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(raw_slots, bundle.raw_state_slots, sizeof(raw_slots),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(compressed_slots, bundle.compressed_state_slots,
                      sizeof(compressed_slots), cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    emit(TargetQsaSeedOperation::kValidation, TargetQsaSeedOutcome::kCuda,
         false);
    throw std::runtime_error("QSA M35 seed slot publication failed");
  }
  if (!validate_target_qsa_m35_seed_slots(main_slots, raw_slots,
                                          compressed_slots)) {
    emit(TargetQsaSeedOperation::kValidation, TargetQsaSeedOutcome::kSlot,
         false);
    throw std::invalid_argument("QSA M35 seed slots changed");
  }
  constexpr int kThreads = 256;
  seed_main_cache<<<(kTargetQsaSeedRows * kMainWidth + kThreads - 1) / kThreads,
                    kThreads, 0, stream>>>(bundle.main_key_fp8,
                                           bundle.key_scale,
                                           state_.main_key_cache);
  seed_main_cache<<<(kTargetQsaSeedRows * kMainWidth + kThreads - 1) / kThreads,
                    kThreads, 0, stream>>>(bundle.main_value_fp8,
                                           bundle.value_scale,
                                           state_.main_value_cache);
  if (cudaPeekAtLastError() != cudaSuccess) {
    emit(TargetQsaSeedOperation::kMainCache, TargetQsaSeedOutcome::kCuda,
         false);
    throw std::runtime_error("QSA M35 main cache seed failed");
  }
  emit(TargetQsaSeedOperation::kMainCache, TargetQsaSeedOutcome::kSuccess,
       true);
  seed_raw_cache<<<(kRawRows * kRawWidth + kThreads - 1) / kThreads,
                   kThreads, 0, stream>>>(bundle.raw_state,
                                          state_.raw_key_cache);
  if (cudaPeekAtLastError() != cudaSuccess) {
    emit(TargetQsaSeedOperation::kRawCache, TargetQsaSeedOutcome::kCuda, false);
    throw std::runtime_error("QSA M35 raw cache seed failed");
  }
  emit(TargetQsaSeedOperation::kRawCache, TargetQsaSeedOutcome::kSuccess, true);
  seed_compressed_cache<<<
      (kCompressedRows * kCompressedWidth + kThreads - 1) / kThreads,
      kThreads, 0, stream>>>(bundle.compressed_state,
                             state_.compressed_key_cache);
  if (cudaPeekAtLastError() != cudaSuccess) {
    emit(TargetQsaSeedOperation::kCompressedCache,
         TargetQsaSeedOutcome::kCuda, false);
    throw std::runtime_error("QSA M35 compressed cache seed failed");
  }
  emit(TargetQsaSeedOperation::kCompressedCache,
       TargetQsaSeedOutcome::kSuccess, true);
  publish_row35_metadata<<<1, 1, 0, stream>>>(state_);
  if (cudaPeekAtLastError() != cudaSuccess) {
    emit(TargetQsaSeedOperation::kMetadata, TargetQsaSeedOutcome::kCuda, false);
    throw std::runtime_error("QSA M35 metadata seed failed");
  }
  seeded_ = true;
  emit(TargetQsaSeedOperation::kMetadata, TargetQsaSeedOutcome::kSuccess, true);
}

}  // namespace rocket::qwen38::attention

extern "C" int qwen38_qsa_m35_state_seed_create(
    const rocket::qwen38::attention::TargetQsaSeedConfig* config,
    const rocket::qwen38::attention::TargetQsaStateView* state,
    void** owner) noexcept {
  if (!config || !state || !owner || *owner) return 1;
  try {
    auto result = std::make_unique<
        rocket::qwen38::attention::TargetQsaM35StateSeedOwner>(*config, *state);
    *owner = result.release();
    return 0;
  } catch (...) {
    return 1;
  }
}

extern "C" int qwen38_qsa_m35_state_seed_launch(
    void* owner, const rocket::qwen38::attention::TargetQsaSeedBundle* bundle,
    cudaStream_t stream) noexcept {
  if (!owner || !bundle) return 1;
  try {
    static_cast<rocket::qwen38::attention::TargetQsaM35StateSeedOwner*>(owner)
        ->seed(*bundle, stream);
    return 0;
  } catch (...) {
    return 1;
  }
}

extern "C" int qwen38_qsa_m35_state_seed_destroy(void* owner) noexcept {
  delete static_cast<rocket::qwen38::attention::TargetQsaM35StateSeedOwner*>(
      owner);
  return 0;
}

// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "attention/qsa_target_state_view.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace rocket::qwen38::attention {

inline constexpr int kTargetQsaSeedRows = 35;

enum class TargetQsaSeedOperation : std::uint8_t {
  kValidation = 0,
  kMainCache = 1,
  kRawCache = 2,
  kCompressedCache = 3,
  kMetadata = 4,
};

enum class TargetQsaSeedOutcome : std::uint8_t {
  kSuccess = 0,
  kIdentity = 1,
  kPointer = 2,
  kSlot = 3,
  kCuda = 4,
  kUnknown = 5,
};

struct TargetQsaSeedRecord {
  TargetQsaSeedOperation operation;
  TargetQsaSeedOutcome outcome;
  std::int8_t rank;
  std::int8_t layer;
  std::uint8_t rows;
  bool success;
};
static_assert(sizeof(TargetQsaSeedRecord) == 6);
static_assert(std::is_standard_layout_v<TargetQsaSeedRecord>);

using TargetQsaSeedPublish = void (*)(void*, TargetQsaSeedRecord) noexcept;

struct TargetQsaSeedConfig {
  int device;
  int rank;
  int layer;
  int rows;
  TargetQsaSeedPublish publish;
  void* publish_context;
};
static_assert(sizeof(TargetQsaSeedConfig) == 32);
static_assert(sizeof(TargetQsaStateView) == 176);

struct TargetQsaSeedBundle {
  const std::uint8_t* main_key_fp8;
  const std::uint8_t* main_value_fp8;
  const std::int64_t* main_slots;
  float key_scale;
  float value_scale;
  const __nv_bfloat16* raw_state;
  const std::int64_t* raw_state_slots;
  const __nv_bfloat16* compressed_state;
  const std::int64_t* compressed_state_slots;
};
static_assert(sizeof(TargetQsaSeedBundle) == 64);

// Pure slot-identity check shared by the CUDA owner and CPU contract test.
bool validate_target_qsa_m35_seed_slots(const std::int64_t* main_slots,
                                        const std::int64_t* raw_slots,
                                        const std::int64_t* compressed_slots)
    noexcept;

class TargetQsaM35StateSeedOwner final {
 public:
  TargetQsaM35StateSeedOwner(const TargetQsaSeedConfig& config,
                             const TargetQsaStateView& state);
  void seed(const TargetQsaSeedBundle& bundle, cudaStream_t stream);

 private:
  void emit(TargetQsaSeedOperation operation, TargetQsaSeedOutcome outcome,
            bool success) const noexcept;
  int device_;
  int rank_;
  int layer_;
  TargetQsaStateView state_;
  TargetQsaSeedPublish publish_;
  void* publish_context_;
  bool seeded_ = false;
};

}  // namespace rocket::qwen38::attention

extern "C" {
int qwen38_qsa_m35_state_seed_create(
    const rocket::qwen38::attention::TargetQsaSeedConfig* config,
    const rocket::qwen38::attention::TargetQsaStateView* state,
    void** owner) noexcept;
int qwen38_qsa_m35_state_seed_launch(
    void* owner, const rocket::qwen38::attention::TargetQsaSeedBundle* bundle,
    cudaStream_t stream) noexcept;
int qwen38_qsa_m35_state_seed_destroy(void* owner) noexcept;
}

// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace rocket::qwen38::linear_attention::prefill {

inline constexpr int kNativePrefillRows = 35;
inline constexpr int kNativePrefillKeyHeads = 8;
inline constexpr int kNativePrefillValueHeads = 24;
inline constexpr int kNativePrefillHeadDim = 128;

enum class NativePrefillPhase : std::uint8_t {
  kValidation = 0,
  kStatePublication = 1,
  kLaunch = 2,
};

enum class NativePrefillStatus : std::uint8_t {
  kSuccess = 0,
  kIdentity = 1,
  kPointer = 2,
  kAliasing = 3,
  kDevice = 4,
  kCuda = 5,
  kUnknown = 6,
};

struct NativePrefillRecord {
  NativePrefillPhase phase;
  NativePrefillStatus status;
  std::int8_t rank;
  std::int8_t layer;
  std::uint8_t rows;
  bool success;
};
static_assert(sizeof(NativePrefillRecord) == 6);
static_assert(std::is_standard_layout_v<NativePrefillRecord>);

using NativePrefillPublish = void (*)(void*, NativePrefillRecord) noexcept;

struct NativePrefillConfig {
  int device;
  int rank;
  int layer;
  int rows;
  NativePrefillPublish publish;
  void* publish_context;
};
static_assert(std::is_standard_layout_v<NativePrefillConfig>);
static_assert(sizeof(NativePrefillPublish) == sizeof(void*));

struct NativePrefillTensors {
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  const float* log_decay;
  const float* beta;
  const float* initial_state;
  __nv_bfloat16* output;
  float* final_state;
};

class NativeGdnM35PrefillOwner final {
 public:
  // The config and callback are borrowed for construction; the callback and
  // context must remain valid through the final launch. Tensor storage is
  // caller-owned. Launch is single-owner, non-reentrant, and ordered only by
  // the supplied CUDA stream. Exact initial/final-state aliasing is supported;
  // any partial state overlap or output/state overlap is rejected.
  explicit NativeGdnM35PrefillOwner(const NativePrefillConfig& config);
  ~NativeGdnM35PrefillOwner() = default;
  NativeGdnM35PrefillOwner(const NativeGdnM35PrefillOwner&) = delete;
  NativeGdnM35PrefillOwner& operator=(const NativeGdnM35PrefillOwner&) = delete;

  void launch(const NativePrefillTensors& tensors, cudaStream_t stream);

 private:
  void emit(NativePrefillPhase phase, NativePrefillStatus status,
            bool success) const noexcept;
  int device_;
  NativePrefillPublish publish_;
  void* publish_context_;
};

}  // namespace rocket::qwen38::linear_attention::prefill

extern "C" {
int qwen38_gdn_m35_prefill_create(
    const rocket::qwen38::linear_attention::prefill::NativePrefillConfig* config,
    void** owner) noexcept;
int qwen38_gdn_m35_prefill_launch(
    void* owner,
    const rocket::qwen38::linear_attention::prefill::NativePrefillTensors* tensors,
    cudaStream_t stream) noexcept;
int qwen38_gdn_m35_prefill_destroy(void* owner) noexcept;
}

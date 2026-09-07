// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string_view>

namespace rocket::qwen38::norm {

inline constexpr int kHiddenFull = 5'120;
inline constexpr int kHiddenTp = 2'560;
inline constexpr float kEpsilon = 1.0e-6F;
inline constexpr int kBuckets[] = {1, 2, 4, 8, 16};

enum class Operation : std::uint8_t { kResidualAdd, kRmsNorm, kFusedAddRmsNorm };
enum class Outcome : std::uint8_t { kOk, kContractError, kCudaError };

struct OtelDimensions {
  std::string_view operation;
  int hidden;
  int m_bucket;
  std::string_view dtype;
  std::string_view outcome;
};

// Valid-path cardinality is 3 operations x 2 widths x 5 M buckets x
// 3 outcomes x 1 dtype = 90. Including collapsed invalid integer dimensions,
// the absolute bound is 3 x 3 x 6 x 3 x 1 = 162.

constexpr bool allowed_hidden(int hidden) noexcept {
  return hidden == kHiddenFull || hidden == kHiddenTp;
}
constexpr bool allowed_m(int m) noexcept {
  for (const int bucket : kBuckets) if (m == bucket) return true;
  return false;
}
constexpr std::string_view operation_name(Operation operation) noexcept {
  switch (operation) {
    case Operation::kResidualAdd: return "residual_add";
    case Operation::kRmsNorm: return "rms_norm";
    case Operation::kFusedAddRmsNorm: return "fused_add_rms_norm";
  }
  return "rms_norm";
}
constexpr std::string_view outcome_name(Outcome outcome) noexcept {
  switch (outcome) {
    case Outcome::kOk: return "ok";
    case Outcome::kContractError: return "contract_error";
    case Outcome::kCudaError: return "cuda_error";
  }
  return "contract_error";
}
constexpr OtelDimensions otel_dimensions(Operation operation, int hidden, int m,
                                          Outcome outcome) noexcept {
  return {operation_name(operation), allowed_hidden(hidden) ? hidden : 0,
          allowed_m(m) ? m : 0, "bf16_fp32", outcome_name(outcome)};
}

// All pointers are borrowed device buffers, 16-byte aligned, and valid through
// completion on stream. Buffers contain m*hidden BF16 elements except weight,
// which contains hidden. Launches are asynchronous, graph-capturable, and
// mutate only their documented output buffers. Inputs and outputs must not
// overlap. Unsupported shapes and invalid pointers fail before launch.
[[nodiscard]] cudaError_t residual_add(const __nv_bfloat16* input, const __nv_bfloat16* residual,
                         __nv_bfloat16* output, int m, int hidden,
                         cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] cudaError_t rms_norm(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                     __nv_bfloat16* output, int m, int hidden,
                     cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] cudaError_t fused_add_rms_norm(const __nv_bfloat16* input,
                               const __nv_bfloat16* residual,
                               const __nv_bfloat16* weight,
                               __nv_bfloat16* residual_output,
                               __nv_bfloat16* norm_output, int m, int hidden,
                               cudaStream_t stream = nullptr) noexcept;

}  // namespace rocket::qwen38::norm

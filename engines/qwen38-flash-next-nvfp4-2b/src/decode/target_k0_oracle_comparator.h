// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <memory>

#include "decode/target_k0_executor.h"
#include "decode/target_layer3_oracle_comparator.h"

namespace rocket::qwen38::decode {

struct TargetK0OracleEvidence {
  TargetK0Boundary boundary = TargetK0Boundary::kEmbedding;
  int row = -1;
  int layer = -1;
  std::uint32_t max_ulp = 0;
  std::size_t mismatch_count = 0;
  bool accepted = false;
};

// Validation-only comparator for the exact 51-artifact c1 K0 oracle. All
// expected files are authenticated and retained in ordinary host memory at
// construction. compare() owns one bounded pinned D2H buffer and fences each
// named boundary. It is not part of the serving graph or production registry.
class NativeTargetK0OracleComparator final : public TargetK0OracleComparator,
                                             public TargetK0LayerBoundaryObserver {
 public:
  NativeTargetK0OracleComparator(
      int rank, const std::filesystem::path& capture,
      pair_reduce::OtelStageSink& telemetry,
      TargetLayer3OracleCudaApi* cuda_api = nullptr);
  ~NativeTargetK0OracleComparator() override;
  NativeTargetK0OracleComparator(const NativeTargetK0OracleComparator&) = delete;

  int rank() const noexcept override;
  int rows() const noexcept override;
  std::string_view manifest_sha256() const noexcept override;
  bool authenticated() const noexcept override;
  std::int32_t expected_input_token(int row) const override;
  void compare(TargetK0Boundary boundary, int row, int layer,
               const void* device_values, std::size_t elements,
               cudaStream_t stream) override;
  void compare_token(std::int32_t token) override;
  void observe(TargetK0LayerBoundary boundary, const void* device_values,
               std::size_t elements, TargetK0DiagnosticDtype dtype,
               cudaStream_t stream,
               TargetK0LayerBoundaryEvidence& evidence) override;
  const TargetK0OracleEvidence& evidence() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rocket::qwen38::decode

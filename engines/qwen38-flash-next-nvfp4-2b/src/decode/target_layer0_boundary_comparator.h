// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace rocket::qwen38::decode {

enum class TargetLayer0BoundaryStage : std::uint8_t {
  kAttentionOutput,
  kHyperconnectionCombineMix,
  kMoeOutput,
};

struct TargetLayer0BoundaryReference {
  TargetLayer0BoundaryStage stage;
  std::vector<std::uint8_t> bytes;
  std::string sha256;
};

struct TargetLayer0BoundaryEvidence {
  TargetLayer0BoundaryStage stage;
  std::string observed_sha256;
  std::size_t mismatch_count = 0;
  std::size_t first_mismatch = 0;
  bool exact = false;
};

std::array<TargetLayer0BoundaryReference, 3>
authenticate_target_layer0_boundary_references(
    const std::filesystem::path& directory);

TargetLayer0BoundaryEvidence compare_target_layer0_boundary_bytes(
    const TargetLayer0BoundaryReference& expected,
    const std::uint8_t* observed, std::size_t bytes);

// Debug-only synchronous D2H comparator. It is not graph captured and does
// not participate in serving publication. Calls must arrive once in attention,
// HC, MoE order for the first layer-0 row.
class TargetLayer0BoundaryComparator final {
 public:
  explicit TargetLayer0BoundaryComparator(
      const std::filesystem::path& directory);
  ~TargetLayer0BoundaryComparator();
  TargetLayer0BoundaryComparator(const TargetLayer0BoundaryComparator&) = delete;
  TargetLayer0BoundaryComparator& operator=(
      const TargetLayer0BoundaryComparator&) = delete;

  bool authenticated() const noexcept { return authenticated_; }
  TargetLayer0BoundaryEvidence compare(
      TargetLayer0BoundaryStage stage, const __nv_bfloat16* device_values,
      cudaStream_t stream);

 private:
  std::array<TargetLayer0BoundaryReference, 3> references_;
  void* pinned_ = nullptr;
  cudaEvent_t ready_ = nullptr;
  std::size_t next_stage_ = 0;
  bool authenticated_ = false;
};

}  // namespace rocket::qwen38::decode

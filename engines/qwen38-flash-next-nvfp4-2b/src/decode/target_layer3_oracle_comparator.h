// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_layer3_prefill.h"

#include <array>
#include <filesystem>
#include <string>

namespace rocket::qwen38::decode {

inline constexpr std::size_t kLayer3OracleWidth = 10'240;
inline constexpr std::size_t kLayer3OracleRowBytes = 20'480;
inline constexpr std::size_t kLayer3OracleRow34Offset = 696'320;
inline constexpr std::string_view kLayer3OracleManifestSha256 =
    "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b";
inline constexpr std::string_view kLayer3OracleFileSha256 =
    "aa2d2a1454f0654ea2082d12e7e3284cad9304a7ede124303f99c52bbf9cddbe";
inline constexpr std::string_view kLayer3OracleRow34Sha256 =
    "914665e731769389dc0f9312a649caf17eaddc90813ff07f63b40ee71353c3df";

struct TargetLayer3OracleEvidence {
  float max_abs = 0;
  double rms = 0;
  std::uint32_t max_ulp = 0;
  std::uint32_t token_max_ulp = 0;
  std::size_t mismatch_count = 0;
  std::size_t nan_count = 0;
  std::size_t max_ulp_index = 0;
  std::array<std::uint32_t, 4> hidden_stream_max_ulp{};
  std::string observed_sha256;
  bool expected_hash_authenticated = false;
  bool observed_exact_hash = false;
  bool accepted = false;
};

std::array<std::uint16_t, kLayer3OracleWidth>
authenticate_layer3_oracle_row34(const std::filesystem::path& capture);
TargetLayer3OracleEvidence compare_layer3_oracle_row34(
    const std::uint16_t* expected, const std::uint16_t* observed);

class TargetLayer3OracleCudaApi {
 public:
  virtual ~TargetLayer3OracleCudaApi() = default;
  virtual cudaError_t host_alloc(void** pointer, std::size_t bytes) noexcept = 0;
  virtual cudaError_t free_host(void* pointer) noexcept = 0;
  virtual cudaError_t event_create(cudaEvent_t* event) noexcept = 0;
  virtual cudaError_t event_destroy(cudaEvent_t event) noexcept = 0;
  virtual cudaError_t copy_d2h(void* destination, const void* source,
                               std::size_t bytes, cudaStream_t stream) noexcept = 0;
  virtual cudaError_t event_record(cudaEvent_t event,
                                   cudaStream_t stream) noexcept = 0;
  virtual cudaError_t event_sync(cudaEvent_t event) noexcept = 0;
};

// Validation-only host comparator. It is never graph-captured or registered as
// a serving participant and owns one fixed 40,960-byte pinned allocation.
class NativeTargetLayer3OracleComparator final : public TargetLayer3Comparator {
 public:
  NativeTargetLayer3OracleComparator(
      int rank, const std::filesystem::path& capture,
      pair_reduce::OtelStageSink& telemetry,
      TargetLayer3OracleCudaApi* cuda_api = nullptr);
  ~NativeTargetLayer3OracleComparator() override;
  NativeTargetLayer3OracleComparator(
      const NativeTargetLayer3OracleComparator&) = delete;
  bool authenticated() const noexcept override { return authenticated_; }
  bool compare_row34(const __nv_bfloat16* replicated_post_layer,
                     cudaStream_t stream) override;
  const TargetLayer3OracleEvidence& evidence() const noexcept { return evidence_; }

 private:
  void emit(pair_reduce::Outcome outcome, std::uint64_t bytes) noexcept;
  int rank_ = -1;
  pair_reduce::OtelStageSink* telemetry_ = nullptr;
  TargetLayer3OracleCudaApi* cuda_api_ = nullptr;
  void* pinned_ = nullptr;
  std::uint16_t* expected_ = nullptr;
  std::uint16_t* observed_ = nullptr;
  cudaEvent_t ready_ = nullptr;
  bool authenticated_ = false;
  bool compared_ = false;
  TargetLayer3OracleEvidence evidence_{};
};

}  // namespace rocket::qwen38::decode

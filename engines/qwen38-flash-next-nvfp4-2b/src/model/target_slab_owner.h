// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>

namespace rocket::qwen38::model {

inline constexpr std::string_view kTargetSlabArtifactKey =
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4";
inline constexpr std::string_view kTargetSlabManifestSha256 =
    "a44a450d9c0b6fe3df904ad1a78ecee959f28d9f055301195181e986bdc7028b";
inline constexpr std::size_t kTargetSlabBytes = 63'212'748'800ULL;
inline constexpr std::size_t kTargetSlabChunks = 236;
inline constexpr std::size_t kTargetSlabPageBytes = 65'536;
inline constexpr std::size_t kTargetSlabChunkBytes = 268'435'456;
inline constexpr std::size_t kTargetSlabRingDepth = 4;
inline constexpr std::size_t kTargetSlabPeakPinnedBytes =
    kTargetSlabRingDepth * (kTargetSlabChunkBytes + kTargetSlabPageBytes - 1);

struct TargetSlabChunk {
  std::uint64_t offset;
  std::uint64_t bytes;
  std::string_view sha256;
};

enum class TargetSlabLoadPhase : std::uint8_t {
  kValidate,
  kManifest,
  kOpen,
  kAllocate,
  kRead,
  kDigest,
  kTransfer,
  kPublish,
  kCleanup,
};

enum class TargetSlabFailureClass : std::uint8_t {
  kNone,
  kContract,
  kIo,
  kCrypto,
  kCuda,
};

struct TargetSlabTelemetryRecord {
  TargetSlabLoadPhase phase;
  TargetSlabFailureClass failure;
  int rank;
  std::uint8_t chunk_bucket;
  bool success;
};

// Host-observed stage durations. Read, digest, enqueue, and fence durations may
// overlap, so their sum is not an alternative wall-clock measurement.
struct TargetSlabStageTimings {
  std::uint64_t open_ns;
  std::uint64_t allocate_ns;
  std::uint64_t direct_read_ns;
  std::uint64_t digest_ns;
  std::uint64_t h2d_enqueue_ns;
  std::uint64_t h2d_fence_ns;
  std::uint64_t transient_cleanup_ns;
};

// OpenTelemetry adapter boundary owned by the engine embedder. Records contain
// only fixed enums, rank {-1,0,1}, chunk bucket {0..7}, and success boolean.
// Implementations synchronously export/copy the record and must not throw.
class TargetSlabTelemetrySink {
 public:
  virtual ~TargetSlabTelemetrySink() = default;
  virtual void emit(const TargetSlabTelemetryRecord& record) noexcept = 0;
};

struct TargetSlabPublication {
  const std::uint8_t* device_base;
  // Owner-borrowed, already-complete readiness event. Consumers may query it;
  // TargetSlabDeviceOwner retains and destroys it.
  cudaEvent_t ready_event;
  std::size_t bytes;
  int device;
  int rank;
  std::string_view artifact_key;
  std::string_view slab_key;
  std::string_view manifest_sha256;
  std::string_view layout_sha256;
  std::uint64_t open_to_publish_ns;
  TargetSlabStageTimings stage_timings;
  std::size_t chunks_authenticated;
  std::size_t peak_host_pinned_bytes;
};

struct TargetSlabMetadata {
  int rank;
  std::filesystem::path payload;
  std::string_view slab_key;
  std::string_view layout_sha256;
  const std::array<TargetSlabChunk, kTargetSlabChunks>* chunks;
};

// Validates the exact content-addressed manifest, rank, payload name, size, and
// 236-entry generated extent table without reading the 63 GB payload.
TargetSlabMetadata authenticate_target_slab_metadata(
    const std::filesystem::path& artifact, int rank);

// Single-owner target-only loader. load() returns an owner only after every
// chunk and the final private-stream publication event have completed. The
// telemetry sink is borrowed for the owner lifetime so cleanup failures remain
// observable without throwing from the destructor.
class TargetSlabDeviceOwner final {
 public:
  static std::unique_ptr<TargetSlabDeviceOwner> load(
      int device, int rank, const std::filesystem::path& artifact,
      TargetSlabTelemetrySink& telemetry);
  ~TargetSlabDeviceOwner();
  TargetSlabDeviceOwner(const TargetSlabDeviceOwner&) = delete;
  TargetSlabDeviceOwner& operator=(const TargetSlabDeviceOwner&) = delete;

  const TargetSlabPublication& publication() const noexcept {
    return publication_;
  }

 private:
  TargetSlabDeviceOwner() = default;
  void release() noexcept;

  int device_ = -1;
  int rank_ = -1;
  TargetSlabTelemetrySink* telemetry_ = nullptr;
  std::uint8_t* allocation_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t publication_event_ = nullptr;
  std::array<void*, kTargetSlabRingDepth> staging_allocations_{};
  std::array<std::uint8_t*, kTargetSlabRingDepth> staging_{};
  std::array<cudaEvent_t, kTargetSlabRingDepth> slot_events_{};
  TargetSlabPublication publication_{};
};

}  // namespace rocket::qwen38::model

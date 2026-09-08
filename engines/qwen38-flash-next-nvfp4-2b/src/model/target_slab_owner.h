// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
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
  std::size_t chunks_authenticated;
  std::size_t peak_host_pinned_bytes;
};

// Owning lifetime boundary for every publication borrower. Implementations
// retain the allocation and ready event until the last shared lease releases.
class TargetSlabLease {
 public:
  enum class Lifetime : std::uint8_t { kOwnerScoped, kProcessLifetime };
  virtual ~TargetSlabLease() = default;
  virtual const TargetSlabPublication& publication() const noexcept = 0;
  virtual Lifetime lifetime() const noexcept = 0;
};

struct TargetSlabChunkReceipt {
  std::uint64_t index;
  std::uint64_t bytes;
  std::uint64_t direct_read_ns;
  std::uint64_t sha256_ns;
  std::uint64_t h2d_fence_ns;
};

struct TargetSlabCudaProbeResult {
  std::uintptr_t allocation_base;
  std::size_t allocation_bytes;
  int device;
  bool device_memory;
  bool ready_event_complete;
};

enum class TargetSlabPublicationValidation : std::uint8_t {
  kAccepted,
  kReceiptHeader,
  kPublicationPointerEvent,
  kPublicationBytes,
  kPublicationRankDevice,
  kSlabKey,
  kArtifactManifest,
  kLayoutIdentity,
  kChunksAuthenticated,
  kPeakPinnedBytes,
  kOpenDuration,
  kProbeMemoryType,
  kProbeDevice,
  kAllocationBase,
  kAllocationExtent,
  kReceiptChunk,
  kReceiptBytes,
};

TargetSlabPublicationValidation diagnose_accepted_loader_publication(
    const TargetSlabPublication& publication,
    const TargetSlabCudaProbeResult& probe, std::uint64_t receipt_started_ns,
    std::uint64_t receipt_completed_ns,
    const TargetSlabChunkReceipt* chunk_receipts,
    std::size_t chunk_receipt_count) noexcept;

// Allocation-free validation seam. Production supplies values obtained from
// CUDA pointer/address-range/event probes. CPU tests inject the same typed
// result without gaining a capability-minting API.
bool validate_accepted_loader_publication(
    const TargetSlabPublication& publication,
    const TargetSlabCudaProbeResult& probe, std::uint64_t receipt_started_ns,
    std::uint64_t receipt_completed_ns,
    const TargetSlabChunkReceipt* chunk_receipts,
    std::size_t chunk_receipt_count) noexcept;

extern "C" int qwen38_target_slab_retain_accepted_loader(
    std::uintptr_t device_base, std::uintptr_t ready_event,
    std::size_t bytes, int device, int rank, const char* slab_key,
    const char* layout_sha256, const char* receipt_sha256,
    std::uint64_t open_to_publish_ns, std::size_t chunks_authenticated,
    std::size_t peak_host_pinned_bytes, std::uint64_t receipt_started_ns,
    std::uint64_t receipt_completed_ns,
    const TargetSlabChunkReceipt* chunk_receipts,
    std::size_t chunk_receipt_count, void** lease_handle) noexcept;

// Token for the accepted Python CudaRankSlabLoader handoff. The startup
// control plane pins the corresponding LoadedRankSlabs object in an
// append-only process-lifetime registry before constructing this token. This
// object copies all publication identities and never releases device memory.
class ProcessLifetimeTargetSlabLease final : public TargetSlabLease {
 public:
  const TargetSlabPublication& publication() const noexcept override {
    return publication_;
  }
  Lifetime lifetime() const noexcept override {
    return Lifetime::kProcessLifetime;
  }

 private:
  friend int qwen38_target_slab_retain_accepted_loader(
      std::uintptr_t, std::uintptr_t, std::size_t, int, int, const char*,
      const char*, const char*, std::uint64_t, std::size_t, std::size_t,
      std::uint64_t, std::uint64_t, const TargetSlabChunkReceipt*, std::size_t,
      void**) noexcept;
  ProcessLifetimeTargetSlabLease(TargetSlabPublication publication,
                                 std::string receipt_sha256);
  std::string artifact_key_;
  std::string slab_key_;
  std::string manifest_sha256_;
  std::string layout_sha256_;
  std::string receipt_sha256_;
  TargetSlabPublication publication_{};
};

class TargetSlabStartupFactory final {
 public:
  static std::shared_ptr<const TargetSlabLease> lease_from_handle(
      void* lease_handle) noexcept;
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
class TargetSlabDeviceOwner final : public TargetSlabLease {
 public:
  static std::unique_ptr<TargetSlabDeviceOwner> load(
      int device, int rank, const std::filesystem::path& artifact,
      TargetSlabTelemetrySink& telemetry);
  ~TargetSlabDeviceOwner();
  TargetSlabDeviceOwner(const TargetSlabDeviceOwner&) = delete;
  TargetSlabDeviceOwner& operator=(const TargetSlabDeviceOwner&) = delete;

  const TargetSlabPublication& publication() const noexcept override {
    return publication_;
  }
  Lifetime lifetime() const noexcept override { return Lifetime::kOwnerScoped; }

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

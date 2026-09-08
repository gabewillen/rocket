// SPDX-License-Identifier: Apache-2.0
#include "model/target_slab_owner.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

extern "C" {
struct evp_md_ctx_st;
struct evp_md_st;
evp_md_ctx_st* EVP_MD_CTX_new();
void EVP_MD_CTX_free(evp_md_ctx_st*);
const evp_md_st* EVP_sha256();
int EVP_DigestInit_ex(evp_md_ctx_st*, const evp_md_st*, void*);
int EVP_DigestUpdate(evp_md_ctx_st*, const void*, std::size_t);
int EVP_DigestFinal_ex(evp_md_ctx_st*, unsigned char*, unsigned int*);
unsigned long OpenSSL_version_num();
}

namespace rocket::qwen38::model {
namespace {

using Clock = std::chrono::steady_clock;

#include "model/target_slab_contract.inc"

class IoError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
class CryptoError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
class CudaError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void cuda_require(cudaError_t status, const char* message) {
  if (status != cudaSuccess) throw CudaError(message);
}

std::array<std::uint8_t, 32> parse_digest(std::string_view text) {
  if (text.size() != 64) throw std::logic_error("target slab digest length changed");
  std::array<std::uint8_t, 32> result{};
  const auto nibble = [](char value) -> std::uint8_t {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    throw std::logic_error("target slab digest alphabet changed");
  };
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<std::uint8_t>(
        nibble(text[2 * index]) * 16 + nibble(text[2 * index + 1]));
  }
  return result;
}

class Sha256 final {
 public:
  Sha256() : context_(EVP_MD_CTX_new()) {
    if (!context_ || EVP_DigestInit_ex(context_, EVP_sha256(), nullptr) != 1)
      throw CryptoError("target slab SHA256 initialization failed");
  }
  ~Sha256() { EVP_MD_CTX_free(context_); }
  void update(const void* data, std::size_t bytes) {
    if (EVP_DigestUpdate(context_, data, bytes) != 1)
      throw CryptoError("target slab SHA256 update failed");
  }
  std::array<std::uint8_t, 32> finish() {
    std::array<std::uint8_t, 32> result{};
    unsigned int bytes = 0;
    if (EVP_DigestFinal_ex(context_, result.data(), &bytes) != 1 ||
        bytes != result.size())
      throw CryptoError("target slab SHA256 finalization failed");
    return result;
  }

 private:
  evp_md_ctx_st* context_;
};

std::array<std::uint8_t, 32> hash_fd(int fd, std::uint64_t offset,
                                    std::uint64_t bytes,
                                    std::uint8_t* direct_buffer = nullptr) {
  Sha256 digest;
  std::vector<std::uint8_t> ordinary;
  if (!direct_buffer) ordinary.resize(1 << 20);
  std::uint64_t consumed = 0;
  while (consumed != bytes) {
    const std::size_t requested = static_cast<std::size_t>(std::min<std::uint64_t>(
        direct_buffer ? bytes : ordinary.size(), bytes - consumed));
    auto* destination = direct_buffer ? direct_buffer + consumed : ordinary.data();
    const ssize_t count = pread(fd, destination, requested, offset + consumed);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0 || static_cast<std::size_t>(count) != requested)
      throw IoError("target slab read was short");
    digest.update(destination, requested);
    consumed += requested;
  }
  return digest.finish();
}

void validate_chunks(const std::array<TargetSlabChunk, kTargetSlabChunks>& chunks) {
  std::uint64_t offset = 0;
  for (const auto& chunk : chunks) {
    if (chunk.offset != offset || chunk.bytes == 0 ||
        chunk.bytes > kTargetSlabChunkBytes ||
        chunk.offset % kTargetSlabPageBytes != 0 ||
        chunk.bytes % kTargetSlabPageBytes != 0)
      throw std::logic_error("target slab chunk extent changed");
    (void)parse_digest(chunk.sha256);
    offset += chunk.bytes;
  }
  if (offset != kTargetSlabBytes)
    throw std::logic_error("target slab chunk coverage changed");
}

TargetSlabFailureClass failure_class(const std::exception& error) noexcept {
  if (dynamic_cast<const IoError*>(&error)) return TargetSlabFailureClass::kIo;
  if (dynamic_cast<const CryptoError*>(&error)) return TargetSlabFailureClass::kCrypto;
  if (dynamic_cast<const CudaError*>(&error)) return TargetSlabFailureClass::kCuda;
  return TargetSlabFailureClass::kContract;
}

void emit(TargetSlabTelemetrySink& sink, TargetSlabLoadPhase phase,
          TargetSlabFailureClass failure, int rank, std::size_t chunk,
          bool success) noexcept {
  sink.emit({phase, failure, (rank == 0 || rank == 1) ? rank : -1,
             static_cast<std::uint8_t>(std::min<std::size_t>(chunk / 32, 7)),
             success});
}

struct DirectoryFd {
  int value = -1;
  ~DirectoryFd() { if (value >= 0) close(value); }
};

}  // namespace

TargetSlabMetadata authenticate_target_slab_metadata(
    const std::filesystem::path& artifact, int rank) {
  if ((rank != 0 && rank != 1) || artifact.filename() != kTargetSlabArtifactKey)
    throw std::invalid_argument("target slab artifact or rank changed");
  if ((OpenSSL_version_num() >> 28) != 3)
    throw CryptoError("target slab requires OpenSSL 3 ABI");
  const auto& chunks = rank == 0 ? kRank0Chunks : kRank1Chunks;
  validate_chunks(chunks);

  DirectoryFd directory{open(artifact.c_str(), O_RDONLY | O_DIRECTORY |
                                                   O_CLOEXEC | O_NOFOLLOW)};
  if (directory.value < 0) throw IoError("target slab artifact open failed");
  DirectoryFd manifest{openat(directory.value, "manifest.json",
                             O_RDONLY | O_CLOEXEC | O_NOFOLLOW)};
  if (manifest.value < 0) throw IoError("target slab manifest open failed");
  struct stat manifest_stat{};
  if (fstat(manifest.value, &manifest_stat) != 0 || !S_ISREG(manifest_stat.st_mode) ||
      manifest_stat.st_size <= 0 || manifest_stat.st_size > 512 * 1024 * 1024)
    throw IoError("target slab manifest extent changed");
  if (hash_fd(manifest.value, 0, static_cast<std::uint64_t>(manifest_stat.st_size)) !=
      parse_digest(kTargetSlabManifestSha256))
    throw std::invalid_argument("target slab manifest hash changed");

  const std::string slab_key = rank == 0 ? "rank0-target" : "rank1-target";
  const std::string file_name = slab_key + ".slab";
  DirectoryFd payload{openat(directory.value, file_name.c_str(),
                            O_RDONLY | O_CLOEXEC | O_NOFOLLOW)};
  if (payload.value < 0) throw IoError("target slab payload open failed");
  struct stat payload_stat{};
  if (fstat(payload.value, &payload_stat) != 0 || !S_ISREG(payload_stat.st_mode) ||
      static_cast<std::uint64_t>(payload_stat.st_size) != kTargetSlabBytes)
    throw IoError("target slab payload extent changed");
  return {rank, artifact / file_name,
          rank == 0 ? std::string_view("rank0-target")
                    : std::string_view("rank1-target"),
          rank == 0 ? kRank0LayoutSha256 : kRank1LayoutSha256, &chunks};
}

std::unique_ptr<TargetSlabDeviceOwner> TargetSlabDeviceOwner::load(
    int device, int rank, const std::filesystem::path& artifact,
    TargetSlabTelemetrySink& telemetry) {
  TargetSlabLoadPhase phase = TargetSlabLoadPhase::kValidate;
  std::size_t chunk_index = 0;
  auto owner = std::unique_ptr<TargetSlabDeviceOwner>(new TargetSlabDeviceOwner);
  owner->device_ = device;
  owner->rank_ = rank;
  owner->telemetry_ = &telemetry;
  try {
    if (device < 0) throw std::invalid_argument("target slab device changed");
    phase = TargetSlabLoadPhase::kManifest;
    const auto metadata = authenticate_target_slab_metadata(artifact, rank);
    phase = TargetSlabLoadPhase::kOpen;
    const auto opened = Clock::now();
    const int fd = open(metadata.payload.c_str(), O_RDONLY | O_DIRECT |
                                                     O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) throw IoError("target slab O_DIRECT open failed");
    struct FdOwner { int fd; ~FdOwner() { if (fd >= 0) close(fd); } } payload{fd};
    struct stat payload_stat{};
    if (fstat(payload.fd, &payload_stat) != 0 ||
        !S_ISREG(payload_stat.st_mode) ||
        static_cast<std::uint64_t>(payload_stat.st_size) != kTargetSlabBytes)
      throw IoError("target slab O_DIRECT payload extent changed");
    phase = TargetSlabLoadPhase::kAllocate;
    cuda_require(cudaSetDevice(device), "target slab device selection failed");
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&owner->allocation_),
                            kTargetSlabBytes),
                 "target slab CUDA allocation failed");
    cuda_require(cudaStreamCreateWithFlags(&owner->stream_, cudaStreamNonBlocking),
                 "target slab private stream creation failed");
    cuda_require(cudaEventCreateWithFlags(&owner->publication_event_,
                                          cudaEventDisableTiming),
                 "target slab publication event creation failed");
    for (std::size_t slot = 0; slot < kTargetSlabRingDepth; ++slot) {
      cuda_require(cudaHostAlloc(&owner->staging_allocations_[slot],
                                 kTargetSlabChunkBytes + kTargetSlabPageBytes - 1,
                                 cudaHostAllocDefault),
                   "target slab pinned staging allocation failed");
      const auto base = reinterpret_cast<std::uintptr_t>(
          owner->staging_allocations_[slot]);
      owner->staging_[slot] = reinterpret_cast<std::uint8_t*>(
          (base + kTargetSlabPageBytes - 1) & ~(kTargetSlabPageBytes - 1));
      cuda_require(cudaEventCreateWithFlags(&owner->slot_events_[slot],
                                            cudaEventDisableTiming),
                   "target slab slot event creation failed");
    }

    std::array<bool, kTargetSlabRingDepth> pending{};
    for (chunk_index = 0; chunk_index < metadata.chunks->size(); ++chunk_index) {
      const auto& chunk = (*metadata.chunks)[chunk_index];
      const std::size_t slot = chunk_index % kTargetSlabRingDepth;
      if (pending[slot])
        cuda_require(cudaEventSynchronize(owner->slot_events_[slot]),
                     "target slab slot reuse fence failed");
      phase = TargetSlabLoadPhase::kRead;
      const auto observed = hash_fd(payload.fd, chunk.offset, chunk.bytes,
                                    owner->staging_[slot]);
      phase = TargetSlabLoadPhase::kDigest;
      if (observed != parse_digest(chunk.sha256))
        throw std::invalid_argument("target slab chunk hash changed");
      phase = TargetSlabLoadPhase::kTransfer;
      cuda_require(cudaMemcpyAsync(owner->allocation_ + chunk.offset,
                                   owner->staging_[slot], chunk.bytes,
                                   cudaMemcpyHostToDevice, owner->stream_),
                   "target slab H2D enqueue failed");
      cuda_require(cudaEventRecord(owner->slot_events_[slot], owner->stream_),
                   "target slab slot event record failed");
      pending[slot] = true;
    }
    phase = TargetSlabLoadPhase::kPublish;
    cuda_require(cudaEventRecord(owner->publication_event_, owner->stream_),
                 "target slab publication event record failed");
    cuda_require(cudaEventSynchronize(owner->publication_event_),
                 "target slab publication fence failed");
    phase = TargetSlabLoadPhase::kCleanup;
    for (auto& event : owner->slot_events_) {
      cuda_require(cudaEventDestroy(event),
                   "target slab slot event cleanup failed");
      event = nullptr;
    }
    cuda_require(cudaStreamDestroy(owner->stream_),
                 "target slab private stream cleanup failed");
    owner->stream_ = nullptr;
    for (auto& allocation : owner->staging_allocations_) {
      cuda_require(cudaFreeHost(allocation),
                   "target slab pinned staging cleanup failed");
      allocation = nullptr;
    }
    owner->staging_.fill(nullptr);
    phase = TargetSlabLoadPhase::kPublish;
    const auto open_to_publish_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - opened)
            .count();
    owner->publication_ = {
        owner->allocation_, owner->publication_event_, kTargetSlabBytes, device,
        rank, kTargetSlabArtifactKey, metadata.slab_key,
        kTargetSlabManifestSha256, metadata.layout_sha256,
        static_cast<std::uint64_t>(open_to_publish_ns),
        kTargetSlabChunks, kTargetSlabPeakPinnedBytes};
    emit(telemetry, TargetSlabLoadPhase::kPublish,
         TargetSlabFailureClass::kNone, rank,
         kTargetSlabChunks - 1, true);
    return owner;
  } catch (const std::exception& error) {
    emit(telemetry, phase, failure_class(error), rank, chunk_index, false);
    owner->release();
    throw;
  } catch (...) {
    emit(telemetry, phase, TargetSlabFailureClass::kContract, rank, chunk_index,
         false);
    owner->release();
    throw;
  }
}

void TargetSlabDeviceOwner::release() noexcept {
  bool failed = false;
  if (device_ >= 0 && cudaSetDevice(device_) != cudaSuccess) failed = true;
  if (publication_event_ && cudaEventDestroy(publication_event_) != cudaSuccess)
    failed = true;
  publication_event_ = nullptr;
  for (auto& event : slot_events_) {
    if (event && cudaEventDestroy(event) != cudaSuccess) failed = true;
    event = nullptr;
  }
  if (stream_ && cudaStreamDestroy(stream_) != cudaSuccess) failed = true;
  stream_ = nullptr;
  for (auto& allocation : staging_allocations_) {
    if (allocation && cudaFreeHost(allocation) != cudaSuccess) failed = true;
    allocation = nullptr;
  }
  staging_.fill(nullptr);
  if (allocation_ && cudaFree(allocation_) != cudaSuccess) failed = true;
  allocation_ = nullptr;
  publication_ = {};
  if (failed && telemetry_)
    emit(*telemetry_, TargetSlabLoadPhase::kCleanup,
         TargetSlabFailureClass::kCuda, rank_, 0, false);
  device_ = -1;
  rank_ = -1;
  telemetry_ = nullptr;
}

TargetSlabDeviceOwner::~TargetSlabDeviceOwner() { release(); }

}  // namespace rocket::qwen38::model

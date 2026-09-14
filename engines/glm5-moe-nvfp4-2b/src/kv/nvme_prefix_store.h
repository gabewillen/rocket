#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocket::engine::kv {

constexpr std::size_t kPrefixIoAlignment = 65536;
constexpr std::uint32_t kPrefixFormatVersion = 1;

struct PrefixRecordKey {
  std::uint64_t namespace_hash = 0;
  std::uint64_t parent_hash = 0;
  std::uint64_t chain_hash = 0;
  std::uint32_t rank = 0;
  std::uint32_t token_count = 0;
  std::uint32_t record_kind = 0;
  std::uint32_t reserved = 0;

  bool operator==(const PrefixRecordKey&) const = default;
};

struct PrefixRecordKeyHash {
  std::size_t operator()(const PrefixRecordKey& k) const noexcept;
};

enum class PrefixComponent : std::uint32_t {
  kTargetLatent = 1,
  kTargetIndexerKey = 2,
  kTargetIndexerGate = 3,
  kKda = 4,
  kDflash2 = 5,
  kNextToken = 6,
};

struct DeviceConstSpan {
  PrefixComponent component;
  const void* data = nullptr;
  std::size_t bytes = 0;
};

struct DeviceSpan {
  PrefixComponent component;
  void* data = nullptr;
  std::size_t bytes = 0;
};

struct NvmePrefixOptions {
  std::filesystem::path directory;
  std::uint64_t capacity_bytes = 0;
  std::size_t staging_bytes = 128ull << 20;
  std::uint64_t segment_bytes = 1ull << 30;
  std::uint64_t free_space_headroom_bytes = 64ull << 30;
  std::uint64_t manifest_compact_bytes = 16ull << 20;
  int queue_depth = 4;
  int rank = 0;
};

struct NvmePrefixStats {
  std::uint64_t read_bytes = 0;
  std::uint64_t write_bytes = 0;
  std::uint64_t hit_records = 0;
  std::uint64_t miss_records = 0;
  std::uint64_t hit_pages = 0;
  std::uint64_t miss_pages = 0;
  std::uint64_t checksum_failures = 0;
  std::uint64_t rejected_records = 0;
  double restore_ms = 0.0;
  double writeback_ms = 0.0;
};

class NvmePrefixStore {
 public:
  explicit NvmePrefixStore(NvmePrefixOptions options, cudaStream_t stream);
  ~NvmePrefixStore();
  NvmePrefixStore(const NvmePrefixStore&) = delete;
  NvmePrefixStore& operator=(const NvmePrefixStore&) = delete;

  void save(const PrefixRecordKey& key, const std::vector<DeviceConstSpan>& spans);
  bool load(const PrefixRecordKey& key, const std::vector<DeviceSpan>& spans);
  bool holds(const PrefixRecordKey& key) const;
  void drop(const PrefixRecordKey& key);
  void note_page_lookup(bool hit) const {
    if (hit) ++stats_.hit_pages; else ++stats_.miss_pages;
  }

  std::size_t staging_bytes() const { return staging_bytes_; }
  std::uint64_t live_bytes() const { return live_bytes_; }
  const NvmePrefixStats& stats() const { return stats_; }
  const NvmePrefixOptions& options() const { return options_; }

  static std::uint64_t parse_bytes(const std::string& text);
  static std::uint64_t available_bytes(const std::filesystem::path& directory);

 private:
  struct ComponentLocation {
    PrefixComponent component{};
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    std::array<std::uint8_t, 32> checksum{};
  };
  struct RecordLocation {
    std::uint32_t segment = 0;
    std::uint64_t header_offset = 0;
    std::uint64_t record_bytes = 0;
    std::uint64_t last_access = 0;
    std::vector<ComponentLocation> components;
  };

  void recover();
  void append_manifest(const PrefixRecordKey& key, const RecordLocation& loc, bool live);
  void maybe_compact_manifest();
  int open_segment(std::uint32_t segment, bool create);
  void rotate_segment(std::uint64_t required);
  void direct_write(int fd, const void* data, std::size_t bytes, std::uint64_t offset);
  void direct_read(int fd, void* data, std::size_t bytes, std::uint64_t offset);

  NvmePrefixOptions options_;
  cudaStream_t stream_ = nullptr;
  void* staging_ = nullptr;
  std::size_t staging_bytes_ = 0;
  int manifest_fd_ = -1;
  std::uint64_t manifest_offset_ = 0;
  int segment_fd_ = -1;
  std::uint32_t segment_id_ = 0;
  std::uint64_t segment_offset_ = 0;
  std::uint64_t live_bytes_ = 0;
  std::uint64_t access_clock_ = 0;
  std::unordered_map<PrefixRecordKey, RecordLocation, PrefixRecordKeyHash> index_;
  mutable NvmePrefixStats stats_;
};

}  // namespace rocket::engine::kv

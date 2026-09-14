#include "kv/nvme_prefix_store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace rocket::engine::kv {
namespace {

constexpr std::uint64_t kHeaderMagic = 0x31584650544b4352ull;  // RCKTPFX1
constexpr std::uint64_t kManifestMagic = 0x31464e4d58465052ull;  // RPFXMNF1
constexpr std::size_t kMaxComponents = 16;

[[noreturn]] void fail(const std::string& what);

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* what) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) fail(what);
  return a + b;
}

std::uint64_t align_up(std::uint64_t n) {
  return checked_add(n, kPrefixIoAlignment - 1, "aligned size overflow") &
         ~(static_cast<std::uint64_t>(kPrefixIoAlignment) - 1);
}

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("nvme prefix: " + what);
}

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}

void io_all(int fd, void* data, std::size_t bytes, std::uint64_t offset, bool write) {
  auto* p = static_cast<std::uint8_t*>(data);
  std::size_t done = 0;
  while (done < bytes) {
    const ssize_t n = write ? ::pwrite(fd, p + done, bytes - done, offset + done)
                            : ::pread(fd, p + done, bytes - done, offset + done);
    if (n < 0) {
      if (errno == EINTR) continue;
      fail(std::string(write ? "pwrite" : "pread") + ": " + std::strerror(errno));
    }
    if (n == 0) fail(write ? "short pwrite" : "unexpected EOF");
    done += static_cast<std::size_t>(n);
  }
}

#pragma pack(push, 1)
struct DiskComponent {
  std::uint32_t kind;
  std::uint32_t reserved;
  std::uint64_t offset;
  std::uint64_t bytes;
  std::uint64_t stored_bytes;
  std::uint8_t checksum[32];
};

struct DiskHeaderPrefix {
  std::uint64_t magic;
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint64_t namespace_hash;
  std::uint64_t parent_hash;
  std::uint64_t chain_hash;
  std::uint32_t rank;
  std::uint32_t token_count;
  std::uint32_t record_kind;
  std::uint32_t key_reserved;
  std::uint32_t component_count;
  std::uint32_t reserved;
  std::uint64_t record_bytes;
  DiskComponent components[kMaxComponents];
};

struct ManifestEntry {
  std::uint64_t magic;
  std::uint32_t version;
  std::uint32_t live;
  PrefixRecordKey key;
  std::uint32_t segment;
  std::uint32_t component_count;
  std::uint64_t header_offset;
  std::uint64_t record_bytes;
  std::uint8_t reserved[32];
};
#pragma pack(pop)

static_assert(sizeof(DiskHeaderPrefix) < kPrefixIoAlignment);
static_assert(sizeof(ManifestEntry) == 112);

void reject_symlink_path(const std::filesystem::path& input) {
  std::error_code ec;
  const auto absolute = std::filesystem::absolute(input, ec);
  if (ec) fail("absolute cache path: " + ec.message());
  std::filesystem::path current;
  for (const auto& part : absolute) {
    current /= part;
    const auto status = std::filesystem::symlink_status(current, ec);
    if (ec) {
      if (ec == std::errc::no_such_file_or_directory) { ec.clear(); continue; }
      fail("inspect cache path " + current.string() + ": " + ec.message());
    }
    if (std::filesystem::is_symlink(status))
      fail("cache path contains symlink: " + current.string());
  }
}

std::filesystem::path segment_path(const std::filesystem::path& dir, std::uint32_t id) {
  char name[40];
  std::snprintf(name, sizeof(name), "segment-%08u.bin", id);
  return dir / name;
}

struct ChecksumState {
  std::uint64_t h[4] = {0xcbf29ce484222325ull, 0x84222325cbf29ce4ull,
                        0x9e3779b97f4a7c15ull, 0xd6e8feb86659fd93ull};
  std::uint64_t position = 0;
};

void checksum_update(ChecksumState* state, const void* data, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  while (bytes >= sizeof(std::uint64_t)) {
    std::uint64_t word = 0;
    std::memcpy(&word, p, sizeof(word));
    const unsigned lane = static_cast<unsigned>((state->position / 8) & 3ull);
    state->h[lane] ^= word + 0x9e3779b97f4a7c15ull + (state->h[lane] << 6) +
                      (state->h[lane] >> 2);
    state->h[lane] *= 0xd6e8feb86659fd93ull;
    state->position += sizeof(word);
    p += sizeof(word);
    bytes -= sizeof(word);
  }
  if (bytes) {
    std::uint64_t tail = 0;
    std::memcpy(&tail, p, bytes);
    const unsigned lane = static_cast<unsigned>((state->position / 8) & 3ull);
    state->h[lane] ^= tail ^ (static_cast<std::uint64_t>(bytes) << 56);
    state->h[lane] *= 0xd6e8feb86659fd93ull;
    state->position += bytes;
  }
}

std::array<std::uint8_t, 32> checksum_finish(const ChecksumState& state) {
  std::array<std::uint8_t, 32> out{};
  std::memcpy(out.data(), state.h, sizeof(state.h));
  return out;
}

}  // namespace

std::size_t PrefixRecordKeyHash::operator()(const PrefixRecordKey& k) const noexcept {
  std::uint64_t h = k.namespace_hash;
  h ^= k.parent_hash + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  h ^= k.chain_hash + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  h ^= (static_cast<std::uint64_t>(k.rank) << 32) | k.token_count;
  h ^= static_cast<std::uint64_t>(k.record_kind) * 0x9e3779b97f4a7c15ull;
  return static_cast<std::size_t>(h ^ (h >> 32));
}

std::uint64_t NvmePrefixStore::available_bytes(const std::filesystem::path& directory) {
  std::error_code ec;
  const auto s = std::filesystem::space(directory, ec);
  if (ec) fail("space(" + directory.string() + "): " + ec.message());
  return s.available;
}

std::uint64_t NvmePrefixStore::parse_bytes(const std::string& text) {
  if (text.empty()) fail("empty byte size");
  std::size_t used = 0;
  unsigned long long base = 0;
  try {
    base = std::stoull(text, &used, 10);
  } catch (...) {
    fail("invalid byte size: " + text);
  }
  const std::string_view suffix(text.data() + used, text.size() - used);
  std::uint64_t mul = 1;
  if (suffix.empty() || suffix == "B") mul = 1;
  else if (suffix == "KiB") mul = 1ull << 10;
  else if (suffix == "MiB") mul = 1ull << 20;
  else if (suffix == "GiB") mul = 1ull << 30;
  else if (suffix == "TiB") mul = 1ull << 40;
  else fail("invalid byte-size suffix: " + text);
  if (base > std::numeric_limits<std::uint64_t>::max() / mul) fail("byte size overflow: " + text);
  return static_cast<std::uint64_t>(base) * mul;
}

NvmePrefixStore::NvmePrefixStore(NvmePrefixOptions options, cudaStream_t stream)
    : options_(std::move(options)), stream_(stream) {
  if (options_.directory.empty()) fail("cache directory is empty");
  if (!options_.capacity_bytes) fail("capacity must be positive");
  if (options_.staging_bytes < kPrefixIoAlignment ||
      options_.staging_bytes % kPrefixIoAlignment != 0)
    fail("staging bytes must be a positive 64 KiB multiple");
  if (options_.staging_bytes > (128ull << 20))
    fail("staging bytes exceed the 128 MiB per-node pinned-memory ceiling");
  if (options_.segment_bytes < 2 * kPrefixIoAlignment ||
      options_.segment_bytes % kPrefixIoAlignment != 0)
    fail("segment bytes must be a 64 KiB multiple");
  if (options_.queue_depth < 1) fail("queue depth must be positive");
  if (options_.staging_bytes <
      static_cast<std::size_t>(options_.queue_depth) * kPrefixIoAlignment)
    fail("staging bytes must cover queue_depth aligned slots");
  reject_symlink_path(options_.directory);
  std::filesystem::create_directories(options_.directory);
  reject_symlink_path(options_.directory);
  const std::uint64_t free = available_bytes(options_.directory);
  if (free <= options_.free_space_headroom_bytes ||
      options_.capacity_bytes > free - options_.free_space_headroom_bytes)
    fail("capacity exceeds free space after safety headroom");
  staging_bytes_ = options_.staging_bytes;
  if (::posix_memalign(&staging_, kPrefixIoAlignment, staging_bytes_) != 0)
    fail("aligned staging allocation failed");
  try {
    cuda_check(cudaHostRegister(staging_, staging_bytes_, cudaHostRegisterDefault),
               "cudaHostRegister staging");
    const auto manifest = options_.directory / "manifest.bin";
    manifest_fd_ = ::open(manifest.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (manifest_fd_ < 0) fail("open manifest: " + std::string(std::strerror(errno)));
    recover();
    segment_fd_ = open_segment(segment_id_, true);
  } catch (...) {
    if (manifest_fd_ >= 0) ::close(manifest_fd_);
    cudaHostUnregister(staging_);
    std::free(staging_);
    staging_ = nullptr;
    throw;
  }
}

NvmePrefixStore::~NvmePrefixStore() {
  if (segment_fd_ >= 0) ::close(segment_fd_);
  if (manifest_fd_ >= 0) ::close(manifest_fd_);
  if (staging_) {
    cudaHostUnregister(staging_);
    std::free(staging_);
  }
}

int NvmePrefixStore::open_segment(std::uint32_t segment, bool create) {
  const auto path = segment_path(options_.directory, segment);
  const int flags = O_RDWR | O_DIRECT | O_CLOEXEC | O_NOFOLLOW | (create ? O_CREAT : 0);
  const int fd = ::open(path.c_str(), flags, 0600);
  if (fd < 0) fail("open " + path.string() + ": " + std::strerror(errno));
  return fd;
}

void NvmePrefixStore::direct_write(int fd, const void* data, std::size_t bytes,
                                   std::uint64_t offset) {
  if ((reinterpret_cast<std::uintptr_t>(data) | bytes | offset) % kPrefixIoAlignment)
    fail("unaligned direct write");
  io_all(fd, const_cast<void*>(data), bytes, offset, true);
}

void NvmePrefixStore::direct_read(int fd, void* data, std::size_t bytes,
                                  std::uint64_t offset) {
  if ((reinterpret_cast<std::uintptr_t>(data) | bytes | offset) % kPrefixIoAlignment)
    fail("unaligned direct read");
  io_all(fd, data, bytes, offset, false);
}

void NvmePrefixStore::rotate_segment(std::uint64_t required) {
  if (required > options_.segment_bytes) fail("record is larger than one segment");
  if (segment_fd_ >= 0 && required <= options_.segment_bytes - segment_offset_) return;
  if (segment_fd_ >= 0) ::close(segment_fd_);
  const std::uint32_t count = std::max<std::uint32_t>(
      1, static_cast<std::uint32_t>(options_.capacity_bytes / options_.segment_bytes));
  const std::uint32_t next = (segment_id_ + 1) % count;
  const bool next_used = std::any_of(index_.begin(), index_.end(),
      [&](const auto& entry) { return entry.second.segment == next; });
  if (!next_used) {
    segment_id_ = next;
  } else {
    std::optional<std::pair<std::uint32_t, std::uint64_t>> victim_segment;
    for (std::uint32_t candidate = 0; candidate < count; ++candidate) {
      bool used = false, protected_ancestor = false;
      std::uint64_t oldest_access = std::numeric_limits<std::uint64_t>::max();
      for (const auto& entry : index_) {
        if (entry.second.segment != candidate) continue;
        used = true;
        oldest_access = std::min(oldest_access, entry.second.last_access);
        if (entry.first.record_kind == 1 &&
            std::any_of(index_.begin(), index_.end(), [&](const auto& child) {
              return child.second.segment != candidate && child.first.record_kind == 1 &&
                     child.first.namespace_hash == entry.first.namespace_hash &&
                     child.first.rank == entry.first.rank &&
                     child.first.parent_hash == entry.first.chain_hash;
            })) protected_ancestor = true;
      }
      if (used && !protected_ancestor &&
          (!victim_segment || oldest_access < victim_segment->second))
        victim_segment = std::make_pair(candidate, oldest_access);
    }
    if (!victim_segment) fail("all cache segments contain protected radix ancestors");
    segment_id_ = victim_segment->first;
  }
  for (auto it = index_.begin(); it != index_.end();) {
    if (it->second.segment != segment_id_) { ++it; continue; }
    const PrefixRecordKey stale_key = it->first;
    const RecordLocation stale_loc = it->second;
    live_bytes_ -= stale_loc.record_bytes;
    it = index_.erase(it);
    append_manifest(stale_key, stale_loc, false);
  }
  segment_offset_ = 0;
  const auto path = segment_path(options_.directory, segment_id_);
  segment_fd_ = ::open(path.c_str(), O_RDWR | O_DIRECT | O_CLOEXEC | O_NOFOLLOW |
                                      O_CREAT | O_TRUNC, 0600);
  if (segment_fd_ < 0) fail("rotate " + path.string() + ": " + std::strerror(errno));
}

void NvmePrefixStore::append_manifest(const PrefixRecordKey& key, const RecordLocation& loc,
                                      bool live) {
  ManifestEntry e{};
  e.magic = kManifestMagic;
  e.version = kPrefixFormatVersion;
  e.live = live ? 1 : 0;
  e.key = key;
  e.segment = loc.segment;
  e.component_count = static_cast<std::uint32_t>(loc.components.size());
  e.header_offset = loc.header_offset;
  e.record_bytes = loc.record_bytes;
  io_all(manifest_fd_, &e, sizeof(e), manifest_offset_, true);
  manifest_offset_ += sizeof(e);
  if (::fdatasync(manifest_fd_) != 0) fail("fdatasync manifest: " + std::string(std::strerror(errno)));
}

void NvmePrefixStore::maybe_compact_manifest() {
  const std::uint64_t live_manifest =
      std::max<std::uint64_t>(1, index_.size()) * sizeof(ManifestEntry);
  if (manifest_offset_ < options_.manifest_compact_bytes ||
      live_manifest > std::numeric_limits<std::uint64_t>::max() / 4 ||
      manifest_offset_ < 4 * live_manifest) return;
  const auto temp = options_.directory / "manifest.compact.tmp";
  const auto final = options_.directory / "manifest.bin";
  const int fd = ::open(temp.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) fail("open compact manifest: " + std::string(std::strerror(errno)));
  std::uint64_t offset = 0;
  try {
    std::vector<std::pair<PrefixRecordKey, RecordLocation>> entries(index_.begin(), index_.end());
    std::sort(entries.begin(), entries.end(), [&](const auto& a, const auto& b) {
      const bool ac = a.second.segment == segment_id_;
      const bool bc = b.second.segment == segment_id_;
      if (ac != bc) return !ac;
      if (a.second.segment != b.second.segment) return a.second.segment < b.second.segment;
      return a.second.header_offset < b.second.header_offset;
    });
    for (const auto& [key, loc] : entries) {
      ManifestEntry e{};
      e.magic = kManifestMagic;
      e.version = kPrefixFormatVersion;
      e.live = 1;
      e.key = key;
      e.segment = loc.segment;
      e.component_count = static_cast<std::uint32_t>(loc.components.size());
      e.header_offset = loc.header_offset;
      e.record_bytes = loc.record_bytes;
      io_all(fd, &e, sizeof(e), offset, true);
      offset += sizeof(e);
    }
    if (::fdatasync(fd) != 0) fail("fdatasync compact manifest");
    ::close(fd);
    if (::rename(temp.c_str(), final.c_str()) != 0)
      fail("rename compact manifest: " + std::string(std::strerror(errno)));
    const int dirfd = ::open(options_.directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dirfd >= 0) { ::fsync(dirfd); ::close(dirfd); }
    ::close(manifest_fd_);
    manifest_fd_ = ::open(final.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (manifest_fd_ < 0) fail("reopen compact manifest");
    manifest_offset_ = offset;
  } catch (...) {
    ::close(fd);
    ::unlink(temp.c_str());
    throw;
  }
}

void NvmePrefixStore::recover() {
  struct stat st{};
  if (::fstat(manifest_fd_, &st) != 0) fail("fstat manifest");
  const std::uint64_t complete = static_cast<std::uint64_t>(st.st_size) / sizeof(ManifestEntry) *
                                 sizeof(ManifestEntry);
  std::uint64_t off = 0;
  while (off < complete) {
    ManifestEntry e{};
    io_all(manifest_fd_, &e, sizeof(e), off, false);
    off += sizeof(e);
    if (e.magic != kManifestMagic || e.version != kPrefixFormatVersion) {
      ++stats_.rejected_records;
      continue;
    }
    if (!e.live) {
      index_.erase(e.key);
      continue;
    }
    int fd = -1;
    try {
      fd = open_segment(e.segment, false);
      struct stat segment_stat{};
      if (::fstat(fd, &segment_stat) != 0) fail("fstat segment");
      direct_read(fd, staging_, kPrefixIoAlignment, e.header_offset);
      ::close(fd);
      fd = -1;
      const auto* h = static_cast<const DiskHeaderPrefix*>(staging_);
      if (e.header_offset % kPrefixIoAlignment || e.record_bytes < 2 * kPrefixIoAlignment ||
          e.record_bytes % kPrefixIoAlignment || h->magic != kHeaderMagic ||
          h->version != kPrefixFormatVersion || h->header_bytes != kPrefixIoAlignment ||
          h->namespace_hash != e.key.namespace_hash ||
          h->parent_hash != e.key.parent_hash || h->chain_hash != e.key.chain_hash ||
          h->rank != e.key.rank || h->token_count != e.key.token_count ||
          h->record_kind != e.key.record_kind || h->component_count == 0 ||
          h->component_count > kMaxComponents || h->record_bytes != e.record_bytes) {
        ++stats_.rejected_records;
        continue;
      }
      RecordLocation loc;
      loc.segment = e.segment;
      loc.header_offset = e.header_offset;
      loc.record_bytes = e.record_bytes;
      loc.last_access = ++access_clock_;
      if (e.header_offset > std::numeric_limits<std::uint64_t>::max() - e.record_bytes) {
        ++stats_.rejected_records;
        continue;
      }
      const std::uint64_t record_end = e.header_offset + e.record_bytes;
      if (segment_stat.st_size < 0 || record_end > static_cast<std::uint64_t>(segment_stat.st_size)) {
        ++stats_.rejected_records;
        continue;
      }
      std::uint64_t previous_end = checked_add(e.header_offset, kPrefixIoAlignment,
                                               "header extent overflow");
      bool valid_components = true;
      for (std::uint32_t i = 0; i < h->component_count; ++i) {
        const DiskComponent& dc = h->components[i];
        if (!dc.bytes || dc.stored_bytes != align_up(dc.bytes) ||
            dc.offset % kPrefixIoAlignment || dc.stored_bytes % kPrefixIoAlignment ||
            dc.offset < previous_end ||
            dc.stored_bytes > std::numeric_limits<std::uint64_t>::max() - dc.offset ||
            dc.offset + dc.stored_bytes > record_end) {
          valid_components = false;
          break;
        }
        ComponentLocation c;
        c.component = static_cast<PrefixComponent>(dc.kind);
        c.offset = dc.offset;
        c.bytes = dc.bytes;
        std::copy_n(dc.checksum, c.checksum.size(), c.checksum.begin());
        loc.components.push_back(c);
        previous_end = dc.offset + dc.stored_bytes;
      }
      if (!valid_components || previous_end != record_end) {
        ++stats_.rejected_records;
        continue;
      }
      const auto prior = index_.find(e.key);
      if (prior != index_.end()) live_bytes_ -= prior->second.record_bytes;
      index_[e.key] = loc;
      live_bytes_ += loc.record_bytes;
      segment_id_ = e.segment;
      segment_offset_ = record_end;
    } catch (...) {
      if (fd >= 0) ::close(fd);
      ++stats_.rejected_records;
    }
  }
  manifest_offset_ = complete;
  if (static_cast<std::uint64_t>(st.st_size) != complete && ::ftruncate(manifest_fd_, complete) != 0)
    fail("truncate partial manifest");
}

bool NvmePrefixStore::holds(const PrefixRecordKey& key) const { return index_.contains(key); }

void NvmePrefixStore::save(const PrefixRecordKey& key,
                           const std::vector<DeviceConstSpan>& spans) {
  if (spans.empty() || spans.size() > kMaxComponents) fail("component count out of range");
  for (std::size_t i = 0; i < spans.size(); ++i)
    for (std::size_t j = i + 1; j < spans.size(); ++j)
      if (spans[i].component == spans[j].component) fail("duplicate component kind");
  std::uint64_t record_bytes = kPrefixIoAlignment;
  for (const auto& span : spans) {
    if (!span.data || !span.bytes) fail("empty component");
    record_bytes = checked_add(record_bytes, align_up(span.bytes), "record size overflow");
  }
  auto old = index_.find(key);
  std::uint64_t old_bytes = old == index_.end() ? 0 : old->second.record_bytes;
  if (old_bytes > live_bytes_) fail("live-byte accounting underflow");
  while (record_bytes > options_.capacity_bytes - (live_bytes_ - old_bytes)) {
    auto evictable = [&](const auto& entry) {
      if (entry.first == key) return false;
      if (entry.first.record_kind != 1) return true;
      return std::none_of(index_.begin(), index_.end(), [&](const auto& child) {
        return child.first.record_kind == 1 &&
               child.first.namespace_hash == entry.first.namespace_hash &&
               child.first.rank == entry.first.rank &&
               child.first.parent_hash == entry.first.chain_hash;
      });
    };
    auto victim = index_.end();
    for (auto it = index_.begin(); it != index_.end(); ++it)
      if (evictable(*it) && (victim == index_.end() ||
                             it->second.last_access < victim->second.last_access))
        victim = it;
    if (victim == index_.end()) fail("record exceeds protected cache capacity");
    drop(victim->first);
    old = index_.find(key);
    old_bytes = old == index_.end() ? 0 : old->second.record_bytes;
  }
  rotate_segment(record_bytes);
  old = index_.find(key);
  const auto started = std::chrono::steady_clock::now();
  RecordLocation loc;
  loc.segment = segment_id_;
  loc.header_offset = segment_offset_;
  loc.record_bytes = record_bytes;
  loc.last_access = ++access_clock_;
  std::uint64_t cursor = checked_add(segment_offset_, kPrefixIoAlignment,
                                     "record cursor overflow");
  for (const auto& span : spans) {
    ComponentLocation c;
    c.component = span.component;
    c.offset = cursor;
    c.bytes = span.bytes;
    ChecksumState checksum{};
    const std::size_t slots = static_cast<std::size_t>(options_.queue_depth);
    const std::size_t slot_bytes =
        (staging_bytes_ / slots / kPrefixIoAlignment) * kPrefixIoAlignment;
    if (!slot_bytes) fail("staging ring is smaller than queue_depth * 64 KiB");
    std::vector<std::future<void>> jobs(slots);
    std::vector<cudaEvent_t> copies(slots, nullptr);
    for (auto& event : copies)
      cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "save copy event");
    std::size_t copied = 0;
    std::size_t chunk = 0;
    while (copied < span.bytes) {
      const std::size_t slot = chunk % slots;
      if (jobs[slot].valid()) jobs[slot].get();
      auto* buffer = static_cast<std::uint8_t*>(staging_) + slot * slot_bytes;
      const std::size_t raw = std::min(slot_bytes, span.bytes - copied);
      const std::size_t stored = static_cast<std::size_t>(align_up(raw));
      cuda_check(cudaMemcpyAsync(buffer, static_cast<const std::uint8_t*>(span.data) + copied,
                                 raw, cudaMemcpyDeviceToHost, stream_), "save device copy");
      cuda_check(cudaEventRecord(copies[slot], stream_), "save copy record");
      cuda_check(cudaEventSynchronize(copies[slot]), "save copy wait");
      checksum_update(&checksum, buffer, raw);
      if (stored > raw) std::memset(buffer + raw, 0, stored - raw);
      const std::uint64_t write_at = cursor;
      jobs[slot] = std::async(std::launch::async, [this, buffer, stored, write_at] {
        direct_write(segment_fd_, buffer, stored, write_at);
      });
      cursor = checked_add(cursor, stored, "component cursor overflow");
      copied += raw;
      stats_.write_bytes += raw;
      ++chunk;
    }
    for (auto& job : jobs) if (job.valid()) job.get();
    for (auto event : copies) cudaEventDestroy(event);
    c.checksum = checksum_finish(checksum);
    loc.components.push_back(c);
  }
  if (::fdatasync(segment_fd_) != 0) fail("fdatasync payload: " + std::string(std::strerror(errno)));
  std::memset(staging_, 0, kPrefixIoAlignment);
  auto* h = static_cast<DiskHeaderPrefix*>(staging_);
  h->magic = kHeaderMagic;
  h->version = kPrefixFormatVersion;
  h->header_bytes = kPrefixIoAlignment;
  h->namespace_hash = key.namespace_hash;
  h->parent_hash = key.parent_hash;
  h->chain_hash = key.chain_hash;
  h->rank = key.rank;
  h->token_count = key.token_count;
  h->record_kind = key.record_kind;
  h->component_count = static_cast<std::uint32_t>(loc.components.size());
  h->record_bytes = loc.record_bytes;
  for (std::size_t i = 0; i < loc.components.size(); ++i) {
    h->components[i].kind = static_cast<std::uint32_t>(loc.components[i].component);
    h->components[i].offset = loc.components[i].offset;
    h->components[i].bytes = loc.components[i].bytes;
    h->components[i].stored_bytes = align_up(loc.components[i].bytes);
    std::copy(loc.components[i].checksum.begin(), loc.components[i].checksum.end(),
              h->components[i].checksum);
  }
  direct_write(segment_fd_, staging_, kPrefixIoAlignment, loc.header_offset);
  if (::fdatasync(segment_fd_) != 0) fail("fdatasync header: " + std::string(std::strerror(errno)));
  append_manifest(key, loc, true);
  if (old != index_.end()) live_bytes_ -= old->second.record_bytes;
  index_[key] = loc;
  live_bytes_ += loc.record_bytes;
  maybe_compact_manifest();
  segment_offset_ += record_bytes;
  stats_.writeback_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
}

bool NvmePrefixStore::load(const PrefixRecordKey& key, const std::vector<DeviceSpan>& spans) {
  const auto it = index_.find(key);
  if (it == index_.end()) {
    ++stats_.miss_records;
    return false;
  }
  if (spans.size() != it->second.components.size()) {
    ++stats_.rejected_records;
    return false;
  }
  int fd = open_segment(it->second.segment, false);
  const auto started = std::chrono::steady_clock::now();
  try {
    for (const auto& requested : spans) {
      const auto found = std::find_if(it->second.components.begin(), it->second.components.end(),
                                     [&](const ComponentLocation& c) {
                                       return c.component == requested.component;
                                     });
      if (found == it->second.components.end() || found->bytes != requested.bytes || !requested.data) {
        ++stats_.rejected_records;
        ::close(fd);
        return false;
      }
      ChecksumState checksum{};
      const std::size_t slots = static_cast<std::size_t>(options_.queue_depth);
      const std::size_t slot_bytes =
          (staging_bytes_ / slots / kPrefixIoAlignment) * kPrefixIoAlignment;
      if (!slot_bytes) fail("staging ring is smaller than queue_depth * 64 KiB");
      struct ReadJob {
        std::future<void> future;
        std::size_t raw = 0;
        std::size_t stored = 0;
        std::size_t copied = 0;
      };
      std::vector<ReadJob> jobs(slots);
      std::vector<cudaEvent_t> copies(slots, nullptr);
      std::vector<bool> copy_pending(slots, false);
      for (auto& event : copies)
        cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "load copy event");
      std::size_t issued = 0, consumed = 0, chunk = 0;
      std::uint64_t cursor = found->offset;
      while (consumed < found->bytes) {
        while (issued < found->bytes && issued - consumed < slots * slot_bytes) {
          const std::size_t slot = chunk % slots;
          if (copy_pending[slot]) {
            cuda_check(cudaEventSynchronize(copies[slot]), "load ring reuse wait");
            copy_pending[slot] = false;
          }
          auto* buffer = static_cast<std::uint8_t*>(staging_) + slot * slot_bytes;
          const std::size_t raw = std::min(slot_bytes, static_cast<std::size_t>(found->bytes - issued));
          const std::size_t stored = static_cast<std::size_t>(align_up(raw));
          const std::uint64_t read_at = cursor;
          jobs[slot].raw = raw;
          jobs[slot].stored = stored;
          jobs[slot].copied = issued;
          jobs[slot].future = std::async(std::launch::async, [this, fd, buffer, stored, read_at] {
            direct_read(fd, buffer, stored, read_at);
          });
          cursor = checked_add(cursor, stored, "read cursor overflow");
          issued += raw;
          ++chunk;
        }
        const std::size_t consume_chunk = consumed / slot_bytes;
        const std::size_t slot = consume_chunk % slots;
        jobs[slot].future.get();
        auto* buffer = static_cast<std::uint8_t*>(staging_) + slot * slot_bytes;
        checksum_update(&checksum, buffer, jobs[slot].raw);
        cuda_check(cudaMemcpyAsync(static_cast<std::uint8_t*>(requested.data) + jobs[slot].copied,
                                   buffer, jobs[slot].raw, cudaMemcpyHostToDevice, stream_),
                   "load device copy");
        cuda_check(cudaEventRecord(copies[slot], stream_), "load copy record");
        copy_pending[slot] = true;
        consumed += jobs[slot].raw;
        stats_.read_bytes += jobs[slot].raw;
      }
      for (std::size_t slot = 0; slot < slots; ++slot)
        if (copy_pending[slot])
          cuda_check(cudaEventSynchronize(copies[slot]), "load copy finish");
      for (auto event : copies) cudaEventDestroy(event);
      if (checksum_finish(checksum) != found->checksum) {
        ++stats_.checksum_failures;
        ::close(fd);
        return false;
      }
    }
    ::close(fd);
  } catch (...) {
    ::close(fd);
    throw;
  }
  it->second.last_access = ++access_clock_;
  ++stats_.hit_records;
  stats_.restore_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  return true;
}

void NvmePrefixStore::drop(const PrefixRecordKey& key) {
  const auto it = index_.find(key);
  if (it == index_.end()) return;
  const RecordLocation loc = it->second;
  append_manifest(key, loc, false);
  live_bytes_ -= loc.record_bytes;
  index_.erase(it);
  maybe_compact_manifest();
}

}  // namespace rocket::engine::kv

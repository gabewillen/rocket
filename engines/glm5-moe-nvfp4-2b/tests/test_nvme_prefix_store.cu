#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <unistd.h>

#include "kv/nvme_prefix_store.h"

namespace kv = rocket::engine::kv;
namespace {
int failures = 0;
void check(const char* what, bool ok) {
  std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}
void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}
std::filesystem::path temp_dir() {
  auto p = std::filesystem::temp_directory_path() /
           ("rocket-prefix-" + std::to_string(::getpid()));
  std::filesystem::remove_all(p);
  std::filesystem::create_directories(p);
  return p;
}
}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  check("512GiB parses without allocating it",
        kv::NvmePrefixStore::parse_bytes("512GiB") == (512ull << 30));
  bool bad_suffix = false;
  try { (void)kv::NvmePrefixStore::parse_bytes("512GB"); } catch (...) { bad_suffix = true; }
  check("ambiguous decimal suffix is rejected", bad_suffix);

  const auto dir = temp_dir();
  cudaStream_t stream = nullptr;
  ck(cudaStreamCreate(&stream), "stream");
  kv::NvmePrefixOptions options;
  options.directory = dir;
  options.capacity_bytes = 32ull << 20;
  options.staging_bytes = 2ull << 20;
  options.segment_bytes = 16ull << 20;
  options.free_space_headroom_bytes = 0;
  options.queue_depth = 2;
  options.manifest_compact_bytes = sizeof(std::uint64_t);
  options.rank = 1;
  const kv::PrefixRecordKey key{0x11, 0x22, 0x33, 1, 8192, 7, 0};
  const std::size_t a_bytes = (5ull << 20) + 12345;
  const std::size_t b_bytes = (3ull << 20) + 777;
  std::vector<std::uint8_t> a(a_bytes), b(b_bytes);
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<std::uint8_t>(i * 17 + 3);
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::uint8_t>(i * 29 + 5);
  void *da = nullptr, *db = nullptr;
  ck(cudaMalloc(&da, a_bytes), "alloc a");
  ck(cudaMalloc(&db, b_bytes), "alloc b");
  ck(cudaMemcpy(da, a.data(), a_bytes, cudaMemcpyHostToDevice), "seed a");
  ck(cudaMemcpy(db, b.data(), b_bytes, cudaMemcpyHostToDevice), "seed b");

  {
    kv::NvmePrefixStore store(options, stream);
    check("pinned staging obeys the configured ceiling",
          store.staging_bytes() == options.staging_bytes);
    store.save(key, {{kv::PrefixComponent::kKda, da, a_bytes},
                     {kv::PrefixComponent::kDflash2, db, b_bytes}});
    check("committed record is indexed", store.holds(key));
    ck(cudaMemset(da, 0, a_bytes), "clear a");
    ck(cudaMemset(db, 0, b_bytes), "clear b");
    check("record restores all components",
          store.load(key, {{kv::PrefixComponent::kKda, da, a_bytes},
                           {kv::PrefixComponent::kDflash2, db, b_bytes}}));
    std::vector<std::uint8_t> ra(a_bytes), rb(b_bytes);
    ck(cudaMemcpy(ra.data(), da, a_bytes, cudaMemcpyDeviceToHost), "read a");
    ck(cudaMemcpy(rb.data(), db, b_bytes, cudaMemcpyDeviceToHost), "read b");
    check("restored bytes match exactly", ra == a && rb == b);
    check("IO counters count unpadded payload bytes",
          store.stats().write_bytes == a_bytes + b_bytes &&
          store.stats().read_bytes == a_bytes + b_bytes);
    check("missing target component rejects the checkpoint",
          !store.load(key, {{kv::PrefixComponent::kTargetLatent, da, a_bytes}}));
    const kv::PrefixRecordKey no_kda{0x11, 0x22, 0x44, 1, 8192, 7, 0};
    store.save(no_kda, {{kv::PrefixComponent::kTargetLatent, da, 65536},
                        {kv::PrefixComponent::kDflash2, db, 65536}});
    check("missing KDA component rejects the checkpoint",
          !store.load(no_kda, {{kv::PrefixComponent::kTargetLatent, da, 65536},
                               {kv::PrefixComponent::kKda, db, 65536},
                               {kv::PrefixComponent::kDflash2, db, 65536}}));
    const kv::PrefixRecordKey no_dflash{0x11, 0x22, 0x55, 1, 8192, 7, 0};
    store.save(no_dflash, {{kv::PrefixComponent::kTargetLatent, da, 65536},
                           {kv::PrefixComponent::kKda, db, 65536}});
    check("missing DFlash2 component rejects the checkpoint",
          !store.load(no_dflash, {{kv::PrefixComponent::kTargetLatent, da, 65536},
                                  {kv::PrefixComponent::kKda, db, 65536},
                                  {kv::PrefixComponent::kDflash2, db, 65536}}));
    bool overflow_rejected = false;
    try {
      const kv::PrefixRecordKey huge{1, 2, 3, 1, 1, 9, 0};
      store.save(huge, {{kv::PrefixComponent::kKda, da,
                         std::numeric_limits<std::size_t>::max()}});
    } catch (...) { overflow_rejected = true; }
    check("overflowing component extent is rejected before IO", overflow_rejected);
    store.note_page_lookup(true);
    store.note_page_lookup(false);
    check("page hit and miss counters are exposed",
          store.stats().hit_pages == 1 && store.stats().miss_pages == 1);
    for (int i = 0; i < 12; ++i) {
      const kv::PrefixRecordKey churn{9, 0, static_cast<std::uint64_t>(100 + i), 1,
                                      static_cast<std::uint32_t>(128 + i), 9, 0};
      store.save(churn, {{kv::PrefixComponent::kKda, da, 65536}});
      store.drop(churn);
    }
    check("manifest compaction bounds journal growth",
          std::filesystem::file_size(dir / "manifest.bin") < 16 * 112);
  }

  {
    std::ofstream manifest(dir / "manifest.bin", std::ios::binary | std::ios::app);
    const char torn[] = "torn-manifest-tail";
    manifest.write(torn, sizeof(torn));
  }
  {
    kv::NvmePrefixStore reopened(options, stream);
    check("restart replays committed manifest and ignores torn tail", reopened.holds(key));
    ck(cudaMemset(da, 0, a_bytes), "clear restart a");
    ck(cudaMemset(db, 0, b_bytes), "clear restart b");
    check("restart restores the committed record after manifest compaction",
          reopened.load(key, {{kv::PrefixComponent::kKda, da, a_bytes},
                              {kv::PrefixComponent::kDflash2, db, b_bytes}}));
    const kv::PrefixRecordKey after_restart{0x11, 0x33, 0x66, 1, 8320, 7, 0};
    reopened.save(after_restart, {{kv::PrefixComponent::kKda, da, 65536}});
    ck(cudaMemset(da, 0, a_bytes), "clear after append");
    check("post-compaction append does not overwrite an earlier live extent",
          reopened.load(key, {{kv::PrefixComponent::kKda, da, a_bytes},
                              {kv::PrefixComponent::kDflash2, db, b_bytes}}));
    {

      std::fstream segment(dir / "segment-00000000.bin",
                           std::ios::in | std::ios::out | std::ios::binary);
      segment.seekg(65536);
      char byte = 0;
      segment.read(&byte, 1);
      byte ^= 0x5a;
      segment.seekp(65536);
      segment.write(&byte, 1);
      segment.flush();
    }
    check("payload corruption fails deterministic checksum validation",
          !reopened.load(key, {{kv::PrefixComponent::kKda, da, a_bytes},
                               {kv::PrefixComponent::kDflash2, db, b_bytes}}) &&
          reopened.stats().checksum_failures == 1);
    reopened.drop(key);
    check("drop removes the live index entry", !reopened.holds(key));
  }
  {
    kv::NvmePrefixStore reopened(options, stream);
    check("drop tombstone survives restart", !reopened.holds(key));
  }

  {
    const auto lru_dir = dir / "lru";
    auto lru_options = options;
    lru_options.directory = lru_dir;
    lru_options.capacity_bytes = 8ull << 20;
    lru_options.segment_bytes = 4ull << 20;
    lru_options.staging_bytes = 1ull << 20;
    kv::NvmePrefixStore lru(lru_options, stream);
    const kv::PrefixRecordKey k1{1, 0, 1, 1, 128, 1, 0};
    const kv::PrefixRecordKey k2{1, 1, 2, 1, 256, 1, 0};
    const kv::PrefixRecordKey k3{1, 0, 3, 1, 128, 1, 0};
    const std::size_t small = 2ull << 20;
    lru.save(k1, {{kv::PrefixComponent::kKda, da, small}});
    lru.save(k2, {{kv::PrefixComponent::kKda, da, small}});
    check("an access refreshes LRU recency",
          lru.load(k1, {{kv::PrefixComponent::kKda, db, small}}));
    lru.save(k3, {{kv::PrefixComponent::kKda, da, small}});
    check("capacity rotation preserves an ancestor and evicts its leaf",
          lru.holds(k1) && !lru.holds(k2) && lru.holds(k3));
  }

  {
    const auto trunc_dir = dir / "truncated";
    auto trunc_options = options;
    trunc_options.directory = trunc_dir;
    trunc_options.capacity_bytes = 8ull << 20;
    trunc_options.segment_bytes = 4ull << 20;
    trunc_options.staging_bytes = 1ull << 20;
    trunc_options.manifest_compact_bytes = 16ull << 20;
    const kv::PrefixRecordKey first{7, 0, 1, 1, 128, 1, 0};
    const kv::PrefixRecordKey torn{7, 1, 2, 1, 256, 1, 0};
    {
      kv::NvmePrefixStore trunc(trunc_options, stream);
      trunc.save(first, {{kv::PrefixComponent::kKda, da, 65536}});
      trunc.save(torn, {{kv::PrefixComponent::kKda, da, 65536}});
    }
    std::filesystem::resize_file(trunc_dir / "segment-00000000.bin", 3 * 65536);
    kv::NvmePrefixStore recovered(trunc_options, stream);
    check("truncated committed tail is ignored on restart",
          recovered.holds(first) && !recovered.holds(torn));
  }

  const auto real_parent = dir / "real-parent";
  std::filesystem::create_directories(real_parent);
  const auto linked_parent = dir / "linked-parent";
  std::filesystem::create_directory_symlink(real_parent, linked_parent);
  bool symlink_rejected = false;
  try {
    auto linked = options;
    linked.directory = linked_parent / "cache";
    kv::NvmePrefixStore rejected(linked, stream);
  } catch (...) { symlink_rejected = true; }
  check("symlinked cache path ancestor is rejected", symlink_rejected);

  bool staging_rejected = false;
  try {
    auto excessive = options;
    excessive.staging_bytes = 128ull * 1024 * 1024 + 65536;
    kv::NvmePrefixStore rejected(excessive, stream);
  } catch (...) { staging_rejected = true; }
  check("staging above 128 MiB is rejected", staging_rejected);

  bool headroom_rejected = false;
  try {
    auto impossible = options;
    impossible.capacity_bytes = kv::NvmePrefixStore::available_bytes(dir);
    impossible.free_space_headroom_bytes = 1;
    kv::NvmePrefixStore rejected(impossible, stream);
  } catch (...) { headroom_rejected = true; }
  check("capacity that consumes safety headroom is rejected", headroom_rejected);

  cudaFree(da);
  cudaFree(db);
  cudaStreamDestroy(stream);
  std::filesystem::remove_all(dir);
  std::printf("%s\n", failures ? "FAILED" : "all ok");
  return failures ? 1 : 0;
}

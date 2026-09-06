// Device side of the paged KV cache: the page slabs, the per-stream block
// table the kernels already read through kv_locate(), the page copy that
// backs copy on extend, and the seam that takes a detached stream's KDA
// state off the device.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kernels.h"
#include "kv/page_pool.h"

namespace rocket::engine::kv {

// Owns latent/key/gate for `num_pages` pages and the [max_streams][max_pages]
// block table. The slab layout is exactly what kernels.cu indexes today:
//   latent[page][layers][page_tokens][kv_lora]
// so nothing in the gather or the indexer scan changes shape; only max_pages
// stops being 1.
class KvArena : public PageStore {
 public:
  KvArena(const KvGeometry& geom, int num_pages, int max_streams, int max_pages_per_stream,
          cudaStream_t stream);
  ~KvArena();
  KvArena(const KvArena&) = delete;
  KvArena& operator=(const KvArena&) = delete;

  void copy_page(int dst, int src) override;

  // Writes one stream slot's block table. `pages` is the sequence's logical
  // page list; entries above its length are filled with `pages.back()` (or 0
  // when empty) rather than left stale, so a kernel that rounds a launch up
  // can never dereference a page another sequence owns.
  void upload_table(int slot, const std::vector<int>& pages);

  const KvPages& pages() const { return kv_; }
  std::size_t bytes() const { return bytes_; }
  int max_pages_per_stream() const { return max_pages_per_stream_; }
  int num_pages() const { return num_pages_; }

  // Host-visible copy of one slot's uploaded table row, for tests.
  std::vector<int> read_table(int slot) const;

 private:
  KvGeometry geom_;
  KvPages kv_;
  int num_pages_ = 0;
  int max_streams_ = 0;
  int max_pages_per_stream_ = 0;
  cudaStream_t stream_ = nullptr;
  std::size_t bytes_ = 0;
  std::vector<void*> owned_;
};

// KDA recurrent + conv state is per stream, 72.8 MiB at BF16 for this fuel
// (fuels/glm-5.3-flash/attention.yaml, bytes.kda_state_per_stream), and is
// not shareable: it is a recurrent hidden state, so a fork has to copy it and
// a detach has to carry it off the device with the slot.
//
// This is the seam. The host implementation below is what detach/resume use
// today; the NVMe tier that replaces it is a second implementation of the
// same three calls and is owned by the restore-path lane.
class KdaStateStore {
 public:
  virtual ~KdaStateStore() = default;
  // `src` and `dst` are device pointers; `bytes` is one stream's slice.
  virtual void save(int session, const void* src, std::size_t bytes) = 0;
  virtual void load(int session, void* dst, std::size_t bytes) = 0;
  virtual void drop(int session) = 0;
  virtual bool holds(int session) const = 0;
  virtual std::size_t resident_bytes() const = 0;
};

// Pinned host memory. Sized by what is detached, not preallocated.
class HostKdaStateStore : public KdaStateStore {
 public:
  explicit HostKdaStateStore(cudaStream_t stream);
  ~HostKdaStateStore() override;

  void save(int session, const void* src, std::size_t bytes) override;
  void load(int session, void* dst, std::size_t bytes) override;
  void drop(int session) override;
  bool holds(int session) const override;
  std::size_t resident_bytes() const override { return resident_; }

 private:
  struct Slab {
    void* host = nullptr;
    std::size_t bytes = 0;
  };
  cudaStream_t stream_ = nullptr;
  std::unordered_map<int, Slab> slabs_;
  std::size_t resident_ = 0;
};

}  // namespace rocket::engine::kv

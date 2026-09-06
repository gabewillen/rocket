#include "kv/kv_arena.h"

#include <stdexcept>
#include <string>

namespace rocket::engine::kv {
namespace {

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("kv: ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

KvArena::KvArena(const KvGeometry& geom, int num_pages, int max_streams, int max_pages_per_stream,
                 cudaStream_t stream)
    : geom_(geom),
      num_pages_(num_pages),
      max_streams_(max_streams),
      max_pages_per_stream_(max_pages_per_stream),
      stream_(stream) {
  const std::string bad = geom.why_invalid();
  if (!bad.empty()) throw std::runtime_error("kv: " + bad);
  if (num_pages <= 0 || max_streams <= 0 || max_pages_per_stream <= 0)
    throw std::runtime_error("kv: arena dimensions must be positive");

  auto alloc = [&](std::size_t bytes) {
    void* p = nullptr;
    cuda_check(cudaMalloc(&p, bytes), "cudaMalloc kv arena");
    cuda_check(cudaMemset(p, 0, bytes), "cudaMemset kv arena");
    owned_.push_back(p);
    bytes_ += bytes;
    return p;
  };

  const std::size_t page_slots =
      static_cast<std::size_t>(num_pages) * geom.layers * geom.page_tokens;
  kv_.latent = static_cast<bf16*>(alloc(page_slots * geom.kv_lora * sizeof(bf16)));
  kv_.key = static_cast<bf16*>(alloc(page_slots * geom.index_head_dim * sizeof(bf16)));
  kv_.gate = static_cast<bf16*>(alloc(page_slots * geom.index_head_dim * sizeof(bf16)));
  kv_.table = static_cast<int*>(
      alloc(static_cast<std::size_t>(max_streams) * max_pages_per_stream * sizeof(int)));
  kv_.max_pages = max_pages_per_stream;
  kv_.page_tokens = geom.page_tokens;
  kv_.layers = geom.layers;
  kv_.kv_lora = geom.kv_lora;
  kv_.index_head_dim = geom.index_head_dim;
}

KvArena::~KvArena() {
  for (void* p : owned_) cudaFree(p);
}

void KvArena::copy_page(int dst, int src) {
  if (dst < 0 || dst >= num_pages_ || src < 0 || src >= num_pages_)
    throw std::runtime_error("kv: copy_page out of range");
  if (dst == src) return;
  const std::size_t slots = static_cast<std::size_t>(geom_.layers) * geom_.page_tokens;
  const std::size_t lat = slots * geom_.kv_lora;
  const std::size_t idx = slots * geom_.index_head_dim;
  cuda_check(cudaMemcpyAsync(kv_.latent + static_cast<std::size_t>(dst) * lat,
                             kv_.latent + static_cast<std::size_t>(src) * lat,
                             lat * sizeof(bf16), cudaMemcpyDeviceToDevice, stream_),
             "copy_page latent");
  cuda_check(cudaMemcpyAsync(kv_.key + static_cast<std::size_t>(dst) * idx,
                             kv_.key + static_cast<std::size_t>(src) * idx,
                             idx * sizeof(bf16), cudaMemcpyDeviceToDevice, stream_),
             "copy_page key");
  cuda_check(cudaMemcpyAsync(kv_.gate + static_cast<std::size_t>(dst) * idx,
                             kv_.gate + static_cast<std::size_t>(src) * idx,
                             idx * sizeof(bf16), cudaMemcpyDeviceToDevice, stream_),
             "copy_page gate");
}

void KvArena::upload_table(int slot, const std::vector<int>& pages) {
  if (slot < 0 || slot >= max_streams_) throw std::runtime_error("kv: upload_table slot range");
  if (static_cast<int>(pages.size()) > max_pages_per_stream_)
    throw std::runtime_error("kv: sequence has more pages than the table row holds");
  std::vector<int> row(max_pages_per_stream_, pages.empty() ? 0 : pages.back());
  for (std::size_t i = 0; i < pages.size(); ++i) row[i] = pages[i];
  cuda_check(cudaMemcpyAsync(const_cast<int*>(kv_.table) +
                                 static_cast<std::size_t>(slot) * max_pages_per_stream_,
                             row.data(), row.size() * sizeof(int), cudaMemcpyHostToDevice, stream_),
             "upload_table");
  cuda_check(cudaStreamSynchronize(stream_), "upload_table sync");
}

std::vector<int> KvArena::read_table(int slot) const {
  std::vector<int> row(max_pages_per_stream_);
  cuda_check(cudaMemcpy(row.data(),
                        kv_.table + static_cast<std::size_t>(slot) * max_pages_per_stream_,
                        row.size() * sizeof(int), cudaMemcpyDeviceToHost),
             "read_table");
  return row;
}

// ------------------------------------------------------------- KDA state

HostKdaStateStore::HostKdaStateStore(cudaStream_t stream) : stream_(stream) {}

HostKdaStateStore::~HostKdaStateStore() {
  for (auto& kv : slabs_) cudaFreeHost(kv.second.host);
}

void HostKdaStateStore::save(int session, const void* src, std::size_t bytes) {
  Slab& s = slabs_[session];
  if (s.bytes != bytes) {
    if (s.host) {
      cudaFreeHost(s.host);
      resident_ -= s.bytes;
    }
    cuda_check(cudaHostAlloc(&s.host, bytes, cudaHostAllocDefault), "kda save alloc");
    s.bytes = bytes;
    resident_ += bytes;
  }
  cuda_check(cudaMemcpyAsync(s.host, src, bytes, cudaMemcpyDeviceToHost, stream_), "kda save");
  cuda_check(cudaStreamSynchronize(stream_), "kda save sync");
}

void HostKdaStateStore::load(int session, void* dst, std::size_t bytes) {
  const auto it = slabs_.find(session);
  if (it == slabs_.end()) throw std::runtime_error("kv: kda load for a session never saved");
  if (it->second.bytes != bytes) throw std::runtime_error("kv: kda load size mismatch");
  cuda_check(cudaMemcpyAsync(dst, it->second.host, bytes, cudaMemcpyHostToDevice, stream_),
             "kda load");
  cuda_check(cudaStreamSynchronize(stream_), "kda load sync");
}

void HostKdaStateStore::drop(int session) {
  const auto it = slabs_.find(session);
  if (it == slabs_.end()) return;
  cudaFreeHost(it->second.host);
  resident_ -= it->second.bytes;
  slabs_.erase(it);
}

bool HostKdaStateStore::holds(int session) const { return slabs_.count(session) != 0; }

}  // namespace rocket::engine::kv

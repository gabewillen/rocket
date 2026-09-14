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
  // Consumer contract, not enforced here: the owner of `stream_` must drain
  // (cudaStreamSynchronize) before the arena is destroyed. Freeing a slab a
  // deferred copy_page/upload_table still addresses is UB. A sync inside this
  // dtor was tried and segfaults inside cuStreamSynchronize on this GB10
  // test-build stack (the rest of the test suite avoids stream syncs for
  // exactly this reason), so the drain lives with the engine lifecycle instead.
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

std::vector<std::uint8_t> KvArena::read_page(int page) const {
  if (page < 0 || page >= num_pages_) throw std::runtime_error("kv: read_page range");
  const std::size_t slots = static_cast<std::size_t>(geom_.layers) * geom_.page_tokens;
  const std::size_t lat_bytes = slots * geom_.kv_lora * sizeof(bf16);
  const std::size_t idx_bytes = slots * geom_.index_head_dim * sizeof(bf16);
  std::vector<std::uint8_t> out(lat_bytes + 2 * idx_bytes);
  cuda_check(cudaMemcpy(out.data(), kv_.latent + static_cast<std::size_t>(page) * slots * geom_.kv_lora,
                        lat_bytes, cudaMemcpyDeviceToHost), "read_page latent");
  cuda_check(cudaMemcpy(out.data() + lat_bytes,
                        kv_.key + static_cast<std::size_t>(page) * slots * geom_.index_head_dim,
                        idx_bytes, cudaMemcpyDeviceToHost), "read_page key");
  cuda_check(cudaMemcpy(out.data() + lat_bytes + idx_bytes,
                        kv_.gate + static_cast<std::size_t>(page) * slots * geom_.index_head_dim,
                        idx_bytes, cudaMemcpyDeviceToHost), "read_page gate");
  return out;
}

// --------------------------------------------------------- NVMe page backing

NvmeArenaPageBacking::NvmeArenaPageBacking(KvArena* arena, const KvGeometry& geom,
                                           NvmePrefixOptions options,
                                           std::uint64_t namespace_hash,
                                           cudaStream_t stream)
    : arena_(arena), geom_(geom), namespace_hash_(namespace_hash),
      store_(std::move(options), stream) {
  if (!arena_) throw std::runtime_error("kv: NVMe page backing needs an arena");
}

PrefixRecordKey NvmeArenaPageBacking::key(std::uint64_t parent_hash,
                                          std::uint64_t chain_hash,
                                          int token_count) const {
  return PrefixRecordKey{namespace_hash_, parent_hash, chain_hash,
                         static_cast<std::uint32_t>(store_.options().rank),
                         static_cast<std::uint32_t>(token_count),
                         /*record_kind=*/1, 0};
}

bool NvmeArenaPageBacking::holds_page(std::uint64_t parent_hash,
                                      std::uint64_t chain_hash,
                                      int token_count) const {
  const bool hit = store_.holds(key(parent_hash, chain_hash, token_count));
  store_.note_page_lookup(hit);
  return hit;
}

bool NvmeArenaPageBacking::persist_page(std::uint64_t parent_hash,
                                        std::uint64_t chain_hash,
                                        int token_count, int page) {
  try {
    if (holds_page(parent_hash, chain_hash, token_count)) return true;
    const std::size_t slots = static_cast<std::size_t>(geom_.layers) * geom_.page_tokens;
    const std::size_t lat = slots * geom_.kv_lora;
    const std::size_t idx = slots * geom_.index_head_dim;
    const KvPages& p = arena_->pages();
    store_.save(key(parent_hash, chain_hash, token_count), {
      {PrefixComponent::kTargetLatent, p.latent + static_cast<std::size_t>(page) * lat,
       lat * sizeof(bf16)},
      {PrefixComponent::kTargetIndexerKey, p.key + static_cast<std::size_t>(page) * idx,
       idx * sizeof(bf16)},
      {PrefixComponent::kTargetIndexerGate, p.gate + static_cast<std::size_t>(page) * idx,
       idx * sizeof(bf16)},
    });
    return true;
  } catch (...) {
    return false;
  }
}

bool NvmeArenaPageBacking::restore_page(std::uint64_t parent_hash,
                                        std::uint64_t chain_hash,
                                        int token_count, int page) {
  try {
    const std::size_t slots = static_cast<std::size_t>(geom_.layers) * geom_.page_tokens;
    const std::size_t lat = slots * geom_.kv_lora;
    const std::size_t idx = slots * geom_.index_head_dim;
    const KvPages& p = arena_->pages();
    return store_.load(key(parent_hash, chain_hash, token_count), {
      {PrefixComponent::kTargetLatent, p.latent + static_cast<std::size_t>(page) * lat,
       lat * sizeof(bf16)},
      {PrefixComponent::kTargetIndexerKey, const_cast<bf16*>(p.key) + static_cast<std::size_t>(page) * idx,
       idx * sizeof(bf16)},
      {PrefixComponent::kTargetIndexerGate, const_cast<bf16*>(p.gate) + static_cast<std::size_t>(page) * idx,
       idx * sizeof(bf16)},
    });
  } catch (...) {
    return false;
  }
}

// ------------------------------------------------------------- KDA state

NvmeKdaStateStore::NvmeKdaStateStore(NvmePrefixStore* store, std::uint64_t namespace_hash)
    : namespace_hash_(namespace_hash), store_(store) {
  if (!store_) throw std::runtime_error("kv: NVMe KDA store needs a prefix store");
}

PrefixRecordKey NvmeKdaStateStore::key(int session) const {
  return PrefixRecordKey{namespace_hash_, 0, static_cast<std::uint64_t>(session),
                         static_cast<std::uint32_t>(store_->options().rank), 0,
                         /*record_kind=*/2, 0};
}

void NvmeKdaStateStore::save(int session, const void* src, std::size_t bytes) {
  store_->save(key(session), {{PrefixComponent::kKda, src, bytes}});
  sizes_[session] = bytes;
}

void NvmeKdaStateStore::load(int session, void* dst, std::size_t bytes) {
  const auto it = sizes_.find(session);
  if (it == sizes_.end() || it->second != bytes ||
      !store_->load(key(session), {{PrefixComponent::kKda, dst, bytes}}))
    throw std::runtime_error("kv: NVMe KDA state missing or invalid");
}

void NvmeKdaStateStore::drop(int session) {
  store_->drop(key(session));
  sizes_.erase(session);
}

bool NvmeKdaStateStore::holds(int session) const {
  return sizes_.count(session) && store_->holds(key(session));
}

void NvmeKdaStateStore::save_prefix(std::uint64_t parent_hash,
                                    std::uint64_t chain_hash, int token_count,
                                    const void* src, std::size_t bytes) {
  const PrefixRecordKey k{namespace_hash_, parent_hash, chain_hash,
                          static_cast<std::uint32_t>(store_->options().rank),
                          static_cast<std::uint32_t>(token_count),
                          /*record_kind=*/2, 0};
  store_->save(k, {{PrefixComponent::kKda, src, bytes}});
}

bool NvmeKdaStateStore::load_prefix(std::uint64_t parent_hash,
                                    std::uint64_t chain_hash, int token_count,
                                    void* dst, std::size_t bytes) {
  const PrefixRecordKey k{namespace_hash_, parent_hash, chain_hash,
                          static_cast<std::uint32_t>(store_->options().rank),
                          static_cast<std::uint32_t>(token_count),
                          /*record_kind=*/2, 0};
  return store_->load(k, {{PrefixComponent::kKda, dst, bytes}});
}

bool NvmeKdaStateStore::holds_prefix(std::uint64_t parent_hash,
                                     std::uint64_t chain_hash,
                                     int token_count) const {
  const PrefixRecordKey k{namespace_hash_, parent_hash, chain_hash,
                          static_cast<std::uint32_t>(store_->options().rank),
                          static_cast<std::uint32_t>(token_count),
                          /*record_kind=*/2, 0};
  return store_->holds(k);
}

// ------------------------------------------------------------- host KDA state

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

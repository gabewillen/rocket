// Refcounted fixed-size KV pages, a page-granularity radix tree over token
// sequences, and the sequence manager that forks, detaches, resumes, and
// destroys streams on top of them.
//
// Stage 1 gave every stream slot one physical page sized to the whole context
// (model.cu, KvPages with max_pages = 1). The kernels were already written
// against a block table: every KV-touching kernel resolves a logical position
// through kv_locate(kv, m, pos) -> (physical page, slot in page), so the
// gather and the indexer scan already iterate through a page table. What was
// missing is a pool behind that table with more than one page per stream, and
// the host bookkeeping that lets two streams name the same physical page.
//
// Sharing model. A page is shared only when two sequences literally point at
// the same bytes, which happens exactly one way: fork(parent, pos) hands the
// child the parent's full pages below `pos` and increments their refcounts.
// The boundary page, partially filled at the fork point, is copied the first
// time either side extends into it (copy on extend). Content-addressed reuse
// across sequences that were computed independently is a separate, opt-in
// entry point; see open_shared().
//
// Everything here is host bookkeeping. It runs once per appended token (an
// index computation) and once per sealed page (a hash over page_tokens ids),
// never per layer and never per gathered token. Device work is confined to
// the PageStore interface: one page copy on extend, and the table upload.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocket::engine::kv {

// Byte geometry of one page, in the layout kernels.cu already reads:
//   latent[page][layers][page_tokens][kv_lora]
//   key   [page][layers][page_tokens][index_head_dim]
//   gate  [page][layers][page_tokens][index_head_dim]
// so one page carries every MLA layer's slice of the same token range and a
// single refcount covers all 11 layers at once.
struct KvGeometry {
  int page_tokens = 0;
  int layers = 0;           // MLA layer count, 11 for this fuel
  int kv_lora = 0;          // 512
  int index_head_dim = 0;   // 128
  int kpool = 0;            // 4, indexer pool width

  // Bytes of latent + key + gate for one page, BF16 throughout.
  std::size_t page_bytes() const {
    const std::size_t per_token = static_cast<std::size_t>(kv_lora) * 2 +
                                  static_cast<std::size_t>(index_head_dim) * 2 * 2;
    return per_token * static_cast<std::size_t>(page_tokens) *
           static_cast<std::size_t>(layers);
  }
  std::size_t bytes_per_token() const { return page_bytes() / static_cast<std::size_t>(page_tokens); }

  // Two constraints on page_tokens, both structural rather than tuning:
  //  - a multiple of kpool, because indexer_pool_kernel reads the kpool
  //    tokens of a pool as one run and a pool must not straddle two pages;
  //  - a multiple of 128, because 16896 B/token * 128 is 2 MiB and every
  //    smaller multiple leaves a page slab unaligned to the platform's
  //    64 KiB quantum (blog/rocket.qmd#alignment).
  bool valid() const;
  std::string why_invalid() const;
};

// Device-side operations the sequence manager needs. Implemented by KvArena
// (kv_arena.h) against the real slabs; a test can substitute a host double.
class PageStore {
 public:
  virtual ~PageStore() = default;
  // Byte copy of every layer's slice of page `src` into page `dst`.
  virtual void copy_page(int dst, int src) = 0;
};

// Physical page allocator. Pages carry a refcount; a page whose refcount
// reaches zero is not returned to the free list, it moves to the tail of an
// LRU of evictable pages so a later sequence with the same prefix can still
// find it in the tree. Reclaiming one is an explicit act.
class PagePool {
 public:
  PagePool(int num_pages, int page_tokens);

  int num_pages() const { return num_pages_; }
  int page_tokens() const { return page_tokens_; }

  // A page with refcount 1, taken from the free list. -1 when the free list
  // is empty; the caller decides whether to reclaim_lru() and try again.
  int allocate();
  void incref(int page);
  // Returns true when the refcount reached zero and the page became
  // evictable. Never frees: the page keeps its contents and its tree node.
  bool decref(int page);
  int refcount(int page) const;

  // Refuses (returns false) while the page is referenced by any sequence.
  // On success the page leaves the evictable LRU and rejoins the free list;
  // the caller must detach its tree node first, or the tree will name a page
  // whose contents have been handed to someone else.
  bool evict(int page);
  // Least recently released evictable page, or -1. Does not evict it.
  int lru_victim() const;

  int free_pages() const { return static_cast<int>(free_.size()); }
  int evictable_pages() const { return static_cast<int>(lru_size_); }
  int pinned_pages() const { return num_pages_ - free_pages() - evictable_pages(); }

 private:
  void drop_from_lru(int page);

  int num_pages_ = 0;
  int page_tokens_ = 0;
  std::vector<int> refcount_;
  std::vector<int> free_;
  // Intrusive doubly-linked LRU over page ids; -1 terminates.
  std::vector<int> lru_prev_, lru_next_;
  int lru_head_ = -1, lru_tail_ = -1;
  std::size_t lru_size_ = 0;
};

// Radix tree over token sequences at page granularity: one node per full
// page, the edge from a parent node keyed by a 64-bit hash of that page's
// token block. A sequence's full pages are exactly a root-to-node path, which
// is why fork sharing and prefix lookup are the same structure.
class PrefixTree {
 public:
  PrefixTree();

  static constexpr int kRoot = 0;
  int root() const { return kRoot; }

  // Child of `node` whose page block hashes to `h`, or -1.
  int find(int node, std::uint64_t h) const;
  // Adds `page` as a child of `node` under `h` and returns the new node.
  int insert(int node, std::uint64_t h, int page);
  // Removes `node` from its parent and recycles it. Only legal for a leaf,
  // which is what eviction produces: a page is evictable only when no
  // sequence holds it, and a node with children has a descendant page a
  // sequence still holds, or a cached descendant that must go first.
  void detach(int node);

  int page_of(int node) const { return nodes_[node].page; }
  int parent_of(int node) const { return nodes_[node].parent; }
  int child_count(int node) const { return static_cast<int>(nodes_[node].children.size()); }
  int node_of_page(int page) const;
  int live_nodes() const { return live_nodes_; }

  // Hash of one page-sized block of token ids. FNV-1a over the ids, seeded
  // with the parent node's own hash so the same block under a different
  // prefix is a different edge.
  static std::uint64_t block_hash(std::uint64_t seed, const int* tokens, int n);
  std::uint64_t hash_of(int node) const { return nodes_[node].hash; }

 private:
  struct Node {
    int parent = -1;
    int page = -1;
    std::uint64_t hash = 0;
    std::unordered_map<std::uint64_t, int> children;
  };
  std::vector<Node> nodes_;
  std::vector<int> free_nodes_;
  std::unordered_map<int, int> node_of_page_;
  int live_nodes_ = 0;
};

// What append_token() decided, so the caller knows where to write this
// step's latent/key/gate and whether the page table changed.
struct AppendSite {
  int page = -1;         // physical page the write lands in
  int slot = -1;         // token slot inside that page
  int logical_page = -1; // index into the sequence's page table
  bool grew_table = false;   // a new page was appended to the table
  bool copied_on_extend = false;  // a shared partial page was copied first
  bool sealed_page = false;  // this token filled `logical_page`
};

// One sequence's view of the pool. `slot` is the engine stream slot it
// currently occupies, or -1 while detached.
struct SeqInfo {
  int length = 0;
  int slot = -1;
  int pages = 0;
  bool live = false;
};

// Sequence manager: the object model.cu holds instead of a table of identity.
class KvCache {
 public:
  // `store` may be null only when no copy on extend can happen, which is not
  // a case worth special-casing; pass the arena.
  KvCache(const KvGeometry& geom, PagePool* pool, PrefixTree* tree, PageStore* store);

  // A fresh empty sequence bound to stream slot `slot` (-1 for detached).
  int open(int slot);
  // A fresh sequence that adopts whatever prefix of `tokens` the tree already
  // holds, then continues as a normal sequence. This is content-addressed
  // reuse across sequences that were computed independently, and it is a
  // separate entry point on purpose: it is exact only if identical token
  // prefixes produce identical KV bytes, which holds for every kernel in this
  // engine except the grouped-GEMM routed-expert path, whose row grouping
  // depends on the batch it ran in (moe_grouped.h). fork() carries no such
  // condition because it shares the bytes themselves.
  int open_shared(int slot, const int* tokens, int n_tokens, int* matched_tokens_out);

  // Child sharing every full page of `parent` below `fork_pos`. The boundary
  // page is shared too and copied by whichever side extends into it first.
  // Returns -1 when the pool cannot back the child's page table.
  int fork(int parent, int fork_pos, int slot);

  // Reserves the site for one more token and returns where to write it.
  // `page` is -1 when the pool is exhausted; nothing is mutated in that case.
  AppendSite append_token(int seq, int token_id);

  // Frees the stream slot, keeps every page. The caller is responsible for
  // the KDA state (KdaStateStore, kv_arena.h): it is per stream, 72.8 MiB,
  // and not shareable, so it has to leave the device with the slot.
  void detach(int seq);
  void attach(int seq, int slot);

  // Drops this sequence's claim on its pages. Pages whose refcount reaches
  // zero stay in the tree as cache until something reclaims them.
  void destroy(int seq);

  const std::vector<int>& page_table(int seq) const;
  const std::vector<int>& tokens(int seq) const;
  SeqInfo info(int seq) const;
  int slot_of(int seq) const;
  int seq_in_slot(int slot) const;

  // Physical pages currently referenced by at least one live sequence.
  int pinned_pages() const { return pool_->pinned_pages(); }
  const KvGeometry& geometry() const { return geom_; }

 private:
  int take_page();
  void seal(int seq, int logical_page);

  struct Seq {
    std::vector<int> pages;   // logical page -> physical page
    std::vector<int> nodes;   // logical page -> tree node, -1 while unsealed
    std::vector<int> tokens;  // full token history, for hashing and forks
    int length = 0;
    int slot = -1;
    bool live = false;
  };

  KvGeometry geom_;
  PagePool* pool_ = nullptr;
  PrefixTree* tree_ = nullptr;
  PageStore* store_ = nullptr;
  std::vector<Seq> seqs_;
  std::vector<int> free_seqs_;
  std::unordered_map<int, int> seq_of_slot_;
};

}  // namespace rocket::engine::kv

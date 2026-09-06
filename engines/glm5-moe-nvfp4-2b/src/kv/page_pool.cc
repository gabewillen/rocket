#include "kv/page_pool.h"

#include <algorithm>
#include <stdexcept>

namespace rocket::engine::kv {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("kv: " + what); }

}  // namespace

// ------------------------------------------------------------------ geometry

bool KvGeometry::valid() const { return why_invalid().empty(); }

std::string KvGeometry::why_invalid() const {
  if (page_tokens <= 0) return "page_tokens must be positive";
  if (layers <= 0) return "layers must be positive";
  if (kv_lora <= 0) return "kv_lora must be positive";
  if (index_head_dim <= 0) return "index_head_dim must be positive";
  if (kpool <= 0) return "kpool must be positive";
  if (page_tokens % kpool != 0)
    return "page_tokens must be a multiple of kpool so an indexer pool never straddles a page";
  if (page_bytes() % 65536 != 0)
    return "page slab must be a multiple of the platform's 64 KiB quantum";
  return {};
}

// ------------------------------------------------------------------ PagePool

PagePool::PagePool(int num_pages, int page_tokens)
    : num_pages_(num_pages), page_tokens_(page_tokens) {
  if (num_pages <= 0) fail("num_pages must be positive");
  if (page_tokens <= 0) fail("page_tokens must be positive");
  refcount_.assign(num_pages, 0);
  lru_prev_.assign(num_pages, -1);
  lru_next_.assign(num_pages, -1);
  free_.reserve(num_pages);
  for (int p = num_pages - 1; p >= 0; --p) free_.push_back(p);
}

int PagePool::allocate() {
  if (free_.empty()) return -1;
  const int p = free_.back();
  free_.pop_back();
  refcount_[p] = 1;
  return p;
}

void PagePool::incref(int page) {
  if (page < 0 || page >= num_pages_) fail("incref: page out of range");
  if (refcount_[page] == 0) drop_from_lru(page);
  ++refcount_[page];
}

bool PagePool::decref(int page) {
  if (page < 0 || page >= num_pages_) fail("decref: page out of range");
  if (refcount_[page] <= 0) fail("decref: page is already unreferenced");
  if (--refcount_[page] > 0) return false;
  // Newly evictable: append at the tail, so lru_head_ is the least recently
  // released page and is what reclaim takes first.
  lru_prev_[page] = lru_tail_;
  lru_next_[page] = -1;
  if (lru_tail_ >= 0) lru_next_[lru_tail_] = page;
  lru_tail_ = page;
  if (lru_head_ < 0) lru_head_ = page;
  ++lru_size_;
  return true;
}

int PagePool::refcount(int page) const {
  if (page < 0 || page >= num_pages_) fail("refcount: page out of range");
  return refcount_[page];
}

int PagePool::lru_victim() const { return lru_head_; }

bool PagePool::evict(int page) {
  if (page < 0 || page >= num_pages_) fail("evict: page out of range");
  if (refcount_[page] > 0) return false;
  // Not on the LRU means never allocated: it is already free.
  if (lru_head_ != page && lru_prev_[page] < 0 && lru_next_[page] < 0 && lru_tail_ != page)
    return false;
  drop_from_lru(page);
  free_.push_back(page);
  return true;
}

void PagePool::drop_from_lru(int page) {
  const int prev = lru_prev_[page], next = lru_next_[page];
  if (prev >= 0) lru_next_[prev] = next; else if (lru_head_ == page) lru_head_ = next;
  if (next >= 0) lru_prev_[next] = prev; else if (lru_tail_ == page) lru_tail_ = prev;
  lru_prev_[page] = -1;
  lru_next_[page] = -1;
  if (lru_size_ > 0) --lru_size_;
}

// ---------------------------------------------------------------- PrefixTree

PrefixTree::PrefixTree() {
  nodes_.push_back(Node{});  // root, page -1
  live_nodes_ = 0;
}

std::uint64_t PrefixTree::block_hash(std::uint64_t seed, const int* tokens, int n) {
  std::uint64_t h = seed ^ 0xcbf29ce484222325ull;
  for (int i = 0; i < n; ++i) {
    h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(tokens[i]));
    h *= 0x100000001b3ull;
  }
  return h;
}

int PrefixTree::find(int node, std::uint64_t h) const {
  const auto& kids = nodes_[node].children;
  const auto it = kids.find(h);
  return it == kids.end() ? -1 : it->second;
}

int PrefixTree::insert(int node, std::uint64_t h, int page) {
  if (find(node, h) >= 0) fail("insert: edge already present");
  int id;
  if (!free_nodes_.empty()) {
    id = free_nodes_.back();
    free_nodes_.pop_back();
    nodes_[id] = Node{};
  } else {
    id = static_cast<int>(nodes_.size());
    nodes_.push_back(Node{});
  }
  nodes_[id].parent = node;
  nodes_[id].page = page;
  nodes_[id].hash = h;
  nodes_[node].children.emplace(h, id);
  node_of_page_[page] = id;
  ++live_nodes_;
  return id;
}

void PrefixTree::detach(int node) {
  if (node == kRoot) fail("detach: the root has no page");
  if (!nodes_[node].children.empty()) fail("detach: node still has children");
  const int parent = nodes_[node].parent;
  nodes_[parent].children.erase(nodes_[node].hash);
  node_of_page_.erase(nodes_[node].page);
  nodes_[node] = Node{};
  free_nodes_.push_back(node);
  --live_nodes_;
}

int PrefixTree::node_of_page(int page) const {
  const auto it = node_of_page_.find(page);
  return it == node_of_page_.end() ? -1 : it->second;
}

// ------------------------------------------------------------------- KvCache

KvCache::KvCache(const KvGeometry& geom, PagePool* pool, PrefixTree* tree, PageStore* store)
    : geom_(geom), pool_(pool), tree_(tree), store_(store) {
  const std::string bad = geom.why_invalid();
  if (!bad.empty()) fail(bad);
  if (!pool || !tree) fail("KvCache needs a pool and a tree");
  if (pool->page_tokens() != geom.page_tokens) fail("pool and geometry disagree on page_tokens");
}

// A free page, reclaiming the least recently released cached page when the
// free list is empty. Reclaim detaches the tree node first, so the tree never
// names a page whose contents have been handed to another sequence.
int KvCache::take_page() {
  int p = pool_->allocate();
  if (p >= 0) return p;
  for (;;) {
    const int victim = pool_->lru_victim();
    if (victim < 0) return -1;
    const int node = tree_->node_of_page(victim);
    if (node >= 0) {
      if (tree_->child_count(node) > 0) {
        // An interior cached page: its descendants must go first. Rotate it
        // to the back of the LRU and try the next victim.
        pool_->incref(victim);
        pool_->decref(victim);
        continue;
      }
      tree_->detach(node);
    }
    if (!pool_->evict(victim)) fail("lru victim refused eviction");
    p = pool_->allocate();
    if (p >= 0) return p;
  }
}

int KvCache::open(int slot) {
  int id;
  if (!free_seqs_.empty()) {
    id = free_seqs_.back();
    free_seqs_.pop_back();
    seqs_[id] = Seq{};
  } else {
    id = static_cast<int>(seqs_.size());
    seqs_.push_back(Seq{});
  }
  seqs_[id].live = true;
  seqs_[id].slot = slot;
  if (slot >= 0) seq_of_slot_[slot] = id;
  return id;
}

int KvCache::open_shared(int slot, const int* tokens, int n_tokens, int* matched_tokens_out) {
  const int id = open(slot);
  Seq& s = seqs_[id];
  const int P = geom_.page_tokens;
  int node = tree_->root();
  std::uint64_t seed = 0;
  int matched = 0;
  while (matched + P <= n_tokens) {
    const std::uint64_t h = PrefixTree::block_hash(seed, tokens + matched, P);
    const int child = tree_->find(node, h);
    if (child < 0) break;
    const int page = tree_->page_of(child);
    pool_->incref(page);
    s.pages.push_back(page);
    s.nodes.push_back(child);
    s.tokens.insert(s.tokens.end(), tokens + matched, tokens + matched + P);
    s.length += P;
    node = child;
    seed = h;
    matched += P;
  }
  if (matched_tokens_out) *matched_tokens_out = matched;
  return id;
}

int KvCache::fork(int parent, int fork_pos, int slot) {
  if (parent < 0 || parent >= static_cast<int>(seqs_.size()) || !seqs_[parent].live)
    fail("fork: parent is not a live sequence");
  if (fork_pos < 0 || fork_pos > seqs_[parent].length)
    fail("fork: fork_pos outside the parent's sequence");

  const int P = geom_.page_tokens;
  const int n_pages = (fork_pos + P - 1) / P;
  const int n_full = fork_pos / P;

  // Snapshot what the child inherits before open(), which can grow seqs_ and
  // invalidate any reference into it.
  std::vector<int> pages(seqs_[parent].pages.begin(), seqs_[parent].pages.begin() + n_pages);
  std::vector<int> nodes(n_pages, -1);
  for (int i = 0; i < n_full; ++i) nodes[i] = seqs_[parent].nodes[i];
  std::vector<int> toks(seqs_[parent].tokens.begin(), seqs_[parent].tokens.begin() + fork_pos);

  const int id = open(slot);
  Seq& c = seqs_[id];
  for (const int page : pages) pool_->incref(page);
  c.pages = std::move(pages);
  c.nodes = std::move(nodes);
  c.tokens = std::move(toks);
  c.length = fork_pos;
  return id;
}

AppendSite KvCache::append_token(int seq, int token_id) {
  if (seq < 0 || seq >= static_cast<int>(seqs_.size()) || !seqs_[seq].live)
    fail("append_token: not a live sequence");
  Seq& s = seqs_[seq];
  const int P = geom_.page_tokens;
  const int li = s.length / P;
  const int off = s.length % P;

  AppendSite site;
  site.logical_page = li;
  site.slot = off;

  if (off == 0) {
    const int p = take_page();
    if (p < 0) return site;  // page == -1, nothing mutated
    s.pages.push_back(p);
    s.nodes.push_back(-1);
    site.page = p;
    site.grew_table = true;
  } else {
    int p = s.pages[li];
    if (pool_->refcount(p) > 1) {
      // Copy on extend: the partial boundary page a fork left shared. The
      // first writer takes a private copy; the other side keeps the original.
      const int q = take_page();
      if (q < 0) return site;
      if (!store_) fail("append_token: copy on extend needs a PageStore");
      store_->copy_page(q, p);
      pool_->decref(p);
      s.pages[li] = q;
      p = q;
      site.copied_on_extend = true;
    }
    site.page = p;
  }

  s.tokens.push_back(token_id);
  ++s.length;
  if (s.length % P == 0) {
    seal(seq, li);
    site.sealed_page = true;
  }
  return site;
}

// A page that just filled becomes a tree node, keyed by the hash of its token
// block seeded with its parent node's hash. The node makes the page findable
// by open_shared() and orders eviction: a page with cached descendants cannot
// be reclaimed before them.
void KvCache::seal(int seq, int logical_page) {
  Seq& s = seqs_[seq];
  const int P = geom_.page_tokens;
  const int parent_node = logical_page == 0 ? tree_->root() : s.nodes[logical_page - 1];
  if (parent_node < 0) return;  // a forked, still-unsealed ancestor: not indexable
  const std::uint64_t seed = parent_node == tree_->root() ? 0ull : tree_->hash_of(parent_node);
  const std::uint64_t h =
      PrefixTree::block_hash(seed, s.tokens.data() + static_cast<std::size_t>(logical_page) * P, P);
  const int existing = tree_->find(parent_node, h);
  if (existing >= 0) {
    // Another sequence already indexed an identical block under this prefix.
    // Keep our own page and leave the edge alone: merging would assume the
    // two pages hold identical bytes, which is only true if the KV was
    // computed batch-invariantly. open_shared() is where that assumption is
    // taken on purpose.
    return;
  }
  s.nodes[logical_page] = tree_->insert(parent_node, h, s.pages[logical_page]);
}

void KvCache::detach(int seq) {
  if (seq < 0 || seq >= static_cast<int>(seqs_.size()) || !seqs_[seq].live)
    fail("detach: not a live sequence");
  Seq& s = seqs_[seq];
  if (s.slot >= 0) seq_of_slot_.erase(s.slot);
  s.slot = -1;
}

void KvCache::attach(int seq, int slot) {
  if (seq < 0 || seq >= static_cast<int>(seqs_.size()) || !seqs_[seq].live)
    fail("attach: not a live sequence");
  if (slot < 0) fail("attach: slot must be non-negative");
  const auto it = seq_of_slot_.find(slot);
  if (it != seq_of_slot_.end() && it->second != seq) fail("attach: slot is already occupied");
  Seq& s = seqs_[seq];
  if (s.slot >= 0) seq_of_slot_.erase(s.slot);
  s.slot = slot;
  seq_of_slot_[slot] = seq;
}

void KvCache::destroy(int seq) {
  if (seq < 0 || seq >= static_cast<int>(seqs_.size()) || !seqs_[seq].live)
    fail("destroy: not a live sequence");
  Seq& s = seqs_[seq];
  for (const int p : s.pages) pool_->decref(p);
  if (s.slot >= 0) seq_of_slot_.erase(s.slot);
  s = Seq{};
  free_seqs_.push_back(seq);
}

const std::vector<int>& KvCache::page_table(int seq) const { return seqs_.at(seq).pages; }
const std::vector<int>& KvCache::tokens(int seq) const { return seqs_.at(seq).tokens; }

SeqInfo KvCache::info(int seq) const {
  const Seq& s = seqs_.at(seq);
  return SeqInfo{s.length, s.slot, static_cast<int>(s.pages.size()), s.live};
}

int KvCache::slot_of(int seq) const { return seqs_.at(seq).slot; }

int KvCache::seq_in_slot(int slot) const {
  const auto it = seq_of_slot_.find(slot);
  return it == seq_of_slot_.end() ? -1 : it->second;
}

}  // namespace rocket::engine::kv

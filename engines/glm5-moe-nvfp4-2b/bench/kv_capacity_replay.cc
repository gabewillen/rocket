// How many streams fit, from the page tables rather than from an average.
//
// scripts/scheduler/agent-workload.py already reports a with/without-sharing
// KV total, but it is cumulative over every session the trace ever created
// and it assumes a session's cost is exactly (context - fork_point) tokens.
// A real pool cannot do that: it allocates whole pages, a fork at an
// arbitrary position leaves a partial boundary page that copy on extend
// duplicates, and a page stays resident until its last referent lets go.
//
// This replays the trace through the actual PagePool / PrefixTree / KvCache
// and reports peak *resident* bytes, which is the number that decides how
// many streams a booster holds.
//
// The trace carries no session_end event, so a pre-pass records each
// session's last event and the replay destroys it there, which is the
// generator's own lifetime semantics. Appends are clamped at the serving cap
// (fuels/glm-5.3-flash/fuel.yaml serving_context_cap, 262144); the count of
// clamped tokens is reported so a trace that leans on the clamp is visible.
//
// Host only: no device memory is allocated, because only the bookkeeping is
// being measured. Copy on extend is counted, not performed.
//
// Usage:
//   bench/kv-capacity-replay trace.jsonl [--page-tokens 128] [--cap 262144]
//                                        [--headroom-gib 32.73] [--kda-bytes N]
//
// --kda-bytes defaults to attention.yaml's bf16 figure (76316672), which is
// what the published unshared stream count was computed from. This engine
// actually keeps the recurrent state in FP32 (model.h, `float* kda_state_`),
// so 147619840 is the number that describes it; both are worth reporting
// because only the first is comparable to the earlier entry.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.h"
#include "kv/page_pool.h"

namespace kv = rocket::engine::kv;
using rocket::fuel::json::JsonParser;
using rocket::fuel::json::JsonValue;

namespace {

// Copy on extend is device traffic in the engine; here it only needs
// counting, so the store records the copies instead of doing them.
class CountingStore : public kv::PageStore {
 public:
  void copy_page(int, int) override { ++copies; }
  long long copies = 0;
};

struct Event {
  std::string kind, session, parent;
  long long n_tokens = 0;
  long long fork_point = 0;
  long long t = 0;
};

std::string str_of(const JsonValue* v) {
  return v && v->kind == JsonValue::Kind::kString ? v->str : std::string();
}
long long num_of(const JsonValue* v) {
  return v && v->kind == JsonValue::Kind::kNumber ? (long long)v->number : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: kv-capacity-replay trace.jsonl [--page-tokens N] [--cap N]"
                         " [--headroom-gib G]\n");
    return 2;
  }
  const char* path = argv[1];
  int page_tokens = 128;
  long long cap = 262144;
  double headroom_gib = 32.73;  // 123.73 booster - 91 weights, attention.yaml
  long long kda_bytes = 76316672;  // attention.yaml bytes.kda_state_per_stream.total, bf16
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--page-tokens") && i + 1 < argc) page_tokens = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--cap") && i + 1 < argc) cap = std::atoll(argv[++i]);
    else if (!std::strcmp(argv[i], "--headroom-gib") && i + 1 < argc)
      headroom_gib = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--kda-bytes") && i + 1 < argc) kda_bytes = std::atoll(argv[++i]);
  }

  // Fuel geometry, fuels/glm-5.3-flash/attention.yaml.
  kv::KvGeometry g{page_tokens, 11, 512, 128, 4};
  const std::string bad = g.why_invalid();
  if (!bad.empty()) { std::fprintf(stderr, "geometry: %s\n", bad.c_str()); return 2; }
  const long long page_bytes = (long long)g.page_bytes();

  std::vector<Event> events;
  {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      JsonParser p(line.data(), line.data() + line.size());
      const JsonValue v = p.parse_document();
      const std::string kind = str_of(v.member("event"));
      if (kind.empty() || kind == "summary") continue;
      Event e;
      e.kind = kind;
      e.session = str_of(v.member("session"));
      e.parent = str_of(v.member("parent_id"));
      e.n_tokens = num_of(v.member("n_tokens"));
      e.fork_point = num_of(v.member("fork_point_tokens"));
      e.t = num_of(v.member("t"));
      events.push_back(std::move(e));
    }
  }

  // Pre-pass: where each session is last seen, so the replay can retire it.
  std::unordered_map<std::string, std::size_t> last_at;
  for (std::size_t i = 0; i < events.size(); ++i) last_at[events[i].session] = i;

  // A pool large enough that nothing is evicted: the question is peak demand,
  // not what an undersized pool would do about it.
  kv::PagePool pool(400000, page_tokens);
  kv::PrefixTree tree;
  CountingStore store;
  kv::KvCache cache(g, &pool, &tree, &store);

  std::unordered_map<std::string, int> seq_of;
  std::unordered_map<std::string, long long> len_of;
  std::unordered_map<std::string, bool> attached;
  int next_slot = 0;

  long long peak_shared = 0, peak_unshared = 0, peak_pages = 0;
  int peak_sessions = 0, peak_attached = 0;
  long long clamped = 0, appended = 0;
  long long peak_t = 0;

  auto append_n = [&](const std::string& sid, long long n) {
    const auto it = seq_of.find(sid);
    if (it == seq_of.end()) return;
    for (long long i = 0; i < n; ++i) {
      if (len_of[sid] >= cap) { ++clamped; continue; }
      const kv::AppendSite site = cache.append_token(it->second, (int)(len_of[sid] & 0x7fffffff));
      if (site.page < 0) { std::fprintf(stderr, "pool exhausted; raise the pool size\n"); std::exit(1); }
      ++len_of[sid];
      ++appended;
    }
  };

  auto observe = [&](long long t) {
    const long long pages = pool.pinned_pages();
    int n_attached = 0;
    long long unshared_pages = 0;
    for (const auto& kvp : seq_of) {
      if (attached[kvp.first]) ++n_attached;
      unshared_pages += (len_of[kvp.first] + page_tokens - 1) / page_tokens;
    }
    const long long shared = pages * page_bytes + (long long)n_attached * kda_bytes;
    const long long unshared = unshared_pages * page_bytes + (long long)n_attached * kda_bytes;
    if (shared > peak_shared) {
      peak_shared = shared;
      peak_unshared = unshared;
      peak_pages = pages;
      peak_sessions = (int)seq_of.size();
      peak_attached = n_attached;
      peak_t = t;
    }
  };

  for (std::size_t i = 0; i < events.size(); ++i) {
    const Event& e = events[i];
    if (e.kind == "session_start") {
      int seq;
      if (e.parent.empty() || !seq_of.count(e.parent)) {
        seq = cache.open(next_slot++);
        len_of[e.session] = 0;
      } else {
        const long long fp = std::min<long long>(e.fork_point, len_of[e.parent]);
        seq = cache.fork(seq_of[e.parent], (int)fp, next_slot++);
        len_of[e.session] = fp;
      }
      seq_of[e.session] = seq;
      attached[e.session] = true;
    } else if (e.kind == "prefill" || e.kind == "decode") {
      append_n(e.session, e.n_tokens);
    } else if (e.kind == "idle") {
      if (seq_of.count(e.session)) { cache.detach(seq_of[e.session]); attached[e.session] = false; }
    } else if (e.kind == "resume") {
      if (seq_of.count(e.session)) {
        cache.attach(seq_of[e.session], next_slot++);
        attached[e.session] = true;
      }
    }
    observe(e.t);
    if (last_at[e.session] == i && seq_of.count(e.session)) {
      cache.destroy(seq_of[e.session]);
      seq_of.erase(e.session);
      len_of.erase(e.session);
      attached.erase(e.session);
    }
  }

  const double gib = 1024.0 * 1024.0 * 1024.0;
  const double headroom = headroom_gib * gib;
  std::printf("trace                 %s\n", path);
  std::printf("page_tokens           %d  (%lld B/page, %.4f MiB)\n", page_tokens, page_bytes,
              page_bytes / (1024.0 * 1024.0));
  std::printf("kda per stream        %lld B (%.4f GiB)\n", kda_bytes, kda_bytes / gib);
  std::printf("serving cap           %lld tokens\n", cap);
  std::printf("tokens appended       %lld  (clamped at the cap: %lld)\n", appended, clamped);
  std::printf("copy-on-extend pages  %lld  (%.2f GiB of device copies over the whole trace)\n",
              store.copies, store.copies * page_bytes / gib);
  std::printf("\npeak resident, at tick %lld\n", peak_t);
  std::printf("  live sessions       %d  (%d attached to a stream slot)\n", peak_sessions,
              peak_attached);
  std::printf("  pages pinned        %lld\n", peak_pages);
  std::printf("  KV with sharing     %.2f GiB\n", peak_shared / gib);
  std::printf("  KV without sharing  %.2f GiB\n", peak_unshared / gib);
  std::printf("  sharing ratio       %.4f\n",
              peak_unshared ? (double)peak_shared / (double)peak_unshared : 0.0);
  std::printf("\nat %.2f GiB of KV headroom\n", headroom_gib);
  if (peak_sessions > 0) {
    const double shared_per = (double)peak_shared / peak_sessions;
    const double unshared_per = (double)peak_unshared / peak_sessions;
    std::printf("  bytes per session   %.4f GiB shared, %.4f GiB unshared\n", shared_per / gib,
                unshared_per / gib);
    std::printf("  sessions held       %d shared, %d unshared\n",
                (int)(headroom / shared_per), (int)(headroom / unshared_per));
  }
  std::printf("  peak fits           %s\n", peak_shared <= headroom ? "yes" : "no");
  return 0;
}

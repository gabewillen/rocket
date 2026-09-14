// Greedy decode on one booster: load the fuel, run the prompt through the
// step, then generate. Prints the text, the per-stage cost of one step, and
// the two validation signals that stand in for a reference engine here (the
// per-layer hidden RMS and the router entropy).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <numeric>
#include <optional>
#include <map>
#include <memory>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_profiler_api.h>

#include "dflash2.h"
#include "fabric/expert_balance.h"
#include "fabric/expert_parallel.h"
#include "model.h"
#include "model_config.h"
#include "tokenizer.h"

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

const char* arg_value(int argc, char** argv, const char* key, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return fallback;
}

std::uint64_t fnv1a(std::string_view text, std::uint64_t seed = 0) {
  std::uint64_t h = seed ^ 0xcbf29ce484222325ull;
  for (unsigned char c : text) { h ^= c; h *= 0x100000001b3ull; }
  return h;
}

std::uint64_t fingerprint_file(const std::filesystem::path& path, std::uint64_t seed) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot fingerprint " + path.string());
  std::uint64_t h = seed;
  char buf[65536];
  while (in) {
    in.read(buf, sizeof(buf));
    h = fnv1a(std::string_view(buf, static_cast<std::size_t>(in.gcount())), h);
  }
  return h;
}

std::uint64_t fingerprint_tree(const std::filesystem::path& root, std::uint64_t seed) {
  if (root.empty() || !std::filesystem::exists(root)) return fnv1a("absent", seed);
  if (std::filesystem::is_regular_file(root)) {
    const auto size = std::filesystem::file_size(root);
    const auto stamp = std::filesystem::last_write_time(root).time_since_epoch().count();
    return fnv1a(root.filename().string() + ":" + std::to_string(size) + ":" +
                 std::to_string(stamp), seed);
  }
  std::vector<std::pair<std::string, std::filesystem::path>> entries;
  for (const auto& e : std::filesystem::recursive_directory_iterator(root)) {
    if (!e.is_regular_file()) continue;
    const auto ext = e.path().extension().string();
    if (ext != ".u8" && ext != ".f32" && ext != ".f8_e4m3" && ext != ".json" &&
        ext != ".blake3" && ext != ".safetensors" && ext != ".bin" && ext != ".slab")
      continue;
    const auto rel = std::filesystem::relative(e.path(), root).string();
    const auto size = e.file_size();
    const auto stamp = e.last_write_time().time_since_epoch().count();
    entries.push_back({rel + ":" + std::to_string(size) + ":" + std::to_string(stamp),
                       e.path()});
  }
  std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& [entry, path] : entries) {
    seed = fnv1a(entry, seed);
    const auto ext = path.extension();
    // Overlay metadata carries source checksums and the deterministic recipe.
    // Draft safetensors lack a sidecar checksum, so fingerprint their payload.
    if (ext == ".json" || ext == ".blake3" || ext == ".safetensors")
      seed = fingerprint_file(path, seed);
  }
  return seed;
}

std::string printable(const std::string& s) {
  std::string out;
  for (const char c : s) {
    if (c == '\n') out += "\\n";
    else if (c == '\t') out += "\\t";
    else out.push_back(c);
  }
  return out;
}

std::string json_escape(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (const unsigned char c : text) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else out.push_back(static_cast<char>(c));
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path snapshot = rocket::fuel::default_nvfp4_snapshot_dir();
  if (snapshot.empty() || !std::filesystem::exists(snapshot / "config.json")) {
    std::fprintf(stderr, "no snapshot; set $ROCKET_FUEL_NVFP4_DIR\n");
    return 77;
  }
  const std::filesystem::path attn = rocket::fuel::default_attention_yaml();
  if (attn.empty() || !std::filesystem::exists(attn)) {
    std::fprintf(stderr, "no attention.yaml; set $ROCKET_ATTENTION_YAML\n");
    return 77;
  }

  std::string prompt = arg_value(argc, argv, "--prompt", "The capital of France is");
  if (const char* prompt_file = arg_value(argc, argv, "--prompt-file", nullptr)) {
    std::ifstream in(prompt_file, std::ios::binary);
    if (!in) { std::fprintf(stderr, "cannot read --prompt-file %s\n", prompt_file); return 2; }
    prompt.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  const char* prompt_list = arg_value(argc, argv, "--prompt-list", nullptr);
  const char* result_json = arg_value(argc, argv, "--result-json", nullptr);
  const char* replacement_prompt = arg_value(argc, argv, "--replacement-prompt", nullptr);
  const char* decode_marker = arg_value(argc, argv, "--decode-marker", nullptr);
  const int prompt_token_limit = std::atoi(arg_value(argc, argv, "--prompt-token-limit", "0"));
  const int prompt_tail_tokens = std::atoi(arg_value(argc, argv, "--prompt-tail-tokens", "512"));
  const int n_new = std::atoi(arg_value(argc, argv, "--tokens", "20"));
  const double cache_gib = std::atof(arg_value(argc, argv, "--expert-cache-gib", "56"));
  const int max_tokens = std::atoi(arg_value(argc, argv, "--max-tokens", "4096"));
  // Chunk the body, then run a sequential tail in the production decode shape.
  // Whole-prompt chunking reduced DFlash2 acceptance; a 32-token tail retained
  // exact long-prompt output and decode throughput in the measured gates.
  const int prefill_chunk = std::atoi(arg_value(argc, argv, "--prefill-chunk", "32"));
  const int prefill_tail = std::atoi(arg_value(argc, argv, "--prefill-tail", "32"));
  if (prefill_chunk < 1 || prefill_chunk > 64 || prefill_tail < 0) {
    std::fprintf(stderr, "--prefill-chunk must be in [1,64] and --prefill-tail must be >= 0\n");
    return 2;
  }
  const int batch = std::atoi(arg_value(argc, argv, "--batch", "1"));
  const int spec_k = std::atoi(arg_value(argc, argv, "--spec", "1"));
  const int rank = std::atoi(arg_value(argc, argv, "--rank", "-1"));
  const bool preload_owned = arg_value(argc, argv, "--preload-owned", "0")[0] == '1';
  const char* draft_file = arg_value(argc, argv, "--draft-file", nullptr);
  const bool telemetry = arg_value(argc, argv, "--telemetry", "0")[0] == '1';
  const char* prefix_cache_dir = arg_value(argc, argv, "--prefix-cache-dir", nullptr);
  const char* prefix_cache_bytes = arg_value(argc, argv, "--prefix-cache-bytes", "0");
  const char* prefix_staging_bytes = arg_value(argc, argv, "--prefix-cache-staging-bytes", "128MiB");
  const int prefix_queue_depth = std::atoi(arg_value(argc, argv, "--prefix-cache-queue-depth", "4"));
  if (n_new <= 0 || max_tokens <= 0 || cache_gib < 0.0 || batch <= 0 ||
      prefix_queue_depth <= 0 || prompt_tail_tokens < 0 ||
      (prompt_token_limit > 0 && prompt_tail_tokens >= prompt_token_limit)) {
    std::fprintf(stderr, "bad sizing: --tokens %d --max-tokens %d --expert-cache-gib %g "
                         "--batch %d --prefix-cache-queue-depth %d\n",
                 n_new, max_tokens, cache_gib, batch, prefix_queue_depth);
    return 2;
  }
  std::optional<rocket::engine::kv::NvmePrefixOptions> prefix_options;
  try {
    if (prefix_cache_dir) {
      rocket::engine::kv::NvmePrefixOptions po;
      po.directory = prefix_cache_dir;
      po.capacity_bytes = rocket::engine::kv::NvmePrefixStore::parse_bytes(prefix_cache_bytes);
      po.staging_bytes = rocket::engine::kv::NvmePrefixStore::parse_bytes(prefix_staging_bytes);
      po.queue_depth = prefix_queue_depth;
      po.rank = std::max(rank, 0);
      std::filesystem::create_directories(po.directory);
      const std::uint64_t free = rocket::engine::kv::NvmePrefixStore::available_bytes(po.directory);
      if (free <= po.free_space_headroom_bytes ||
          po.capacity_bytes > free - po.free_space_headroom_bytes)
        throw std::runtime_error("prefix cache capacity exceeds free space after 64GiB headroom");
      prefix_options = po;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "prefix cache configuration: %s\n", e.what());
    return 2;
  }

  try {

  const rocket::fuel::ModelConfig cfg = rocket::fuel::load_model_config(attn, snapshot);
  std::printf("fuel      glm-5.3-flash, %d text layers (%d KDA, %d sparse-MLA), %d experts top-%d\n",
              cfg.text_layers, cfg.kda_layer_count(), cfg.mla_layer_count(), cfg.n_routed_experts,
              cfg.num_experts_per_tok);

  auto t_tok = Clock::now();
  const rocket::fuel::Tokenizer tok(snapshot / "tokenizer.json");
  const double tok_ms = ms_since(t_tok);

  auto t_load = Clock::now();
  std::unique_ptr<rocket::fabric::ExpertParallel> ep;
  if (rank >= 0) {
    rocket::fabric::Config fc;
    fc.rank = rank;
    fc.bootstrap_host = arg_value(argc, argv, "--host", "192.168.100.10");
    fc.bootstrap_port = std::atoi(arg_value(argc, argv, "--port", "18779"));
    const rocket::fabric::ExpertPartition part =
        rocket::fabric::contiguous_partition(cfg.n_routed_experts, {});
    // Experimental prefill widths can route more rows than decode's fixed
    // spec-8 path, so the exchange staging follows the larger width.
    ep = std::make_unique<rocket::fabric::ExpertParallel>(
        fc, part.owner, batch * std::max(rocket::engine::DecodeEngine::kSpecMax, prefill_chunk) *
                            cfg.num_experts_per_tok,
        cfg.hidden_size);
    std::printf("rank %d: fabric up, %d owned experts\n", ep->rank(), ep->expert_count());
  }
  rocket::engine::DecodeEngine engine(cfg, snapshot,
                                      static_cast<std::size_t>(cache_gib * (1ull << 30)),
                                      max_tokens, batch, 0, 128, prefill_chunk);
  engine.set_telemetry(telemetry);
  std::uint64_t prefix_namespace = 0;
  if (prefix_options) {
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(snapshot, ec);
    const auto env_or_empty = [](const char* name) {
      const char* value = std::getenv(name);
      return value ? std::string(value) : std::string();
    };
    const std::string fp8_root = env_or_empty("ROCKET_FP8_ATTN_DIR");
    const std::string kda_fp8_root = env_or_empty("ROCKET_KDA_QKV_FP8_DIR");
    const std::string dflash_root = env_or_empty("ROCKET_DFLASH2_DIR");
    const std::string packed_weights = env_or_empty("ROCKET_PACKED_WEIGHTS");
    const std::string expert_slabs = env_or_empty("ROCKET_EXPERT_SLAB_DIR");
    const std::string ns = (ec ? snapshot.string() : canonical.string()) + "|" +
        attn.string() + "|batch=" + std::to_string(batch) + "|slot-group=8|page=128|nvfp4|fp32-kda|" +
        env_or_empty("ROCKET_KDA_CUBLAS") + "|" + env_or_empty("ROCKET_CUBLAS_ALL") + "|" +
        packed_weights + "|" + expert_slabs + "|" + fp8_root + "|" + kda_fp8_root + "|" +
        dflash_root;
    prefix_namespace = fnv1a(ns);
    prefix_namespace = fingerprint_file(snapshot / "checksums.blake3", prefix_namespace);
    prefix_namespace = fingerprint_file(snapshot / "config.json", prefix_namespace);
    prefix_namespace = fingerprint_file(snapshot / "tokenizer.json", prefix_namespace);
    prefix_namespace = fingerprint_file(attn, prefix_namespace);
    prefix_namespace = fingerprint_tree(packed_weights, prefix_namespace);
    prefix_namespace = fingerprint_tree(expert_slabs, prefix_namespace);
    prefix_namespace = fingerprint_tree(fp8_root, prefix_namespace);
    prefix_namespace = fingerprint_tree(kda_fp8_root, prefix_namespace);
    prefix_namespace = fingerprint_tree(dflash_root, prefix_namespace);
    engine.enable_nvme_prefix_cache(std::move(*prefix_options), prefix_namespace);
    std::printf("prefix    dir=%s capacity=%s staging=%s queue=%d namespace=%016llx\n",
                prefix_cache_dir, prefix_cache_bytes, prefix_staging_bytes, prefix_queue_depth,
                static_cast<unsigned long long>(prefix_namespace));
  }
  if (ep) {
    engine.weights().set_expert_set(ep->owned_experts());
    engine.set_expert_parallel(ep.get());
  }
  std::unique_ptr<rocket::engine::DFlash2DraftEngine> dflash;
  if (const char* draft_dir = std::getenv("ROCKET_DFLASH2_DIR")) {
    if (spec_k > 8) throw std::runtime_error("DFlash2 requires --spec <= 8");
    if (spec_k >= 2) {
      const auto td = Clock::now();
      dflash = std::make_unique<rocket::engine::DFlash2DraftEngine>(
          draft_dir, engine.weights().embed(), engine.weights().lm_head(), batch, max_tokens, 7,
          ep.get());
      std::printf("dflash2   loaded in %.2f s\n", ms_since(td) / 1000.0);
    } else {
      std::printf("dflash2   disabled for no-speculation control\n");
    }
  }
  if (ep && preload_owned) {
    const auto t_pre = Clock::now();
    const size_t pre0 = engine.weights().expert_bytes_streamed();
    const size_t loaded = engine.weights().preload_owned_experts(nullptr);
    std::printf("rank %d: preloaded %zu owned experts, %.2f GiB in %.1f s; streamed %.2f GiB so far\n",
                rank, loaded,
                (engine.weights().expert_bytes_streamed() - pre0) / 1073741824.0,
                ms_since(t_pre) / 1000.0,
                engine.weights().expert_bytes_streamed() / 1073741824.0);
    // A rank that finishes loading first reaches its first MoE exchange while
    // the peer can still be ~90 s into the preload; the timed-out spurious
    // doorbell kills the pair before a step runs. Hold both ranks here.
    ep->barrier();
  }
  if (arg_value(argc, argv, "--cuda-graph", "0")[0] == '1') engine.set_use_cuda_graph(true);
  const double load_ms = ms_since(t_load);
  std::printf("resident  %.2f GiB   expert cache %zu slots x %.2f MiB = %.2f GiB\n",
              engine.weights().resident_bytes() / 1073741824.0, engine.weights().expert_slots(),
              engine.weights().expert_slot_bytes() / 1048576.0,
              engine.weights().expert_slots() * engine.weights().expert_slot_bytes() / 1073741824.0);
  std::printf("load      %.1f s weights, %.1f s tokenizer\n", load_ms / 1000.0, tok_ms / 1000.0);

  std::vector<std::string> prompts(batch, prompt);
  if (prompt_list) {
    std::ifstream list(prompt_list);
    if (!list) { std::fprintf(stderr, "cannot read --prompt-list %s\n", prompt_list); return 2; }
    prompts.clear();
    std::string path;
    while (std::getline(list, path)) {
      if (path.empty()) continue;
      std::ifstream in(path, std::ios::binary);
      if (!in) { std::fprintf(stderr, "cannot read prompt path %s\n", path.c_str()); return 2; }
      prompts.emplace_back(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    if (static_cast<int>(prompts.size()) != batch) {
      std::fprintf(stderr, "--prompt-list has %zu prompts, expected batch %d\n", prompts.size(), batch);
      return 2;
    }
  }
  std::vector<std::vector<int>> prompt_ids(batch);
  std::size_t common_prompt_tokens = static_cast<std::size_t>(-1);
  for (int m = 0; m < batch; ++m) {
    prompt_ids[m] = tok.encode(prompts[m]);
    if (prompt_token_limit > 0 && prompt_ids[m].size() > static_cast<std::size_t>(prompt_token_limit)) {
      const std::size_t tail = std::min<std::size_t>(prompt_tail_tokens, prompt_ids[m].size());
      std::vector<int> limited;
      limited.reserve(prompt_token_limit);
      limited.insert(limited.end(), prompt_ids[m].begin(),
                     prompt_ids[m].begin() + (prompt_token_limit - tail));
      limited.insert(limited.end(), prompt_ids[m].end() - tail, prompt_ids[m].end());
      prompt_ids[m] = std::move(limited);
    }
    common_prompt_tokens = std::min(common_prompt_tokens, prompt_ids[m].size());
  }
  for (auto& ids : prompt_ids) ids.resize(common_prompt_tokens);
  for (int a = 0; a < batch; ++a)
    for (int b = a + 1; b < batch; ++b)
      if (prompt_ids[a] == prompt_ids[b])
        throw std::runtime_error("tokenized production prompts are replicated");
  const std::size_t max_prompt_tokens = common_prompt_tokens;
  std::string prompt_print = printable(prompts[0]);
  if (prompt_print.size() > 512) prompt_print.resize(512), prompt_print += "...";
  std::printf("prompt    %zu..%zu tokens across %d distinct-capable streams\n",
              std::min_element(prompt_ids.begin(), prompt_ids.end(), [](const auto& a, const auto& b) { return a.size() < b.size(); })->size(),
              max_prompt_tokens, batch);

  engine.reset();
  const auto t_request_prompt = Clock::now();
  std::vector<int> last(batch, 0);
  int restored_prefix = 0;
  const int prefix_page_tokens = 128;
  if (prefix_cache_dir) {
    int candidate = static_cast<int>(prompt_ids[0].size()) / prefix_page_tokens * prefix_page_tokens;
    for (int m = 0; m < batch; ++m)
      candidate = std::min(candidate, engine.kv_match_shared(
          m, prompt_ids[m].data(), static_cast<int>(prompt_ids[m].size())));
    if (dflash) {
      while (candidate > 0) {
        bool all_dflash = true;
        for (int m = 0; m < batch; ++m) {
          const auto dk = engine.kv_prefix_key(m, prompt_ids[m].data(), candidate,
                                               /*record_kind=*/3);
          all_dflash = engine.kv_nvme_store()->holds(dk) && all_dflash;
        }
        if (all_dflash) break;
        candidate -= prefix_page_tokens;
      }
    }
    if (ep) ep->sync_prefix_boundary(candidate);
    while (candidate > 0) {
      int local_boundary = candidate;
      for (int m = 0; m < batch; ++m) {
        int cached_next = 0;
        const int got = engine.kv_open_shared(m, prompt_ids[m].data(), candidate, &cached_next);
        if (got != candidate) local_boundary = std::min(local_boundary, got);
        if (got > 0) last[m] = cached_next;
      }
      if (ep) ep->sync_prefix_boundary(local_boundary);
      if (local_boundary != candidate) {
        engine.reset();
        candidate = local_boundary > 0 ? local_boundary : candidate - prefix_page_tokens;
        continue;
      }
      bool dflash_ok = true;
      if (dflash) {
        for (int m = 0; m < batch; ++m) {
          const auto dk = engine.kv_prefix_key(m, prompt_ids[m].data(), candidate,
                                               /*record_kind=*/3);
          dflash_ok = dflash->load_prefix_state(
              *engine.kv_nvme_store(), dk, m, candidate, nullptr) && dflash_ok;
        }
      }
      int dflash_boundary = dflash_ok ? candidate : candidate - prefix_page_tokens;
      if (ep) ep->sync_prefix_boundary(dflash_boundary);
      if (dflash_boundary != candidate) {
        engine.reset();
        candidate = dflash_boundary;
        continue;
      }
      restored_prefix = candidate;
      break;
    }
    std::printf("prefix    restored %d/%zu prompt tokens, %d physical GPU pages at batch %d\n",
                restored_prefix, prompt_ids[0].size(), engine.kv_pinned_pages(), batch);
  }

  // Prefill runs the decode path once per prompt token; every stream gets
  // the same prompt so the batch is full from the first step.
  const bool profile_prefill = std::getenv("ROCKET_PROFILE_PREFILL") != nullptr;
  if (profile_prefill) cudaProfilerStart();
  auto t_prefill = Clock::now();
  std::vector<int> next_batch;
  // Keep at least one prompt token after the checkpoint. A terminal full page
  // can still have an unsealed radix parent after transactional chunk replay;
  // the preceding boundary is always a live, reusable prefix.
  const int checkpoint_target =
      static_cast<int>(prompt_ids[0].size() - 1) / prefix_page_tokens * prefix_page_tokens;
  for (std::size_t begin = static_cast<std::size_t>(restored_prefix); begin < prompt_ids[0].size();) {
    const int remaining = static_cast<int>(prompt_ids[0].size() - begin);
    int count = remaining <= prefill_tail ? 1 : std::min(prefill_chunk, remaining - prefill_tail);
    if (prefix_cache_dir && static_cast<int>(begin) < checkpoint_target)
      count = std::min(count, checkpoint_target - static_cast<int>(begin));
    std::vector<int> base(batch);
    for (int m = 0; m < batch; ++m) base[m] = engine.position(m);
    if (count == 1) {
      std::vector<int> input(batch);
      for (int m = 0; m < batch; ++m) input[m] = prompt_ids[m][begin];
      engine.step(input, next_batch, false);
    } else {
      std::vector<int> chunk(static_cast<std::size_t>(count) * batch);
      for (int j = 0; j < count; ++j)
        for (int m = 0; m < batch; ++m)
          chunk[static_cast<std::size_t>(j) * batch + m] = prompt_ids[m][begin + j];
      engine.step_spec(chunk, count, next_batch, false);
      engine.commit_positions(std::vector<int>(batch, count));
    }
    if (dflash) {
      for (int off = 0; off < count; off += rocket::engine::DecodeEngine::kSpecMax) {
        const int n = std::min(rocket::engine::DecodeEngine::kSpecMax, count - off);
        std::vector<int> slice_base(batch);
        for (int m = 0; m < batch; ++m) slice_base[m] = base[m] + off;
        const rocket::engine::bf16* aux =
            engine.dflash_aux_hidden() + static_cast<std::size_t>(off) * batch * cfg.hidden_size;
        dflash->append_context(aux, engine.dflash_aux_stride_rows(), n, batch, slice_base,
                               std::vector<int>(batch, n), nullptr);
      }
    }
    for (int m = 0; m < batch; ++m)
      last[m] = next_batch[static_cast<std::size_t>(count - 1) * batch + m];
    begin += count;
    if (prefix_cache_dir && static_cast<int>(begin) == checkpoint_target &&
        checkpoint_target > restored_prefix) {
      for (int m = 0; m < batch; ++m) {
        engine.kv_checkpoint(m, last[m]);
        if (dflash) {
          auto dk = engine.kv_prefix_key(m, prompt_ids[m].data(), checkpoint_target,
                                         /*record_kind=*/3);
          dflash->save_prefix_state(*engine.kv_nvme_store(), dk, m,
                                    checkpoint_target, nullptr);
        }
      }
      std::printf("prefix    checkpointed %d prompt tokens\n", checkpoint_target);
    }
  }
  const double prefill_ms = ms_since(t_prefill);
  const double request_prompt_ms = ms_since(t_request_prompt);
  if (profile_prefill) cudaProfilerStop();

  if (decode_marker) {
    std::ofstream marker(decode_marker);
    marker << "start\n";
  }
  const auto t_decode_total = Clock::now();
  std::vector<std::vector<int>> generated(batch);
  std::vector<double> token_ms;
  std::vector<double> draft_ms;
  std::vector<double> draft_context_ms;
  const int instrument_at = n_new > 12 ? n_new - 6 : std::max(0, n_new - 2);
  rocket::engine::StageMs stages;
  std::vector<float> rms;
  std::vector<int> spec_tokens;
  std::vector<int> verify_out;
  std::vector<int> accepted_cnt(batch);
  long long sl_accepted = 0, sl_rounds = 0;  // spec tallies hoisted for the timing print
  std::vector<long long> accepted_drafts_by_stream(batch, 0);
  std::vector<long long> drafted_by_stream(batch, 0);
  std::vector<long long> active_rounds_by_stream(batch, 0);
  std::vector<long long> accepted_by_position(std::max(spec_k - 1, 0), 0);
  double entropy = 0.0;

  std::vector<int>* draft_ids = nullptr;
  std::vector<int> draft_storage;
  if (draft_file) {
    std::ifstream in(draft_file);
    if (!in) {
      std::fprintf(stderr, "cannot read --draft-file %s\n", draft_file);
      return 2;
    }
    int v;
    while (in >> v) draft_storage.push_back(v);
    draft_ids = &draft_storage;
    std::printf("draft     %zu reference token ids from %s\n", draft_storage.size(), draft_file);
  }

  if (spec_k <= 1) {
    // Plain decode: one token per stream per step.
    for (int i = 0; i < n_new; ++i) {
      for (int m = 0; m < batch; ++m) generated[m].push_back(last[m]);
      const bool last_tok = (i == n_new - 1);
      if (last_tok) break;
      const bool instrument = (i == instrument_at);
      const auto t0 = Clock::now();
      engine.step(last, next_batch, instrument);
      for (int m = 0; m < batch; ++m) last[m] = next_batch[m];
      const double dt = ms_since(t0);
      if (instrument) {
        stages = engine.stages();
        rms = engine.layer_rms();
        entropy = engine.router_entropy();
      } else {
        token_ms.push_back(dt);
      }
    }
  } else {
    // Spec-decode: each round drafts spec_k tokens per stream (position 0 is
    // the committed argmax, 1..spec_k-1 are n-gram continuations), verifies
    // them in one batched step_spec call, and accepts the longest prefix the
    // target agrees with. The bonus argmax after the last accepted position
    // becomes the next round's committed token.
    const int K = spec_k;
    long long drafted = 0;
    // Alias into the hoisted tallies so the loop body and the summary print
    // keep their original names while the timing section can read them.
    long long& accepted_total = sl_accepted;
    long long& rounds = sl_rounds;
    bool spec_instrumented = false;
  while (true) {
      std::vector<unsigned char> active(batch, 0);
      bool any_active = false;
      // Commit one pending greedy token only for unfinished sessions. Finished
      // slots remain transactionally paused while shorter-acceptance peers run.
      for (int m = 0; m < batch; ++m) {
        active[m] = static_cast<int>(generated[m].size()) < n_new;
        any_active = any_active || active[m];
        if (active[m]) generated[m].push_back(last[m]);
      }
      if (!any_active) break;
      bool any_left = false;
      for (int m = 0; m < batch; ++m) {
        active[m] = static_cast<int>(generated[m].size()) < n_new;
        any_left = any_left || active[m];
      }
      if (!any_left) break;

      const auto round_t0 = Clock::now();
      spec_tokens.assign(batch * K, 0);
      std::vector<int> dflash_tokens;
      if (dflash && !draft_ids) {
        const auto draft_t0 = Clock::now();
        std::vector<int> draft_pos(batch);
        for (int m = 0; m < batch; ++m) draft_pos[m] = engine.position(m);
        dflash->propose(last, draft_pos, batch, K - 1, dflash_tokens, nullptr);
        draft_ms.push_back(ms_since(draft_t0));
      }
      for (int m = 0; m < batch; ++m) {
        spec_tokens[m] = last[m];
        if (!active[m]) {
          for (int j = 1; j < K; ++j) spec_tokens[j * batch + m] = last[m];
          continue;
        }
        // Draft sources: --draft-file draws the verify chain from a reference
        // generation (perfect-draft correctness probe); otherwise an n-gram
        // lookup proposes the continuation, padded with the committed token.
        const std::vector<int>& g = generated[m];
        std::vector<int> d;
        if (draft_ids) {
          for (int j = 1; j < K && g.size() + j - 1 < draft_ids->size(); ++j)
            d.push_back((*draft_ids)[g.size() + j - 1]);
        } else if (dflash) {
          for (int j = 0; j < K - 1; ++j)
            d.push_back(dflash_tokens[static_cast<std::size_t>(j) * batch + m]);
        } else if (g.size() >= 2) {
          const int a = g[g.size() - 2], b = g[g.size() - 1];
          for (int i = static_cast<int>(g.size()) - 3; i >= 0 && static_cast<int>(d.size()) < K - 1; --i) {
            if (g[i] == a && i + 1 < static_cast<int>(g.size()) && g[i + 1] == b) {
              for (int j = i + 2; j < static_cast<int>(g.size()) && static_cast<int>(d.size()) < K - 1; ++j)
                d.push_back(g[j]);
              break;
            }
          }
        }
        for (int j = 1; j < K; ++j) {
          const int t = (static_cast<int>(d.size()) >= j) ? d[j - 1] : last[m];
          spec_tokens[j * batch + m] = t;
        }
        drafted += K - 1;
        drafted_by_stream[m] += K - 1;
        ++active_rounds_by_stream[m];
      }
      if (ep && dflash) ep->sync_draft_tokens(spec_tokens);

      const bool instrument = !spec_instrumented &&
          static_cast<int>(generated[0].size()) + K >= instrument_at;
      engine.step_spec(spec_tokens, K, verify_out, instrument);
      spec_instrumented = spec_instrumented || instrument;
      if (std::getenv("ROCKET_SPEC_TRACE")) {
        std::fprintf(stderr, "[trace] r%lld:", rounds);
        for (int j = 0; j < K; ++j)
          std::fprintf(stderr, " d%d=%d a%d=%d", j, spec_tokens[static_cast<std::size_t>(j) * batch],
                       j, verify_out[static_cast<std::size_t>(j) * batch]);
        std::fprintf(stderr, "\n");
      }


      // Greedy acceptance per stream. In pair mode, rank 0's decision is
      // authoritative: tiny last-bit differences between otherwise equal
      // replicated logits must never let one rank enter an extra model round.
      for (int m = 0; m < batch; ++m) {
        if (!active[m]) {
          accepted_cnt[m] = 0;
          continue;
        }
        int acc = 1;  // position 0 is the committed argmax, always correct
        for (int j = 0; j + 1 < K; ++j) {
          if (verify_out[j * batch + m] == spec_tokens[(j + 1) * batch + m]) ++acc;
          else break;
        }
        accepted_cnt[m] = acc;
        accepted_drafts_by_stream[m] += acc - 1;
        for (int j = 0; j < acc - 1; ++j) ++accepted_by_position[j];
        last[m] = verify_out[(acc - 1) * batch + m];
      }
      if (ep) ep->sync_round_decision(accepted_cnt, last);
      for (int m = 0; m < batch; ++m) {
        for (int j = 1; j < accepted_cnt[m]; ++j)
          if (static_cast<int>(generated[m].size()) < n_new)
            generated[m].push_back(spec_tokens[j * batch + m]);
        accepted_total += accepted_cnt[m];
      }
      if (dflash) {
        const auto context_t0 = Clock::now();
        std::vector<int> base(batch);
        for (int m = 0; m < batch; ++m) base[m] = engine.position(m);
        dflash->append_context(engine.dflash_aux_hidden(), engine.dflash_aux_stride_rows(), K, batch,
                               base, accepted_cnt, nullptr);
        draft_context_ms.push_back(ms_since(context_t0));
      }
      engine.commit_positions(accepted_cnt);
      const double dt = ms_since(round_t0);  // draft, verify, acceptance, state commit
      ++rounds;
      sl_accepted = accepted_total;
      sl_rounds = rounds;
      if (instrument) {
        stages = engine.stages();
        rms = engine.layer_rms();
        entropy = engine.router_entropy();
      } else {
        token_ms.push_back(dt);
      }
    }
    const long long accepted_drafts = accepted_total - rounds * batch;
    std::printf("spec       rounds=%lld drafted=%lld accepted=%lld + committed=%lld (%.1f%% draft acceptance)\n",
                rounds, drafted, accepted_drafts, rounds * batch,
                drafted > 0 ? 100.0 * accepted_drafts / drafted : 0.0);
  }

  const double decode_total_ms = ms_since(t_decode_total);
  if (decode_marker) {
    std::ofstream marker(decode_marker);
    marker << "end\n";
  }
  bool replacement_preserved_active_slots = true;
  if (replacement_prompt && batch > 1) {
    std::ifstream in(replacement_prompt, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot read --replacement-prompt ") + replacement_prompt);
    const std::string replacement_text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    std::vector<int> replacement_ids = tok.encode(replacement_text);
    if (replacement_ids.size() > 128) replacement_ids.resize(128);
    std::vector<int> held_positions(batch);
    for (int m = 0; m < batch; ++m) held_positions[m] = engine.position(m);
    engine.reset_slot(0);
    if (dflash) dflash->reset_slot(0, nullptr);
    int replacement_next = 0;
    for (std::size_t begin = 0; begin < replacement_ids.size();) {
      const int count = std::min<int>(rocket::engine::DecodeEngine::kSpecMax,
                                      replacement_ids.size() - begin);
      std::vector<int> chunk(static_cast<std::size_t>(count) * batch, last[1]);
      for (int j = 0; j < count; ++j)
        chunk[static_cast<std::size_t>(j) * batch] = replacement_ids[begin + j];
      std::vector<int> base(batch);
      for (int m = 0; m < batch; ++m) base[m] = engine.position(m);
      engine.step_spec(chunk, count, verify_out, false);
      replacement_next = verify_out[static_cast<std::size_t>(count - 1) * batch];
      std::vector<int> accepted(batch, 0);
      accepted[0] = count;
      if (dflash)
        dflash->append_context(engine.dflash_aux_hidden(), engine.dflash_aux_stride_rows(), count,
                               batch, base, accepted, nullptr);
      engine.commit_positions(accepted);
      begin += count;
    }
    last[0] = replacement_next;
    for (int m = 1; m < batch; ++m)
      replacement_preserved_active_slots = replacement_preserved_active_slots &&
                                           engine.position(m) == held_positions[m];
    std::printf("continuous replacement preserved %d active slots: %s\n", batch - 1,
                replacement_preserved_active_slots ? "yes" : "NO");
  }
  std::printf("\n--- output ------------------------------------------------------\n");
  if (batch == 1)
    std::printf("%s%s\n", prompt_print.c_str(), tok.decode(generated[0]).c_str());
  else
    std::printf("batch %d, stream 0: %s%s\n", batch, prompt_print.c_str(),
                tok.decode(generated[0]).c_str());
  if (batch == 1 || std::getenv("ROCKET_TOKEN_IDS")) {
    std::printf("--- token ids ---------------------------------------------------\n");
    for (std::size_t i = 0; i < generated[0].size(); ++i)
      std::printf("%d:%d |%s|%s", static_cast<int>(i), generated[0][i],
                  printable(tok.decode_one(generated[0][i])).c_str(), (i % 4 == 3) ? "\n" : "  ");
    std::printf("\n");
  }

  std::sort(token_ms.begin(), token_ms.end());
  const double median = token_ms.empty() ? 0.0 : token_ms[token_ms.size() / 2];
  std::printf("\n--- timing ------------------------------------------------------\n");
  std::printf("prefill              %8.1f ms for %zu tokens (%.1f ms/token)\n", prefill_ms,
              prompt_ids[0].size(), prefill_ms / static_cast<double>(prompt_ids[0].size()));
  if (spec_k > 1 && sl_rounds > 0) {
    // Spec rounds absorb more than one token per stream (accepted prefix +
    // committed token), so batch/median undercounts by the acceptance length.
    const double tpr = static_cast<double>(sl_accepted) / static_cast<double>(sl_rounds);
  }
  std::printf("decode median        %8.1f ms/step  (%.2f agg tok/s at batch %d, %.2f tok/s/stream)\n",
              median,
              median > 0 ? (spec_k > 1 && sl_rounds > 0
                                ? 1000.0 * static_cast<double>(sl_accepted) / static_cast<double>(sl_rounds) / median
                                : 1000.0 * batch / median)
                         : 0.0,
              batch,
              median > 0 ? (spec_k > 1 && sl_rounds > 0
                                ? 1000.0 * static_cast<double>(sl_accepted) / static_cast<double>(sl_rounds) / median / static_cast<double>(batch)
                                : 1000.0 / median)
                         : 0.0);
  if (!token_ms.empty())
    std::printf("decode min/max       %8.1f / %.1f ms\n", token_ms.front(), token_ms.back());
  if (!draft_ms.empty()) {
    std::sort(draft_ms.begin(), draft_ms.end());
    std::sort(draft_context_ms.begin(), draft_context_ms.end());
    std::printf("dflash2 propose      %8.1f ms median\n", draft_ms[draft_ms.size() / 2]);
    std::printf("dflash2 context      %8.1f ms median\n",
                draft_context_ms[draft_context_ms.size() / 2]);
  }

  std::printf("\nper-stage, one instrumented step (a sync per stage, so the sum\n"
              "exceeds an uninstrumented step):\n");
  std::printf("  %-22s %10s\n", "stage", "ms");
  const struct {
    const char* name;
    double v;
  } rows[] = {
      {"embed", stages.embed},         {"hyper-connections", stages.hyper_connection},
      {"norms", stages.norms},         {"KDA (34 layers)", stages.kda},
      {"sparse MLA + indexer", stages.mla}, {"dense MLP (3 layers)", stages.dense_mlp},
      {"MoE incl. experts", stages.moe_experts}, {"  of which streaming", stages.expert_stream},
      {"lm_head + argmax", stages.lm_head},
  };
  for (const auto& r : rows) std::printf("  %-22s %10.2f\n", r.name, r.v);
  std::printf("  %-22s %10.2f\n", "sum", stages.sum());

  std::printf("\n--- validation --------------------------------------------------\n");
  if (rms.empty()) {
    std::printf("hidden RMS: not collected (no instrumented step ran)\n");
  } else {
    float lo = rms[0], hi = rms[0];
    bool finite = true;
    for (const float v : rms) {
      lo = std::min(lo, v);
      hi = std::max(hi, v);
      finite = finite && std::isfinite(v);
    }
    std::printf("hidden RMS per layer:");
    for (std::size_t i = 0; i < rms.size(); ++i)
      std::printf("%s%5.2f", (i % 15 == 0) ? "\n  " : " ", rms[i]);
    std::printf("\nhidden RMS min / max over %zu layers: %.3f / %.3f%s\n", rms.size(), lo, hi,
                (finite && hi < 1e4f && lo > 1e-4f) ? "  (finite, in band)" : "  OUT OF BAND");
  }
  std::printf("router entropy, mean over 42 MoE layers: %.4f nats (max ln 8 = %.4f)\n", entropy,
              std::log(8.0));
  std::printf("expert cache: %llu hits, %llu misses, %.2f GiB streamed\n",
              static_cast<unsigned long long>(engine.weights().expert_hits()),
              static_cast<unsigned long long>(engine.weights().expert_misses()),
              engine.weights().expert_bytes_streamed() / 1073741824.0);
  if (const auto* ps = engine.kv_nvme_stats()) {
    std::printf("prefix IO: read %.3f GiB write %.3f GiB record_hits %llu record_misses %llu "
                "page_hits %llu page_misses %llu restore %.1f ms writeback %.1f ms "
                "checksum_failures %llu rejected %llu\n",
                ps->read_bytes / 1073741824.0, ps->write_bytes / 1073741824.0,
                static_cast<unsigned long long>(ps->hit_records),
                static_cast<unsigned long long>(ps->miss_records),
                static_cast<unsigned long long>(ps->hit_pages),
                static_cast<unsigned long long>(ps->miss_pages), ps->restore_ms,
                ps->writeback_ms,
                static_cast<unsigned long long>(ps->checksum_failures),
                static_cast<unsigned long long>(ps->rejected_records));
  }
  if (telemetry) {
    std::map<std::string, float> ordered(engine.telemetry_absmax().begin(),
                                         engine.telemetry_absmax().end());
    std::printf("\n--- quantization telemetry (fixed family vocabulary) -----------\n");
    for (const auto& [name, value] : ordered)
      std::printf("telemetry_absmax %-48s %.9g\n", name.c_str(), value);
  }
  if (result_json) {
    std::ofstream out(result_json);
    if (!out) throw std::runtime_error(std::string("cannot write --result-json ") + result_json);
    const long long useful_tokens = static_cast<long long>(batch) * n_new;
    out << "{\n  \"schema\":\"rocket.glm53.coding-bench.v1\",\n";
    out << "  \"batch\":" << batch << ",\n  \"spec_k\":" << spec_k << ",\n";
    out << "  \"prompt_tokens_per_stream\":" << prompt_ids[0].size() << ",\n";
    out << "  \"useful_output_tokens\":" << useful_tokens << ",\n";
    out << "  \"prefill_ms\":" << prefill_ms << ",\n  \"prompt_restore_and_prefill_ms\":" << request_prompt_ms << ",\n  \"decode_ms\":" << decode_total_ms << ",\n";
    out << "  \"ttft_ms_p50\":" << request_prompt_ms << ",\n  \"ttft_ms_p95\":" << request_prompt_ms << ",\n";
    out << "  \"completion_ms_p50\":" << (request_prompt_ms + decode_total_ms) << ",\n";
    out << "  \"completion_ms_p95\":" << (request_prompt_ms + decode_total_ms) << ",\n";
    out << "  \"inter_token_ms_p50\":" << (useful_tokens > 0 ? decode_total_ms * batch / useful_tokens : 0.0) << ",\n";
    out << "  \"inter_token_ms_p95\":" << (useful_tokens > 0 ? decode_total_ms * batch / useful_tokens : 0.0) << ",\n";
    out << "  \"aggregate_useful_tok_s\":" << (decode_total_ms > 0 ? useful_tokens * 1000.0 / decode_total_ms : 0.0) << ",\n";
    out << "  \"router_entropy_nats\":" << entropy << ",\n";
    out << "  \"stage_ms\":{\"embed\":" << stages.embed
        << ",\"hyper_connection\":" << stages.hyper_connection
        << ",\"norms\":" << stages.norms << ",\"kda\":" << stages.kda
        << ",\"mla_indexer\":" << stages.mla << ",\"dense_mlp\":" << stages.dense_mlp
        << ",\"moe\":" << stages.moe_experts << ",\"expert_stream\":" << stages.expert_stream
        << ",\"lm_head\":" << stages.lm_head << "},\n";
    out << "  \"expert_cache_hits\":" << engine.weights().expert_hits() << ",\n";
    out << "  \"expert_cache_misses\":" << engine.weights().expert_misses() << ",\n";
    out << "  \"rounds\":" << sl_rounds << ",\n";
    out << "  \"replacement_preserved_active_slots\":"
        << (replacement_preserved_active_slots ? "true" : "false") << ",\n";
    out << "  \"accepted_drafts_by_stream\":[";
    for (int m = 0; m < batch; ++m) out << (m ? "," : "") << accepted_drafts_by_stream[m];
    out << "],\n  \"drafted_by_stream\":[";
    for (int m = 0; m < batch; ++m) out << (m ? "," : "") << drafted_by_stream[m];
    out << "],\n  \"active_rounds_by_stream\":[";
    for (int m = 0; m < batch; ++m) out << (m ? "," : "") << active_rounds_by_stream[m];
    out << "],\n  \"accepted_by_position\":[";
    for (std::size_t j = 0; j < accepted_by_position.size(); ++j) out << (j ? "," : "") << accepted_by_position[j];
    out << "],\n  \"generated_token_ids\":[";
    for (int m = 0; m < batch; ++m) {
      out << (m ? "," : "") << "[";
      for (std::size_t i = 0; i < generated[m].size(); ++i) out << (i ? "," : "") << generated[m][i];
      out << "]";
    }
    out << "],\n  \"generated_text\":[";
    for (int m = 0; m < batch; ++m)
      out << (m ? "," : "") << "\"" << json_escape(tok.decode(generated[m])) << "\"";
    out << "]";
    if (const auto* ps = engine.kv_nvme_stats()) {
      out << ",\n  \"prefix_record_hits\":" << ps->hit_records;
      out << ",\n  \"prefix_record_misses\":" << ps->miss_records;
      out << ",\n  \"prefix_restore_ms\":" << ps->restore_ms;
    }
    out << "\n}\n";
  }
  return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
}

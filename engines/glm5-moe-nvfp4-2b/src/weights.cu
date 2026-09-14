#include "weights.h"

#include "nvfp4.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

extern "C" unsigned char* SHA256(const unsigned char*, std::size_t, unsigned char*);

namespace rocket::engine {
namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("rocket::engine::weights: " + what);
}

void cuda_check(cudaError_t e, const std::string& what) {
  if (e != cudaSuccess) fail(what + ": " + cudaGetErrorString(e));
}

std::string layer_prefix(int l) {
  return "model.language_model.layers." + std::to_string(l) + ".";
}

std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

constexpr std::size_t kPageBytes = 65536;
constexpr std::size_t kDirectChunkBytes = 256u << 20;
constexpr int kDirectPipelineSlots = 4;
constexpr std::size_t kSha256Bytes = 32;

#pragma pack(push, 1)
struct ExpertSlabHeader {
  char magic[16];
  std::uint32_t version;
  std::uint32_t page_bytes;
  std::uint64_t metadata_bytes;
  std::uint64_t payload_offset;
  std::uint64_t payload_bytes;
  std::uint64_t slot_bytes;
  std::uint64_t slot_count;
  std::uint32_t hidden;
  std::uint32_t intermediate;
  std::uint32_t layers;
  std::uint32_t experts;
  std::uint32_t first_expert;
  std::uint32_t expert_count;
  std::uint32_t sparse_layers;
  std::uint8_t config_sha256[32];
  std::uint8_t index_sha256[32];
};
#pragma pack(pop)
static_assert(sizeof(ExpertSlabHeader) == 156);

void read_exact_at(int fd, void* dst, std::size_t bytes, std::uint64_t offset,
                   const std::filesystem::path& path) {
  auto* out = static_cast<std::uint8_t*>(dst);
  std::size_t done = 0;
  while (done < bytes) {
    const ssize_t got = ::pread(fd, out + done, bytes - done,
                                static_cast<off_t>(offset + done));
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0)
      fail("pread " + path.string() + ": " +
           (got == 0 ? std::string("unexpected EOF") : std::strerror(errno)));
    done += static_cast<std::size_t>(got);
  }
}

float optional_input_scale(fuel::Checkpoint& ckpt, const std::string& name) {
  if (!ckpt.has(name)) return 1.0f;
  const fuel::TensorView& scale = ckpt.tensor(name);
  if (scale.dtype != fuel::DType::kF32 || scale.numel() != 1)
    fail(name + ": input_scale must be one F32 value");
  float value = 1.0f;
  std::memcpy(&value, scale.data, sizeof(value));
  if (!std::isfinite(value) || value <= 0.0f)
    fail(name + ": input_scale must be finite and positive");
  return value;
}

std::array<unsigned char, kSha256Bytes> sha256_file(const std::filesystem::path& path) {
  const auto bytes = std::filesystem::file_size(path);
  std::vector<std::uint8_t> data(bytes);
  std::ifstream in(path, std::ios::binary);
  if (!in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size())))
    fail("read " + path.string() + " for SHA-256");
  std::array<unsigned char, kSha256Bytes> digest{};
  SHA256(data.data(), data.size(), digest.data());
  return digest;
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path, std::size_t expected) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in || static_cast<std::size_t>(in.tellg()) != expected)
    fail(path.string() + " has the wrong byte count");
  std::vector<std::uint8_t> bytes(expected);
  in.seekg(0);
  if (!in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(expected)))
    fail("read " + path.string());
  return bytes;
}

}  // namespace

void* WeightStore::device_alloc(std::size_t bytes) {
  const auto started = std::chrono::steady_clock::now();
  void* p = nullptr;
  cuda_check(cudaMalloc(&p, bytes), "cudaMalloc " + std::to_string(bytes) + " B");
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  allocation_ms_ += elapsed_ms;
  if (std::getenv("ROCKET_LOAD_PROFILE") && bytes >= (512u << 20))
    std::fprintf(stderr, "[alloc] %.2f GiB in %.3f s\n",
                 static_cast<double>(bytes) / (1ull << 30), elapsed_ms / 1000.0);
  owned_.push_back(p);
  resident_bytes_ += bytes;
  return p;
}

void WeightStore::copy_in(void* dst, const void* src, std::size_t bytes) {
  const auto started = std::chrono::steady_clock::now();
  static const bool trace = std::getenv("ROCKET_TRACE_COPY") != nullptr;
  static unsigned seq = 0;
  if (trace)
    std::fprintf(stderr, "[copy %u] dst=%p bytes=%zu\n", ++seq, dst, bytes);
  std::size_t done = 0;
  while (done < bytes) {
    const std::size_t chunk = std::min(pinned_bytes_, bytes - done);
    const auto stage_started = std::chrono::steady_clock::now();
    std::memcpy(pinned_, static_cast<const std::uint8_t*>(src) + done, chunk);
    upload_stage_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - stage_started).count();
    const auto cuda_started = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(dst) + done, pinned_, chunk,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy weight");
    upload_cuda_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cuda_started).count();
    done += chunk;
  }
  upload_ms_ += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
}

void WeightStore::copy_in_direct(void* dst, const fuel::TensorView& tensor) {
  const auto started = std::chrono::steady_clock::now();
  const std::string key = tensor.source_file.string();
  int fd = -1;
  if (const auto it = direct_fds_.find(key); it != direct_fds_.end()) {
    fd = it->second;
  } else {
    fd = ::open(tensor.source_file.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECT);
    if (fd < 0)
      fail("open O_DIRECT " + tensor.source_file.string() + ": " + std::strerror(errno));
    direct_fds_.emplace(key, fd);
  }
  std::size_t done = 0;
  while (done < tensor.nbytes) {
    const std::uint64_t position = tensor.file_offset + done;
    const std::uint64_t aligned = position / kPageBytes * kPageBytes;
    const std::size_t leading = static_cast<std::size_t>(position - aligned);
    const std::size_t logical = std::min(tensor.nbytes - done, pinned_bytes_ - leading);
    const std::size_t direct_bytes = align_up(leading + logical, kPageBytes);
    const auto stage_started = std::chrono::steady_clock::now();
    read_exact_at(fd, pinned_, direct_bytes, aligned, tensor.source_file);
    upload_stage_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - stage_started).count();
    const auto cuda_started = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(dst) + done, pinned_ + leading, logical,
                          cudaMemcpyHostToDevice), "cudaMemcpy direct weight");
    upload_cuda_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cuda_started).count();
    done += logical;
  }
  upload_ms_ += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
}

void WeightStore::record_dtype(std::string_view name, fuel::DType dt) {
  dtype_registry_.emplace(std::string(name), dt);
}

fuel::DType WeightStore::dtype_of(std::string_view tensor_name) const {
  const auto it = dtype_registry_.find(std::string(tensor_name));
  if (it == dtype_registry_.end()) fail(std::string(tensor_name) + " was never uploaded");
  return it->second;
}

const bf16* WeightStore::upload_bf16(std::string_view name, std::int64_t expect_numel) {
  if (std::getenv("ROCKET_TRACE_COPY"))
    std::fprintf(stderr, "[up-bf16] %.*s\n", (int)name.size(), name.data());
  const fuel::TensorView& t = ckpt_.tensor(name);
  if (t.dtype != fuel::DType::kBF16) fail(std::string(name) + " is not BF16");
  if (t.numel() != expect_numel)
    fail(std::string(name) + " has " + std::to_string(t.numel()) + " elements, expected " +
         std::to_string(expect_numel));
  void* d = device_alloc(t.nbytes);
  if (std::getenv("ROCKET_RESIDENT_DIRECT")) copy_in_direct(d, t);
  else copy_in(d, t.data, t.nbytes);
  record_dtype(name, fuel::DType::kBF16);
  return static_cast<const bf16*>(d);
}

const float* WeightStore::upload_f32(std::string_view name, std::int64_t expect_numel) {
  if (std::getenv("ROCKET_TRACE_COPY"))
    std::fprintf(stderr, "[up-f32] %.*s\n", (int)name.size(), name.data());
  const fuel::TensorView& t = ckpt_.tensor(name);
  // Hub and NIM snapshots of this fuel disagree on the dtype of tiny scalar
  // tensors (hyper-connection bases/scales ship BF16 on the hub, F32 in the
  // NIM bundle). The upcast is exact, so accepting BF16 here is lossless -
  // but only for the whitelisted tensors, and the served family is still
  // recorded as F32 because that is the serving dtype.
  static const std::vector<std::string> kF32FromBf16 = {
      "hc_attn_base", "hc_attn_scale", "hc_ffn_base", "hc_ffn_scale",
  };
  const bool needs_upcast =
      t.dtype == fuel::DType::kBF16 &&
      std::any_of(kF32FromBf16.begin(), kF32FromBf16.end(), [&](const std::string& s) {
        return std::string_view(name).ends_with(s);
      });
  if (t.dtype != fuel::DType::kF32 && !needs_upcast)
    fail(std::string(name) + " is not F32");
  if (t.numel() != expect_numel)
    fail(std::string(name) + " has the wrong element count");
  if (needs_upcast) {
    // Exact widening: f32 bits(p0xx...) per bf16 element.
    const std::size_t bytes = static_cast<std::size_t>(expect_numel) * 4;
    std::vector<float> host(static_cast<std::size_t>(expect_numel));
    const auto* src = reinterpret_cast<const std::uint16_t*>(t.data);
    for (int64_t i = 0; i < expect_numel; ++i) {
      const std::uint16_t raw = src[i];
      const std::uint32_t bits = std::uint32_t(raw) << 16;
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      host[i] = f;
    }
    void* d = device_alloc(bytes);
    copy_in(d, host.data(), bytes);
    record_dtype(name, fuel::DType::kF32);
    return static_cast<const float*>(d);
  }
  if (t.dtype != fuel::DType::kF32)
    fail(std::string(name) + " is not F32");
  void* d = device_alloc(t.nbytes);
  if (std::getenv("ROCKET_RESIDENT_DIRECT")) copy_in_direct(d, t);
  else copy_in(d, t.data, t.nbytes);
  record_dtype(name, fuel::DType::kF32);
  return static_cast<const float*>(d);
}

const bf16* WeightStore::upload_concat(const std::vector<std::string>& names,
                                       std::int64_t expect_numel) {
  if (std::getenv("ROCKET_TRACE_COPY"))
    std::fprintf(stderr, "[up-concat] %s\n", names.front().c_str());
  std::size_t total = 0;
  std::int64_t numel = 0;
  // Hub fuel ships KDA conv taps F32 where the NIM bundle had BF16. Round to
  // BF16 here (lossy); the family journal records it and the decode token-
  // divergence check is the gate.
  // F32→BF16 conversion only when the source is actually F32; the EXL3 fuel
  // ships conv taps as BF16 and must take the plain copy path (halving the
  // slab for already-BF16 tensors overflows it on the third copy).
  const bool f32_from_conv =
      std::all_of(names.begin(), names.end(), [](const std::string& n) {
        return n.ends_with("conv1d.weight");
      }) &&
      ckpt_.tensor(names.front()).dtype == fuel::DType::kF32;
  for (const std::string& n : names) {
    const fuel::TensorView& t = ckpt_.tensor(n);
    if (!(t.dtype == fuel::DType::kBF16 || (f32_from_conv && t.dtype == fuel::DType::kF32)))
      fail(n + " is not BF16");
    total += f32_from_conv ? t.nbytes / 2 : t.nbytes;
    numel += t.numel();
  }
  if (numel != expect_numel) fail(names.front() + " concat has the wrong element count");
  auto* d = static_cast<std::uint8_t*>(device_alloc(total));
  std::size_t off = 0;
  for (const std::string& n : names) {
    const fuel::TensorView& t = ckpt_.tensor(n);
    if (f32_from_conv && t.dtype == fuel::DType::kF32) {
      std::vector<std::uint16_t> host(t.numel());
      const auto* src = reinterpret_cast<const std::uint32_t*>(t.data);
      if (std::getenv("ROCKET_DEBUG_LAYER0") != nullptr && n.find("layers.0.") != std::string::npos) {
        std::printf("[dbg-up] %s data=%p nbytes=%zu numel=%lld f32[0..3]=%.6g %.6g %.6g %.6g\n",
                    n.c_str(), (const void*)t.data, t.nbytes, (long long)t.numel(),
                    static_cast<float>(src[0]), static_cast<float>(src[1]),
                    static_cast<float>(src[2]), static_cast<float>(src[3]));
        ckpt_.debug_probe(n);
      }
      for (std::size_t i = 0; i < t.numel(); ++i) {
        float bits;
        std::memcpy(&bits, &src[i], 4);  // bit-reinterpret, not integer conversion
        host[i] = rocket::fuel::float_to_bf16(bits);
      }
      if (std::getenv("ROCKET_DEBUG_LAYER0") != nullptr && n.find("layers.0.") != std::string::npos) {
        std::printf("[dbg-up] %s host bf16[0..7] =", n.c_str());
        for (int i = 0; i < 8; ++i) std::printf(" %04x", host[i]);
        std::printf("\n");
      }
      if (std::getenv("ROCKET_TRACE_COPY"))
        std::fprintf(stderr, "[conv-copy] %s off=%zu bytes=%zu\n", n.c_str(), off,
                     host.size() * 2);
      copy_in(d + off, host.data(), host.size() * 2);
      if (std::getenv("ROCKET_DEBUG_LAYER0") != nullptr && n.find("layers.0.") != std::string::npos) {
        std::vector<std::uint16_t> back(8);
        cudaMemcpy(back.data(), d + off, 16, cudaMemcpyDeviceToHost);
        std::printf("[dbg-up] %s device readback =", n.c_str());
        for (int i = 0; i < 8; ++i) std::printf(" %04x", back[i]);
        std::printf("  (d=%p off=%zu)\n", (void*)d, off);
      }
      record_dtype(n, fuel::DType::kBF16);
      off += host.size() * 2;
      continue;
    }
    if (std::getenv("ROCKET_RESIDENT_DIRECT")) copy_in_direct(d + off, t);
    else copy_in(d + off, t.data, t.nbytes);
    record_dtype(n, t.dtype);
    off += t.nbytes;
  }
  return reinterpret_cast<const bf16*>(d);
}

WeightStore::WeightStore(const fuel::ModelConfig& cfg, const std::filesystem::path& snapshot_dir,
                         std::size_t expert_cache_bytes)
    : cfg_(cfg), ckpt_(snapshot_dir) {
  // Frees every resident allocation if the ctor throws mid-load. Without it a
  // cuda_check failure past the first cudaMalloc leaks tens of GiB in an
  // embedder or test process that catches the error and keeps running;
  // ~WeightStore only runs after a completed construction.
  struct CtorGuard {
    std::vector<void*>& owned;
    std::uint8_t*& pinned_raw;
    bool armed = true;
    ~CtorGuard() {
      if (!armed) return;
      for (void* p : owned) cudaFree(p);
      if (pinned_raw != nullptr) cudaFreeHost(pinned_raw);
    }
  } guard{owned_, pinned_raw_};

  // FP8-per-row overlay loader: reads {weight.u8, weight_scale.f32} produced
  // by materialize-nvfp4-overlay --fp8-row. A successful load is the serving
  // representation, so callers must not also upload the BF16 source tensor.
  auto try_fp8_row = [&](const std::string& tensor_name, std::int64_t rows,
                         std::int64_t k_true, Fp8RowW& out) -> bool {
    const char* root = std::getenv("ROCKET_FP8_ATTN_DIR");
    if (root == nullptr) return false;
    // tensor_name -> overlay subdir: layers.N.self_attn.o_proj.weight -> l-N/o_proj
    const std::size_t layers = tensor_name.find("layers.");
    if (layers == std::string::npos) return false;
    const std::string rest = tensor_name.substr(layers + 7);
    const std::size_t dot = rest.find('.');
    if (dot == std::string::npos) return false;
    const std::string layer_id = rest.substr(0, dot);
    std::string leaf = rest.substr(dot + 1);
    if (leaf.starts_with("self_attn.")) leaf.erase(0, std::string("self_attn.").size());
    const std::string suffix = ".weight";
    const std::size_t at = leaf.rfind(suffix);
    if (at != std::string::npos && at + suffix.size() == leaf.size()) leaf.resize(at);
    const std::filesystem::path root_path(root);
    const std::filesystem::path layer = "l-" + layer_id;
    std::filesystem::path dir;
    for (const std::filesystem::path candidate : {
             root_path / layer / leaf,
             root_path / "kda-o-fp8" / layer / leaf,
             root_path / "kda-o-fp8" / layer,
             root_path / "mla-fp8" / layer / leaf}) {
      if (std::filesystem::exists(candidate / "weight.u8")) {
        dir = candidate;
        break;
      }
    }
    if (dir.empty()) return false;
    const auto packed = read_file(dir / "weight.u8", static_cast<std::size_t>(rows) * k_true);
    const auto scales = read_file(dir / "weight_scale.f32", static_cast<std::size_t>(rows) * 4);
    auto* pd = static_cast<std::uint8_t*>(device_alloc(packed.size()));
    copy_in(pd, packed.data(), packed.size());
    auto* sd = static_cast<float*>(device_alloc(scales.size()));
    copy_in(sd, scales.data(), scales.size());
    out.packed = pd;
    out.scales = sd;
    return true;
  };

  pinned_bytes_ = 64u << 20;
  cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&pinned_raw_), pinned_bytes_ + kPageBytes,
                           cudaHostAllocDefault), "cudaHostAlloc staging");
  pinned_ = reinterpret_cast<std::uint8_t*>(
      align_up(reinterpret_cast<std::uintptr_t>(pinned_raw_), kPageBytes));

  const int H = cfg_.hidden_size;
  const int qkv = cfg_.kda_qkv_dim();
  const int hd = cfg_.kda_head_dim;
  const int qk = cfg_.qk_head_dim();

  embed_ = upload_bf16("model.language_model.embed_tokens.weight",
                       static_cast<std::int64_t>(cfg_.vocab_size) * H);
  final_norm_ = upload_bf16("model.language_model.norm.weight", H);
  lm_head_ = upload_bf16("lm_head.weight", static_cast<std::int64_t>(cfg_.vocab_size) * H);

  layers_.resize(cfg_.text_layers);
  for (int l = 0; l < cfg_.text_layers; ++l) {
    const std::string p = layer_prefix(l);
    LayerW& w = layers_[l];

    w.attn_hc.fn = upload_bf16(p + "hc_attn_fn", static_cast<std::int64_t>(cfg_.hc_mix()) * cfg_.hc_mult * H);
    w.attn_hc.base = upload_f32(p + "hc_attn_base", cfg_.hc_mix());
    w.attn_hc.scale = upload_f32(p + "hc_attn_scale", 3);
    w.ffn_hc.fn = upload_bf16(p + "hc_ffn_fn", static_cast<std::int64_t>(cfg_.hc_mix()) * cfg_.hc_mult * H);
    w.ffn_hc.base = upload_f32(p + "hc_ffn_base", cfg_.hc_mix());
    w.ffn_hc.scale = upload_f32(p + "hc_ffn_scale", 3);

    w.input_norm = upload_bf16(p + "input_layernorm.weight", H);
    w.post_attn_norm = upload_bf16(p + "post_attention_layernorm.weight", H);

    if (cfg_.layers[l].attn == fuel::AttnKind::kKda) {
      const std::string a = p + "self_attn.";
      bool qkv_overlay_loaded = false;
      {
        if (const char* fp4_dir = std::getenv("ROCKET_KDA_QKV_NVFP4_DIR")) {
          const std::filesystem::path layer_dir =
              std::filesystem::path(fp4_dir) / ("l-" + std::to_string(l));
          if (std::filesystem::exists(layer_dir / "weight.u8")) {
            const auto packed = read_file(layer_dir / "weight.u8",
                static_cast<std::size_t>(3) * qkv * H / 2);
            const auto scale = read_file(layer_dir / "weight_scale.f8_e4m3",
                static_cast<std::size_t>(3) * qkv * H / 16);
            const auto global = read_file(layer_dir / "weight_scale_2.f32", sizeof(float));
            auto* pd = static_cast<std::uint8_t*>(device_alloc(packed.size()));
            auto* sd = static_cast<std::uint8_t*>(device_alloc(scale.size()));
            copy_in(pd, packed.data(), packed.size());
            copy_in(sd, scale.data(), scale.size());
            w.kda.qkv_fp4.packed = pd;
            w.kda.qkv_fp4.scale = sd;
            std::memcpy(&w.kda.qkv_fp4.global, global.data(), sizeof(float));
            qkv_overlay_loaded = true;
          }
        }
        if (!qkv_overlay_loaded) {
          if (const char* fp8_dir = std::getenv("ROCKET_KDA_QKV_FP8_DIR")) {
            const std::filesystem::path layer_dir =
                std::filesystem::path(fp8_dir) / ("l-" + std::to_string(l));
            if (std::filesystem::exists(layer_dir / "weight.u8")) {
              const auto packed =
                  read_file(layer_dir / "weight.u8", static_cast<std::size_t>(3) * qkv * H);
              const auto scales =
                  read_file(layer_dir / "weight_scale.f32", static_cast<std::size_t>(3) * qkv * 4);
              auto* pd = static_cast<std::uint8_t*>(device_alloc(packed.size()));
              auto* sd = static_cast<float*>(device_alloc(scales.size()));
              copy_in(pd, packed.data(), packed.size());
              copy_in(sd, scales.data(), scales.size());
              w.kda.qkv_fp8.packed = pd;
              w.kda.qkv_fp8.scales = sd;
              qkv_overlay_loaded = true;
            }
          }
        }
      }
      const char* q_only_overlay = std::getenv("ROCKET_KDA_Q_NVFP4_OBJECT");
      const int q_only_layer = std::atoi(std::getenv("ROCKET_KDA_Q_NVFP4_LAYER")
                                             ? std::getenv("ROCKET_KDA_Q_NVFP4_LAYER") : "-1");
      const bool use_q_only_overlay = !qkv_overlay_loaded && q_only_overlay != nullptr &&
                                      l == q_only_layer;
      if (!qkv_overlay_loaded) {
        if (use_q_only_overlay)
          w.kda.kv = upload_concat({a + "k_proj.weight", a + "v_proj.weight"},
                                   static_cast<std::int64_t>(2) * qkv * H);
        else
          w.kda.qkv = upload_concat({a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"},
                                    static_cast<std::int64_t>(3) * qkv * H);
      }
      if (use_q_only_overlay) {
        const char* overlay = q_only_overlay;
        {
          const std::filesystem::path object(overlay);
          const auto packed = read_file(object / "weight.u8", static_cast<std::size_t>(qkv) * H / 2);
          const auto scale = read_file(object / "weight_scale.f8_e4m3",
                                       static_cast<std::size_t>(qkv) * H / 16);
          const auto global = read_file(object / "weight_scale_2.f32", sizeof(float));
          auto* packed_dev = static_cast<std::uint8_t*>(device_alloc(packed.size()));
          auto* scale_dev = static_cast<std::uint8_t*>(device_alloc(scale.size()));
          copy_in(packed_dev, packed.data(), packed.size());
          copy_in(scale_dev, scale.data(), scale.size());
          w.kda.q_overlay.packed = packed_dev;
          w.kda.q_overlay.scale = scale_dev;
          std::memcpy(&w.kda.q_overlay.global, global.data(), sizeof(float));
        }
      }
      { const cudaError_t pre = cudaGetLastError();
        if (pre != cudaSuccess)
          std::fprintf(stderr, "[pre-conv] sticky CUDA error: %s\n", cudaGetErrorString(pre)); }
      w.kda.conv = upload_concat({a + "q_conv1d.weight", a + "k_conv1d.weight", a + "v_conv1d.weight"},
                                 static_cast<std::int64_t>(3) * qkv * cfg_.kda_conv_kernel);
      w.kda.f_a = upload_bf16(a + "f_a_proj.weight", static_cast<std::int64_t>(hd) * H);
      w.kda.f_b = upload_bf16(a + "f_b_proj.weight", static_cast<std::int64_t>(qkv) * hd);
      w.kda.g_a = upload_bf16(a + "g_a_proj.weight", static_cast<std::int64_t>(hd) * H);
      w.kda.g_b = upload_bf16(a + "g_b_proj.weight", static_cast<std::int64_t>(qkv) * hd);
      w.kda.b_proj = upload_bf16(a + "b_proj.weight", static_cast<std::int64_t>(cfg_.kda_heads) * H);
      w.kda.a_log = upload_f32(a + "A_log", cfg_.kda_heads);
      w.kda.dt_bias = upload_f32(a + "dt_bias", qkv);
      w.kda.o_norm = upload_bf16(a + "o_norm.weight", hd);
      if (!try_fp8_row(a + "o_proj.weight", H, qkv, w.kda.o_proj_fp8))
        w.kda.o_proj = upload_bf16(a + "o_proj.weight", static_cast<std::int64_t>(H) * qkv);
    } else {
      const std::string a = p + "self_attn.";
      if (!try_fp8_row(a + "q_a_proj.weight", cfg_.q_lora_rank, H, w.mla.q_a_fp8))
        w.mla.q_a = upload_bf16(a + "q_a_proj.weight", static_cast<std::int64_t>(cfg_.q_lora_rank) * H);
      w.mla.q_a_norm = upload_bf16(a + "q_a_layernorm.weight", cfg_.q_lora_rank);
      if (!try_fp8_row(a + "q_b_proj.weight", cfg_.mla_heads * qk,
                       cfg_.q_lora_rank, w.mla.q_b_fp8))
        w.mla.q_b = upload_bf16(a + "q_b_proj.weight",
                                static_cast<std::int64_t>(cfg_.mla_heads) * qk * cfg_.q_lora_rank);
      if (!try_fp8_row(a + "kv_a_proj_with_mqa.weight",
                       cfg_.kv_lora_rank + cfg_.qk_rope_head_dim, H, w.mla.kv_a_fp8))
        w.mla.kv_a = upload_bf16(a + "kv_a_proj_with_mqa.weight",
                                 static_cast<std::int64_t>(cfg_.kv_lora_rank + cfg_.qk_rope_head_dim) * H);
      w.mla.kv_a_norm = upload_bf16(a + "kv_a_layernorm.weight", cfg_.kv_lora_rank);
      // kv_b is consumed directly by the absorb/expand kernels. Keep BF16
      // until those kernels have an FP8 operand path; do not load an unused overlay.
      w.mla.kv_b = upload_bf16(a + "kv_b_proj.weight",
                               static_cast<std::int64_t>(cfg_.mla_kv_b_out()) * cfg_.kv_lora_rank);
      if (!try_fp8_row(a + "o_proj.weight", H,
                       cfg_.mla_heads * cfg_.v_head_dim, w.mla.o_proj_fp8))
        w.mla.o_proj = upload_bf16(a + "o_proj.weight",
                                   static_cast<std::int64_t>(H) * cfg_.mla_heads * cfg_.v_head_dim);
      const std::string ix = a + "indexer.";
      w.mla.idx_wk = upload_bf16(ix + "wk.weight", static_cast<std::int64_t>(cfg_.index_head_dim) * H);
      if (!try_fp8_row(ix + "wq_b.weight", cfg_.index_n_heads * cfg_.index_head_dim,
                       cfg_.q_lora_rank, w.mla.idx_wq_b_fp8))
        w.mla.idx_wq_b = upload_bf16(
            ix + "wq_b.weight",
            static_cast<std::int64_t>(cfg_.index_n_heads) * cfg_.index_head_dim * cfg_.q_lora_rank);
      w.mla.idx_k_norm_w = upload_bf16(ix + "k_norm.weight", cfg_.index_head_dim);
      w.mla.idx_k_norm_b = upload_bf16(ix + "k_norm.bias", cfg_.index_head_dim);
      w.mla.idx_weights = upload_bf16(ix + "weights_proj.weight",
                                      static_cast<std::int64_t>(cfg_.index_n_heads) * H);
      w.mla.idx_gate = upload_bf16(ix + "index_kpool_compress_gate",
                                   static_cast<std::int64_t>(cfg_.index_head_dim) * H);
      w.mla.idx_ape = upload_bf16(ix + "index_kpool_compress_ape",
                                  static_cast<std::int64_t>(cfg_.index_kpool) * cfg_.index_head_dim);
    }

    if (cfg_.layers[l].mlp == fuel::MlpKind::kDense) {
      const int I = cfg_.intermediate_size;
      const fuel::TensorView& probe = ckpt_.tensor(p + "mlp.gate_proj.weight");
      if (probe.dtype == fuel::DType::kU8) {
        // Hub fuel packs the dense MLP NVFP4. Scale is raw per-16-block e4m3
        // plus one f32 global and one f32 activation scale per projection,
        // matching ExpertDev's gemv operand convention.
        auto dense_fp4 = [&](int n, int k_true, Nvfp4W& w4, const std::uint8_t** sw,
                             float& a_in, const char* base) {
          const std::string p_full = p + base;
          const fuel::TensorView& wt = ckpt_.tensor(p_full + ".weight");
          if (wt.numel() != static_cast<std::int64_t>(n) * k_true / 2)
            fail(p_full + " has unexpected packed element count");
          auto* pd = static_cast<std::uint8_t*>(device_alloc(wt.numel()));
          if (std::getenv("ROCKET_RESIDENT_DIRECT")) copy_in_direct(pd, wt);
          else copy_in(pd, wt.data, wt.nbytes);
          w4.packed = pd;
          const fuel::TensorView& sc = ckpt_.tensor(p_full + ".weight_scale");
          auto* sd = static_cast<std::uint8_t*>(device_alloc(sc.numel()));
          if (std::getenv("ROCKET_RESIDENT_DIRECT")) copy_in_direct(sd, sc);
          else copy_in(sd, sc.data, sc.nbytes);
          w4.scale = sd;
          const fuel::SfLayout layout = fuel::sf_layout(n, k_true, 16);
          if (layout.bytes() != sc.nbytes)
            fail(p_full + ": scale size does not match the grouped SfLayout");
          std::vector<std::uint8_t> swizzled(layout.bytes());
          fuel::swizzle_block_scales(sc.data, layout, swizzled.data());
          auto* swd = static_cast<std::uint8_t*>(device_alloc(swizzled.size()));
          copy_in(swd, swizzled.data(), swizzled.size());
          *sw = swd;
          const fuel::TensorView& g2 = ckpt_.tensor(p_full + ".weight_scale_2");
          std::memcpy(&w4.global, g2.data, sizeof(float));
          const fuel::TensorView& is = ckpt_.tensor(p_full + ".input_scale");
          std::memcpy(&a_in, is.data, sizeof(float));
        };
        w.dense = DenseMlpW{};
        // gate|up in one contiguous slab so the grouped GEMM reads them as
        // the fused w13 operand, exactly like the MoE expert slots.
        {
          const fuel::TensorView& gw = ckpt_.tensor(p + "mlp.gate_proj.weight");
          const fuel::TensorView& uw = ckpt_.tensor(p + "mlp.up_proj.weight");
          const std::size_t per = static_cast<std::size_t>(I) * H / 2;
          auto* w13 = static_cast<std::uint8_t*>(device_alloc(2 * per));
          if (std::getenv("ROCKET_RESIDENT_DIRECT")) {
            copy_in_direct(w13, gw);
            copy_in_direct(w13 + per, uw);
          } else {
            copy_in(w13, gw.data, per);
            copy_in(w13 + per, uw.data, per);
          }
          w.dense.fp4_gate.packed = w13;
          w.dense.fp4_up.packed = w13 + per;
          const fuel::SfLayout lay = fuel::sf_layout(I, H, 16);
          auto* w13s = static_cast<std::uint8_t*>(device_alloc(2 * lay.bytes()));
          std::vector<std::uint8_t> sw(lay.bytes());
          fuel::swizzle_block_scales(ckpt_.tensor(p + "mlp.gate_proj.weight_scale").data, lay,
                                     sw.data());
          copy_in(w13s, sw.data(), sw.size());
          fuel::swizzle_block_scales(ckpt_.tensor(p + "mlp.up_proj.weight_scale").data, lay,
                                     sw.data());
          copy_in(w13s + lay.bytes(), sw.data(), sw.size());
          w.dense.fp4_gate_sw = w13s;
          w.dense.fp4_up_sw = w13s + lay.bytes();
          const fuel::TensorView& gg = ckpt_.tensor(p + "mlp.gate_proj.weight_scale_2");
          std::memcpy(&w.dense.fp4_gate.global, gg.data, sizeof(float));
          const fuel::TensorView& gi = ckpt_.tensor(p + "mlp.gate_proj.input_scale");
          std::memcpy(&w.dense.fp4_gate_in, gi.data, sizeof(float));
          const fuel::TensorView& ug = ckpt_.tensor(p + "mlp.up_proj.weight_scale_2");
          std::memcpy(&w.dense.fp4_up.global, ug.data, sizeof(float));
          const fuel::TensorView& ui = ckpt_.tensor(p + "mlp.up_proj.input_scale");
          std::memcpy(&w.dense.fp4_up_in, ui.data, sizeof(float));
        }
        dense_fp4(H, I, w.dense.fp4_down, &w.dense.fp4_down_sw, w.dense.fp4_down_in,
                  "mlp.down_proj");
      } else {
      w.dense.gate = upload_bf16(p + "mlp.gate_proj.weight", static_cast<std::int64_t>(I) * H);
      w.dense.up = upload_bf16(p + "mlp.up_proj.weight", static_cast<std::int64_t>(I) * H);
      w.dense.down = upload_bf16(p + "mlp.down_proj.weight", static_cast<std::int64_t>(H) * I);
      }
    } else {
      const int SI = cfg_.moe_intermediate_size * cfg_.n_shared_experts;
      w.moe.router = upload_bf16(p + "mlp.gate.weight",
                                 static_cast<std::int64_t>(cfg_.n_routed_experts) * H);
      w.moe.router_bias = upload_f32(p + "mlp.gate.e_score_correction_bias", cfg_.n_routed_experts);
      const char* shared_fp4_root = std::getenv("ROCKET_FP4_SHARED_DIR");
      const std::filesystem::path shared_fp4_dir = shared_fp4_root == nullptr
          ? std::filesystem::path{}
          : std::filesystem::path(shared_fp4_root) / ("l-" + std::to_string(l));
      const bool shared_fp4_available = shared_fp4_root != nullptr &&
          std::filesystem::exists(shared_fp4_dir / "gate_proj" / "weight.u8");
      if (!shared_fp4_available) {
        w.moe.shared.gate = upload_bf16(p + "mlp.shared_experts.gate_proj.weight",
                                        static_cast<std::int64_t>(SI) * H);
        w.moe.shared.up = upload_bf16(p + "mlp.shared_experts.up_proj.weight",
                                      static_cast<std::int64_t>(SI) * H);
        w.moe.shared.down = upload_bf16(p + "mlp.shared_experts.down_proj.weight",
                                        static_cast<std::int64_t>(H) * SI);
      }
      // Shared experts to NVFP4 (same operand convention as the routed
      // experts): a fused gate|up packed slab with pre-swizzled SFBs, down
      // separate, all under ROCKET_FP4_SHARED_DIR/l-<layer>/<proj>/.
      if (shared_fp4_available) {
        const std::filesystem::path& ld = shared_fp4_dir;
          const fuel::SfLayout lay = fuel::sf_layout(SI, H, 16);
          const auto gp = read_file(ld / "gate_proj" / "weight.u8",
                                    static_cast<std::size_t>(SI) * H / 2);
          const auto up = read_file(ld / "up_proj" / "weight.u8",
                                    static_cast<std::size_t>(SI) * H / 2);
          const auto dp = read_file(ld / "down_proj" / "weight.u8",
                                    static_cast<std::size_t>(H) * SI / 2);
          const auto gsc = read_file(ld / "gate_proj" / "weight_scale",
                                     static_cast<std::size_t>(SI) * H / 16);
          const auto usc = read_file(ld / "up_proj" / "weight_scale",
                                     static_cast<std::size_t>(SI) * H / 16);
          const auto dsc = read_file(ld / "down_proj" / "weight_scale",
                                     static_cast<std::size_t>(H) * SI / 16);
          const auto gg = read_file(ld / "gate_proj" / "weight_scale_2", sizeof(float));
          const auto ug = read_file(ld / "up_proj" / "weight_scale_2", sizeof(float));
          const auto dg = read_file(ld / "down_proj" / "weight_scale_2", sizeof(float));
          auto* w13 = static_cast<std::uint8_t*>(device_alloc(gp.size() + up.size()));
          copy_in(w13, gp.data(), gp.size());
          copy_in(w13 + gp.size(), up.data(), up.size());
          w.moe.shared.fp4_gate.packed = w13;
          w.moe.shared.fp4_up.packed = w13 + gp.size();
          auto* w13s = static_cast<std::uint8_t*>(device_alloc(2 * lay.bytes()));
          std::vector<std::uint8_t> sw(lay.bytes());
          fuel::swizzle_block_scales(gsc.data(), lay, sw.data());
          copy_in(w13s, sw.data(), sw.size());
          fuel::swizzle_block_scales(usc.data(), lay, sw.data());
          copy_in(w13s + lay.bytes(), sw.data(), sw.size());
          w.moe.shared.fp4_gate_sw = w13s;
          w.moe.shared.fp4_up_sw = w13s + lay.bytes();
          auto* dpd = static_cast<std::uint8_t*>(device_alloc(dp.size()));
          copy_in(dpd, dp.data(), dp.size());
          w.moe.shared.fp4_down.packed = dpd;
          auto* dsd = static_cast<std::uint8_t*>(device_alloc(dsc.size()));
          copy_in(dsd, dsc.data(), dsc.size());
          w.moe.shared.fp4_down_sw = dsd;
          std::memcpy(&w.moe.shared.fp4_gate.global, gg.data(), sizeof(float));
          std::memcpy(&w.moe.shared.fp4_up.global, ug.data(), sizeof(float));
          std::memcpy(&w.moe.shared.fp4_down.global, dg.data(), sizeof(float));
      }
    }
  }

  // --- streamed expert cache ----------------------------------------------
  // --- streamed NVFP4 expert cache -----------------------------------------
  // Layout, in order: gate_packed and up_packed sit back to back so the pair
  // doubles as the fused w13 grouped-GEMM B operand (weights.h::ExpertDev);
  // gate_scale/up_scale are the checkpoint's own linear layout for the GEMV
  // fallback; w13_scale is the fused, swizzled SFB for the grouped path;
  // down_packed/down_scale are the GEMV layout for down_proj; down_scale
  // gets a second, swizzled copy for the grouped path's w2 GEMM.
  const std::size_t gate_packed = static_cast<std::size_t>(cfg_.moe_intermediate_size) * H / 2;
  const std::size_t gate_scale = static_cast<std::size_t>(cfg_.moe_intermediate_size) * H / 16;
  const std::size_t down_packed = static_cast<std::size_t>(H) * cfg_.moe_intermediate_size / 2;
  const std::size_t down_scale = static_cast<std::size_t>(H) * cfg_.moe_intermediate_size / 16;
  // Swizzled scale bytes equal linear scale bytes for these dims: both
  // moe_intermediate_size and hidden_size divide the SfLayout 128-row atom
  // exactly (nvfp4.h::SfLayout), so there is no tile padding to pay for.
  const std::size_t w13_scale_sw = 2 * gate_scale;  // fused gate+up SFB
  slot_bytes_ =
      align_up(2 * (gate_packed + gate_scale) + w13_scale_sw + down_packed + 2 * down_scale, 512);

  std::size_t n_slots = expert_cache_bytes / slot_bytes_;
  if (n_slots < static_cast<std::size_t>(cfg_.num_experts_per_tok))
    fail("expert cache is smaller than one token's top-k working set");
  slots_.resize(n_slots);
  slot_view_.resize(n_slots);
  lru_pos_.resize(n_slots);
  // One cache slab keeps preload destinations contiguous. Besides avoiding
  // thousands of large cudaMalloc calls, this permits multi-expert DMA runs.
  expert_cache_slab_ = static_cast<std::uint8_t*>(device_alloc(n_slots * slot_bytes_));
  for (std::size_t i = 0; i < n_slots; ++i) {
    auto* base = expert_cache_slab_ + i * slot_bytes_;
    slots_[i].base = base;
    ExpertDev& v = slot_view_[i];
    std::size_t off = 0;
    v.gate_packed = base + off; off += gate_packed;
    v.up_packed = base + off;   off += gate_packed;  // contiguous: fused w13 B
    v.gate_scale = base + off;  off += gate_scale;
    v.up_scale = base + off;    off += gate_scale;
    v.w13_scale = base + off;   off += w13_scale_sw;
    v.down_packed = base + off; off += down_packed;
    v.down_scale = base + off;  off += down_scale;
    v.down_scale_swizzled = base + off; off += down_scale;
    lru_.push_front(static_cast<int>(i));
    lru_pos_[i] = lru_.begin();
  }
  { const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess)
      std::fprintf(stderr, "[ctor-end] sticky CUDA error: %s\n", cudaGetErrorString(ce));
    else
      std::fprintf(stderr,
                   "[ctor-end] clean; allocations %.3f s, resident uploads %.3f s "
                   "(stage %.3f s, H2D %.3f s)\n",
                   allocation_ms_ / 1000.0, upload_ms_ / 1000.0,
                   upload_stage_ms_ / 1000.0, upload_cuda_ms_ / 1000.0); }
  guard.armed = false;
}

WeightStore::~WeightStore() {
  for (void* p : owned_) cudaFree(p);
  for (const auto& [path, fd] : direct_fds_) {
    (void)path;
    ::close(fd);
  }
  if (pinned_raw_ != nullptr) cudaFreeHost(pinned_raw_);
}

void WeightStore::set_expert_range(int first, int count) {
  if (first < 0 || count <= 0 || first + count > cfg_.n_routed_experts)
    fail("expert range [" + std::to_string(first) + ", " + std::to_string(first + count) +
         ") does not fit " + std::to_string(cfg_.n_routed_experts) + " routed experts");
  std::vector<int> ids;
  ids.reserve(static_cast<std::size_t>(count));
  for (int e = first; e < first + count; ++e) ids.push_back(e);
  set_expert_set(ids);
}

void WeightStore::set_expert_set(const std::vector<int>& expert_ids) {
  if (expert_ids.empty()) fail("this rank was given no routed experts to own");
  build_expert_ownership(expert_ids);

  // The foreign half's down-projection weight_scale_2, read once here. These
  // are 4 bytes each out of tensors whose packed weights are never touched on
  // this rank, so this faults one checkpoint page per expert and nothing more.
  down_global_.assign(static_cast<std::size_t>(cfg_.text_layers) * cfg_.n_routed_experts, 0.0f);
  down_input_scale_.assign(static_cast<std::size_t>(cfg_.text_layers) * cfg_.n_routed_experts,
                           1.0f);
  // The slab carries all ranks' down scalars. Avoid faulting one 64 KiB
  // checkpoint page per expert when the slab fast path will replace them.
  if (std::getenv("ROCKET_EXPERT_SLAB_DIR") != nullptr) return;
  for (int l = 0; l < cfg_.text_layers; ++l) {
    if (cfg_.layers[l].mlp != fuel::MlpKind::kSparse) continue;
    const std::string p = layer_prefix(l) + "mlp.experts.";
    for (int e = 0; e < cfg_.n_routed_experts; ++e) {
      const fuel::TensorView& g2 = ckpt_.tensor(p + std::to_string(e) + ".down_proj.weight_scale_2");
      std::memcpy(&down_global_[static_cast<std::size_t>(l) * cfg_.n_routed_experts + e], g2.data,
                  sizeof(float));
      down_input_scale_[static_cast<std::size_t>(l) * cfg_.n_routed_experts + e] =
          optional_input_scale(ckpt_, p + std::to_string(e) + ".down_proj.input_scale");
    }
  }
}

void WeightStore::build_expert_ownership(const std::vector<int>& expert_ids) {
  owned_expert_.assign(static_cast<std::size_t>(cfg_.n_routed_experts), 0);
  expert_count_ = 0;
  for (const int e : expert_ids) {
    if (e < 0 || e >= cfg_.n_routed_experts)
      fail("expert id " + std::to_string(e) + " is outside the " +
           std::to_string(cfg_.n_routed_experts) + " routed experts");
    if (owned_expert_[static_cast<std::size_t>(e)] != 0)
      fail("expert id " + std::to_string(e) + " was given to this rank twice");
    owned_expert_[static_cast<std::size_t>(e)] = 1;
    ++expert_count_;
  }
}

float WeightStore::expert_down_global(int layer, int expert_id) const {
  const std::size_t i = static_cast<std::size_t>(layer) * cfg_.n_routed_experts + expert_id;
  return i < down_global_.size() ? down_global_[i] : 0.0f;
}

float WeightStore::expert_down_input_scale(int layer, int expert_id) const {
  const std::size_t i = static_cast<std::size_t>(layer) * cfg_.n_routed_experts + expert_id;
  return i < down_input_scale_.size() ? down_input_scale_[i] : 1.0f;
}

std::size_t WeightStore::preload_expert_slab(const std::filesystem::path& path,
                                              cudaStream_t s) {
  const int meta_fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (meta_fd < 0) fail("open " + path.string() + ": " + std::strerror(errno));
  ExpertSlabHeader h{};
  read_exact_at(meta_fd, &h, sizeof(h), 0, path);
  constexpr char magic[16] = {'R','O','C','K','E','T','E','X','P','E','R','T','1',0,0,0};
  if (std::memcmp(h.magic, magic, sizeof(magic)) != 0 || h.version != 1)
    fail(path.string() + " is not a version-1 Rocket expert slab");
  const auto config_digest = sha256_file(ckpt_.dir() / "config.json");
  const auto index_digest = sha256_file(ckpt_.dir() / "model.safetensors.index.json");
  if (std::memcmp(h.config_sha256, config_digest.data(), kSha256Bytes) != 0 ||
      std::memcmp(h.index_sha256, index_digest.data(), kSha256Bytes) != 0)
    fail(path.string() + " was materialized from a different checkpoint");
  if (h.page_bytes != kPageBytes || h.payload_offset % kPageBytes != 0 ||
      h.payload_bytes % kPageBytes != 0 || h.slot_bytes != slot_bytes_ ||
      h.hidden != static_cast<std::uint32_t>(cfg_.hidden_size) ||
      h.intermediate != static_cast<std::uint32_t>(cfg_.moe_intermediate_size) ||
      h.layers != static_cast<std::uint32_t>(cfg_.text_layers) ||
      h.experts != static_cast<std::uint32_t>(cfg_.n_routed_experts) ||
      h.expert_count != static_cast<std::uint32_t>(expert_count_))
    fail(path.string() + " geometry does not match this engine");
  int first = 0;
  while (first < cfg_.n_routed_experts && !owns_expert(first)) ++first;
  for (int e = 0; e < cfg_.n_routed_experts; ++e)
    if (owns_expert(e) != (e >= first && e < first + expert_count_))
      fail("expert slab requires one contiguous ownership range");
  if (h.first_expert != static_cast<std::uint32_t>(first))
    fail(path.string() + " belongs to a different expert rank");
  std::size_t sparse_layers = 0;
  for (const auto& layer : cfg_.layers)
    if (layer.mlp == fuel::MlpKind::kSparse) ++sparse_layers;
  const std::size_t needed = sparse_layers * static_cast<std::size_t>(expert_count_);
  if (h.sparse_layers != sparse_layers || h.slot_count != needed ||
      h.payload_bytes != needed * slot_bytes_ || slots_.size() < needed)
    fail(path.string() + " slot count does not match this engine");
  struct stat st{};
  if (::fstat(meta_fd, &st) != 0 || !S_ISREG(st.st_mode) ||
      static_cast<std::uint64_t>(st.st_size) != h.payload_offset + h.payload_bytes)
    fail(path.string() + " has the wrong byte count");
  std::vector<std::uint8_t> metadata(h.metadata_bytes);
  read_exact_at(meta_fd, metadata.data(), metadata.size(), kPageBytes, path);
  ::close(meta_fd);
  const std::size_t down_meta_bytes =
      static_cast<std::size_t>(cfg_.text_layers) * cfg_.n_routed_experts * 2 * sizeof(float);
  const std::size_t slot_meta_bytes = needed * 4 * sizeof(float);
  const std::size_t chunk_count =
      (static_cast<std::size_t>(h.payload_bytes) + kDirectChunkBytes - 1) /
      kDirectChunkBytes;
  const std::size_t digest_offset = down_meta_bytes + slot_meta_bytes;
  if (h.metadata_bytes < digest_offset + chunk_count * kSha256Bytes)
    fail(path.string() + " metadata is truncated");
  for (std::size_t i = 0; i < static_cast<std::size_t>(cfg_.text_layers) * cfg_.n_routed_experts;
       ++i) {
    std::memcpy(&down_global_[i], metadata.data() + i * 8, sizeof(float));
    std::memcpy(&down_input_scale_[i], metadata.data() + i * 8 + 4, sizeof(float));
  }

  const int direct_fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECT);
  if (direct_fd < 0) fail("open O_DIRECT " + path.string() + ": " + std::strerror(errno));
  std::array<void*, kDirectPipelineSlots> staging{};
  std::array<cudaEvent_t, kDirectPipelineSlots> ready{};
  std::array<bool, kDirectPipelineSlots> used{};
  std::array<std::future<std::array<unsigned char, kSha256Bytes>>,
             kDirectPipelineSlots> hashes{};
  std::array<std::size_t, kDirectPipelineSlots> hash_chunks{};
  for (int i = 0; i < kDirectPipelineSlots; ++i) {
    if (::posix_memalign(&staging[i], kPageBytes, kDirectChunkBytes) != 0)
      fail("allocate aligned expert slab staging");
    cuda_check(cudaHostRegister(staging[i], kDirectChunkBytes, cudaHostRegisterDefault),
               "cudaHostRegister expert slab staging");
    cuda_check(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming),
               "cudaEventCreate expert slab staging");
  }
  const auto started = std::chrono::steady_clock::now();
  double read_ms = 0.0;
  auto verify_hash = [&](int b) {
    if (!used[b]) return;
    cuda_check(cudaEventSynchronize(ready[b]), "expert slab staging reuse");
    const auto digest = hashes[b].get();
    const std::uint8_t* expected =
        metadata.data() + digest_offset + hash_chunks[b] * kSha256Bytes;
    if (std::memcmp(digest.data(), expected, kSha256Bytes) != 0)
      fail(path.string() + " chunk " + std::to_string(hash_chunks[b]) +
           " failed SHA-256 authentication");
  };
  std::size_t done = 0, chunk_index = 0;
  while (done < h.payload_bytes) {
    const int b = static_cast<int>(chunk_index % kDirectPipelineSlots);
    verify_hash(b);
    const std::size_t chunk = std::min(kDirectChunkBytes,
                                      static_cast<std::size_t>(h.payload_bytes) - done);
    const auto read_started = std::chrono::steady_clock::now();
    read_exact_at(direct_fd, staging[b], chunk, h.payload_offset + done, path);
    read_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - read_started).count();
    hashes[b] = std::async(std::launch::async, [src = staging[b], chunk] {
      std::array<unsigned char, kSha256Bytes> digest{};
      SHA256(static_cast<const unsigned char*>(src), chunk, digest.data());
      return digest;
    });
    hash_chunks[b] = chunk_index;
    cuda_check(cudaMemcpyAsync(expert_cache_slab_ + done, staging[b], chunk,
                               cudaMemcpyHostToDevice, s), "expert slab H2D");
    cuda_check(cudaEventRecord(ready[b], s), "expert slab staging ready");
    used[b] = true;
    done += chunk;
    ++chunk_index;
  }
  for (int b = 0; b < kDirectPipelineSlots; ++b) verify_hash(b);
  cuda_check(cudaStreamSynchronize(s), "expert slab final sync");
  const double total_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  for (int i = 0; i < kDirectPipelineSlots; ++i) {
    cudaEventDestroy(ready[i]);
    cudaHostUnregister(staging[i]);
    std::free(staging[i]);
  }
  ::close(direct_fd);

  const std::uint8_t* slot_meta = metadata.data() + down_meta_bytes;
  std::size_t slot = 0;
  for (int l = 0; l < cfg_.text_layers; ++l) {
    if (cfg_.layers[l].mlp != fuel::MlpKind::kSparse) continue;
    for (int e = first; e < first + expert_count_; ++e, ++slot) {
      const long long key = static_cast<long long>(l) * cfg_.n_routed_experts + e;
      ExpertDev& v = slot_view_[slot];
      std::memcpy(&v.gate_global, slot_meta + slot * 16, 4);
      std::memcpy(&v.gate_input_scale, slot_meta + slot * 16 + 4, 4);
      std::memcpy(&v.up_global, slot_meta + slot * 16 + 8, 4);
      std::memcpy(&v.up_input_scale, slot_meta + slot * 16 + 12, 4);
      v.down_global = down_global_[static_cast<std::size_t>(l) * cfg_.n_routed_experts + e];
      v.down_input_scale =
          down_input_scale_[static_cast<std::size_t>(l) * cfg_.n_routed_experts + e];
      slots_[slot].key = key;
      resident_expert_[key] = static_cast<int>(slot);
      lru_.erase(lru_pos_[slot]);
      lru_.push_front(static_cast<int>(slot));
      lru_pos_[slot] = lru_.begin();
    }
  }
  expert_slab_loaded_ = true;
  streamed_bytes_ += h.payload_bytes;
  std::fprintf(stderr,
               "expert slab: %.2f GiB in %.3f s (direct reads %.3f s, %.2f GiB/s)\n",
               static_cast<double>(h.payload_bytes) / (1ull << 30), total_ms / 1000.0,
               read_ms / 1000.0,
               static_cast<double>(h.payload_bytes) / (1ull << 30) / (total_ms / 1000.0));
  return needed;
}

std::size_t WeightStore::preload_owned_experts(cudaStream_t s) {
  if (expert_count_ <= 0) fail("preload_owned_experts needs an owned expert set");
  if (const char* dir = std::getenv("ROCKET_EXPERT_SLAB_DIR")) {
    int first = 0;
    while (first < cfg_.n_routed_experts && !owns_expert(first)) ++first;
    const int rank = first == 0 ? 0 : 1;
    return preload_expert_slab(std::filesystem::path(dir) /
                                   ("rank" + std::to_string(rank) + "-experts.slab"),
                               s);
  }
  std::size_t needed = 0;
  for (int l = 0; l < cfg_.text_layers; ++l)
    if (cfg_.layers[l].mlp == fuel::MlpKind::kSparse)
      needed += static_cast<std::size_t>(expert_count_);
  const std::size_t before = resident_expert_.size();
  if (slots_.size() < needed)
    fail("expert cache has " + std::to_string(slots_.size()) + " slots; preload needs " +
         std::to_string(needed));
  for (int l = 0; l < cfg_.text_layers; ++l) {
    if (cfg_.layers[l].mlp != fuel::MlpKind::kSparse) continue;
    for (int e = 0; e < cfg_.n_routed_experts; ++e)
      if (owns_expert(e)) (void)expert(l, e, s);
  }
  return resident_expert_.size() - before;
}

const ExpertDev& WeightStore::expert(int layer, int expert_id, cudaStream_t s) {
  const long long key = static_cast<long long>(layer) * cfg_.n_routed_experts + expert_id;
  if (!owns_expert(expert_id))
    fail("expert " + std::to_string(expert_id) + " of layer " + std::to_string(layer) +
         " is not in this rank's set of " + std::to_string(expert_count_) +
         " routed experts; the peer owns it");
  if (const auto it = resident_expert_.find(key); it != resident_expert_.end()) {
    ++hits_;
    const int slot = it->second;
    lru_.erase(lru_pos_[slot]);
    lru_.push_front(slot);
    lru_pos_[slot] = lru_.begin();
    return slot_view_[slot];
  }
  ++misses_;
  const int slot = lru_.back();
  lru_.pop_back();
  if (slots_[slot].key >= 0) resident_expert_.erase(slots_[slot].key);

  const std::string p = layer_prefix(layer) + "mlp.experts." + std::to_string(expert_id) + ".";
  ExpertDev& v = slot_view_[slot];
  const int MI = cfg_.moe_intermediate_size, H = cfg_.hidden_size;
  struct Piece {
    const char* proj;
    const std::uint8_t* dst_packed;
    const std::uint8_t* dst_scale;     // linear, GEMV
    const std::uint8_t* dst_scale_sw;  // swizzled, grouped GEMM
    float* global;
    float* input_scale;
    std::int64_t n, k;  // this projection's [out_features, in_features]
  };
  const Piece pieces[3] = {
      {"gate_proj", v.gate_packed, v.gate_scale, v.w13_scale, &v.gate_global,
       &v.gate_input_scale, MI, H},
      {"up_proj", v.up_packed, v.up_scale, v.w13_scale, &v.up_global, &v.up_input_scale, MI, H},
      {"down_proj", v.down_packed, v.down_scale, v.down_scale_swizzled, &v.down_global,
       &v.down_input_scale, H, MI},
  };
  const auto* slot_base = slots_[slot].base;
  if (slot_bytes_ > pinned_bytes_) fail("expert slot exceeds pinned staging buffer");
  for (int pi = 0; pi < 3; ++pi) {
    const Piece& pc = pieces[pi];
    const fuel::TensorView& packed = ckpt_.tensor(p + pc.proj + ".weight");
    const fuel::TensorView& scale = ckpt_.tensor(p + pc.proj + ".weight_scale");
    const fuel::TensorView& g2 = ckpt_.tensor(p + pc.proj + ".weight_scale_2");
    const fuel::SfLayout layout = fuel::sf_layout(pc.n, pc.k, 16);
    if (layout.bytes() != scale.nbytes)
      fail(p + pc.proj + ": swizzled scale size does not match the linear checkpoint size");
    const std::uint8_t* dst_sw =
        (pi == 1) ? v.w13_scale + layout.bytes() : pc.dst_scale_sw;
    const auto packed_off = static_cast<std::size_t>(pc.dst_packed - slot_base);
    const auto scale_off = static_cast<std::size_t>(pc.dst_scale - slot_base);
    const auto sw_off = static_cast<std::size_t>(dst_sw - slot_base);
    std::memcpy(pinned_ + packed_off, packed.data, packed.nbytes);
    std::memcpy(pinned_ + scale_off, scale.data, scale.nbytes);
    fuel::swizzle_block_scales(pinned_ + scale_off, layout, pinned_ + sw_off);
    std::memcpy(pc.global, g2.data, sizeof(float));
    *pc.input_scale = optional_input_scale(ckpt_, p + pc.proj + ".input_scale");
    streamed_bytes_ += packed.nbytes + scale.nbytes + layout.bytes();
  }
  // The slot is already in its final device layout. One transfer and one
  // synchronization replace nine copies and three synchronizations.
  cuda_check(cudaMemcpyAsync(slots_[slot].base, pinned_, slot_bytes_, cudaMemcpyHostToDevice, s),
             "expert slot H2D");
  cuda_check(cudaStreamSynchronize(s), "expert slot sync");

  slots_[slot].key = key;
  resident_expert_[key] = slot;
  lru_.push_front(slot);
  lru_pos_[slot] = lru_.begin();
  return v;
}

}  // namespace rocket::engine

#include "weights.h"

#include "nvfp4.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

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

}  // namespace

void* WeightStore::device_alloc(std::size_t bytes) {
  void* p = nullptr;
  cuda_check(cudaMalloc(&p, bytes), "cudaMalloc " + std::to_string(bytes) + " B");
  owned_.push_back(p);
  resident_bytes_ += bytes;
  return p;
}

void WeightStore::copy_in(void* dst, const void* src, std::size_t bytes) {
  std::size_t done = 0;
  while (done < bytes) {
    const std::size_t chunk = std::min(pinned_bytes_, bytes - done);
    std::memcpy(pinned_, static_cast<const std::uint8_t*>(src) + done, chunk);
    cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(dst) + done, pinned_, chunk,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy weight");
    done += chunk;
  }
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
  const fuel::TensorView& t = ckpt_.tensor(name);
  if (t.dtype != fuel::DType::kBF16) fail(std::string(name) + " is not BF16");
  if (t.numel() != expect_numel)
    fail(std::string(name) + " has " + std::to_string(t.numel()) + " elements, expected " +
         std::to_string(expect_numel));
  void* d = device_alloc(t.nbytes);
  copy_in(d, t.data, t.nbytes);
  record_dtype(name, fuel::DType::kBF16);
  return static_cast<const bf16*>(d);
}

const float* WeightStore::upload_f32(std::string_view name, std::int64_t expect_numel) {
  const fuel::TensorView& t = ckpt_.tensor(name);
  if (t.dtype != fuel::DType::kF32) fail(std::string(name) + " is not F32");
  if (t.numel() != expect_numel)
    fail(std::string(name) + " has the wrong element count");
  void* d = device_alloc(t.nbytes);
  copy_in(d, t.data, t.nbytes);
  record_dtype(name, fuel::DType::kF32);
  return static_cast<const float*>(d);
}

const bf16* WeightStore::upload_concat(const std::vector<std::string>& names,
                                       std::int64_t expect_numel) {
  std::size_t total = 0;
  std::int64_t numel = 0;
  for (const std::string& n : names) {
    const fuel::TensorView& t = ckpt_.tensor(n);
    if (t.dtype != fuel::DType::kBF16) fail(n + " is not BF16");
    total += t.nbytes;
    numel += t.numel();
  }
  if (numel != expect_numel) fail(names.front() + " concat has the wrong element count");
  auto* d = static_cast<std::uint8_t*>(device_alloc(total));
  std::size_t off = 0;
  for (const std::string& n : names) {
    const fuel::TensorView& t = ckpt_.tensor(n);
    copy_in(d + off, t.data, t.nbytes);
    off += t.nbytes;
  }
  return reinterpret_cast<const bf16*>(d);
}

WeightStore::WeightStore(const fuel::ModelConfig& cfg, const std::filesystem::path& snapshot_dir,
                         std::size_t expert_cache_bytes)
    : cfg_(cfg), ckpt_(snapshot_dir) {
  pinned_bytes_ = 64u << 20;
  cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&pinned_), pinned_bytes_, cudaHostAllocDefault),
             "cudaHostAlloc staging");

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
      w.kda.qkv = upload_concat({a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"},
                                static_cast<std::int64_t>(3) * qkv * H);
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
      w.kda.o_proj = upload_bf16(a + "o_proj.weight", static_cast<std::int64_t>(H) * qkv);
    } else {
      const std::string a = p + "self_attn.";
      w.mla.q_a = upload_bf16(a + "q_a_proj.weight", static_cast<std::int64_t>(cfg_.q_lora_rank) * H);
      w.mla.q_a_norm = upload_bf16(a + "q_a_layernorm.weight", cfg_.q_lora_rank);
      w.mla.q_b = upload_bf16(a + "q_b_proj.weight",
                              static_cast<std::int64_t>(cfg_.mla_heads) * qk * cfg_.q_lora_rank);
      w.mla.kv_a = upload_bf16(a + "kv_a_proj_with_mqa.weight",
                               static_cast<std::int64_t>(cfg_.kv_lora_rank + cfg_.qk_rope_head_dim) * H);
      w.mla.kv_a_norm = upload_bf16(a + "kv_a_layernorm.weight", cfg_.kv_lora_rank);
      w.mla.kv_b = upload_bf16(a + "kv_b_proj.weight",
                               static_cast<std::int64_t>(cfg_.mla_kv_b_out()) * cfg_.kv_lora_rank);
      w.mla.o_proj = upload_bf16(a + "o_proj.weight",
                                 static_cast<std::int64_t>(H) * cfg_.mla_heads * cfg_.v_head_dim);
      const std::string ix = a + "indexer.";
      w.mla.idx_wk = upload_bf16(ix + "wk.weight", static_cast<std::int64_t>(cfg_.index_head_dim) * H);
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
      w.dense.gate = upload_bf16(p + "mlp.gate_proj.weight", static_cast<std::int64_t>(I) * H);
      w.dense.up = upload_bf16(p + "mlp.up_proj.weight", static_cast<std::int64_t>(I) * H);
      w.dense.down = upload_bf16(p + "mlp.down_proj.weight", static_cast<std::int64_t>(H) * I);
    } else {
      const int SI = cfg_.moe_intermediate_size * cfg_.n_shared_experts;
      w.moe.router = upload_bf16(p + "mlp.gate.weight",
                                 static_cast<std::int64_t>(cfg_.n_routed_experts) * H);
      w.moe.router_bias = upload_f32(p + "mlp.gate.e_score_correction_bias", cfg_.n_routed_experts);
      w.moe.shared.gate =
          upload_bf16(p + "mlp.shared_experts.gate_proj.weight", static_cast<std::int64_t>(SI) * H);
      w.moe.shared.up =
          upload_bf16(p + "mlp.shared_experts.up_proj.weight", static_cast<std::int64_t>(SI) * H);
      w.moe.shared.down =
          upload_bf16(p + "mlp.shared_experts.down_proj.weight", static_cast<std::int64_t>(H) * SI);
    }
  }

  // --- streamed expert cache ----------------------------------------------
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
  for (std::size_t i = 0; i < n_slots; ++i) {
    auto* base = static_cast<std::uint8_t*>(device_alloc(slot_bytes_));
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
}

WeightStore::~WeightStore() {
  for (void* p : owned_) cudaFree(p);
  if (pinned_ != nullptr) cudaFreeHost(pinned_);
}

const ExpertDev& WeightStore::expert(int layer, int expert_id, cudaStream_t s) {
  const long long key = static_cast<long long>(layer) * cfg_.n_routed_experts + expert_id;
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
    std::int64_t n, k;  // this projection's [out_features, in_features]
  };
  const Piece pieces[3] = {
      {"gate_proj", v.gate_packed, v.gate_scale, v.w13_scale, &v.gate_global, MI, H},
      {"up_proj", v.up_packed, v.up_scale, v.w13_scale, &v.up_global, MI, H},
      {"down_proj", v.down_packed, v.down_scale, v.down_scale_swizzled, &v.down_global, H, MI},
  };
  for (int pi = 0; pi < 3; ++pi) {
    const Piece& pc = pieces[pi];
    const fuel::TensorView& packed = ckpt_.tensor(p + pc.proj + ".weight");
    const fuel::TensorView& scale = ckpt_.tensor(p + pc.proj + ".weight_scale");
    const fuel::TensorView& g2 = ckpt_.tensor(p + pc.proj + ".weight_scale_2");
    const fuel::SfLayout layout = fuel::sf_layout(pc.n, pc.k, 16);
    if (layout.bytes() != scale.nbytes)
      fail(p + pc.proj + ": swizzled scale size does not match the linear checkpoint size");
    // Staged through pinned memory: the source is a page in the mmap'd
    // checkpoint, so this is the disk-or-page-cache read plus one DMA. The
    // swizzle is a host-side byte permutation (nvfp4.cc::swizzle_block_scales)
    // between the linear and swizzled copies, both staged in the same pinned
    // buffer; test-loader-swizzle measures it at ~1 GB/s of scale bytes,
    // which hides under the NVMe read of the packed weight it swizzles for.
    std::memcpy(pinned_, packed.data, packed.nbytes);
    std::memcpy(pinned_ + packed.nbytes, scale.data, scale.nbytes);
    std::vector<std::uint8_t> swizzled(layout.bytes());
    fuel::swizzle_block_scales(pinned_ + packed.nbytes, layout, swizzled.data());
    std::memcpy(pinned_ + packed.nbytes + scale.nbytes, swizzled.data(), swizzled.size());

    cuda_check(cudaMemcpyAsync(const_cast<std::uint8_t*>(pc.dst_packed), pinned_, packed.nbytes,
                               cudaMemcpyHostToDevice, s),
               "expert packed H2D");
    cuda_check(cudaMemcpyAsync(const_cast<std::uint8_t*>(pc.dst_scale), pinned_ + packed.nbytes,
                               scale.nbytes, cudaMemcpyHostToDevice, s),
               "expert scale H2D");
    // up_proj's swizzled half lands after gate's in the fused w13_scale slab
    // (weights.h::ExpertDev), which is exactly one gate-sized swizzle.
    const std::uint8_t* dst_sw = (pi == 1) ? v.w13_scale + swizzled.size() : pc.dst_scale_sw;
    cuda_check(cudaMemcpyAsync(const_cast<std::uint8_t*>(dst_sw),
                               pinned_ + packed.nbytes + scale.nbytes, swizzled.size(),
                               cudaMemcpyHostToDevice, s),
               "expert scale (swizzled) H2D");
    cuda_check(cudaStreamSynchronize(s), "expert stage sync");
    std::memcpy(pc.global, g2.data, sizeof(float));
    streamed_bytes_ += packed.nbytes + scale.nbytes + swizzled.size();
  }

  slots_[slot].key = key;
  resident_expert_[key] = slot;
  lru_.push_front(slot);
  lru_pos_[slot] = lru_.begin();
  return v;
}

}  // namespace rocket::engine

// Torch-free DFlash2 safetensors loader. The 2.2 GiB checkpoint is one shard;
// all tensors are resident because the draft runs every generation round.
#include "dflash2.h"
#include "safetensors.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

namespace rocket::engine {
namespace {
constexpr std::size_t kPageBytes = 65536;
constexpr std::size_t kChunkBytes = 256u << 20;
std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}
void cuda_ok(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("dflash2: ") + what + ": " +
                                                 cudaGetErrorString(e));
}
}  // namespace

void DFlash2Weights::load(const std::filesystem::path& ckpt_file) {
  if (!tensors_dev_.empty()) throw std::runtime_error("dflash2: weights already loaded");
  rocket::fuel::Shard shard(ckpt_file);
  const std::uint8_t* first = nullptr;
  const std::uint8_t* last = nullptr;
  for (const auto& [name, tv] : shard.tensors()) {
    if (name == "__metadata__" || tv.nbytes == 0) continue;
    if (tv.dtype != rocket::fuel::DType::kBF16)
      throw std::runtime_error("dflash2: non-BF16 tensor " + name);
    first = first ? std::min(first, tv.data) : tv.data;
    last = last ? std::max(last, tv.data + tv.nbytes) : tv.data + tv.nbytes;
  }
  if (!first || !last || last <= first) throw std::runtime_error("dflash2: empty tensor payload");
  const std::size_t slab_bytes = static_cast<std::size_t>(last - first);
  const std::uint64_t first_file_offset = [&] {
    for (const auto& [name, tv] : shard.tensors())
      if (tv.data == first) return tv.file_offset;
    throw std::runtime_error("dflash2: first tensor file offset missing");
  }();
  cuda_ok(cudaMalloc(&device_slab_, slab_bytes), "slab allocation");
  void* raw = nullptr;
  cuda_ok(cudaHostAlloc(&raw, kChunkBytes + kPageBytes, cudaHostAllocDefault),
          "direct staging allocation");
  auto* staging = reinterpret_cast<std::uint8_t*>(
      align_up(reinterpret_cast<std::uintptr_t>(raw), kPageBytes));
  const int fd = ::open(ckpt_file.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECT);
  if (fd < 0) {
    cudaFreeHost(raw);
    throw std::runtime_error("dflash2: O_DIRECT open: " + std::string(std::strerror(errno)));
  }
  const auto upload_started = std::chrono::steady_clock::now();
  std::size_t done = 0;
  while (done < slab_bytes) {
    const std::uint64_t position = first_file_offset + done;
    const std::uint64_t aligned = position / kPageBytes * kPageBytes;
    const std::size_t leading = static_cast<std::size_t>(position - aligned);
    const std::size_t logical = std::min(slab_bytes - done, kChunkBytes - leading);
    const std::size_t request = align_up(leading + logical, kPageBytes);
    ssize_t got;
    do {
      got = ::pread(fd, staging, request, static_cast<off_t>(aligned));
    } while (got < 0 && errno == EINTR);
    if (got < 0 || static_cast<std::size_t>(got) < leading + logical) {
      ::close(fd);
      cudaFreeHost(raw);
      throw std::runtime_error("dflash2: short O_DIRECT read");
    }
    cuda_ok(cudaMemcpy(static_cast<std::uint8_t*>(device_slab_) + done, staging + leading,
                       logical, cudaMemcpyHostToDevice), "direct slab upload");
    done += logical;
  }
  ::close(fd);
  cudaFreeHost(raw);
  const double upload_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - upload_started).count();
  std::fprintf(stderr, "dflash2 direct: %.2f GiB in %.3f s (%.2f GiB/s)\n",
               static_cast<double>(slab_bytes) / (1ull << 30), upload_s,
               static_cast<double>(slab_bytes) / (1ull << 30) / upload_s);
  for (const auto& [name, tv] : shard.tensors()) {
    if (name == "__metadata__" || tv.nbytes == 0) continue;
    void* dev = static_cast<std::uint8_t*>(device_slab_) + (tv.data - first);
    tensors_dev_.emplace(name, dev);
    bytes_.emplace(name, tv.nbytes);
    shapes_.emplace(name, tv.shape);
  }
  auto require_shape = [&](const std::string& name,
                           std::initializer_list<std::int64_t> expected) {
    const auto it = shapes_.find(name);
    if (it == shapes_.end() || it->second != std::vector<std::int64_t>(expected))
      throw std::runtime_error("dflash2: incompatible or missing tensor " + name);
  };
  require_shape("fc.weight", {cfg.hidden_size, 5ll * cfg.hidden_size});
  require_shape("hidden_norm.weight", {cfg.hidden_size});
  require_shape("norm.weight", {cfg.hidden_size});
  require_shape("candidate_selector.hidden_projection.weight",
                {cfg.selector_rank, cfg.hidden_size});
  require_shape("candidate_selector.predecessor_codebook", {cfg.vocab_size, cfg.selector_rank});
  require_shape("candidate_selector.successor_codebook", {cfg.vocab_size, cfg.selector_rank});
  for (int l = 0; l < cfg.num_layers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    for (const char* n : {"input_layernorm.weight", "post_attention_layernorm.weight"})
      require_shape(p + n, {cfg.hidden_size});
    for (const char* site : {"attention_conv.", "mlp_conv."}) {
      require_shape(p + site + "base_kernel",
                    {cfg.conv_kernel_size, cfg.conv_kernel_size, cfg.hidden_size});
      require_shape(p + site + "kernel_projection.weight",
                    {cfg.hidden_size / cfg.conv_group_size * 4, cfg.hidden_size});
    }
    require_shape(p + "self_attn.q_proj.weight", {cfg.hidden_size, cfg.hidden_size});
    require_shape(p + "self_attn.k_proj.weight",
                  {cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size});
    require_shape(p + "self_attn.v_proj.weight",
                  {cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size});
    require_shape(p + "self_attn.o_proj.weight", {cfg.hidden_size, cfg.hidden_size});
    require_shape(p + "self_attn.q_norm.weight", {cfg.head_dim});
    require_shape(p + "self_attn.k_norm.weight", {cfg.head_dim});
    require_shape(p + "mlp.gate_proj.weight", {cfg.intermediate_size, cfg.hidden_size});
    require_shape(p + "mlp.up_proj.weight", {cfg.intermediate_size, cfg.hidden_size});
    require_shape(p + "mlp.down_proj.weight", {cfg.hidden_size, cfg.intermediate_size});
  }
}

const void* DFlash2Weights::tensor_data(const char* name) const {
  const auto it = tensors_dev_.find(name);
  return it == tensors_dev_.end() ? nullptr : it->second;
}

std::size_t DFlash2Weights::tensor_bytes(const char* name) const {
  const auto it = bytes_.find(name);
  return it == bytes_.end() ? 0 : it->second;
}

const std::vector<std::int64_t>& DFlash2Weights::tensor_shape(const char* name) const {
  const auto it = shapes_.find(name);
  if (it == shapes_.end()) throw std::runtime_error(std::string("dflash2: missing tensor ") + name);
  return it->second;
}

DFlash2Weights::~DFlash2Weights() { cudaFree(device_slab_); }

}  // namespace rocket::engine

// Torch-free DFlash2 safetensors loader. The 2.2 GiB checkpoint is one shard;
// all tensors are resident because the draft runs every generation round.
#include "dflash2.h"
#include "safetensors.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace rocket::engine {
namespace {
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
  cuda_ok(cudaMalloc(&device_slab_, slab_bytes), "slab allocation");
  cuda_ok(cudaMemcpy(device_slab_, first, slab_bytes, cudaMemcpyHostToDevice), "slab upload");
  for (const auto& [name, tv] : shard.tensors()) {
    if (name == "__metadata__" || tv.nbytes == 0) continue;
    void* dev = static_cast<std::uint8_t*>(device_slab_) + (tv.data - first);
    tensors_dev_.emplace(name, dev);
    bytes_.emplace(name, tv.nbytes);
    shapes_.emplace(name, tv.shape);
  }
  if (tensors_dev_.size() != 81)
    throw std::runtime_error("dflash2: expected 81 tensors, loaded " +
                             std::to_string(tensors_dev_.size()));
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

// Minimal safetensors reader for the fuel loader.
//
// Serving must not depend on Python, so the checkpoint is read here: mmap the
// shard, parse the 8-byte little-endian header length followed by that many
// bytes of JSON, and hand out tensor spans that point straight into the map.
// No tensor bytes are copied by this file.
//
// The checkpoint lives in an HF/NGC cache snapshot directory whose entries are
// symlinks into ../../blobs/<hash>. open() and mmap() follow those links, so
// the layout needs no special handling beyond resolving the real path for
// reporting (see Shard::blob_path).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rocket::fuel {

enum class DType {
  kBool, kU8, kI8, kF8E4M3, kF8E5M2,
  kU16, kI16, kF16, kBF16,
  kU32, kI32, kF32,
  kU64, kI64, kF64,
};

std::size_t dtype_bytes(DType dt);
std::string_view dtype_name(DType dt);
// Throws std::runtime_error on an unknown safetensors dtype string.
DType dtype_from_string(std::string_view s);

// A span into an mmap'd shard. Valid as long as the owning Shard is alive.
struct TensorView {
  std::string name;
  DType dtype = DType::kU8;
  std::vector<std::int64_t> shape;
  const std::uint8_t* data = nullptr;
  std::size_t nbytes = 0;

  std::int64_t numel() const;
};

// One mmap'd .safetensors file.
class Shard {
 public:
  // Throws std::runtime_error if the file cannot be opened, mapped, or parsed.
  explicit Shard(const std::filesystem::path& file);
  ~Shard();
  Shard(Shard&&) noexcept;
  Shard& operator=(Shard&&) noexcept;
  Shard(const Shard&) = delete;
  Shard& operator=(const Shard&) = delete;

  const TensorView* find(std::string_view name) const;
  // Throws std::runtime_error if absent.
  const TensorView& at(std::string_view name) const;

  const std::map<std::string, TensorView, std::less<>>& tensors() const { return tensors_; }
  const std::filesystem::path& path() const { return path_; }
  // The resolved target, i.e. the HF-cache blob the snapshot entry links to.
  const std::filesystem::path& blob_path() const { return blob_; }
  std::size_t file_bytes() const { return map_bytes_; }
  std::size_t header_bytes() const { return header_bytes_; }

 private:
  std::filesystem::path path_;
  std::filesystem::path blob_;
  void* map_ = nullptr;
  std::size_t map_bytes_ = 0;
  std::size_t header_bytes_ = 0;
  std::map<std::string, TensorView, std::less<>> tensors_;
};

// A sharded checkpoint: model.safetensors.index.json plus its shard files.
// Shards are mmap'd on first use, so touching one expert maps one shard.
class Checkpoint {
 public:
  // Throws std::runtime_error if the directory or its index is missing.
  explicit Checkpoint(std::filesystem::path snapshot_dir);

  bool has(std::string_view name) const;
  // Maps the owning shard if needed. Throws std::runtime_error if absent.
  const TensorView& tensor(std::string_view name);
  std::vector<std::string> names_with_prefix(std::string_view prefix) const;

  const std::filesystem::path& dir() const { return dir_; }
  std::size_t tensor_count() const { return weight_map_.size(); }
  std::size_t mapped_shard_count() const { return shards_.size(); }
  const Shard& mapped_shard_for(std::string_view name) const;

 private:
  Shard& shard_for(std::string_view name);

  std::filesystem::path dir_;
  std::map<std::string, std::string, std::less<>> weight_map_;
  std::map<std::string, std::unique_ptr<Shard>, std::less<>> shards_;
};

// The default on-disk location of the NVFP4 GLM-5.3-Flash checkpoint, with ~
// expanded. Empty if $HOME is unset. Override with $ROCKET_FUEL_NVFP4_DIR.
std::filesystem::path default_nvfp4_snapshot_dir();

}  // namespace rocket::fuel

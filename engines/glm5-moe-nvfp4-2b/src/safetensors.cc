#include "safetensors.h"

#include "json.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

namespace rocket::fuel {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("rocket::fuel: " + what); }

using json::JsonParser;
using json::JsonValue;


}  // namespace

// ---------------------------------------------------------------------------
// dtypes
// ---------------------------------------------------------------------------

std::size_t dtype_bytes(DType dt) {
  switch (dt) {
    case DType::kBool: case DType::kU8: case DType::kI8:
    case DType::kF8E4M3: case DType::kF8E5M2:
      return 1;
    case DType::kU16: case DType::kI16: case DType::kF16: case DType::kBF16:
      return 2;
    case DType::kU32: case DType::kI32: case DType::kF32:
      return 4;
    case DType::kU64: case DType::kI64: case DType::kF64:
      return 8;
  }
  return 0;
}

std::string_view dtype_name(DType dt) {
  switch (dt) {
    case DType::kBool: return "BOOL";
    case DType::kU8: return "U8";
    case DType::kI8: return "I8";
    case DType::kF8E4M3: return "F8_E4M3";
    case DType::kF8E5M2: return "F8_E5M2";
    case DType::kU16: return "U16";
    case DType::kI16: return "I16";
    case DType::kF16: return "F16";
    case DType::kBF16: return "BF16";
    case DType::kU32: return "U32";
    case DType::kI32: return "I32";
    case DType::kF32: return "F32";
    case DType::kU64: return "U64";
    case DType::kI64: return "I64";
    case DType::kF64: return "F64";
  }
  return "?";
}

DType dtype_from_string(std::string_view s) {
  if (s == "BOOL") return DType::kBool;
  if (s == "U8") return DType::kU8;
  if (s == "I8") return DType::kI8;
  if (s == "F8_E4M3") return DType::kF8E4M3;
  if (s == "F8_E5M2") return DType::kF8E5M2;
  if (s == "U16") return DType::kU16;
  if (s == "I16") return DType::kI16;
  if (s == "F16") return DType::kF16;
  if (s == "BF16") return DType::kBF16;
  if (s == "U32") return DType::kU32;
  if (s == "I32") return DType::kI32;
  if (s == "F32") return DType::kF32;
  if (s == "U64") return DType::kU64;
  if (s == "I64") return DType::kI64;
  if (s == "F64") return DType::kF64;
  fail("unknown safetensors dtype '" + std::string(s) + "'");
}

std::int64_t TensorView::numel() const {
  std::int64_t n = 1;
  for (std::int64_t d : shape) n *= d;
  return n;
}

// ---------------------------------------------------------------------------
// Shard
// ---------------------------------------------------------------------------

Shard::Shard(const std::filesystem::path& file) : path_(file) {
  std::error_code ec;
  blob_ = std::filesystem::canonical(file, ec);
  if (ec) blob_ = file;

  int fd = ::open(file.c_str(), O_RDONLY);
  if (fd < 0) fail("open " + file.string() + ": " + std::strerror(errno));

  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    fail("fstat " + file.string() + ": " + std::strerror(errno));
  }
  map_bytes_ = static_cast<std::size_t>(st.st_size);
  if (map_bytes_ < 8) {
    ::close(fd);
    fail(file.string() + " is too small to be a safetensors file");
  }

  map_ = ::mmap(nullptr, map_bytes_, PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (map_ == MAP_FAILED) {
    map_ = nullptr;
    fail("mmap " + file.string() + ": " + std::strerror(errno));
  }

  const auto* base = static_cast<const std::uint8_t*>(map_);
  std::uint64_t header_len = 0;
  std::memcpy(&header_len, base, 8);  // little-endian, and so is every target here
  if (header_len > map_bytes_ - 8) fail(file.string() + ": header length runs past end of file");
  header_bytes_ = static_cast<std::size_t>(header_len);

  const char* json_begin = reinterpret_cast<const char*>(base + 8);
  JsonParser parser(json_begin, json_begin + header_len);
  JsonValue header = parser.parse_document();
  if (header.kind != JsonValue::Kind::kObject) fail(file.string() + ": header is not a JSON object");

  const std::uint8_t* payload = base + 8 + header_len;
  const std::size_t payload_bytes = map_bytes_ - 8 - header_len;

  for (const auto& [name, entry] : *header.object) {
    if (name == "__metadata__") continue;
    const JsonValue* dtype = entry.member("dtype");
    const JsonValue* shape = entry.member("shape");
    const JsonValue* offsets = entry.member("data_offsets");
    if (!dtype || !shape || !offsets) continue;  // not a tensor record

    TensorView tv;
    tv.name = name;
    tv.dtype = dtype_from_string(dtype->str);
    if (shape->array) {
      for (const JsonValue& d : *shape->array) tv.shape.push_back(static_cast<std::int64_t>(d.number));
    }
    if (!offsets->array || offsets->array->size() != 2)
      fail(file.string() + ": tensor '" + name + "' has malformed data_offsets");
    const auto begin = static_cast<std::size_t>((*offsets->array)[0].number);
    const auto end = static_cast<std::size_t>((*offsets->array)[1].number);
    if (end < begin || end > payload_bytes)
      fail(file.string() + ": tensor '" + name + "' offsets run past end of file");

    const std::size_t expect_bytes =
        static_cast<std::size_t>(tv.numel()) * dtype_bytes(tv.dtype);
    if (end - begin != expect_bytes)
      fail(file.string() + ": tensor '" + name + "' byte span does not match shape x dtype");

    tv.data = payload + begin;
    tv.nbytes = end - begin;
    tensors_.emplace(name, std::move(tv));
  }
}

Shard::~Shard() {
  if (map_ != nullptr) ::munmap(map_, map_bytes_);
}

Shard::Shard(Shard&& other) noexcept
    : path_(std::move(other.path_)),
      blob_(std::move(other.blob_)),
      map_(other.map_),
      map_bytes_(other.map_bytes_),
      header_bytes_(other.header_bytes_),
      tensors_(std::move(other.tensors_)) {
  other.map_ = nullptr;
  other.map_bytes_ = 0;
}

Shard& Shard::operator=(Shard&& other) noexcept {
  if (this != &other) {
    if (map_ != nullptr) ::munmap(map_, map_bytes_);
    path_ = std::move(other.path_);
    blob_ = std::move(other.blob_);
    map_ = other.map_;
    map_bytes_ = other.map_bytes_;
    header_bytes_ = other.header_bytes_;
    tensors_ = std::move(other.tensors_);
    other.map_ = nullptr;
    other.map_bytes_ = 0;
  }
  return *this;
}

const TensorView* Shard::find(std::string_view name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView& Shard::at(std::string_view name) const {
  const TensorView* tv = find(name);
  if (tv == nullptr) fail("tensor '" + std::string(name) + "' not in " + path_.string());
  return *tv;
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

Checkpoint::Checkpoint(std::filesystem::path snapshot_dir) : dir_(std::move(snapshot_dir)) {
  if (!std::filesystem::exists(dir_)) fail("checkpoint dir does not exist: " + dir_.string());

  const std::filesystem::path index = dir_ / "model.safetensors.index.json";
  if (!std::filesystem::exists(index)) {
    // Single-file checkpoint: map it and take its tensor list as the map.
    const std::filesystem::path single = dir_ / "model.safetensors";
    if (!std::filesystem::exists(single))
      fail("no model.safetensors.index.json or model.safetensors in " + dir_.string());
    auto shard = std::make_unique<Shard>(single);
    for (const auto& [name, tv] : shard->tensors()) weight_map_.emplace(name, "model.safetensors");
    shards_.emplace("model.safetensors", std::move(shard));
    return;
  }

  int fd = ::open(index.c_str(), O_RDONLY);
  if (fd < 0) fail("open " + index.string() + ": " + std::strerror(errno));
  std::string text;
  char buf[65536];
  ssize_t n;
  while ((n = ::read(fd, buf, sizeof buf)) > 0) text.append(buf, static_cast<std::size_t>(n));
  ::close(fd);
  if (text.empty()) fail(index.string() + " is empty");

  JsonParser parser(text.data(), text.data() + text.size());
  JsonValue root = parser.parse_document();
  const JsonValue* wm = root.member("weight_map");
  if (wm == nullptr || wm->kind != JsonValue::Kind::kObject)
    fail(index.string() + " has no weight_map object");
  for (const auto& [name, file] : *wm->object) weight_map_.emplace(name, file.str);
}

bool Checkpoint::has(std::string_view name) const {
  return weight_map_.find(name) != weight_map_.end();
}

Shard& Checkpoint::shard_for(std::string_view name) {
  auto it = weight_map_.find(name);
  if (it == weight_map_.end()) fail("tensor '" + std::string(name) + "' not in checkpoint index");
  const std::string& file = it->second;
  auto sit = shards_.find(file);
  if (sit == shards_.end())
    sit = shards_.emplace(file, std::make_unique<Shard>(dir_ / file)).first;
  return *sit->second;
}

const TensorView& Checkpoint::tensor(std::string_view name) { return shard_for(name).at(name); }

const Shard& Checkpoint::mapped_shard_for(std::string_view name) const {
  auto it = weight_map_.find(name);
  if (it == weight_map_.end()) fail("tensor '" + std::string(name) + "' not in checkpoint index");
  auto sit = shards_.find(it->second);
  if (sit == shards_.end()) fail("shard for '" + std::string(name) + "' is not mapped yet");
  return *sit->second;
}

std::vector<std::string> Checkpoint::names_with_prefix(std::string_view prefix) const {
  std::vector<std::string> out;
  for (auto it = weight_map_.lower_bound(prefix); it != weight_map_.end(); ++it) {
    if (std::string_view(it->first).substr(0, prefix.size()) != prefix) break;
    out.push_back(it->first);
  }
  return out;
}

std::filesystem::path default_nvfp4_snapshot_dir() {
  if (const char* override_dir = std::getenv("ROCKET_FUEL_NVFP4_DIR"); override_dir != nullptr)
    return std::filesystem::path(override_dir);
  const char* home = std::getenv("HOME");
  if (home == nullptr) return {};
  return std::filesystem::path(home) /
         ".cache/nim-glm53-2.1.2/ngc/hub/models--nim--zai-org--glm-5.3-flash/snapshots/nim-aa28e1f-nvfp4";
}

}  // namespace rocket::fuel

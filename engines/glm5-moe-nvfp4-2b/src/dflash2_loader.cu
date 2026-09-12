// DFlash2 draft model loader: reads the safetensors checkpoint, uploads
// all tensors as BF16, and exposes named accessors for the forward pass.
#include "dflash2.h"
#include <cuda_runtime.h>
#include <fstream>
#include <cstring>
#include <stdexcept>

namespace rocket::engine {

namespace {
// Minimal safetensors header parser: extracts name -> (offset, shape, dtype)
struct STEntry {
    std::size_t offset = 0, nbytes = 0;
    std::string dtype;
    std::vector<std::int64_t> shape;
};

std::map<std::string, STEntry> parse_st_header(const char* data, std::uint64_t header_len) {
    // The header is JSON: {"name": {"dtype": "BF16", "shape": [...], "data_offsets": [s, e]}, ...}
    std::map<std::string, STEntry> out;
    // Simple JSON scan: find "name":{"dtype":"X","shape":[...],"data_offsets":[s,e]}
    const char* p = data;
    const char* end = data + header_len;
    while (p < end) {
        // Find next quoted key
        const char* q0 = static_cast<const char*>(memchr(p, '"', end - p));
        if (!q0) break;
        const char* q1 = static_cast<const char*>(memchr(q0 + 1, '"', end - q0 - 1));
        if (!q1) break;
        std::string name(q0 + 1, q1 - q0 - 1);
        if (name == "__metadata__") { p = q1 + 1; continue; }
        // Find data_offsets
        const char* doff = strstr(q1, "\"data_offsets\":[");
        if (!doff || doff >= end) break;
        const char* n0 = strchr(doff + 16, ',');
        const char* n1 = strchr(n0 + 1, ']');
        if (!n0 || !n1) break;
        std::size_t s = strtoull(n0 + 1, nullptr, 10);
        std::size_t e = strtoull(n0 + 1, nullptr, 10);
        s = strtoull(strchr(doff, '[') + 1, nullptr, 10);
        e = strtoull(strchr(strchr(doff, '[') + 1, ',') + 1, nullptr, 10);
        // Find dtype
        const char* dtp = strstr(q1, "\"dtype\":\"");
        std::string dtype = "BF16";
        if (dtp && dtp < end) {
            const char* ds = dtp + 9;
            const char* de = strchr(ds, '"');
            if (de) dtype = std::string(ds, de - ds);
        }
        // Find shape
        std::vector<std::int64_t> shape;
        const char* shp = strstr(q1, "\"shape\":[");
        if (shp && shp < end) {
            const char* sp = shp + 9;
            while (*sp && *sp != ']') {
                shape.push_back(strtoll(sp, nullptr, 10));
                while (*sp && *sp != ',' && *sp != ']') ++sp;
                if (*sp == ',') ++sp;
            }
        }
        STEntry entry;
        entry.offset = s;
        entry.nbytes = e - s;
        entry.dtype = dtype;
        entry.shape = shape;
        out[name] = entry;
        p = q1 + 1;
        // Skip past the closing brace of this entry
        while (p < end && *p != '}') ++p;
        if (p < end) ++p;
``    }
    return out;
}
}  // namespace

void DFlash2Weights::load(const std::filesystem::path& ckpt_file) {
  std::ifstream in(ckpt_file, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + ckpt_file.string());
  const std::size_t fsize = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  blob_.resize(fsize);
  if (!in.read(blob_.data(), fsize)) throw std::runtime_error("short read");
  in.close();

  std::uint64_t hl = 0;
  std::memcpy(&hl, blob_.data(), 8);
  auto entries = parse_st_header(blob_.data() + 8, hl);
  data_ = blob_.data() + 8 + hl;

  // Upload all tensors to device
  for (const auto& [name, entry] : entries) {
    if (entry.nbytes == 0) continue;
    void* dev = nullptr;
    cudaMalloc(&dev, entry.nbytes);
    cudaMemcpy(dev, data_ + entry.offset, entry.nbytes, cudaMemcpyHostToDevice);
    tensors_dev_[name] = dev;
    shapes_[name] = entry.shape;
  }
}

const void* DFlash2Weights::tensor_data(const char* name) const {
  auto it = tensors_dev_.find(name);
  return it != tensors_dev_.end() ? it->second : nullptr;
}

std::size_t DFlash2Weights::tensor_bytes(const char* name) const {
  auto it = tensors_dev_.find(name);
  if (it == tensors_dev_.end()) return 0;
  std::size_t bytes = 1;
  for (auto d : shapes_.at(name)) bytes *= d;
  return bytes * 2;  // BF16
}

DFlash2Weights::~DFlash2Weights() {
  for (auto& [name, ptr] : tensors_dev_) cudaFree(ptr);
}

}  // namespace rocket::engine

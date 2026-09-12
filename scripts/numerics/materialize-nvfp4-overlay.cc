// Materialize one NVFP4 overlay object from one or more BF16 checkpoint
// tensors. Multiple tensor names are comma-joined and row-concatenated before
// quantization (the KDA loader's q/k/v weight is one concat matrix, so its
// overlay object must be one concat too, with a single global scale).
//
// Output schema rocket.nvfp4-overlay-object.v1 per source tensor:
//   weight.u8            nibble-packed e2m1, rows-major
//   weight_scale.f8_e4m3 one e4m3 scale per 16-element block
//   weight_scale_2.f32   per-tensor global scale
//   metadata.json        provenance and per-tensor weight-error metrics
//
// The object directory is named by the caller: the directory name is what the
// loader env var points at, so the layout is a caller decision, not ours here.
//
// usage: materialize-nvfp4-overlay SNAPSHOT OUTPUT SNAPSHOT_KEY T0=SHA0,T1=SHA1,...
// Each comma-separated member names a BF16 checkpoint tensor and its SHA-256
// from the overlay inventory (provenance only; not re-hashed here).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <unistd.h>

#include "nvfp4.h"
#include "safetensors.h"

namespace fs = std::filesystem;
using rocket::fuel::Checkpoint;
using rocket::fuel::DType;
using rocket::fuel::TensorView;

static void write_file(const fs::path& path, const void* data, std::size_t bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes)))
    throw std::runtime_error("write failed: " + path.string());
  out.close();
  if (!out) throw std::runtime_error("close failed: " + path.string());
}

// Row-concat the given BF16 matrices into one contiguous buffer. All shapes
// must share the same K.todo
static std::vector<std::uint16_t> concat_bf16(const std::vector<TensorView>& ts,
                                              std::int64_t& n_out, std::int64_t& k_out) {
  std::vector<std::uint16_t> out;
  for (const auto& t : ts) {
    if (t.dtype != DType::kBF16 || t.shape.size() != 2)
      throw std::runtime_error(t.name + " is not a BF16 matrix");
    if (t.shape[0] <= 0 || t.shape[1] <= 0)
      throw std::runtime_error(t.name + " has a zero extent");
    const std::uint16_t* src = reinterpret_cast<const std::uint16_t*>(t.data);
    out.insert(out.end(), src, src + static_cast<std::size_t>(t.shape[0]) * t.shape[1]);
  }
  const std::int64_t k = ts.front().shape[1];
  for (const auto& t : ts)
    if (t.shape[1] != k)
      throw std::runtime_error("concat members disagree on K: " + t.name);
  std::int64_t n = 0;
  for (const auto& t : ts) n += t.shape[0];
  n_out = n;
  k_out = k;
  return out;
}

static std::vector<TensorView> load_tensors(Checkpoint& ckpt, const std::string& spec,
                                            std::vector<std::string>& shas_out) {
  std::vector<TensorView> ts;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    const std::size_t eq = item.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 == item.size())
      throw std::runtime_error("tensor spec must be NAME=SHA256: " + item);
    const std::string name = item.substr(0, eq);
    const std::string sha = item.substr(eq + 1);
    if (sha.size() != 64 ||
        !std::all_of(sha.begin(), sha.end(), [](unsigned char c) { return std::isxdigit(c); }))
      throw std::runtime_error("sha256 is not 64 hex characters: " + name);
    ts.push_back(ckpt.tensor(name));
    shas_out.push_back(sha);
  }
  if (ts.empty()) throw std::runtime_error("no tensors requested");
  return ts;
}

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    std::cerr << "usage: materialize-nvfp4-overlay SNAPSHOT OUTPUT SNAPSHOT_KEY T0=SHA0,T1=SHA1,... [--fp8-row]\n";
    return 2;
  }
  const bool fp8_row = argc == 6 && std::string(argv[5]) == "--fp8-row";
  try {
    Checkpoint checkpoint(argv[1]);
    const fs::path output(argv[2]);
    const std::string snapshot_key = argv[3];
    std::vector<std::string> shas;
    const auto tensors = load_tensors(checkpoint, argv[4], shas);
    fs::create_directories(output);

    std::int64_t n = 0, k = 0;
    const std::vector<std::uint16_t> source = concat_bf16(tensors, n, k);
    if (fp8_row) {
      // FP8 e4m3 payload with one f32 scale per output row: half the bytes
      // of BF16 at ~1.7 percent relative L2 (versus 9.4 percent for NVFP4),
      // the recipe the KDA-Q NVFP4 rejection pointed at as the gentler next
      // step for the attention families.
      const std::size_t elements = static_cast<std::size_t>(n) * k;
      std::vector<std::uint8_t> payload(elements);
      std::vector<float> row_scales(static_cast<std::size_t>(n));
      double error_sq = 0.0, reference_sq = 0.0;
      std::vector<double> member_err(tensors.size(), 0.0), member_ref(tensors.size(), 0.0);
      std::vector<std::int64_t> row_of(tensors.size() + 1, 0);
      for (std::size_t i = 0; i < tensors.size(); ++i)
        row_of[i + 1] = row_of[i] + tensors[i].shape[0];
      for (std::int64_t r = 0; r < n; ++r) {
        std::size_t member = 0;
        while (member + 1 < tensors.size() && r >= row_of[member + 1]) ++member;
        const std::size_t base = static_cast<std::size_t>(r) * k;
        float rowmax = 0.0f;
        for (std::int64_t j = 0; j < k; ++j)
          rowmax = std::max(rowmax, std::fabs(rocket::fuel::bf16_to_float(source[base + static_cast<std::size_t>(j)])));
        const float scale = rowmax / 448.0f;
        row_scales[static_cast<std::size_t>(r)] = scale;
        for (std::int64_t j = 0; j < k; ++j) {
          const float original = rocket::fuel::bf16_to_float(source[base + static_cast<std::size_t>(j)]);
          const float target = scale > 0.0f ? original / scale : 0.0f;
          const std::uint8_t bits = rocket::fuel::float_to_e4m3(target);
          payload[base + static_cast<std::size_t>(j)] = bits;
          const double error = static_cast<double>(rocket::fuel::e4m3_to_float(bits)) * scale - original;
          member_err[member] += error * error;
          member_ref[member] += static_cast<double>(original) * original;
          error_sq += error * error;
          reference_sq += static_cast<double>(original) * original;
        }
      }
      write_file(output / "weight.u8", payload.data(), payload.size());
      write_file(output / "weight_scale.f32", row_scales.data(), row_scales.size() * sizeof(float));
      const float one = 1.0f;
      write_file(output / "weight_scale_2.f32", &one, sizeof(one));
      std::ofstream meta(output / "metadata.json", std::ios::trunc);
      meta << "{\n"
           << "  \"schema\": \"rocket.fp8-row-overlay.v1\",\n"
           << "  \"source_tensors\": [";
      for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (i) meta << ",\n                    ";
        meta << "\"" << tensors[i].name << "\"";
      }
      meta << "],\n"
           << "  \"source_sha256\": [";
      for (std::size_t i = 0; i < shas.size(); ++i) {
        if (i) meta << ",\n                    ";
        meta << "\"" << shas[i] << "\"";
      }
      meta << "],\n"
           << "  \"source_snapshot_key\": \"" << snapshot_key << "\",\n"
           << "  \"source_dtype\": \"BF16\",\n"
           << "  \"shape\": [" << n << ", " << k << "],\n"
           << "  \"payload\": \"e4m3\",\n"
           << "  \"scale\": \"f32_per_row\",\n"
           << "  \"relative_l2_weight_error\": "
           << std::sqrt(error_sq / std::max(reference_sq, 1e-300)) << ",\n"
           << "  \"per_tensor_relative_l2\": [";
      for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (i) meta << ", ";
        meta << std::sqrt(member_err[i] / std::max(member_ref[i], 1e-300));
      }
      meta << "]\n}\n";
      if (!meta) throw std::runtime_error("metadata write failed");
      std::cout << "fp8-row\t" << elements << "\t"
                << std::sqrt(error_sq / std::max(reference_sq, 1e-300)) << '\n';
      return 0;
    }
    if (k % 16 != 0)
      throw std::runtime_error("concat does not satisfy K%16=0");
    const std::size_t elements = static_cast<std::size_t>(n) * k;
    std::vector<std::uint8_t> packed(elements / 2);
    std::vector<std::uint8_t> scales(elements / 16);

    float amax = 0.0f;
    for (std::size_t i = 0; i < elements; ++i)
      amax = std::max(amax, std::fabs(rocket::fuel::bf16_to_float(source[i])));
    const float global = amax > 0.0f ? amax / (448.0f * 6.0f) : 1.0f;

    double error_sq = 0.0, reference_sq = 0.0;
    double max_abs_error = 0.0;
    // Per-source row boundaries: members are concatenated in order. Each row
    // maps to one member tensor by a binary-search-free forward walk.
    std::vector<std::int64_t> row_of(tensors.size() + 1, 0);
    for (std::size_t i = 0; i < tensors.size(); ++i)
      row_of[i + 1] = row_of[i] + tensors[i].shape[0];
    std::vector<double> member_err(tensors.size(), 0.0), member_ref(tensors.size(), 0.0);

    for (std::int64_t r = 0; r < n; ++r) {
      std::size_t member = 0;
      while (member + 1 < tensors.size() && r >= row_of[member + 1]) ++member;
      for (std::int64_t b = 0; b < k / 16; ++b) {
        const std::size_t base = static_cast<std::size_t>(r) * k + b * 16;
        float block_max = 0.0f;
        for (int j = 0; j < 16; ++j)
          block_max = std::max(block_max, std::fabs(rocket::fuel::bf16_to_float(source[base + j])));
        const std::uint8_t scale_bits =
            rocket::fuel::float_to_e4m3(block_max / (6.0f * global));
        scales[base / 16] = scale_bits;
        const float step = rocket::fuel::e4m3_to_float(scale_bits) * global;
        for (int j = 0; j < 16; ++j) {
          const std::size_t index = base + j;
          const float original = rocket::fuel::bf16_to_float(source[index]);
          const std::uint8_t nibble = step > 0.0f
              ? rocket::fuel::float_to_e2m1(original / step) : 0;
          std::uint8_t& byte = packed[index / 2];
          if ((index & 1) == 0) byte = nibble;
          else byte = static_cast<std::uint8_t>(byte | (nibble << 4));
          const double error = static_cast<double>(rocket::fuel::e2m1_to_float(nibble)) * step - original;
          member_err[member] += error * error;
          member_ref[member] += static_cast<double>(original) * original;
          error_sq += error * error;
          reference_sq += static_cast<double>(original) * original;
          max_abs_error = std::max(max_abs_error, std::fabs(error));
        }
      }
    }

    const auto write_json = [&](const fs::path& dir) {
      std::ofstream meta(dir / "metadata.json", std::ios::trunc);
      meta << "{\n"
           << "  \"schema\": \"rocket.nvfp4-overlay-object.v1\",\n"
           << "  \"source_tensors\": [";
      for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (i) meta << ",\n                    ";
        meta << "\"" << tensors[i].name << "\"";
      }
      meta << "],\n"
           << "  " << "\"source_sha256\": [";
      for (std::size_t i = 0; i < shas.size(); ++i) {
        if (i) meta << ",\n                    ";
        meta << "\"" << shas[i] << "\"";
      }
      meta << "],\n"
           << "  \"source_snapshot_key\": \"" << snapshot_key << "\",\n"
           << "  \"source_dtype\": \"BF16\",\n"
           << "  \"shape\": [" << n << ", " << k << "],\n"
           << "  \"block_size\": 16,\n"
           << "  \"packed_bytes\": " << packed.size() << ",\n"
           << "  \"scale_bytes\": " << scales.size() << ",\n"
           << "  \"global_scale\": " << global << ",\n"
           << "  \"relative_l2_weight_error\": "
           << std::sqrt(error_sq / std::max(reference_sq, 1e-300)) << ",\n"
           << "  \"max_abs_weight_error\": " << max_abs_error << ",\n"
           << "  \"per_tensor_relative_l2\": [";
      for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (i) meta << ", ";
        meta << std::sqrt(member_err[i] / std::max(member_ref[i], 1e-300));
      }
      meta << "]\n}\n";
      if (!meta) throw std::runtime_error("metadata write failed");
      meta.close();
    };

    // Object payloads and metadata land directly in OUTPUT (the caller owns
    // the directory layout and points the loader env var at it).
    write_file(output / "weight.u8", packed.data(), packed.size());
    write_file(output / "weight_scale.f8_e4m3", scales.data(), scales.size());
    write_file(output / "weight_scale_2.f32", &global, sizeof(global));
    write_json(output);
  } catch (const std::exception& error) {
    std::cerr << "materialize failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

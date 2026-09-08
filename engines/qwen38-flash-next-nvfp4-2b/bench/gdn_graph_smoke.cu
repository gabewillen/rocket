// SPDX-License-Identifier: Apache-2.0
// Live-only proof for the authenticated rank-0/layer-0 fixed GDN graph.
#include "linear_attention/gdn_cutlass.h"
#include "linear_attention/gdn_flashinfer_wheel.h"
#include "linear_attention/gdn_verifier.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
constexpr std::uint64_t kSlabBytes = 63'212'748'800ULL;
constexpr int kRows = 16, kHidden = 2'560, kConvRows = 6;
constexpr int kConvWidth = 5'120, kHeads = 24, kDim = 128, kOut = 2'560;
constexpr float kLocalRoofGbps = 238.0F;
struct Extent { std::uint64_t offset, bytes; };
constexpr Extent kALog{1'297'735'680, 48};
constexpr Extent kConv{1'297'735'936, 40'960};
constexpr Extent kDt{1'297'776'896, 48};
constexpr Extent kAWeight{1'297'777'152, 30'720};
constexpr Extent kAScale{1'297'807'872, 20'480};
constexpr Extent kAGlobal{1'297'828'352, 4};
constexpr Extent kBWeight{1'297'828'864, 30'720};
constexpr Extent kBScale{1'297'859'584, 20'480};
constexpr Extent kBGlobal{1'297'880'064, 4};
constexpr Extent kQWeight{1'297'880'576, 6'553'600};
constexpr Extent kQScale{1'304'434'176, 819'200};
constexpr Extent kQGlobal{1'305'253'376, 4};
constexpr Extent kZWeight{1'305'253'888, 3'932'160};
constexpr Extent kZScale{1'309'186'048, 491'520};
constexpr Extent kZGlobal{1'309'677'568, 4};
constexpr Extent kNorm{1'309'678'080, 256};
constexpr Extent kOWeight{1'309'678'336, 3'932'160};
constexpr Extent kOScale{1'313'610'496, 491'520};
constexpr Extent kOGlobal{1'314'102'016, 4};

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

struct DeviceBlob {
  void* pointer = nullptr;
  explicit DeviceBlob(std::size_t bytes) { check(cudaMalloc(&pointer, bytes), "cudaMalloc"); }
  ~DeviceBlob() { cudaFree(pointer); }
  DeviceBlob(const DeviceBlob&) = delete;
  DeviceBlob& operator=(const DeviceBlob&) = delete;
};

std::vector<std::uint8_t> read_exact(int fd, Extent extent) {
  std::vector<std::uint8_t> result(extent.bytes);
  std::size_t done = 0;
  while (done != result.size()) {
    const ssize_t count = pread(fd, result.data() + done, result.size() - done,
                                static_cast<off_t>(extent.offset + done));
    if (count <= 0) throw std::runtime_error("short authenticated slab read");
    done += static_cast<std::size_t>(count);
  }
  return result;
}

void load_blob(int fd, Extent extent, DeviceBlob& device) {
  const auto host = read_exact(fd, extent);
  check(cudaMemcpy(device.pointer, host.data(), host.size(),
                   cudaMemcpyHostToDevice), "copy slab extent");
}

float load_scalar(int fd, Extent extent) {
  const auto bytes = read_exact(fd, extent);
  float value;
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}

__global__ void initialize(__nv_bfloat16* input, std::int32_t* indices) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kRows * kHidden) {
    input[index] = __float2bfloat16(static_cast<float>((index % 29) - 14) / 32.0F);
  }
  if (index < kRows) indices[index] = index + 1;
}

__global__ void initialize_rows(__nv_bfloat16* input, int rows) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < rows * kHidden) {
    input[index] = __float2bfloat16(
        static_cast<float>(((index * 17 + index / kHidden * 7) % 61) - 30) /
        64.0F);
  }
}

__global__ void initialize_elements(__nv_bfloat16* input,
                                    std::size_t elements) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    input[index] = __float2bfloat16(
        static_cast<float>((index * 17 % 61) - 30) / 64.0F);
  }
}

struct Telemetry final : rocket::qwen38::pair_reduce::OtelStageSink {
  std::uint64_t spans = 0, metrics = 0;
  void emit_span_and_log(
      const rocket::qwen38::pair_reduce::SpanRecord&) noexcept override {
    ++spans;
  }
  void record_duration(
      const rocket::qwen38::pair_reduce::MetricPoint&) noexcept override {
    ++metrics;
  }
};

std::uint64_t hash(const void* data, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  std::uint64_t value = 14'695'981'039'346'656'037ULL;
  for (std::size_t i = 0; i < bytes; ++i) value = (value ^ p[i]) * 1'099'511'628'211ULL;
  return value;
}

std::uint64_t device_hash(const void* data, std::size_t bytes) {
  std::vector<std::uint8_t> host(bytes);
  check(cudaMemcpy(host.data(), data, bytes, cudaMemcpyDeviceToHost),
        "copy projection hash input");
  return hash(host.data(), host.size());
}

bool device_equal(const void* left, const void* right, std::size_t bytes) {
  std::vector<std::uint8_t> left_host(bytes);
  std::vector<std::uint8_t> right_host(bytes);
  check(cudaMemcpy(left_host.data(), left, bytes, cudaMemcpyDeviceToHost),
        "copy parity left input");
  check(cudaMemcpy(right_host.data(), right, bytes, cudaMemcpyDeviceToHost),
        "copy parity right input");
  return left_host == right_host;
}

void write_device_file(const std::filesystem::path& path, const void* data,
                       std::size_t bytes) {
  std::vector<std::uint8_t> host(bytes);
  check(cudaMemcpy(host.data(), data, bytes, cudaMemcpyDeviceToHost),
        "copy fixture output");
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(host.data()), host.size());
  if (!output) throw std::runtime_error("write projection fixture");
}

void load_fixture_file(const std::filesystem::path& path, DeviceBlob& device,
                       std::size_t expected_bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || static_cast<std::size_t>(input.tellg()) != expected_bytes)
    throw std::runtime_error("projection fixture byte count changed");
  input.seekg(0);
  std::vector<std::uint8_t> host(expected_bytes);
  input.read(reinterpret_cast<char*>(host.data()), host.size());
  if (!input) throw std::runtime_error("read projection fixture");
  check(cudaMemcpy(device.pointer, host.data(), host.size(), cudaMemcpyHostToDevice),
        "copy projection fixture to device");
}

struct JsonValue {
  enum class Kind { kNull, kBoolean, kInteger, kString, kArray, kObject };
  Kind kind = Kind::kNull;
  bool boolean = false;
  std::int64_t integer = 0;
  std::string string;
  std::vector<JsonValue> array;
  std::map<std::string, JsonValue> object;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view source) : source_(source) {}
  JsonValue parse() {
    auto value = parse_value();
    whitespace();
    if (position_ != source_.size()) fail();
    return value;
  }

 private:
  [[noreturn]] void fail() const {
    throw std::runtime_error("projection fixture JSON changed");
  }
  void whitespace() {
    while (position_ < source_.size() &&
           std::isspace(static_cast<unsigned char>(source_[position_]))) ++position_;
  }
  bool take(char expected) {
    whitespace();
    if (position_ < source_.size() && source_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }
  std::string parse_string() {
    if (!take('"')) fail();
    std::string result;
    while (position_ < source_.size()) {
      const char character = source_[position_++];
      if (character == '"') return result;
      if (static_cast<unsigned char>(character) < 0x20) fail();
      if (character != '\\') {
        result.push_back(character);
        continue;
      }
      if (position_ == source_.size()) fail();
      const char escaped = source_[position_++];
      switch (escaped) {
        case '"': case '\\': case '/': result.push_back(escaped); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        default: fail();
      }
    }
    fail();
  }
  JsonValue parse_value() {
    whitespace();
    if (position_ == source_.size()) fail();
    if (source_[position_] == '"') {
      JsonValue value; value.kind = JsonValue::Kind::kString;
      value.string = parse_string(); return value;
    }
    if (source_[position_] == '{') return parse_object();
    if (source_[position_] == '[') return parse_array();
    if (source_.substr(position_, 4) == "null") {
      position_ += 4; return {};
    }
    if (source_.substr(position_, 4) == "true" ||
        source_.substr(position_, 5) == "false") {
      JsonValue value; value.kind = JsonValue::Kind::kBoolean;
      value.boolean = source_[position_] == 't';
      position_ += value.boolean ? 4 : 5; return value;
    }
    const std::size_t begin = position_;
    if (source_[position_] == '-') ++position_;
    while (position_ < source_.size() &&
           std::isdigit(static_cast<unsigned char>(source_[position_]))) ++position_;
    if (begin == position_ || (source_[begin] == '-' && begin + 1 == position_)) fail();
    JsonValue value; value.kind = JsonValue::Kind::kInteger;
    try { value.integer = std::stoll(std::string(source_.substr(begin, position_ - begin))); }
    catch (...) { fail(); }
    return value;
  }
  JsonValue parse_array() {
    if (!take('[')) fail();
    JsonValue value; value.kind = JsonValue::Kind::kArray;
    if (take(']')) return value;
    do { value.array.push_back(parse_value()); } while (take(','));
    if (!take(']')) fail();
    return value;
  }
  JsonValue parse_object() {
    if (!take('{')) fail();
    JsonValue value; value.kind = JsonValue::Kind::kObject;
    if (take('}')) return value;
    do {
      const std::string key = parse_string();
      if (!take(':')) fail();
      if (!value.object.emplace(key, parse_value()).second)
        throw std::runtime_error("projection fixture JSON duplicate key");
    } while (take(','));
    if (!take('}')) fail();
    return value;
  }
  std::string_view source_;
  std::size_t position_ = 0;
};

const JsonValue& member(const JsonValue& value, std::string_view key) {
  if (value.kind != JsonValue::Kind::kObject) throw std::runtime_error("projection fixture object changed");
  const auto found = value.object.find(std::string(key));
  if (found == value.object.end()) throw std::runtime_error("projection fixture member missing");
  return found->second;
}

void exact_keys(const JsonValue& value,
                std::initializer_list<std::string_view> expected) {
  if (value.kind != JsonValue::Kind::kObject || value.object.size() != expected.size())
    throw std::runtime_error("projection fixture object keys changed");
  for (const auto key : expected)
    if (value.object.find(std::string(key)) == value.object.end())
      throw std::runtime_error("projection fixture object key missing");
}

void exact_integers(const JsonValue& value,
                    std::initializer_list<std::int64_t> expected) {
  if (value.kind != JsonValue::Kind::kArray || value.array.size() != expected.size())
    throw std::runtime_error("projection fixture integer array changed");
  std::size_t index = 0;
  for (const auto number : expected) {
    if (value.array[index].kind != JsonValue::Kind::kInteger ||
        value.array[index].integer != number)
      throw std::runtime_error("projection fixture integer changed");
    ++index;
  }
}

void exact_layout(const JsonValue& layouts, std::string_view name,
                  std::initializer_list<std::int64_t> shape,
                  std::initializer_list<std::int64_t> stride,
                  std::string_view dtype) {
  const auto& layout = member(layouts, name);
  exact_keys(layout, {"shape", "stride", "dtype"});
  exact_integers(member(layout, "shape"), shape);
  exact_integers(member(layout, "stride"), stride);
  const auto& observed_dtype = member(layout, "dtype");
  if (observed_dtype.kind != JsonValue::Kind::kString ||
      observed_dtype.string != dtype)
    throw std::runtime_error("projection fixture dtype changed");
}

struct FixtureManifest {
  std::string raw;
  JsonValue root;
  int ba_physical_n = 0;
};

FixtureManifest read_fixture_manifest(const std::filesystem::path& directory,
                                      int tokens) {
  std::ifstream input(directory / "manifest.json");
  FixtureManifest manifest;
  manifest.raw.assign(std::istreambuf_iterator<char>(input), {});
  if (!input) throw std::runtime_error("projection fixture manifest changed");
  manifest.root = JsonParser(manifest.raw).parse();
  exact_keys(manifest.root, {"format", "provenance", "tokens", "qkvz_mnk",
                             "ba_logical_mnk", "ba_physical_mnk",
                             "ba_physical_n", "layouts", "files"});
  const auto& format = member(manifest.root, "format");
  const auto& token_value = member(manifest.root, "tokens");
  if (format.kind != JsonValue::Kind::kString ||
      format.string != "rocket-gdn-fp4-fixture-v1" ||
      token_value.kind != JsonValue::Kind::kInteger || token_value.integer != tokens)
    throw std::runtime_error("projection fixture identity changed");
  exact_integers(member(manifest.root, "qkvz_mnk"), {tokens, 8192, 2560});
  exact_integers(member(manifest.root, "ba_logical_mnk"), {tokens, 48, 2560});
  const auto& physical_n = member(manifest.root, "ba_physical_n");
  if (physical_n.kind != JsonValue::Kind::kInteger ||
      (physical_n.integer != 48 && physical_n.integer != 64))
    throw std::runtime_error("projection fixture BA ABI incompatible");
  manifest.ba_physical_n = static_cast<int>(physical_n.integer);
  exact_integers(member(manifest.root, "ba_physical_mnk"),
                 {tokens, manifest.ba_physical_n, 2560});
  const auto& provenance = member(manifest.root, "provenance");
  if (provenance.kind != JsonValue::Kind::kString ||
      (provenance.string != "python-synthetic-seed-7" &&
       provenance.string != "authenticated-rank0-layer0"))
    throw std::runtime_error("projection fixture provenance changed");
  const auto& layouts = member(manifest.root, "layouts");
  exact_keys(layouts, {"hidden", "packed_a", "sfa", "qkvz_b", "qkvz_sfb",
                       "ba_b", "ba_sfb"});
  exact_layout(layouts, "hidden", {tokens, 2560}, {2560, 1}, "bfloat16");
  exact_layout(layouts, "packed_a", {tokens, 1280}, {1280, 1}, "uint8");
  exact_layout(layouts, "sfa", {((tokens + 127) / 128) * 128, 160},
               {160, 1}, "uint8");
  exact_layout(layouts, "qkvz_b", {8192, 1280}, {1280, 1}, "uint8");
  exact_layout(layouts, "qkvz_sfb", {8192, 160}, {160, 1}, "uint8");
  exact_layout(layouts, "ba_b", {manifest.ba_physical_n, 1280}, {1280, 1},
               "uint8");
  exact_layout(layouts, "ba_sfb", {128, 160}, {160, 1}, "uint8");
  exact_keys(member(manifest.root, "files"),
             {"hidden.bin", "qkvz_a.bin", "qkvz_sfa.bin", "ba_a.bin",
              "ba_sfa.bin", "qkvz_b.bin", "qkvz_sfb.bin", "ba_b.bin",
              "ba_sfb.bin", "alpha.bin"});
  return manifest;
}

void validate_hashed_fixture_file(const std::filesystem::path& directory,
                                  const FixtureManifest& manifest,
                                  const char* name, std::size_t bytes) {
  const auto path = directory / name;
  if (!std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) != bytes)
    throw std::runtime_error("projection fixture byte count changed");
  const std::string digest =
      rocket::qwen38::linear_attention::gdn_sha256_file(path.string());
  const auto& record = member(member(manifest.root, "files"), name);
  exact_keys(record, {"bytes", "sha256"});
  const auto& byte_count = member(record, "bytes");
  const auto& expected_digest = member(record, "sha256");
  if (byte_count.kind != JsonValue::Kind::kInteger ||
      byte_count.integer != static_cast<std::int64_t>(bytes) ||
      expected_digest.kind != JsonValue::Kind::kString ||
      expected_digest.string != digest)
    throw std::runtime_error("projection fixture hash mismatch");
}

void load_hashed_fixture_file(const std::filesystem::path& directory,
                              const FixtureManifest& manifest, const char* name,
                              DeviceBlob& storage, std::size_t bytes) {
  validate_hashed_fixture_file(directory, manifest, name, bytes);
  const auto path = directory / name;
  load_fixture_file(path, storage, bytes);
}

void validate_fixture_root(const std::filesystem::path& root) {
  for (const int tokens : std::array<int, 2>{300, 8'192}) {
    const auto directory = root / ("tokens-" + std::to_string(tokens));
    const FixtureManifest manifest = read_fixture_manifest(directory, tokens);
    const std::size_t a_bytes = static_cast<std::size_t>(tokens) * kHidden / 2;
    const std::size_t sfa_bytes =
        rocket::qwen38::linear_attention::prefill_sfa_bytes(tokens, kHidden);
    struct FileSize { const char* name; std::size_t bytes; };
    const std::array files{
        FileSize{"hidden.bin", static_cast<std::size_t>(tokens) * kHidden * 2},
        FileSize{"qkvz_a.bin", a_bytes}, FileSize{"qkvz_sfa.bin", sfa_bytes},
        FileSize{"ba_a.bin", a_bytes}, FileSize{"ba_sfa.bin", sfa_bytes},
        FileSize{"qkvz_b.bin", 8'192ULL * kHidden / 2},
        FileSize{"qkvz_sfb.bin", 8'192ULL * kHidden / 16},
        FileSize{"ba_b.bin",
                 static_cast<std::size_t>(manifest.ba_physical_n) * kHidden / 2},
        FileSize{"ba_sfb.bin", 128ULL * kHidden / 16},
        FileSize{"alpha.bin", sizeof(float)},
    };
    for (const auto& file : files)
      validate_hashed_fixture_file(directory, manifest, file.name, file.bytes);
    std::cout << "fixture_validation=pass tokens=" << tokens
              << " ba_physical_n=" << manifest.ba_physical_n
              << " manifest_sha256="
              << rocket::qwen38::linear_attention::gdn_sha256_file(
                     (directory / "manifest.json").string())
              << '\n';
  }
}

void dump_projection_fixture(
    const std::filesystem::path& root, int tokens,
    const __nv_bfloat16* hidden,
    const rocket::qwen38::linear_attention::CutlassGdnPrefillProjection& projection) {
  namespace fs = std::filesystem;
  const fs::path directory = root / ("tokens-" + std::to_string(tokens));
  if (!fs::create_directories(directory))
    throw std::runtime_error("projection fixture directory already exists");
  struct Entry { const char* name; const void* data; std::size_t bytes; };
  const std::array entries{
      Entry{"hidden.bin", hidden,
            static_cast<std::size_t>(tokens) * kHidden * 2},
      Entry{"qkvz_a.bin", projection.input_packed(tokens),
            static_cast<std::size_t>(tokens) * kHidden / 2},
      Entry{"qkvz_sfa.bin", projection.input_sfa(tokens),
            rocket::qwen38::linear_attention::prefill_sfa_bytes(tokens, kHidden)},
      Entry{"ba_a.bin", projection.input_packed(tokens),
            static_cast<std::size_t>(tokens) * kHidden / 2},
      Entry{"ba_sfa.bin", projection.input_sfa(tokens),
            rocket::qwen38::linear_attention::prefill_sfa_bytes(tokens, kHidden)},
      Entry{"qkvz_b.bin", projection.qkvz_weight(), 8'192ULL * kHidden / 2},
      Entry{"qkvz_sfb.bin", projection.qkvz_sfb(), 8'192ULL * kHidden / 16},
      Entry{"ba_b.bin", projection.ba_weight(), 48ULL * kHidden / 2},
      Entry{"ba_sfb.bin", projection.ba_sfb(), 128ULL * kHidden / 16},
      Entry{"alpha.bin", projection.projection_alpha(), sizeof(float)},
  };
  for (const auto& entry : entries)
    write_device_file(directory / entry.name, entry.data, entry.bytes);
  std::ofstream manifest(directory / "manifest.json", std::ios::trunc);
  manifest << "{\n  \"format\": \"rocket-gdn-fp4-fixture-v1\",\n"
           << "  \"provenance\": \"authenticated-rank0-layer0\",\n"
           << "  \"tokens\": " << tokens << ",\n"
           << "  \"qkvz_mnk\": [" << tokens << ", 8192, 2560],\n"
           << "  \"ba_logical_mnk\": [" << tokens << ", 48, 2560],\n"
           << "  \"ba_physical_mnk\": [" << tokens << ", 48, 2560],\n"
           << "  \"ba_physical_n\": 48,\n"
           << "  \"layouts\": {\"hidden\": {\"shape\": [" << tokens
           << ", 2560], \"stride\": [2560, 1], \"dtype\": \"bfloat16\"},"
           << " \"packed_a\": {\"shape\": [" << tokens
           << ", 1280], \"stride\": [1280, 1], \"dtype\": \"uint8\"},"
           << " \"sfa\": {\"shape\": [" << ((tokens + 127) / 128) * 128
           << ", 160], \"stride\": [160, 1], \"dtype\": \"uint8\"},"
           << " \"qkvz_b\": {\"shape\": [8192, 1280], \"stride\": [1280, 1], \"dtype\": \"uint8\"},"
           << " \"qkvz_sfb\": {\"shape\": [8192, 160], \"stride\": [160, 1], \"dtype\": \"uint8\"},"
           << " \"ba_b\": {\"shape\": [48, 1280], \"stride\": [1280, 1], \"dtype\": \"uint8\"},"
           << " \"ba_sfb\": {\"shape\": [128, 160], \"stride\": [160, 1], \"dtype\": \"uint8\"}},\n"
           << "  \"files\": {\n";
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const auto& entry = entries[i];
    manifest << "    \"" << entry.name << "\": {\"bytes\": " << entry.bytes
             << ", \"sha256\": \""
             << rocket::qwen38::linear_attention::gdn_sha256_file(
                    (directory / entry.name).string())
             << "\"}" << (i + 1 == entries.size() ? "\n" : ",\n");
  }
  manifest << "  }\n}\n";
  if (!manifest) throw std::runtime_error("write projection fixture manifest");
}

struct RawMeasurement {
  float p50 = 0.0F;
  float p95 = 0.0F;
  std::vector<float> samples;
};

template <typename Launch>
RawMeasurement capture_measure_python_scope(cudaStream_t stream, Launch launch) {
  for (int iteration = 0; iteration < 5; ++iteration) launch();
  check(cudaStreamSynchronize(stream), "synchronize fixture eager warmup");
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "begin fixture capture");
  launch();
  check(cudaStreamEndCapture(stream, &graph), "end fixture capture");
  check(cudaGraphInstantiate(&executable, graph, 0), "instantiate fixture graph");
  check(cudaGraphLaunch(executable, stream), "warm fixture graph");
  check(cudaStreamSynchronize(stream), "synchronize fixture graph warmup");
  cudaEvent_t begin = nullptr, end = nullptr;
  check(cudaEventCreate(&begin), "create fixture begin event");
  check(cudaEventCreate(&end), "create fixture end event");
  RawMeasurement result;
  for (int iteration = 0; iteration < 50; ++iteration) {
    check(cudaEventRecord(begin, stream), "record fixture begin");
    check(cudaGraphLaunch(executable, stream), "replay fixture graph");
    check(cudaEventRecord(end, stream), "record fixture end");
    check(cudaEventSynchronize(end), "synchronize fixture end");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, begin, end), "time fixture graph");
    result.samples.push_back(milliseconds * 1000.0F);
  }
  auto ordered = result.samples;
  std::sort(ordered.begin(), ordered.end());
  result.p50 = (ordered[24] + ordered[25]) / 2.0F;
  result.p95 = ordered[47];
  cudaEventDestroy(end); cudaEventDestroy(begin);
  cudaGraphExecDestroy(executable); cudaGraphDestroy(graph);
  return result;
}

void run_wheel_fixture(std::string_view wheel_shared_object,
                       const std::filesystem::path& root) {
  using rocket::qwen38::linear_attention::GdnFlashInferWheelGemm;
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream), "create fixture stream");
  for (const int tokens : std::array<int, 2>{300, 8'192}) {
    const auto directory = root / ("tokens-" + std::to_string(tokens));
    const FixtureManifest manifest = read_fixture_manifest(directory, tokens);
    const int ba_physical_n = manifest.ba_physical_n;
    const std::size_t a_bytes = static_cast<std::size_t>(tokens) * kHidden / 2;
    const std::size_t sfa_bytes =
        rocket::qwen38::linear_attention::prefill_sfa_bytes(tokens, kHidden);
    DeviceBlob hidden(static_cast<std::size_t>(tokens) * kHidden * 2),
        qkvz_a(a_bytes), qkvz_sfa(sfa_bytes), ba_a(a_bytes),
        ba_sfa(sfa_bytes), qkvz_b(8'192ULL * kHidden / 2),
        qkvz_sfb(8'192ULL * kHidden / 16),
        ba_b(static_cast<std::size_t>(ba_physical_n) * kHidden / 2),
        ba_sfb(128ULL * kHidden / 16), alpha(sizeof(float)),
        qkvz_out(static_cast<std::size_t>(tokens) * 8'192 * 2),
        ba_out(static_cast<std::size_t>(tokens) * ba_physical_n * 2);
    struct FixtureInput { const char* name; DeviceBlob* storage; std::size_t bytes; };
    const std::array<FixtureInput, 10> files{{
        {"hidden.bin", &hidden, static_cast<std::size_t>(tokens) * kHidden * 2},
        {"qkvz_a.bin", &qkvz_a, a_bytes},
        {"qkvz_sfa.bin", &qkvz_sfa, sfa_bytes},
        {"ba_a.bin", &ba_a, a_bytes},
        {"ba_sfa.bin", &ba_sfa, sfa_bytes},
        {"qkvz_b.bin", &qkvz_b, 8'192ULL * kHidden / 2},
        {"qkvz_sfb.bin", &qkvz_sfb, 8'192ULL * kHidden / 16},
        {"ba_b.bin", &ba_b,
         static_cast<std::size_t>(ba_physical_n) * kHidden / 2},
        {"ba_sfb.bin", &ba_sfb, 128ULL * kHidden / 16},
        {"alpha.bin", &alpha, sizeof(float)},
    }};
    for (const auto& file : files)
      load_hashed_fixture_file(directory, manifest, file.name, *file.storage,
                               file.bytes);
    GdnFlashInferWheelGemm qkvz, ba;
    qkvz.init(wheel_shared_object, tokens, 8'192, kHidden,
              static_cast<std::uint8_t*>(qkvz_a.pointer),
              static_cast<std::uint8_t*>(qkvz_sfa.pointer),
              static_cast<std::uint8_t*>(qkvz_b.pointer),
              static_cast<std::uint8_t*>(qkvz_sfb.pointer),
              static_cast<float*>(alpha.pointer),
              static_cast<__nv_bfloat16*>(qkvz_out.pointer));
    ba.init(wheel_shared_object, tokens, ba_physical_n, kHidden,
            static_cast<std::uint8_t*>(ba_a.pointer),
            static_cast<std::uint8_t*>(ba_sfa.pointer),
            static_cast<std::uint8_t*>(ba_b.pointer),
            static_cast<std::uint8_t*>(ba_sfb.pointer),
            static_cast<float*>(alpha.pointer),
            static_cast<__nv_bfloat16*>(ba_out.pointer));
    const auto timing = capture_measure_python_scope(stream, [&] {
      qkvz.run(stream); ba.run(stream);
    });
    std::cout << "{\"fixture_backend\":\"flashinfer_wheel_raw\","
              << "\"scope\":\"two_raw_gemms_no_quant_no_ba_slice\","
              << "\"ba_logical_n\":48,\"ba_physical_n\":" << ba_physical_n << ","
              << "\"tokens\":" << tokens << ",\"p50_us\":" << timing.p50
              << ",\"p95_us\":" << timing.p95 << ",\"samples_us\":[";
    for (std::size_t i = 0; i < timing.samples.size(); ++i)
      std::cout << (i ? "," : "") << timing.samples[i];
    std::cout << "],\"pointer_alignment_mod128\":["
              << reinterpret_cast<std::uintptr_t>(qkvz_a.pointer) % 128 << ","
              << reinterpret_cast<std::uintptr_t>(qkvz_sfa.pointer) % 128 << ","
              << reinterpret_cast<std::uintptr_t>(ba_a.pointer) % 128 << ","
              << reinterpret_cast<std::uintptr_t>(ba_sfa.pointer) % 128
              << "],\"manifest_sha256\":\""
              << rocket::qwen38::linear_attention::gdn_sha256_file(
                     (directory / "manifest.json").string())
              << "\",\"qkvz_hash\":"
              << device_hash(qkvz_out.pointer,
                             static_cast<std::size_t>(tokens) * 8'192 * 2)
              << ",\"ba_hash\":"
              << device_hash(ba_out.pointer,
                             static_cast<std::size_t>(tokens) * ba_physical_n * 2)
              << ",\"graph_capture\":\"pass\"}\n";
  }
  cudaStreamDestroy(stream);
}

void print_raw_samples(const RawMeasurement& timing) {
  std::cout << "\"p50_us\":" << timing.p50 << ",\"p95_us\":" << timing.p95
            << ",\"samples_us\":[";
  for (std::size_t i = 0; i < timing.samples.size(); ++i)
    std::cout << (i ? "," : "") << timing.samples[i];
  std::cout << "]";
}

void run_wheel_qkvz_cross(std::string_view wheel_shared_object,
                          const std::filesystem::path& synthetic_root,
                          const std::filesystem::path& authenticated_root) {
  using rocket::qwen38::linear_attention::GdnFlashInferWheelGemm;
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream), "create QKVZ crossing stream");
  for (const int tokens : std::array<int, 2>{300, 8'192}) {
    const auto synthetic_directory =
        synthetic_root / ("tokens-" + std::to_string(tokens));
    const auto authenticated_directory =
        authenticated_root / ("tokens-" + std::to_string(tokens));
    const FixtureManifest synthetic_manifest =
        read_fixture_manifest(synthetic_directory, tokens);
    const FixtureManifest authenticated_manifest =
        read_fixture_manifest(authenticated_directory, tokens);
    const std::size_t a_bytes = static_cast<std::size_t>(tokens) * kHidden / 2;
    const std::size_t sfa_bytes =
        rocket::qwen38::linear_attention::prefill_sfa_bytes(tokens, kHidden);
    constexpr std::size_t b_bytes = 8'192ULL * kHidden / 2;
    constexpr std::size_t sfb_bytes = 8'192ULL * kHidden / 16;
    DeviceBlob synthetic_a(a_bytes), synthetic_sfa(sfa_bytes),
        synthetic_b(b_bytes), synthetic_sfb(sfb_bytes),
        authenticated_a(a_bytes), authenticated_sfa(sfa_bytes),
        authenticated_b(b_bytes), authenticated_sfb(sfb_bytes),
        alpha(sizeof(float)), authenticated_alpha(sizeof(float)),
        output(static_cast<std::size_t>(tokens) * 8'192 * 2);
    load_hashed_fixture_file(synthetic_directory, synthetic_manifest,
                             "qkvz_a.bin", synthetic_a, a_bytes);
    load_hashed_fixture_file(synthetic_directory, synthetic_manifest,
                             "qkvz_sfa.bin", synthetic_sfa, sfa_bytes);
    load_hashed_fixture_file(synthetic_directory, synthetic_manifest,
                             "qkvz_b.bin", synthetic_b, b_bytes);
    load_hashed_fixture_file(synthetic_directory, synthetic_manifest,
                             "qkvz_sfb.bin", synthetic_sfb, sfb_bytes);
    load_hashed_fixture_file(synthetic_directory, synthetic_manifest,
                             "alpha.bin", alpha, sizeof(float));
    load_hashed_fixture_file(authenticated_directory, authenticated_manifest,
                             "qkvz_a.bin", authenticated_a, a_bytes);
    load_hashed_fixture_file(authenticated_directory, authenticated_manifest,
                             "qkvz_sfa.bin", authenticated_sfa, sfa_bytes);
    load_hashed_fixture_file(authenticated_directory, authenticated_manifest,
                             "qkvz_b.bin", authenticated_b, b_bytes);
    load_hashed_fixture_file(authenticated_directory, authenticated_manifest,
                             "qkvz_sfb.bin", authenticated_sfb, sfb_bytes);
    load_hashed_fixture_file(authenticated_directory, authenticated_manifest,
                             "alpha.bin", authenticated_alpha, sizeof(float));
    if (!device_equal(alpha.pointer, authenticated_alpha.pointer, sizeof(float)))
      throw std::runtime_error("QKVZ crossing alpha identity changed");

    GdnFlashInferWheelGemm qkvz;
    qkvz.init(wheel_shared_object, tokens, 8'192, kHidden,
              static_cast<std::uint8_t*>(synthetic_a.pointer),
              static_cast<std::uint8_t*>(synthetic_sfa.pointer),
              static_cast<std::uint8_t*>(synthetic_b.pointer),
              static_cast<std::uint8_t*>(synthetic_sfb.pointer),
              static_cast<float*>(alpha.pointer),
              static_cast<__nv_bfloat16*>(output.pointer));
    struct Crossing {
      const char* activation;
      const char* weight;
      const std::uint8_t* a;
      const std::uint8_t* sfa;
      const std::uint8_t* b;
      const std::uint8_t* sfb;
    };
    const std::array crossings{
        Crossing{"synthetic", "synthetic",
                 static_cast<std::uint8_t*>(synthetic_a.pointer),
                 static_cast<std::uint8_t*>(synthetic_sfa.pointer),
                 static_cast<std::uint8_t*>(synthetic_b.pointer),
                 static_cast<std::uint8_t*>(synthetic_sfb.pointer)},
        Crossing{"synthetic", "authenticated",
                 static_cast<std::uint8_t*>(synthetic_a.pointer),
                 static_cast<std::uint8_t*>(synthetic_sfa.pointer),
                 static_cast<std::uint8_t*>(authenticated_b.pointer),
                 static_cast<std::uint8_t*>(authenticated_sfb.pointer)},
        Crossing{"authenticated", "synthetic",
                 static_cast<std::uint8_t*>(authenticated_a.pointer),
                 static_cast<std::uint8_t*>(authenticated_sfa.pointer),
                 static_cast<std::uint8_t*>(synthetic_b.pointer),
                 static_cast<std::uint8_t*>(synthetic_sfb.pointer)},
        Crossing{"authenticated", "authenticated",
                 static_cast<std::uint8_t*>(authenticated_a.pointer),
                 static_cast<std::uint8_t*>(authenticated_sfa.pointer),
                 static_cast<std::uint8_t*>(authenticated_b.pointer),
                 static_cast<std::uint8_t*>(authenticated_sfb.pointer)},
    };
    for (std::size_t index = 0; index < crossings.size(); ++index) {
      const auto& crossing = crossings[index];
      qkvz.bind_inputs(crossing.a, crossing.sfa, crossing.b, crossing.sfb);
      const auto timing = capture_measure_python_scope(stream, [&] {
        qkvz.run(stream);
      });
      std::cout << "{\"fixture_backend\":\"flashinfer_wheel_raw_qkvz_cross\","
                << "\"tokens\":" << tokens << ",\"order_index\":" << index
                << ",\"activation\":\"" << crossing.activation
                << "\",\"weight\":\"" << crossing.weight << "\","
                << "\"wheel_sha256\":\""
                << rocket::qwen38::linear_attention::kGdnFlashInferWheelSha256
                << "\",\"tactic_id\":-1,\"scheduler\":\"dp_static_persistent\","
                << "\"swap_ab\":false,\"tile_mnk\":[128,128,256],"
                << "\"synthetic_manifest_sha256\":\""
                << rocket::qwen38::linear_attention::gdn_sha256_file(
                       (synthetic_directory / "manifest.json").string())
                << "\",\"authenticated_manifest_sha256\":\""
                << rocket::qwen38::linear_attention::gdn_sha256_file(
                       (authenticated_directory / "manifest.json").string())
                << "\",";
      print_raw_samples(timing);
      std::cout << ",\"output_hash\":"
                << device_hash(output.pointer,
                               static_cast<std::size_t>(tokens) * 8'192 * 2)
                << ",\"output_workspace_alpha_fixed\":true,"
                << "\"graph_capture\":\"pass\"}\n";
    }
  }
  cudaStreamDestroy(stream);
}

template <typename Launch>
std::pair<float, float> capture_measure(cudaStream_t stream, Launch launch) {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "begin prefill projection capture");
  launch();
  check(cudaStreamEndCapture(stream, &graph), "end prefill projection capture");
  check(cudaGraphInstantiate(&executable, graph, 0),
        "instantiate prefill projection graph");
  for (int iteration = 0; iteration < 10; ++iteration)
    check(cudaGraphLaunch(executable, stream), "warm prefill projection");
  check(cudaStreamSynchronize(stream), "synchronize prefill projection warmup");
  cudaEvent_t begin = nullptr, end = nullptr;
  check(cudaEventCreate(&begin), "create prefill begin event");
  check(cudaEventCreate(&end), "create prefill end event");
  std::vector<float> samples;
  for (int iteration = 0; iteration < 50; ++iteration) {
    check(cudaEventRecord(begin, stream), "record prefill begin");
    check(cudaGraphLaunch(executable, stream), "replay prefill projection");
    check(cudaEventRecord(end, stream), "record prefill end");
    check(cudaEventSynchronize(end), "synchronize prefill end");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, begin, end),
          "time prefill projection");
    samples.push_back(milliseconds * 1000.0F);
  }
  std::sort(samples.begin(), samples.end());
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  cudaGraphExecDestroy(executable);
  cudaGraphDestroy(graph);
  return {samples[24], samples[47]};
}

void run_prefill_projection(int device,
                            rocket::qwen38::linear_attention::GdnWeights weights,
                            rocket::qwen38::linear_attention::GdnPrefillInputBackend backend,
                            std::string_view wheel_shared_object,
                            std::string_view fixture_dump_root = {}) {
  using rocket::qwen38::linear_attention::CutlassGdnPrefillProjection;
  using rocket::qwen38::linear_attention::GdnPrefillInputBackend;
  CutlassGdnPrefillProjection projection(device, weights, true, backend,
                                         wheel_shared_object);
  if (backend == GdnPrefillInputBackend::kFlashInferWheelBenchmark) {
    std::cout
        << "prefill_wheel_sha256="
        << rocket::qwen38::linear_attention::kGdnFlashInferWheelSha256
        << " wheel_architecture="
        << rocket::qwen38::linear_attention::kGdnFlashInferWheelArchitecture
        << " fallback_tactic=-1 fallback_cta=128x128x256"
        << " fallback_scheduler=dp fallback_swap_ab=false"
        << " fallback_cluster=1x1x1\n";
  }
  DeviceBlob hidden(8'192ULL * kHidden * 2);
  DeviceBlob normalized(8'192ULL * kHeads * kDim * 2);
  constexpr std::size_t hidden_elements = 8'192ULL * kHidden;
  constexpr std::size_t normalized_elements = 8'192ULL * kHeads * kDim;
  initialize_elements<<<(hidden_elements + 255) / 256, 256>>>(
      static_cast<__nv_bfloat16*>(hidden.pointer), hidden_elements);
  initialize_elements<<<(normalized_elements + 255) / 256, 256>>>(
      static_cast<__nv_bfloat16*>(normalized.pointer), normalized_elements);
  check(cudaDeviceSynchronize(), "initialize prefill projection inputs");
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream), "create prefill projection stream");
  for (const int tokens : std::array<int, 2>{300, 8'192}) {
    projection.launch_input_quantize(
        static_cast<__nv_bfloat16*>(hidden.pointer), tokens, stream);
    check(cudaStreamSynchronize(stream), "prepare projection phase timing");
    if (!fixture_dump_root.empty())
      dump_projection_fixture(
          fixture_dump_root, tokens,
          static_cast<__nv_bfloat16*>(hidden.pointer), projection);
    const auto quantize = capture_measure(stream, [&] {
      projection.launch_input_quantize(
          static_cast<__nv_bfloat16*>(hidden.pointer), tokens, stream);
    });
    const auto qkvz = capture_measure(stream, [&] {
      projection.launch_qkvz(tokens, stream);
    });
    const auto ba = capture_measure(stream, [&] {
      projection.launch_ba(tokens, stream);
    });
    const auto input = capture_measure(stream, [&] {
      projection.launch_input(static_cast<__nv_bfloat16*>(hidden.pointer),
                              tokens, stream);
    });
    const auto reference = capture_measure(stream, [&] {
      projection.launch_reference_input(
          static_cast<__nv_bfloat16*>(hidden.pointer), tokens, stream);
    });
    const auto output = capture_measure(stream, [&] {
      projection.launch_output(
          static_cast<__nv_bfloat16*>(normalized.pointer), tokens, stream);
    });
    const auto input_hash =
        device_hash(projection.qkvz(tokens),
                    static_cast<std::size_t>(tokens) * 8'192 * 2);
    const auto output_hash =
        device_hash(projection.output(tokens),
                    static_cast<std::size_t>(tokens) * kOut * 2);
    projection.launch_input(static_cast<__nv_bfloat16*>(hidden.pointer), tokens,
                            stream);
    projection.launch_reference_input(
        static_cast<__nv_bfloat16*>(hidden.pointer), tokens, stream);
    check(cudaStreamSynchronize(stream), "complete projection parity paths");
    const std::size_t packed_bytes =
        static_cast<std::size_t>(tokens) * kHidden / 2;
    const std::size_t sfa_bytes =
        rocket::qwen38::linear_attention::prefill_sfa_bytes(tokens, kHidden);
    const bool packed_parity =
        device_equal(projection.input_packed(tokens),
                     projection.reference_qkvz_packed(tokens), packed_bytes) &&
        device_equal(projection.input_packed(tokens),
                     projection.reference_ba_packed(tokens), packed_bytes);
    const bool sfa_parity =
        device_equal(projection.input_sfa(tokens),
                     projection.reference_qkvz_sfa(tokens), sfa_bytes) &&
        device_equal(projection.input_sfa(tokens),
                     projection.reference_ba_sfa(tokens), sfa_bytes);
    const bool qkvz_parity = device_equal(
        projection.qkvz(tokens), projection.reference_qkvz(tokens),
        static_cast<std::size_t>(tokens) * 8'192 * 2);
    const bool ba_parity = device_equal(
        projection.ba(tokens), projection.reference_ba(tokens),
        static_cast<std::size_t>(tokens) * 48 * 2);
    std::cout << "prefill_backend="
              << (backend == GdnPrefillInputBackend::kB12x
                      ? "b12x"
                      : backend ==
                                GdnPrefillInputBackend::kFlashInferWheelBenchmark
                            ? "flashinfer_wheel_0.6.17_sm120f_fallback"
                            : "flashinfer_cutlass_91bda04")
              << " prefill_tokens=" << tokens
              << " shared_input_quantizations=1 input_p50_us=" << input.first
              << " input_p95_us=" << input.second
              << " quantize_once_p50_us=" << quantize.first
              << " quantize_once_p95_us=" << quantize.second
              << " qkvz_p50_us=" << qkvz.first
              << " qkvz_p95_us=" << qkvz.second
              << " ba_p50_us=" << ba.first
              << " ba_p95_us=" << ba.second
              << " two_quant_reference_p50_us=" << reference.first
              << " two_quant_reference_p95_us=" << reference.second
              << " output_p50_us=" << output.first
              << " output_p95_us=" << output.second
              << " qkvz_hash=" << input_hash << " output_hash=" << output_hash
              << " packed_parity=" << (packed_parity ? "pass" : "fail")
              << " sfa_parity=" << (sfa_parity ? "pass" : "fail")
              << " qkvz_parity=" << (qkvz_parity ? "pass" : "fail")
              << " ba_parity=" << (ba_parity ? "pass" : "fail")
              << " graph_capture=pass\n";
  }
  cudaStreamDestroy(stream);
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc != 3 && argc != 4 && argc != 5 && argc != 6 && argc != 7)
    throw std::invalid_argument(
        "usage: qwen38-gdn-graph-smoke SLAB DEVICE "
        "[--prefill-projection|--prefill-projection-b12x|"
        "--prefill-projection-flashinfer-wheel SO [--dump-fixtures ROOT]|"
        "--prefill-projection-flashinfer-wheel-fixture SO FIXTURE_ROOT|"
        "--prefill-projection-flashinfer-wheel-qkvz-cross SO SYN_ROOT AUTH_ROOT|"
        "--validate-projection-fixtures ROOT]");
  if (argc == 4 && std::string(argv[3]) != "--prefill-projection" &&
      std::string(argv[3]) != "--prefill-projection-b12x")
    throw std::invalid_argument("unknown GDN graph smoke mode");
  if (argc == 5 &&
      std::string(argv[3]) != "--prefill-projection-flashinfer-wheel" &&
      std::string(argv[3]) != "--validate-projection-fixtures")
    throw std::invalid_argument("unknown GDN graph smoke mode");
  if (argc == 6 &&
      std::string(argv[3]) != "--prefill-projection-flashinfer-wheel-fixture")
    throw std::invalid_argument("unknown GDN fixture smoke mode");
  if (argc == 7 &&
      !((std::string(argv[3]) == "--prefill-projection-flashinfer-wheel" &&
         std::string(argv[5]) == "--dump-fixtures") ||
        std::string(argv[3]) ==
            "--prefill-projection-flashinfer-wheel-qkvz-cross"))
    throw std::invalid_argument("unknown GDN fixture dump mode");
  if (argc == 5 && std::string(argv[3]) == "--validate-projection-fixtures") {
    validate_fixture_root(argv[4]);
    return 0;
  }
  const int device = std::stoi(argv[2]);
  check(cudaSetDevice(device), "cudaSetDevice");
  if (argc == 6 &&
      std::string(argv[3]) == "--prefill-projection-flashinfer-wheel-fixture") {
    run_wheel_fixture(argv[4], argv[5]);
    return 0;
  }
  if (argc == 7 && std::string(argv[3]) ==
                           "--prefill-projection-flashinfer-wheel-qkvz-cross") {
    run_wheel_qkvz_cross(argv[4], argv[5], argv[6]);
    return 0;
  }
  const int fd = open(argv[1], O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error(std::string("open slab: ") + std::strerror(errno));
  struct stat info {};
  if (fstat(fd, &info) || static_cast<std::uint64_t>(info.st_size) != kSlabBytes) {
    close(fd); throw std::runtime_error("rank0-target slab byte identity changed");
  }

  DeviceBlob qw(kQWeight.bytes), qs(kQScale.bytes), zw(kZWeight.bytes), zs(kZScale.bytes);
  DeviceBlob bw(kBWeight.bytes), bs(kBScale.bytes), aw(kAWeight.bytes), as(kAScale.bytes);
  DeviceBlob ow(kOWeight.bytes), os(kOScale.bytes), conv(kConv.bytes), alog(kALog.bytes);
  DeviceBlob dt(kDt.bytes), norm(kNorm.bytes);
  load_blob(fd, kQWeight, qw); load_blob(fd, kQScale, qs);
  load_blob(fd, kZWeight, zw); load_blob(fd, kZScale, zs);
  load_blob(fd, kBWeight, bw); load_blob(fd, kBScale, bs);
  load_blob(fd, kAWeight, aw); load_blob(fd, kAScale, as);
  load_blob(fd, kOWeight, ow); load_blob(fd, kOScale, os);
  load_blob(fd, kConv, conv); load_blob(fd, kALog, alog);
  load_blob(fd, kDt, dt); load_blob(fd, kNorm, norm);
  const float qg = load_scalar(fd, kQGlobal), zg = load_scalar(fd, kZGlobal);
  const float bg = load_scalar(fd, kBGlobal), ag = load_scalar(fd, kAGlobal);
  const float og = load_scalar(fd, kOGlobal);
  close(fd);

  const rocket::qwen38::linear_attention::GdnWeights weights{
      {static_cast<std::uint8_t*>(qw.pointer),
       static_cast<std::uint8_t*>(qs.pointer), qg},
      {static_cast<std::uint8_t*>(zw.pointer),
       static_cast<std::uint8_t*>(zs.pointer), zg},
      {static_cast<std::uint8_t*>(bw.pointer),
       static_cast<std::uint8_t*>(bs.pointer), bg},
      {static_cast<std::uint8_t*>(aw.pointer),
       static_cast<std::uint8_t*>(as.pointer), ag},
      {static_cast<std::uint8_t*>(ow.pointer),
       static_cast<std::uint8_t*>(os.pointer), og},
      static_cast<__nv_bfloat16*>(conv.pointer),
      static_cast<__nv_bfloat16*>(alog.pointer),
      static_cast<__nv_bfloat16*>(dt.pointer),
      static_cast<__nv_bfloat16*>(norm.pointer)};
  if (argc >= 4) {
    using rocket::qwen38::linear_attention::GdnPrefillInputBackend;
    const std::string mode(argv[3]);
    const auto backend =
        mode == "--prefill-projection-b12x"
            ? GdnPrefillInputBackend::kB12x
            : mode == "--prefill-projection-flashinfer-wheel"
                  ? GdnPrefillInputBackend::kFlashInferWheelBenchmark
                  : GdnPrefillInputBackend::kFlashInferCutlass;
    run_prefill_projection(
        device, weights, backend,
        argc >= 5 ? std::string_view(argv[4]) : std::string_view{},
        argc == 7 ? std::string_view(argv[6]) : std::string_view{});
    return 0;
  }

  void* graph = nullptr;
  if (qwen38_gdn_graph_create(
          device, static_cast<std::uint8_t*>(qw.pointer), static_cast<std::uint8_t*>(qs.pointer), qg,
          static_cast<std::uint8_t*>(zw.pointer), static_cast<std::uint8_t*>(zs.pointer), zg,
          static_cast<std::uint8_t*>(bw.pointer), static_cast<std::uint8_t*>(bs.pointer), bg,
          static_cast<std::uint8_t*>(aw.pointer), static_cast<std::uint8_t*>(as.pointer), ag,
          static_cast<std::uint8_t*>(ow.pointer), static_cast<std::uint8_t*>(os.pointer), og,
          static_cast<__nv_bfloat16*>(conv.pointer), static_cast<__nv_bfloat16*>(alog.pointer),
          static_cast<__nv_bfloat16*>(dt.pointer), static_cast<__nv_bfloat16*>(norm.pointer), &graph)) {
    throw std::runtime_error(qwen38_gdn_graph_last_error());
  }
  DeviceBlob input(kRows * kHidden * 2), conv_state(17ULL * kConvRows * kConvWidth * 2);
  DeviceBlob recurrent(17ULL * kHeads * kDim * kDim * 4), indices(kRows * 4);
  initialize<<<(kRows * kHidden + 255) / 256, 256>>>(
      static_cast<__nv_bfloat16*>(input.pointer), static_cast<std::int32_t*>(indices.pointer));
  check(cudaMemset(conv_state.pointer, 0, 17ULL * kConvRows * kConvWidth * 2), "clear conv state");
  check(cudaMemset(recurrent.pointer, 0, 17ULL * kHeads * kDim * kDim * 4), "clear recurrent state");
  cudaStream_t stream; check(cudaStreamCreate(&stream), "create stream");

  void* output = nullptr; std::size_t elements = 0;
  if (qwen38_gdn_graph_output(graph, &output, &elements)) throw std::runtime_error(qwen38_gdn_graph_last_error());
  std::vector<std::uint8_t> host(elements * 2);
  constexpr std::size_t conv_bytes = 17ULL * kConvRows * kConvWidth * 2;
  constexpr std::size_t recurrent_bytes = 17ULL * kHeads * kDim * kDim * 4;
  std::vector<std::uint8_t> first_output(host.size());
  std::vector<std::uint8_t> first_conv(conv_bytes), first_recurrent(recurrent_bytes);
  std::vector<std::uint8_t> second_conv(conv_bytes), second_recurrent(recurrent_bytes);
  for (const int m : std::array<int, 5>{1, 2, 4, 8, 16}) {
    cudaGraph_t captured; cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    if (qwen38_gdn_graph_launch(graph, static_cast<__nv_bfloat16*>(input.pointer),
                               static_cast<__nv_bfloat16*>(conv_state.pointer),
                               static_cast<float*>(recurrent.pointer),
                               static_cast<std::int32_t*>(indices.pointer), m, stream)) {
      throw std::runtime_error(qwen38_gdn_graph_last_error());
    }
    check(cudaStreamEndCapture(stream, &captured), "end capture");
    check(cudaGraphInstantiate(&executable, captured, 0), "instantiate graph");
    check(cudaMemsetAsync(conv_state.pointer, 0, 17ULL * kConvRows * kConvWidth * 2, stream), "reset conv");
    check(cudaMemsetAsync(recurrent.pointer, 0, 17ULL * kHeads * kDim * kDim * 4, stream), "reset recurrent");
    check(cudaGraphLaunch(executable, stream), "graph replay 1"); check(cudaStreamSynchronize(stream), "sync replay 1");
    check(cudaMemcpy(host.data(), output, host.size(), cudaMemcpyDeviceToHost), "copy output 1");
    first_output = host;
    check(cudaMemcpy(first_conv.data(), conv_state.pointer, conv_bytes,
                     cudaMemcpyDeviceToHost), "copy conv state 1");
    check(cudaMemcpy(first_recurrent.data(), recurrent.pointer, recurrent_bytes,
                     cudaMemcpyDeviceToHost), "copy recurrent state 1");
    const auto first = hash(host.data(), static_cast<std::size_t>(m) * kOut * 2);
    check(cudaMemsetAsync(conv_state.pointer, 0, 17ULL * kConvRows * kConvWidth * 2, stream), "reset conv 2");
    check(cudaMemsetAsync(recurrent.pointer, 0, 17ULL * kHeads * kDim * kDim * 4, stream), "reset recurrent 2");
    check(cudaGraphLaunch(executable, stream), "graph replay 2"); check(cudaStreamSynchronize(stream), "sync replay 2");
    check(cudaMemcpy(host.data(), output, host.size(), cudaMemcpyDeviceToHost), "copy output 2");
    check(cudaMemcpy(second_conv.data(), conv_state.pointer, conv_bytes,
                     cudaMemcpyDeviceToHost), "copy conv state 2");
    check(cudaMemcpy(second_recurrent.data(), recurrent.pointer, recurrent_bytes,
                     cudaMemcpyDeviceToHost), "copy recurrent state 2");
    const auto second = hash(host.data(), static_cast<std::size_t>(m) * kOut * 2);
    const auto conv_hash = hash(first_conv.data(), conv_bytes);
    const auto recurrent_hash = hash(first_recurrent.data(), recurrent_bytes);
    if (first != second || first_conv != second_conv ||
        first_recurrent != second_recurrent) {
      std::size_t differing_output = 0;
      for (std::size_t i = 0; i < static_cast<std::size_t>(m) * kOut * 2; ++i) {
        differing_output += first_output[i] != host[i];
      }
      throw std::runtime_error(
          "GDN graph replay differs: m=" + std::to_string(m) +
          " output_bytes=" + std::to_string(differing_output) +
          " conv=" + (first_conv == second_conv ? "same" : "different") +
          " recurrent=" +
          (first_recurrent == second_recurrent ? "same" : "different"));
    }
    std::array<float, 5> samples{};
    cudaEvent_t begin, end; check(cudaEventCreate(&begin), "create begin event");
    check(cudaEventCreate(&end), "create end event");
    for (auto& sample : samples) {
      check(cudaMemsetAsync(conv_state.pointer, 0, 17ULL * kConvRows * kConvWidth * 2, stream), "profile reset conv");
      check(cudaMemsetAsync(recurrent.pointer, 0, 17ULL * kHeads * kDim * kDim * 4, stream), "profile reset recurrent");
      check(cudaEventRecord(begin, stream), "record begin");
      check(cudaGraphLaunch(executable, stream), "profile graph replay");
      check(cudaEventRecord(end, stream), "record end");
      check(cudaEventSynchronize(end), "sync profile replay");
      check(cudaEventElapsedTime(&sample, begin, end), "elapsed profile replay");
    }
    cudaEventDestroy(end); cudaEventDestroy(begin);
    const float mean = std::accumulate(samples.begin(), samples.end(), 0.0F) /
                       static_cast<float>(samples.size());
    const auto [minimum, maximum] =
        std::minmax_element(samples.begin(), samples.end());
    constexpr std::uint64_t weights =
        static_cast<std::uint64_t>(8'240) * 2'560 * 9 / 16 +
        static_cast<std::uint64_t>(2'560) * 3'072 * 9 / 16 + 41'312;
    constexpr std::uint64_t state_per_row =
        2ULL * kHeads * kDim * kDim * 4 + 6ULL * kConvWidth * 2;
    const std::uint64_t traffic = weights + static_cast<std::uint64_t>(m) *
        (state_per_row + 2ULL * kHidden + 2ULL * kOut);
    const float gbps = static_cast<float>(traffic) / (mean * 1.0e6F);
    std::cout << "m=" << m << " output_fnv64=" << first
              << " conv_fnv64=" << conv_hash
              << " recurrent_fnv64=" << recurrent_hash
              << " replay=bit-exact mean_ms=" << mean
              << " min_ms=" << *minimum << " max_ms=" << *maximum
              << " spread_percent=" << ((*maximum - *minimum) / mean * 100.0F)
              << " traffic_bytes="
              << traffic << " effective_GBps=" << gbps
              << " local_roof_fraction=" << gbps / kLocalRoofGbps << '\n';
    cudaGraphExecDestroy(executable); cudaGraphDestroy(captured);
  }
  // The fixed output GEMM reads 16 rows for every bucket. A c1 launch after
  // c16 must clear the inactive recurrent rows before that read.
  if (qwen38_gdn_graph_launch(graph, static_cast<__nv_bfloat16*>(input.pointer),
                             static_cast<__nv_bfloat16*>(conv_state.pointer),
                             static_cast<float*>(recurrent.pointer),
                             static_cast<std::int32_t*>(indices.pointer), 1,
                             stream)) {
    throw std::runtime_error(qwen38_gdn_graph_last_error());
  }
  check(cudaStreamSynchronize(stream), "sync c16-to-c1 isolation launch");
  check(cudaMemcpy(host.data(), output, host.size(), cudaMemcpyDeviceToHost),
        "copy c16-to-c1 output");
  for (std::size_t byte = static_cast<std::size_t>(kOut) * 2;
       byte < host.size(); ++byte) {
    if (host[byte] != 0) {
      throw std::runtime_error("c16-to-c1 inactive output retained data");
    }
  }
  std::cout << "c16_to_c1=inactive-output-zero\n";

  // The verifier consumes position-major rows and advances private state one
  // position at a time. Compare its per-sequence prefix publication with the
  // unchanged K0 graph executing exactly the accepted positions.
  DeviceBlob verify_input((8 * kRows + kRows) * kHidden * 2ULL);
  DeviceBlob reference_conv(conv_bytes), reference_recurrent(recurrent_bytes);
  DeviceBlob reference_indices(kRows * sizeof(std::int32_t));
  DeviceBlob compact_input(kRows * kHidden * 2ULL);
  DeviceBlob reference_output(8ULL * kRows * kHidden * 2ULL);
  initialize_rows<<<((8 * kRows + kRows) * kHidden + 255) / 256, 256>>>(
      static_cast<__nv_bfloat16*>(verify_input.pointer), 8 * kRows + kRows);
  initialize<<<(kRows * kHidden + 255) / 256, 256>>>(
      static_cast<__nv_bfloat16*>(input.pointer),
      static_cast<std::int32_t*>(indices.pointer));
  Telemetry telemetry;
  auto* typed_graph =
      static_cast<rocket::qwen38::linear_attention::CutlassGdnGraph*>(graph);
  rocket::qwen38::linear_attention::GdnVerifier verifier(
      device, *typed_graph, 17, telemetry);

  const auto prove_shape = [&](int sequences, int width, bool profile) {
    check(cudaMemsetAsync(conv_state.pointer, 0, conv_bytes, stream),
          "clear verifier accepted conv");
    check(cudaMemsetAsync(recurrent.pointer, 0, recurrent_bytes, stream),
          "clear verifier accepted recurrent");
    check(cudaMemsetAsync(reference_conv.pointer, 0, conv_bytes, stream),
          "clear verifier reference conv");
    check(cudaMemsetAsync(reference_recurrent.pointer, 0, recurrent_bytes,
                          stream),
          "clear verifier reference recurrent");
    check(cudaStreamSynchronize(stream), "sync verifier reset");
    std::vector<std::uint8_t> before_conv(conv_bytes), before_recurrent(recurrent_bytes);
    check(cudaMemcpy(before_conv.data(), conv_state.pointer, conv_bytes,
                     cudaMemcpyDeviceToHost),
          "copy verifier accepted conv before");
    check(cudaMemcpy(before_recurrent.data(), recurrent.pointer, recurrent_bytes,
                     cudaMemcpyDeviceToHost),
          "copy verifier accepted recurrent before");

    const rocket::qwen38::linear_attention::VerifierShape shape{sequences, width};
    verifier.stage(static_cast<__nv_bfloat16*>(verify_input.pointer),
                   static_cast<__nv_bfloat16*>(conv_state.pointer),
                   static_cast<float*>(recurrent.pointer),
                   static_cast<std::int32_t*>(indices.pointer), shape,
                   "gdn-verifier-proof", "shape", stream);
    check(cudaStreamSynchronize(stream), "sync verifier stage");
    std::vector<std::uint8_t> staged_conv(conv_bytes), staged_recurrent(recurrent_bytes);
    check(cudaMemcpy(staged_conv.data(), conv_state.pointer, conv_bytes,
                     cudaMemcpyDeviceToHost),
          "copy accepted conv after stage");
    check(cudaMemcpy(staged_recurrent.data(), recurrent.pointer, recurrent_bytes,
                     cudaMemcpyDeviceToHost),
          "copy accepted recurrent after stage");
    if (staged_conv != before_conv || staged_recurrent != before_recurrent) {
      throw std::runtime_error("GDN verifier stage mutated accepted state");
    }
    std::vector<std::uint8_t> staged_output(
        static_cast<std::size_t>(sequences) * width * kHidden * 2);
    check(cudaMemcpy(staged_output.data(), verifier.staged_output(),
                     staged_output.size(), cudaMemcpyDeviceToHost),
          "copy verifier staged output");

    std::array<std::int32_t, kRows> prefixes{};
    for (int sequence = 0; sequence < sequences; ++sequence) {
      prefixes[sequence] = width == 1 ? 1 : sequence % (width + 1);
    }
    verifier.accept(prefixes.data(), "gdn-verifier-proof", "shape", stream);

    // Full causal sequential K0 is the numeric oracle for every staged row.
    for (int prefix = 0; prefix < width; ++prefix) {
      check(cudaMemsetAsync(compact_input.pointer, 0,
                            kRows * kHidden * 2ULL, stream),
            "clear full sequential verifier input");
      check(cudaMemcpyAsync(
                compact_input.pointer,
                static_cast<__nv_bfloat16*>(verify_input.pointer) +
                    static_cast<std::size_t>(prefix) * sequences * kHidden,
                static_cast<std::size_t>(sequences) * kHidden * 2,
                cudaMemcpyDeviceToDevice, stream),
            "copy full sequential verifier input");
      if (qwen38_gdn_graph_launch(
              graph, static_cast<__nv_bfloat16*>(compact_input.pointer),
              static_cast<__nv_bfloat16*>(reference_conv.pointer),
              static_cast<float*>(reference_recurrent.pointer),
              static_cast<std::int32_t*>(indices.pointer), sequences, stream)) {
        throw std::runtime_error(qwen38_gdn_graph_last_error());
      }
      check(cudaMemcpyAsync(
                static_cast<__nv_bfloat16*>(reference_output.pointer) +
                    static_cast<std::size_t>(prefix) * sequences * kHidden,
                output,
                static_cast<std::size_t>(sequences) * kHidden * 2,
                cudaMemcpyDeviceToDevice, stream),
            "retain full sequential verifier output");
    }
    check(cudaStreamSynchronize(stream), "sync full sequential verifier");
    std::vector<std::uint8_t> direct_output(staged_output.size());
    check(cudaMemcpy(direct_output.data(), reference_output.pointer,
                     direct_output.size(), cudaMemcpyDeviceToHost),
          "copy full sequential verifier output");
    if (direct_output != staged_output) {
      throw std::runtime_error("GDN verifier output differs from sequential K0");
    }
    check(cudaMemsetAsync(reference_conv.pointer, 0, conv_bytes, stream),
          "reset prefix reference conv");
    check(cudaMemsetAsync(reference_recurrent.pointer, 0, recurrent_bytes,
                          stream),
          "reset prefix reference recurrent");

    for (int prefix = 0; prefix < width; ++prefix) {
      std::array<std::int32_t, kRows> active{};
      for (int sequence = 0; sequence < sequences; ++sequence) {
        active[sequence] = prefixes[sequence] > prefix ? sequence + 1 : 0;
      }
      check(cudaMemcpyAsync(reference_indices.pointer, active.data(),
                            sizeof(active), cudaMemcpyHostToDevice, stream),
            "copy verifier reference indices");
      check(cudaMemsetAsync(compact_input.pointer, 0,
                            kRows * kHidden * 2ULL, stream),
            "clear verifier reference input");
      check(cudaMemcpyAsync(
                compact_input.pointer,
                static_cast<__nv_bfloat16*>(verify_input.pointer) +
                    static_cast<std::size_t>(prefix) * sequences * kHidden,
                static_cast<std::size_t>(sequences) * kHidden * 2,
                cudaMemcpyDeviceToDevice, stream),
            "copy verifier reference input");
      if (qwen38_gdn_graph_launch(
              graph, static_cast<__nv_bfloat16*>(compact_input.pointer),
              static_cast<__nv_bfloat16*>(reference_conv.pointer),
              static_cast<float*>(reference_recurrent.pointer),
              static_cast<std::int32_t*>(reference_indices.pointer), sequences,
              stream)) {
        throw std::runtime_error(qwen38_gdn_graph_last_error());
      }
    }
    check(cudaStreamSynchronize(stream), "sync verifier accept and reference");
    std::vector<std::uint8_t> accepted_conv(conv_bytes), accepted_recurrent(recurrent_bytes);
    std::vector<std::uint8_t> expected_conv(conv_bytes), expected_recurrent(recurrent_bytes);
    check(cudaMemcpy(accepted_conv.data(), conv_state.pointer, conv_bytes,
                     cudaMemcpyDeviceToHost), "copy accepted verifier conv");
    check(cudaMemcpy(accepted_recurrent.data(), recurrent.pointer, recurrent_bytes,
                     cudaMemcpyDeviceToHost), "copy accepted verifier recurrent");
    check(cudaMemcpy(expected_conv.data(), reference_conv.pointer, conv_bytes,
                     cudaMemcpyDeviceToHost), "copy expected verifier conv");
    check(cudaMemcpy(expected_recurrent.data(), reference_recurrent.pointer,
                     recurrent_bytes, cudaMemcpyDeviceToHost),
          "copy expected verifier recurrent");
    if (accepted_conv != expected_conv || accepted_recurrent != expected_recurrent) {
      throw std::runtime_error("GDN verifier published a non-prefix state");
    }

    std::array<float, 20> samples{};
    std::uint64_t profiled_bytes = 0;
    if (profile) {
      cudaEvent_t begin, end;
      check(cudaEventCreate(&begin), "create verifier begin event");
      check(cudaEventCreate(&end), "create verifier end event");
      for (auto& sample : samples) {
        check(cudaMemsetAsync(conv_state.pointer, 0, conv_bytes, stream),
              "profile verifier conv reset");
        check(cudaMemsetAsync(recurrent.pointer, 0, recurrent_bytes, stream),
              "profile verifier recurrent reset");
        check(cudaEventRecord(begin, stream), "record verifier begin");
        verifier.stage(static_cast<__nv_bfloat16*>(verify_input.pointer),
                       static_cast<__nv_bfloat16*>(conv_state.pointer),
                       static_cast<float*>(recurrent.pointer),
                       static_cast<std::int32_t*>(indices.pointer), shape,
                       "gdn-verifier-profile", "shape", stream);
        profiled_bytes = verifier.logical_stage_bytes();
        check(cudaEventRecord(end, stream), "record verifier end");
        check(cudaEventSynchronize(end), "sync verifier profile");
        check(cudaEventElapsedTime(&sample, begin, end),
              "elapsed verifier profile");
        verifier.reset("gdn-verifier-profile", "shape");
      }
      cudaEventDestroy(end);
      cudaEventDestroy(begin);
    }
    const float mean = profile
        ? std::accumulate(samples.begin(), samples.end(), 0.0F) / samples.size()
        : 0.0F;
    const auto [minimum, maximum] =
        std::minmax_element(samples.begin(), samples.end());
    const auto bytes = profiled_bytes;
    const float gbps = profile ? static_cast<float>(bytes) / (mean * 1.0e6F) : 0.0F;
    std::cout << "verifier_sequences=" << sequences << " verify_width=" << width
              << " token_rows=" << shape.token_rows()
              << " acceptance=exact-prefix stage_state=isolated"
              << " output_drift_bytes=0"
              << " output_fnv64=" << hash(staged_output.data(), staged_output.size());
    if (profile) {
      std::cout << " mean_ms=" << mean << " min_ms=" << *minimum
                << " max_ms=" << *maximum
                << " spread_percent=" << ((*maximum - *minimum) / mean * 100.0F)
                << " traffic_bytes=" << bytes << " effective_GBps=" << gbps
                << " local_roof_fraction=" << gbps / kLocalRoofGbps;
    }
    std::cout << '\n';
  };

  // K0 equivalence is covered by width 1 and the same unchanged graph. K1,
  // K4, and K7 are widths 2, 5, and 8 respectively.
  prove_shape(16, 1, false);
  prove_shape(16, 2, true);
  prove_shape(16, 5, true);
  prove_shape(16, 8, true);
  prove_shape(1, 8, true);
  prove_shape(2, 8, true);
  prove_shape(4, 8, true);
  prove_shape(8, 8, true);
  prove_shape(1, 1, false);
  std::vector<std::uint8_t> verifier_tail(8ULL * kRows * kHidden * 2ULL);
  check(cudaMemcpy(verifier_tail.data(), typed_graph->verifier_output(),
                   verifier_tail.size(), cudaMemcpyDeviceToHost),
        "copy K7-to-K0 verifier output");
  for (std::size_t byte = static_cast<std::size_t>(kHidden) * 2;
       byte < verifier_tail.size(); ++byte) {
    if (verifier_tail[byte] != 0) {
      throw std::runtime_error("K7-to-K0 verifier output retained tail data");
    }
  }
  std::cout << "verifier_k7_to_k0=exact inactive-output-zero\n";

  // Reset must discard the speculative fork without publishing it.
  check(cudaMemsetAsync(conv_state.pointer, 0x35, conv_bytes, stream),
        "seed verifier reset conv");
  check(cudaMemsetAsync(recurrent.pointer, 0x5a, recurrent_bytes, stream),
        "seed verifier reset recurrent");
  check(cudaStreamSynchronize(stream), "sync verifier reset seed");
  std::vector<std::uint8_t> reset_conv_before(conv_bytes), reset_recurrent_before(recurrent_bytes);
  check(cudaMemcpy(reset_conv_before.data(), conv_state.pointer, conv_bytes,
                   cudaMemcpyDeviceToHost), "copy reset conv before");
  check(cudaMemcpy(reset_recurrent_before.data(), recurrent.pointer,
                   recurrent_bytes, cudaMemcpyDeviceToHost),
        "copy reset recurrent before");
  verifier.stage(static_cast<__nv_bfloat16*>(verify_input.pointer),
                 static_cast<__nv_bfloat16*>(conv_state.pointer),
                 static_cast<float*>(recurrent.pointer),
                 static_cast<std::int32_t*>(indices.pointer), {4, 8},
                 "gdn-verifier-proof", "reset", stream);
  check(cudaStreamSynchronize(stream), "sync verifier reset stage");
  std::vector<std::uint8_t> reset_output_first(32ULL * kHidden * 2ULL);
  check(cudaMemcpy(reset_output_first.data(), verifier.staged_output(),
                   reset_output_first.size(), cudaMemcpyDeviceToHost),
        "copy verifier reset output 1");
  verifier.reset("gdn-verifier-proof", "reset");
  verifier.stage(static_cast<__nv_bfloat16*>(verify_input.pointer),
                 static_cast<__nv_bfloat16*>(conv_state.pointer),
                 static_cast<float*>(recurrent.pointer),
                 static_cast<std::int32_t*>(indices.pointer), {4, 8},
                 "gdn-verifier-proof", "reset-replay", stream);
  check(cudaStreamSynchronize(stream), "sync verifier reset replay");
  std::vector<std::uint8_t> reset_output_second(reset_output_first.size());
  check(cudaMemcpy(reset_output_second.data(), verifier.staged_output(),
                   reset_output_second.size(), cudaMemcpyDeviceToHost),
        "copy verifier reset output 2");
  if (reset_output_first != reset_output_second) {
    throw std::runtime_error("GDN verifier reset replay differs");
  }
  verifier.reset("gdn-verifier-proof", "reset-replay");
  std::vector<std::uint8_t> reset_conv_after(conv_bytes), reset_recurrent_after(recurrent_bytes);
  check(cudaMemcpy(reset_conv_after.data(), conv_state.pointer, conv_bytes,
                   cudaMemcpyDeviceToHost), "copy reset conv after");
  check(cudaMemcpy(reset_recurrent_after.data(), recurrent.pointer,
                   recurrent_bytes, cudaMemcpyDeviceToHost),
        "copy reset recurrent after");
  if (reset_conv_before != reset_conv_after ||
      reset_recurrent_before != reset_recurrent_after) {
    throw std::runtime_error("GDN verifier reset published speculative state");
  }
  std::cout << "verifier_reset=discarded replay=bit-exact telemetry_spans=" << telemetry.spans
            << " telemetry_metrics=" << telemetry.metrics << '\n';

  check(cudaMemsetAsync(indices.pointer, 0, kRows * 4, stream), "set null slots");
  check(cudaMemsetAsync(conv_state.pointer, 0x35, conv_bytes, stream), "seed null conv");
  check(cudaMemsetAsync(recurrent.pointer, 0x5a, recurrent_bytes, stream), "seed null recurrent");
  std::vector<std::uint8_t> conv_before(conv_bytes), recurrent_before(recurrent_bytes);
  check(cudaMemcpyAsync(conv_before.data(), conv_state.pointer, conv_bytes,
                        cudaMemcpyDeviceToHost, stream), "copy null conv before");
  check(cudaMemcpyAsync(recurrent_before.data(), recurrent.pointer, recurrent_bytes,
                        cudaMemcpyDeviceToHost, stream), "copy null recurrent before");
  if (qwen38_gdn_graph_launch(graph, static_cast<__nv_bfloat16*>(input.pointer),
                             static_cast<__nv_bfloat16*>(conv_state.pointer),
                             static_cast<float*>(recurrent.pointer),
                             static_cast<std::int32_t*>(indices.pointer), 16, stream)) {
    throw std::runtime_error(qwen38_gdn_graph_last_error());
  }
  check(cudaStreamSynchronize(stream), "sync null-slot launch");
  std::vector<std::uint8_t> conv_after(conv_bytes), recurrent_after(recurrent_bytes);
  check(cudaMemcpy(conv_after.data(), conv_state.pointer, conv_bytes,
                   cudaMemcpyDeviceToHost), "copy null conv after");
  check(cudaMemcpy(recurrent_after.data(), recurrent.pointer, recurrent_bytes,
                   cudaMemcpyDeviceToHost), "copy null recurrent after");
  check(cudaMemcpy(host.data(), output, host.size(), cudaMemcpyDeviceToHost),
        "copy null output");
  if (conv_before != conv_after || recurrent_before != recurrent_after) {
    throw std::runtime_error("slot 0 mutated recurrent state");
  }
  for (const auto byte : host) {
    if (byte != 0) throw std::runtime_error("slot 0 produced nonzero output");
  }
  std::cout << "slot0=null state=unchanged output=zero\n";
  cudaStreamDestroy(stream);
  if (qwen38_gdn_graph_destroy(graph)) throw std::runtime_error(qwen38_gdn_graph_last_error());
  return 0;
} catch (const std::exception& error) {
  std::cerr << "gdn_graph_smoke: " << error.what() << '\n';
  return 1;
}

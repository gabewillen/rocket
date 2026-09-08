// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_oracle_comparator.h"
#include "decode/target_layer0_boundary_comparator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
struct evp_md_ctx_st;
struct evp_md_st;
evp_md_ctx_st* EVP_MD_CTX_new();
void EVP_MD_CTX_free(evp_md_ctx_st*);
const evp_md_st* EVP_sha256();
int EVP_DigestInit_ex(evp_md_ctx_st*, const evp_md_st*, void*);
int EVP_DigestUpdate(evp_md_ctx_st*, const void*, std::size_t);
int EVP_DigestFinal_ex(evp_md_ctx_st*, unsigned char*, unsigned int*);
struct json_object;
struct json_tokener;
json_tokener* json_tokener_new();
void json_tokener_free(json_tokener*);
json_object* json_tokener_parse_ex(json_tokener*, const char*, int);
int json_tokener_get_error(json_tokener*);
std::size_t json_tokener_get_parse_end(json_tokener*);
int json_object_put(json_object*);
int json_object_get_type(const json_object*);
int json_object_object_get_ex(const json_object*, const char*, json_object**);
std::size_t json_object_array_length(const json_object*);
json_object* json_object_array_get_idx(const json_object*, std::size_t);
const char* json_object_get_string(const json_object*);
std::int64_t json_object_get_int64(const json_object*);
int json_object_get_boolean(const json_object*);
}

namespace rocket::qwen38::decode {
namespace {

constexpr std::size_t kMaxObservedBytes =
    kTargetK0LocalVocab * sizeof(float);
constexpr int kJsonBoolean = 1;
constexpr int kJsonInt = 3;
constexpr int kJsonObject = 4;
constexpr int kJsonArray = 5;
constexpr int kJsonString = 6;

std::string sha256(const std::vector<std::uint8_t>& bytes) {
  std::unique_ptr<evp_md_ctx_st, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  std::array<unsigned char, 32> digest{};
  unsigned int size = 0;
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1 ||
      EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1 ||
      size != digest.size())
    throw std::runtime_error("K0 oracle digest failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result(64, '0');
  for (std::size_t index = 0; index < digest.size(); ++index) {
    result[index * 2] = hex[digest[index] >> 4];
    result[index * 2 + 1] = hex[digest[index] & 15];
  }
  return result;
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path,
                                    std::size_t maximum) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::invalid_argument("K0 oracle file unavailable");
  const auto end = input.tellg();
  if (end <= 0 || static_cast<std::uint64_t>(end) > maximum)
    throw std::invalid_argument("K0 oracle file size changed");
  std::vector<std::uint8_t> result(static_cast<std::size_t>(end));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(result.size()));
  if (!input || input.peek() != std::char_traits<char>::eof())
    throw std::invalid_argument("K0 oracle file read changed");
  return result;
}

json_object* field(json_object* object, const char* name, int type) {
  json_object* value = nullptr;
  if (!json_object_object_get_ex(object, name, &value) ||
      json_object_get_type(value) != type)
    throw std::invalid_argument("K0 oracle manifest schema changed");
  return value;
}

std::string string_field(json_object* object, const char* name) {
  return json_object_get_string(field(object, name, kJsonString));
}

int int_field(json_object* object, const char* name) {
  const auto value = json_object_get_int64(field(object, name, kJsonInt));
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max())
    throw std::invalid_argument("K0 oracle integer changed");
  return static_cast<int>(value);
}

std::uint16_t float_to_bf16(float value) noexcept {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounding = 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

bool is_nan(std::uint16_t bits) noexcept {
  return (bits & 0x7f80U) == 0x7f80U && (bits & 0x007fU) != 0;
}

std::uint32_t ordered(std::uint16_t bits) noexcept {
  return (bits & 0x8000U) ? 0x8000U - (bits & 0x7fffU)
                          : 0x8000U + bits;
}

class RuntimeCudaApi final : public TargetLayer3OracleCudaApi {
 public:
  cudaError_t host_alloc(void** pointer, std::size_t bytes) noexcept override {
    return cudaHostAlloc(pointer, bytes, cudaHostAllocDefault);
  }
  cudaError_t free_host(void* pointer) noexcept override {
    return cudaFreeHost(pointer);
  }
  cudaError_t event_create(cudaEvent_t* event) noexcept override {
    return cudaEventCreateWithFlags(event, cudaEventDisableTiming);
  }
  cudaError_t event_destroy(cudaEvent_t event) noexcept override {
    return cudaEventDestroy(event);
  }
  cudaError_t copy_d2h(void* destination, const void* source,
                       std::size_t bytes, cudaStream_t stream) noexcept override {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost,
                           stream);
  }
  cudaError_t event_record(cudaEvent_t event,
                           cudaStream_t stream) noexcept override {
    return cudaEventRecord(event, stream);
  }
  cudaError_t event_sync(cudaEvent_t event) noexcept override {
    return cudaEventSynchronize(event);
  }
};

RuntimeCudaApi runtime_cuda;

}  // namespace

struct NativeTargetK0OracleComparator::Impl {
  struct Artifact {
    std::string name;
    int rows = 0;
    int columns = 0;
    std::vector<std::uint16_t> values;
  };

  int rank = -1;
  pair_reduce::OtelStageSink* telemetry = nullptr;
  TargetLayer3OracleCudaApi* cuda = nullptr;
  std::string manifest;
  std::vector<std::int32_t> tokens;
  std::array<Artifact, 51> artifacts{};
  std::optional<std::array<TargetLayer0BoundaryReference, 3>>
      layer0_boundaries;
  std::int32_t greedy_token = -1;
  void* observed = nullptr;
  cudaEvent_t ready = nullptr;
  bool valid = false;
  bool token_compared = false;
  TargetK0OracleEvidence evidence{};

  void emit(pair_reduce::Outcome outcome, std::uint64_t bytes) noexcept {
    if (!telemetry) return;
    telemetry->emit_span_and_log({
        "rocket.qwen38.k0_oracle.lifecycle", "k0-oracle", "oracle",
        rank == 0 || rank == 1 ? rank : -1, 1, pair_reduce::kDtype, outcome,
        0, bytes});
    if (rank == 0 || rank == 1)
      telemetry->record_duration({rank, 1, pair_reduce::kDtype, outcome, 0});
  }

  void emit_boundary(TargetK0LayerBoundary boundary,
                     pair_reduce::Outcome outcome,
                     std::uint64_t bytes) noexcept {
    if (!telemetry) return;
    static constexpr std::array<std::string_view,
                                kTargetK0LayerBoundaryCount>
        stages = {"rocket.qwen38.k0_boundary.attention_output",
                  "rocket.qwen38.k0_boundary.attention_reduction",
                  "rocket.qwen38.k0_boundary.hc_combine_mix",
                  "rocket.qwen38.k0_boundary.moe_output",
                  "rocket.qwen38.k0_boundary.moe_reduction",
                  "rocket.qwen38.k0_boundary.final_hc"};
    const auto index = static_cast<std::size_t>(boundary);
    if (index >= stages.size()) return;
    telemetry->emit_span_and_log({stages[index], "k0-boundary", "row0-layer0",
                                  rank == 0 || rank == 1 ? rank : -1, 1,
                                  pair_reduce::kDtype, outcome, 0, bytes});
  }
};

NativeTargetK0OracleComparator::NativeTargetK0OracleComparator(
    int rank, const std::filesystem::path& capture,
    pair_reduce::OtelStageSink& telemetry, TargetLayer3OracleCudaApi* cuda_api)
    : impl_(std::make_unique<Impl>()) {
  impl_->rank = rank;
  impl_->telemetry = &telemetry;
  impl_->cuda = cuda_api ? cuda_api : &runtime_cuda;
  try {
    if (rank != 0 && rank != 1)
      throw std::invalid_argument("K0 oracle rank changed");
    if (const char* layer0_directory =
            std::getenv("ROCKET_QWEN38_K0_LAYER0_BOUNDARY_DIR")) {
      if (!*layer0_directory)
        throw std::invalid_argument("K0 layer0 boundary directory changed");
      impl_->layer0_boundaries = authenticate_target_layer0_boundary_references(
          std::filesystem::path(layer0_directory));
    }
    auto manifest_bytes = read_file(capture / "manifest.json", 1 << 20);
    impl_->manifest = sha256(manifest_bytes);
    if (impl_->manifest != kTargetK0OracleManifestSha256)
      throw std::invalid_argument("K0 oracle manifest identity changed");
    json_tokener* tokener = json_tokener_new();
    if (!tokener) throw std::runtime_error("K0 oracle JSON allocation failed");
    json_object* raw = json_tokener_parse_ex(
        tokener, reinterpret_cast<const char*>(manifest_bytes.data()),
        static_cast<int>(manifest_bytes.size()));
    const bool parsed = raw && json_tokener_get_error(tokener) == 0 &&
                        json_tokener_get_parse_end(tokener) ==
                            manifest_bytes.size();
    json_tokener_free(tokener);
    std::unique_ptr<json_object, decltype(&json_object_put)> root(raw,
                                                                 json_object_put);
    if (!parsed || json_object_get_type(root.get()) != kJsonObject ||
        string_field(root.get(), "schema") !=
            "rocket.qwen38.k0-target-oracle.v1" ||
        !json_object_get_boolean(field(root.get(), "complete", kJsonBoolean)) ||
        !json_object_get_boolean(field(root.get(), "valid", kJsonBoolean)) ||
        int_field(root.get(), "generation_index") != 0)
      throw std::invalid_argument("K0 oracle manifest contract changed");
    impl_->greedy_token = int_field(root.get(), "greedy_token_id");
    auto* ids = field(root.get(), "input_token_ids", kJsonArray);
    if (json_object_array_length(ids) != 35)
      throw std::invalid_argument("K0 oracle prompt length changed");
    for (std::size_t row = 0; row < 35; ++row) {
      auto* id = json_object_array_get_idx(ids, row);
      if (!id || json_object_get_type(id) != kJsonInt)
        throw std::invalid_argument("K0 oracle prompt schema changed");
      impl_->tokens.push_back(static_cast<std::int32_t>(
          json_object_get_int64(id)));
    }
    auto* artifacts = field(root.get(), "artifacts", kJsonArray);
    if (json_object_array_length(artifacts) != impl_->artifacts.size())
      throw std::invalid_argument("K0 oracle artifact count changed");
    for (std::size_t index = 0; index < impl_->artifacts.size(); ++index) {
      auto* item = json_object_array_get_idx(artifacts, index);
      if (!item || json_object_get_type(item) != kJsonObject ||
          string_field(item, "dtype") != "bfloat16")
        throw std::invalid_argument("K0 oracle artifact schema changed");
      auto& artifact = impl_->artifacts[index];
      artifact.name = string_field(item, "name");
      const std::string file = string_field(item, "file");
      auto* shape = field(item, "shape", kJsonArray);
      if (json_object_array_length(shape) != 2)
        throw std::invalid_argument("K0 oracle artifact rank changed");
      artifact.rows = static_cast<int>(
          json_object_get_int64(json_object_array_get_idx(shape, 0)));
      artifact.columns = static_cast<int>(
          json_object_get_int64(json_object_array_get_idx(shape, 1)));
      const int expected_rows = index == 50 ? 1 : 35;
      const int expected_columns = index == 0 || index == 49
                                       ? kTargetK0Hidden
                                       : index == 50 ? 2 * kTargetK0LocalVocab
                                                     : kTargetK0HyperHidden;
      if (artifact.rows != expected_rows ||
          artifact.columns != expected_columns ||
          int_field(item, "bytes") !=
              artifact.rows * artifact.columns *
                  static_cast<int>(sizeof(std::uint16_t)))
        throw std::invalid_argument("K0 oracle artifact extent changed");
      auto bytes = read_file(capture / file, 1 << 20);
      if (sha256(bytes) != string_field(item, "sha256"))
        throw std::invalid_argument("K0 oracle artifact digest changed");
      artifact.values.resize(bytes.size() / sizeof(std::uint16_t));
      std::memcpy(artifact.values.data(), bytes.data(), bytes.size());
    }
    if (impl_->artifacts[0].name != "embedding" ||
        impl_->artifacts[49].name != "final_norm" ||
        impl_->artifacts[50].name != "logits")
      throw std::invalid_argument("K0 oracle boundary names changed");
    for (int layer = 0; layer < 48; ++layer) {
      char name[9]{};
      std::snprintf(name, sizeof(name), "layer.%02d", layer);
      if (impl_->artifacts[layer + 1].name != name)
        throw std::invalid_argument("K0 oracle layer order changed");
    }
    if (impl_->cuda->host_alloc(&impl_->observed, kMaxObservedBytes) !=
            cudaSuccess ||
        impl_->cuda->event_create(&impl_->ready) != cudaSuccess)
      throw std::runtime_error("K0 oracle CUDA owner creation failed");
    impl_->valid = true;
  } catch (...) {
    if (impl_->ready) impl_->cuda->event_destroy(impl_->ready);
    if (impl_->observed) impl_->cuda->free_host(impl_->observed);
    impl_->emit(pair_reduce::Outcome::kContractError, 0);
    throw;
  }
}

NativeTargetK0OracleComparator::~NativeTargetK0OracleComparator() {
  if (!impl_) return;
  bool failed = false;
  if (impl_->ready && impl_->cuda->event_destroy(impl_->ready) != cudaSuccess)
    failed = true;
  if (impl_->observed && impl_->cuda->free_host(impl_->observed) != cudaSuccess)
    failed = true;
  if (failed) impl_->emit(pair_reduce::Outcome::kCudaError, 0);
}

int NativeTargetK0OracleComparator::rank() const noexcept { return impl_->rank; }
int NativeTargetK0OracleComparator::rows() const noexcept {
  return static_cast<int>(impl_->tokens.size());
}
std::string_view NativeTargetK0OracleComparator::manifest_sha256() const
    noexcept { return impl_->manifest; }
bool NativeTargetK0OracleComparator::authenticated() const noexcept {
  return impl_->valid;
}

void NativeTargetK0OracleComparator::observe(
    TargetK0LayerBoundary boundary, const void* device_values,
    std::size_t elements, TargetK0DiagnosticDtype dtype, cudaStream_t stream,
    TargetK0LayerBoundaryEvidence& evidence) {
  const auto index = static_cast<std::size_t>(boundary);
  const bool wide = boundary == TargetK0LayerBoundary::kHyperconnectionCombineMix ||
                    boundary == TargetK0LayerBoundary::kFinalHyperconnection;
  const bool fp32 = boundary == TargetK0LayerBoundary::kAttentionReduction ||
                    boundary == TargetK0LayerBoundary::kMoeReduction;
  const std::size_t expected_elements =
      wide ? static_cast<std::size_t>(kTargetK0HyperHidden)
           : static_cast<std::size_t>(kTargetK0Hidden);
  if (!impl_->valid || index >= kTargetK0LayerBoundaryCount ||
      !device_values || !stream || elements != expected_elements ||
      (fp32 ? dtype != TargetK0DiagnosticDtype::kFloat32
            : dtype != TargetK0DiagnosticDtype::kBfloat16) ||
      evidence.elements[index] != 0)
    throw std::invalid_argument("K0 boundary diagnostic contract changed");
  const std::size_t element_bytes =
      dtype == TargetK0DiagnosticDtype::kFloat32 ? sizeof(float)
                                                 : sizeof(std::uint16_t);
  const std::size_t observed_bytes = elements * element_bytes;
  if (observed_bytes > kMaxObservedBytes ||
      impl_->cuda->copy_d2h(impl_->observed, device_values, observed_bytes,
                            stream) != cudaSuccess ||
      impl_->cuda->event_record(impl_->ready, stream) != cudaSuccess ||
      impl_->cuda->event_sync(impl_->ready) != cudaSuccess) {
    impl_->emit_boundary(boundary, pair_reduce::Outcome::kCudaError,
                         observed_bytes);
    throw std::runtime_error("K0 boundary diagnostic D2H fence failed");
  }

  constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  std::uint64_t hash = kFnvOffset;
  const auto* bytes = static_cast<const std::uint8_t*>(impl_->observed);
  for (std::size_t byte = 0; byte < observed_bytes; ++byte)
    hash = (hash ^ bytes[byte]) * kFnvPrime;
  std::uint32_t zero_count = 0;
  std::uint32_t nonfinite_count = 0;
  if (dtype == TargetK0DiagnosticDtype::kFloat32) {
    const auto* values = static_cast<const float*>(impl_->observed);
    for (std::size_t element = 0; element < elements; ++element) {
      zero_count += values[element] == 0.0F;
      nonfinite_count += !std::isfinite(values[element]);
    }
  } else {
    const auto* values = static_cast<const std::uint16_t*>(impl_->observed);
    for (std::size_t element = 0; element < elements; ++element) {
      zero_count += (values[element] & 0x7fffU) == 0;
      nonfinite_count += (values[element] & 0x7f80U) == 0x7f80U;
    }
  }
  evidence.hashes[index] = hash;
  evidence.elements[index] = static_cast<std::uint32_t>(elements);
  evidence.zero_counts[index] = zero_count;
  evidence.nonfinite_counts[index] = nonfinite_count;
  if (boundary == TargetK0LayerBoundary::kAttentionReduction &&
      impl_->layer0_boundaries) {
    const auto rounded = round_target_layer0_attention_to_bf16(
        static_cast<const float*>(impl_->observed), elements);
    const auto comparison = compare_target_layer0_boundary_bytes(
        (*impl_->layer0_boundaries)[0], rounded.data(), rounded.size());
    evidence.reference_compared[index] = 1;
    evidence.reference_exact[index] = comparison.exact ? 1 : 0;
    evidence.reference_mismatch_counts[index] =
        static_cast<std::uint32_t>(comparison.mismatch_count);
    evidence.reference_first_mismatches[index] =
        static_cast<std::uint32_t>(comparison.first_mismatch);
  } else if (boundary == TargetK0LayerBoundary::kHyperconnectionCombineMix &&
             impl_->layer0_boundaries) {
    const auto comparison = compare_target_layer0_boundary_bytes(
        (*impl_->layer0_boundaries)[1], bytes, observed_bytes);
    evidence.reference_compared[index] = 1;
    evidence.reference_exact[index] = comparison.exact ? 1 : 0;
    evidence.reference_mismatch_counts[index] =
        static_cast<std::uint32_t>(comparison.mismatch_count);
    evidence.reference_first_mismatches[index] =
        static_cast<std::uint32_t>(comparison.first_mismatch);
  }
  impl_->emit_boundary(boundary, pair_reduce::Outcome::kOk, observed_bytes);
}
std::int32_t NativeTargetK0OracleComparator::expected_input_token(int row) const {
  if (!impl_->valid || row < 0 || row >= rows())
    throw std::invalid_argument("K0 oracle prompt row changed");
  return impl_->tokens[static_cast<std::size_t>(row)];
}

void NativeTargetK0OracleComparator::compare(
    TargetK0Boundary boundary, int row, int layer, const void* device_values,
    std::size_t elements, cudaStream_t stream) {
  if (!impl_->valid || !device_values || !stream || row < 0 || row >= rows())
    throw std::invalid_argument("K0 oracle comparison contract changed");
  int artifact_index = -1;
  std::size_t observed_bytes = elements * sizeof(std::uint16_t);
  if (boundary == TargetK0Boundary::kEmbedding && layer == -1 &&
      elements == kTargetK0Hidden) {
    artifact_index = 0;
  } else if (boundary == TargetK0Boundary::kLayer && layer >= 0 && layer < 48 &&
             elements == kTargetK0HyperHidden) {
    artifact_index = layer + 1;
  } else if (boundary == TargetK0Boundary::kFinalNorm && layer == -1 &&
             row == rows() - 1 && elements == kTargetK0Hidden) {
    artifact_index = 49;
  } else if (boundary == TargetK0Boundary::kLocalLogits && layer == -1 &&
             row == rows() - 1 && elements == kTargetK0LocalVocab) {
    artifact_index = 50;
    observed_bytes = elements * sizeof(float);
  } else {
    throw std::invalid_argument("K0 oracle boundary changed");
  }
  if (impl_->cuda->copy_d2h(impl_->observed, device_values, observed_bytes,
                             stream) != cudaSuccess ||
      impl_->cuda->event_record(impl_->ready, stream) != cudaSuccess ||
      impl_->cuda->event_sync(impl_->ready) != cudaSuccess) {
    impl_->emit(pair_reduce::Outcome::kCudaError, observed_bytes);
    throw std::runtime_error("K0 oracle D2H fence failed");
  }
  const auto& artifact = impl_->artifacts[artifact_index];
  const std::size_t expected_offset =
      artifact_index == 50
          ? static_cast<std::size_t>(impl_->rank) * kTargetK0LocalVocab
          : static_cast<std::size_t>(row) * elements;
  impl_->evidence = {boundary, row, layer, 0, 0, false};
  for (std::size_t index = 0; index < elements; ++index) {
    const auto expected = artifact.values[expected_offset + index];
    const auto observed = artifact_index == 50
                              ? float_to_bf16(
                                    static_cast<const float*>(impl_->observed)[index])
                              : static_cast<const std::uint16_t*>(impl_->observed)[index];
    const std::uint32_t ulp =
        is_nan(expected) || is_nan(observed)
            ? std::numeric_limits<std::uint16_t>::max()
            : std::max(ordered(expected), ordered(observed)) -
                  std::min(ordered(expected), ordered(observed));
    impl_->evidence.max_ulp = std::max(impl_->evidence.max_ulp, ulp);
    impl_->evidence.mismatch_count += ulp > 1;
  }
  impl_->evidence.accepted = impl_->evidence.mismatch_count == 0;
  impl_->emit(impl_->evidence.accepted ? pair_reduce::Outcome::kOk
                                      : pair_reduce::Outcome::kContractError,
              observed_bytes);
  if (!impl_->evidence.accepted)
    throw std::logic_error("K0 oracle boundary mismatch");
}

void NativeTargetK0OracleComparator::compare_token(std::int32_t token) {
  if (!impl_->valid || impl_->token_compared || token != impl_->greedy_token)
    throw std::logic_error("K0 oracle token mismatch");
  impl_->token_compared = true;
}

const TargetK0OracleEvidence& NativeTargetK0OracleComparator::evidence() const
    noexcept { return impl_->evidence; }

}  // namespace rocket::qwen38::decode

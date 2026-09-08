// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer0_boundary_comparator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

extern "C" {
struct evp_md_ctx_st;
struct evp_md_st;
evp_md_ctx_st* EVP_MD_CTX_new();
void EVP_MD_CTX_free(evp_md_ctx_st*);
const evp_md_st* EVP_sha256();
int EVP_DigestInit_ex(evp_md_ctx_st*, const evp_md_st*, void*);
int EVP_DigestUpdate(evp_md_ctx_st*, const void*, std::size_t);
int EVP_DigestFinal_ex(evp_md_ctx_st*, unsigned char*, unsigned int*);
}

namespace rocket::qwen38::decode {
namespace {

struct Identity {
  TargetLayer0BoundaryStage stage;
  const char* stem;
  std::size_t bytes;
  const char* payload_sha256;
  const char* metadata_sha256;
};

constexpr std::array<Identity, 3> kIdentities{{
    {TargetLayer0BoundaryStage::kAttentionOutput, "attention_output", 5'120,
     "50c355cb6c75e64e9929ca920d88f8da50bbd5f329ed915c2739bd6b465d2a08",
     "8e8058a8fc2c311c980b610cde74efb31fd39eae0d7bd0c0aad339a76325ebe8"},
    {TargetLayer0BoundaryStage::kHyperconnectionCombineMix, "hc_combine_mix", 20'480,
     "6f0dbabe0d74f72dfdfe15ece373dbce80fec62a9e6b6bf1241e5e391fe57f97",
     "6704902bcacaba506600fb0146e3a7d4dd37cea214c666c40600bb97c5e7fbcb"},
    {TargetLayer0BoundaryStage::kMoeOutput, "moe_output", 5'120,
     "5bf23508725ef92715a9d8a43666316f8fb20b7a7244b6c38355c802fc569acb",
     "f5f4790d1972f748e1f648cdeefd365be91d5dfdec3250beab02f3d9af68ada0"},
}};

struct Fd {
  int value = -1;
  ~Fd() { if (value >= 0) close(value); }
};

std::string sha256(const void* bytes, std::size_t size) {
  auto* context = EVP_MD_CTX_new();
  std::array<unsigned char, 32> digest{};
  unsigned int length = 0;
  const bool ok = context &&
      EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
      EVP_DigestUpdate(context, bytes, size) == 1 &&
      EVP_DigestFinal_ex(context, digest.data(), &length) == 1 &&
      length == digest.size();
  if (context) EVP_MD_CTX_free(context);
  if (!ok) throw std::runtime_error("layer0 boundary SHA256 failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result(64, '0');
  for (std::size_t i = 0; i < digest.size(); ++i) {
    result[2 * i] = hex[digest[i] >> 4];
    result[2 * i + 1] = hex[digest[i] & 15];
  }
  return result;
}

std::vector<std::uint8_t> read_exact(int directory, std::string_view name,
                                     std::size_t bytes) {
  Fd file{openat(directory, std::string(name).c_str(),
                 O_RDONLY | O_CLOEXEC | O_NOFOLLOW)};
  struct stat status{};
  std::vector<std::uint8_t> result(bytes);
  if (file.value < 0 || fstat(file.value, &status) != 0 ||
      !S_ISREG(status.st_mode) || status.st_size != static_cast<off_t>(bytes) ||
      pread(file.value, result.data(), bytes, 0) != static_cast<ssize_t>(bytes))
    throw std::invalid_argument("layer0 boundary file identity changed");
  return result;
}

}  // namespace

std::array<TargetLayer0BoundaryReference, 3>
authenticate_target_layer0_boundary_references(
    const std::filesystem::path& directory) {
  Fd root{open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                      O_NOFOLLOW)};
  if (root.value < 0)
    throw std::invalid_argument("layer0 boundary directory changed");
  std::array<TargetLayer0BoundaryReference, 3> result;
  for (std::size_t i = 0; i < kIdentities.size(); ++i) {
    const auto& identity = kIdentities[i];
    const auto metadata = read_exact(
        root.value, std::string(identity.stem) + ".json",
        identity.stage == TargetLayer0BoundaryStage::kHyperconnectionCombineMix
            ? 237 : (identity.stage == TargetLayer0BoundaryStage::kAttentionOutput ? 236 : 230));
    auto payload = read_exact(root.value,
                              std::string(identity.stem) + ".bin",
                              identity.bytes);
    if (sha256(metadata.data(), metadata.size()) != identity.metadata_sha256 ||
        sha256(payload.data(), payload.size()) != identity.payload_sha256)
      throw std::invalid_argument("layer0 boundary digest changed");
    result[i] = {identity.stage, std::move(payload), identity.payload_sha256};
  }
  return result;
}

TargetLayer0BoundaryEvidence compare_target_layer0_boundary_bytes(
    const TargetLayer0BoundaryReference& expected,
    const std::uint8_t* observed, std::size_t bytes) {
  if (!observed || bytes != expected.bytes.size())
    throw std::invalid_argument("layer0 boundary observation extent changed");
  TargetLayer0BoundaryEvidence result{};
  result.stage = expected.stage;
  result.observed_sha256 = sha256(observed, bytes);
  bool first = true;
  for (std::size_t i = 0; i < bytes; ++i) {
    if (expected.bytes[i] == observed[i]) continue;
    if (first) { result.first_mismatch = i; first = false; }
    ++result.mismatch_count;
  }
  result.exact = result.mismatch_count == 0 &&
                 result.observed_sha256 == expected.sha256;
  return result;
}

std::vector<std::uint8_t> round_target_layer0_attention_to_bf16(
    const float* observed, std::size_t elements) {
  if (!observed || elements != 2'560)
    throw std::invalid_argument("layer0 attention observation extent changed");
  std::vector<std::uint8_t> result(elements * sizeof(std::uint16_t));
  for (std::size_t i = 0; i < elements; ++i) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(observed[i]);
    const auto rounded = static_cast<std::uint16_t>(
        (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
    std::memcpy(result.data() + i * sizeof(rounded), &rounded, sizeof(rounded));
  }
  return result;
}

TargetLayer0BoundaryComparator::TargetLayer0BoundaryComparator(
    const std::filesystem::path& directory)
    : references_(authenticate_target_layer0_boundary_references(directory)) {
  if (cudaHostAlloc(&pinned_, 20'480, cudaHostAllocDefault) != cudaSuccess ||
      cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess) {
    if (ready_) cudaEventDestroy(ready_);
    if (pinned_) cudaFreeHost(pinned_);
    throw std::runtime_error("layer0 boundary debug allocation failed");
  }
  authenticated_ = true;
}

TargetLayer0BoundaryComparator::~TargetLayer0BoundaryComparator() {
  if (ready_) cudaEventDestroy(ready_);
  if (pinned_) cudaFreeHost(pinned_);
}

TargetLayer0BoundaryEvidence TargetLayer0BoundaryComparator::compare(
    TargetLayer0BoundaryStage stage, const __nv_bfloat16* device_values,
    cudaStream_t stream) {
  if (!authenticated_ || next_stage_ >= references_.size() ||
      references_[next_stage_].stage != stage || !device_values || !stream)
    throw std::logic_error("layer0 boundary debug order changed");
  const auto bytes = references_[next_stage_].bytes.size();
  if (cudaMemcpyAsync(pinned_, device_values, bytes, cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaEventRecord(ready_, stream) != cudaSuccess ||
      cudaEventSynchronize(ready_) != cudaSuccess)
    throw std::runtime_error("layer0 boundary debug copy failed");
  auto result = compare_target_layer0_boundary_bytes(
      references_[next_stage_], static_cast<const std::uint8_t*>(pinned_), bytes);
  ++next_stage_;
  return result;
}

}  // namespace rocket::qwen38::decode

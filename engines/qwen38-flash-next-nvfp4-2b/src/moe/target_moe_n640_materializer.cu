// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_n640_materializer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

extern "C" {
struct evp_md_ctx_st;
struct evp_md_st;
evp_md_ctx_st* EVP_MD_CTX_new();
void EVP_MD_CTX_free(evp_md_ctx_st*);
const evp_md_st* EVP_sha256();
int EVP_DigestInit_ex(evp_md_ctx_st*, const evp_md_st*, void*);
int EVP_DigestUpdate(evp_md_ctx_st*, const void*, std::size_t);
int EVP_DigestFinal_ex(evp_md_ctx_st*, unsigned char*, unsigned int*);
unsigned long OpenSSL_version_num();
}

namespace rocket::qwen38::moe {
namespace {

constexpr std::size_t kExperts = 256;
constexpr std::size_t kH = kTargetMoeHidden;
constexpr std::size_t kN = kTargetMoeLogicalIntermediate;
constexpr std::size_t kNp = kTargetMoePhysicalIntermediate;
constexpr std::size_t kW13Rows = 2 * kN;
constexpr std::size_t kW13PhysicalRows = 2 * kNp;

void require_size(std::size_t actual, std::size_t expected,
                  const char* name) {
  if (actual != expected)
    throw std::invalid_argument(std::string("target MoE logical extent changed: ") + name);
}

std::size_t swizzled_scale_offset(std::size_t row, std::size_t column_block,
                                  std::size_t rows, std::size_t columns) {
  const std::size_t row_tiles = (rows + 127) / 128;
  const std::size_t column_tiles = (columns + 3) / 4;
  const std::size_t tile_row = row / 128;
  const std::size_t inner_m = (row % 128) / 32;
  const std::size_t outer_m = row % 32;
  const std::size_t tile_column = column_block / 4;
  const std::size_t inner_column = column_block % 4;
  (void)row_tiles;
  return ((((tile_row * column_tiles + tile_column) * 32 + outer_m) * 4 +
            inner_m) * 4 + inner_column);
}

void copy_scale_row(const std::uint8_t* source, std::uint8_t* destination,
                    std::size_t source_row, std::size_t destination_row,
                    std::size_t source_rows, std::size_t destination_rows,
                    std::size_t source_columns, std::size_t destination_columns) {
  for (std::size_t column = 0; column < source_columns; ++column) {
    destination[swizzled_scale_offset(destination_row, column,
                                      destination_rows, destination_columns)] =
        source[swizzled_scale_offset(source_row, column, source_rows,
                                     source_columns)];
  }
}

template <typename T>
T* allocate_copy(std::vector<void*>& allocations, std::span<const T> host,
                 cudaStream_t stream) {
  T* pointer = nullptr;
  const auto bytes = host.size_bytes();
  if (cudaMalloc(reinterpret_cast<void**>(&pointer), bytes) != cudaSuccess)
    throw std::runtime_error("target MoE N768 device allocation failed");
  allocations.push_back(pointer);
  if (cudaMemcpyAsync(pointer, host.data(), bytes, cudaMemcpyHostToDevice,
                      stream) != cudaSuccess)
    throw std::runtime_error("target MoE N768 H2D copy failed");
  return pointer;
}

bool nonzero(const std::array<std::uint8_t, 32>& value) {
  return std::any_of(value.begin(), value.end(), [](std::uint8_t byte) {
    return byte != 0;
  });
}

std::array<std::uint8_t, 32> parse_digest(std::string_view text) {
  if (text.size() != 64) throw std::logic_error("target MoE digest constant changed");
  std::array<std::uint8_t, 32> result{};
  const auto nibble = [](char value) -> std::uint8_t {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    throw std::logic_error("target MoE digest constant is invalid");
  };
  for (std::size_t index = 0; index < result.size(); ++index)
    result[index] = static_cast<std::uint8_t>(
        nibble(text[2 * index]) * 16 + nibble(text[2 * index + 1]));
  return result;
}

template <typename... Spans>
std::array<std::uint8_t, 32> digest(Spans... spans) {
  auto* context = EVP_MD_CTX_new();
  if (!context) throw std::runtime_error("target MoE SHA256 context failed");
  std::array<std::uint8_t, 32> output{};
  unsigned int bytes = 0;
  const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                  (... && (EVP_DigestUpdate(context, spans.data(),
                                            spans.size_bytes()) == 1)) &&
                  EVP_DigestFinal_ex(context, output.data(), &bytes) == 1;
  EVP_MD_CTX_free(context);
  if (!ok || bytes != output.size())
    throw std::runtime_error("target MoE SHA256 failed");
  return output;
}

}  // namespace

TargetMoeTransformedIdentity target_moe_layer3_transformed_identity(int rank) {
  if (rank != 0 && rank != 1)
    throw std::invalid_argument("target MoE transformed rank identity changed");
  static constexpr std::string_view layouts[] = {
      "ebf6db24c257c3516f7ff8c94bb2ba70d692c62c4cbf1b2577ebe99a3f56875b",
      "6e20c303b336f980e94e7aa3d897009ac527e1810279c09c8cbab1bd7867f841"};
  static constexpr std::string_view source[] = {
      "5e62ca24366069fbbfb05b234730a72046976d87269f0739da6a29e7ed50e93d",
      "fd4c46b152a4a8199b3370e27c4e825f79f54ebdcd8f5e808074d18fac5101e9"};
  static constexpr std::string_view physical[] = {
      "e0d9089c5976a77d3a6d460713b5be5dae962c02fc64b6a84c5db7c8cf3558fc",
      "54fc2b8b2df62686e0a40a46ae6b9e3568de866dbeea2794a669c0849c83dc25"};
  return {parse_digest(layouts[rank]), parse_digest(source[rank]),
          parse_digest(physical[rank]), kTargetMoeLogicalIntermediate,
          kTargetMoePhysicalIntermediate, 256, kTargetMoeHidden, rank, 3};
}

TargetMoePhysicalN768Host materialize_target_moe_n640_host(
    const TargetMoeLogicalN640& source) {
  require_size(source.w13_packed.size(), kExperts * kW13Rows * kH / 2,
               "w13 packed");
  require_size(source.w13_scale.size(), kExperts * kW13Rows * (kH / 16),
               "w13 scale");
  require_size(source.down_packed.size(), kExperts * kH * kN / 2,
               "down packed");
  require_size(source.down_scale.size(), kExperts * kH * (kN / 16),
               "down scale");
  require_size(source.input_global_scale.size(), kExperts, "input scale");
  require_size(source.w1_alpha.size(), kExperts, "w1 alpha");
  require_size(source.w2_alpha.size(), kExperts, "w2 alpha");
  require_size(source.down_input_scale.size(), kExperts, "down input scale");

  TargetMoePhysicalN768Host result;
  result.w13_packed.assign(kExperts * kW13PhysicalRows * kH / 2, 0);
  result.w13_scale.assign(kExperts * kW13PhysicalRows * (kH / 16), 0);
  result.down_packed.assign(kExperts * kH * kNp / 2, 0);
  result.down_scale.assign(kExperts * kH * (kNp / 16), 0);
  result.input_global_scale.assign(source.input_global_scale.begin(),
                                   source.input_global_scale.end());
  result.w2_alpha.assign(source.w2_alpha.begin(), source.w2_alpha.end());
  result.down_input_scale.assign(source.down_input_scale.begin(),
                                 source.down_input_scale.end());
  result.folded_w1_alpha.resize(kExperts);

  constexpr std::size_t source_w13_bytes = kW13Rows * kH / 2;
  constexpr std::size_t physical_w13_bytes = kW13PhysicalRows * kH / 2;
  constexpr std::size_t half_bytes = kN * kH / 2;
  constexpr std::size_t source_w13_scale_bytes = kW13Rows * (kH / 16);
  constexpr std::size_t physical_w13_scale_bytes =
      kW13PhysicalRows * (kH / 16);
  constexpr std::size_t source_down_bytes = kH * kN / 2;
  constexpr std::size_t physical_down_bytes = kH * kNp / 2;
  constexpr std::size_t source_down_scale_bytes = kH * (kN / 16);
  constexpr std::size_t physical_down_scale_bytes = kH * (kNp / 16);

  for (std::size_t expert = 0; expert < kExperts; ++expert) {
    const auto* w13 = source.w13_packed.data() + expert * source_w13_bytes;
    auto* w13p = result.w13_packed.data() + expert * physical_w13_bytes;
    std::memcpy(w13p, w13, half_bytes);
    std::memcpy(w13p + kNp * kH / 2, w13 + half_bytes, half_bytes);

    const auto* s13 = source.w13_scale.data() +
                      expert * source_w13_scale_bytes;
    auto* s13p = result.w13_scale.data() +
                 expert * physical_w13_scale_bytes;
    for (std::size_t row = 0; row < kW13Rows; ++row) {
      const std::size_t destination_row = row < kN ? row : row + (kNp - kN);
      copy_scale_row(s13, s13p, row, destination_row, kW13Rows,
                     kW13PhysicalRows, kH / 16, kH / 16);
    }

    const auto* down = source.down_packed.data() + expert * source_down_bytes;
    auto* downp = result.down_packed.data() + expert * physical_down_bytes;
    for (std::size_t row = 0; row < kH; ++row)
      std::memcpy(downp + row * kNp / 2, down + row * kN / 2, kN / 2);

    const auto* sd = source.down_scale.data() +
                     expert * source_down_scale_bytes;
    auto* sdp = result.down_scale.data() +
                expert * physical_down_scale_bytes;
    for (std::size_t row = 0; row < kH; ++row)
      copy_scale_row(sd, sdp, row, row, kH, kH, kN / 16, kNp / 16);

    result.folded_w1_alpha[expert] =
        source.w1_alpha[expert] * source.input_global_scale[expert];
  }
  return result;
}

std::array<std::uint8_t, 32> target_moe_n640_sha256(
    const TargetMoeLogicalN640& source) {
  return digest(source.w13_packed, source.w13_scale, source.down_packed,
                source.down_scale, source.input_global_scale, source.w1_alpha,
                source.w2_alpha, source.down_input_scale);
}

std::array<std::uint8_t, 32> target_moe_n768_sha256(
    const TargetMoePhysicalN768Host& physical) {
  return digest(std::span<const std::uint8_t>(physical.w13_packed),
                std::span<const std::uint8_t>(physical.w13_scale),
                std::span<const std::uint8_t>(physical.down_packed),
                std::span<const std::uint8_t>(physical.down_scale),
                std::span<const float>(physical.input_global_scale),
                std::span<const float>(physical.folded_w1_alpha),
                std::span<const float>(physical.w2_alpha),
                std::span<const float>(physical.down_input_scale));
}

std::array<std::array<std::uint8_t, 32>, 8> target_moe_n768_plane_sha256(
    const TargetMoePhysicalN768Host& p) {
  return {digest(std::span<const std::uint8_t>(p.w13_packed)),
          digest(std::span<const std::uint8_t>(p.w13_scale)),
          digest(std::span<const std::uint8_t>(p.down_packed)),
          digest(std::span<const std::uint8_t>(p.down_scale)),
          digest(std::span<const float>(p.input_global_scale)),
          digest(std::span<const float>(p.folded_w1_alpha)),
          digest(std::span<const float>(p.w2_alpha)),
          digest(std::span<const float>(p.down_input_scale))};
}

TargetMoeN768DeviceOwner::TargetMoeN768DeviceOwner(
    int device, const TargetMoeLogicalN640& source,
    TargetMoeTransformedIdentity identity)
    : device_(device), identity_(identity) {
  if (device < 0 || !nonzero(identity.source_layout_sha256) ||
      !nonzero(identity.source_planes_sha256) ||
      !nonzero(identity.physical_planes_sha256) ||
      identity.logical_intermediate != static_cast<int>(kN) ||
      identity.physical_intermediate != static_cast<int>(kNp) ||
      identity.experts != static_cast<int>(kExperts) ||
      identity.hidden != static_cast<int>(kH) ||
      (identity.rank != 0 && identity.rank != 1) || identity.layer != 3)
    throw std::invalid_argument("target MoE transformed identity changed");
  const auto expected = target_moe_layer3_transformed_identity(identity.rank);
  if (identity.source_layout_sha256 != expected.source_layout_sha256 ||
      identity.source_planes_sha256 != expected.source_planes_sha256 ||
      identity.physical_planes_sha256 != expected.physical_planes_sha256)
    throw std::invalid_argument("target MoE transformed digest identity changed");
  if ((OpenSSL_version_num() >> 28) != 3)
    throw std::runtime_error("target MoE SHA256 requires OpenSSL 3 ABI");
  if (cudaSetDevice(device) != cudaSuccess)
    throw std::runtime_error("target MoE materializer device selection failed");

  cudaStream_t stream = nullptr;
  cudaEvent_t published = nullptr;
  try {
    if (target_moe_n640_sha256(source) != identity.source_planes_sha256)
      throw std::invalid_argument("target MoE logical source hash changed");
    const auto host = materialize_target_moe_n640_host(source);
    if (target_moe_n768_sha256(host) != identity.physical_planes_sha256)
      throw std::invalid_argument("target MoE N768 transformed hash changed");
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&published, cudaEventDisableTiming) != cudaSuccess)
      throw std::runtime_error("target MoE materializer fence creation failed");
    weights_.w13_packed = allocate_copy(
        allocations_, std::span<const std::uint8_t>(host.w13_packed), stream);
    weights_.w13_scale = allocate_copy(
        allocations_, std::span<const std::uint8_t>(host.w13_scale), stream);
    weights_.down_packed = allocate_copy(
        allocations_, std::span<const std::uint8_t>(host.down_packed), stream);
    weights_.down_scale = allocate_copy(
        allocations_, std::span<const std::uint8_t>(host.down_scale), stream);
    weights_.input_global_scale = allocate_copy(
        allocations_, std::span<const float>(host.input_global_scale), stream);
    weights_.folded_w1_alpha = allocate_copy(
        allocations_, std::span<const float>(host.folded_w1_alpha), stream);
    weights_.w2_alpha = allocate_copy(
        allocations_, std::span<const float>(host.w2_alpha), stream);
    weights_.down_input_scale = allocate_copy(
        allocations_, std::span<const float>(host.down_input_scale), stream);
    if (cudaEventRecord(published, stream) != cudaSuccess ||
        cudaEventSynchronize(published) != cudaSuccess)
      throw std::runtime_error("target MoE materializer publication fence failed");
    cudaEventDestroy(published);
    cudaStreamDestroy(stream);
  } catch (...) {
    if (published) cudaEventDestroy(published);
    if (stream) cudaStreamDestroy(stream);
    for (auto* pointer : allocations_) cudaFree(pointer);
    allocations_.clear();
    throw;
  }
}

TargetMoeN768DeviceOwner::~TargetMoeN768DeviceOwner() {
  if (device_ >= 0) cudaSetDevice(device_);
  for (auto* pointer : allocations_) cudaFree(pointer);
}

}  // namespace rocket::qwen38::moe

namespace {
thread_local std::string materializer_last_error;
}

extern "C" int rocket_qwen38_target_moe_n640_hash(
    const RocketQwen38TargetMoeLogicalN640* source,
    std::uint8_t source_sha256[32],
    std::uint8_t physical_sha256[32]) noexcept {
  materializer_last_error.clear();
  if (!source || !source_sha256 || !physical_sha256) return 1;
  try {
    namespace moe = rocket::qwen38::moe;
    constexpr std::size_t e = 256, h = 2560, n = 640;
    if (!source->w13_packed || !source->w13_scale || !source->down_packed ||
        !source->down_scale || !source->input_global_scale ||
        !source->w1_alpha || !source->w2_alpha || !source->down_input_scale)
      throw std::invalid_argument("target MoE logical C ABI pointer changed");
    const moe::TargetMoeLogicalN640 view{
        {source->w13_packed, e * 2 * n * h / 2},
        {source->w13_scale, e * 2 * n * (h / 16)},
        {source->down_packed, e * h * n / 2},
        {source->down_scale, e * h * (n / 16)},
        {source->input_global_scale, e}, {source->w1_alpha, e},
        {source->w2_alpha, e}, {source->down_input_scale, e}};
    const auto source_digest = moe::target_moe_n640_sha256(view);
    const auto physical = moe::materialize_target_moe_n640_host(view);
    const auto physical_digest = moe::target_moe_n768_sha256(physical);
    std::copy(source_digest.begin(), source_digest.end(), source_sha256);
    std::copy(physical_digest.begin(), physical_digest.end(), physical_sha256);
    return 0;
  } catch (const std::exception& error) {
    materializer_last_error = error.what();
  } catch (...) {
    materializer_last_error = "unknown target MoE N640 materializer failure";
  }
  return 1;
}

extern "C" const char* rocket_qwen38_target_moe_n640_last_error() noexcept {
  return materializer_last_error.c_str();
}

extern "C" int rocket_qwen38_target_moe_n640_plane_hashes(
    const RocketQwen38TargetMoeLogicalN640* source,
    std::uint8_t output[8][32]) noexcept {
  materializer_last_error.clear();
  if (!source || !output || !source->w13_packed || !source->w13_scale ||
      !source->down_packed || !source->down_scale ||
      !source->input_global_scale || !source->w1_alpha || !source->w2_alpha ||
      !source->down_input_scale)
    return 1;
  try {
    namespace moe = rocket::qwen38::moe;
    constexpr std::size_t e = 256, h = 2560, n = 640;
    const moe::TargetMoeLogicalN640 view{
        {source->w13_packed, e * 2 * n * h / 2},
        {source->w13_scale, e * 2 * n * (h / 16)},
        {source->down_packed, e * h * n / 2},
        {source->down_scale, e * h * (n / 16)},
        {source->input_global_scale, e}, {source->w1_alpha, e},
        {source->w2_alpha, e}, {source->down_input_scale, e}};
    const auto p = moe::materialize_target_moe_n640_host(view);
    const auto digests = moe::target_moe_n768_plane_sha256(p);
    for (std::size_t plane = 0; plane < digests.size(); ++plane)
      std::copy(digests[plane].begin(), digests[plane].end(), output[plane]);
    return 0;
  } catch (const std::exception& error) {
    materializer_last_error = error.what();
  }
  return 1;
}

// SPDX-License-Identifier: Apache-2.0
#include "attention/qsa_sidecar_owner.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
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
unsigned long OpenSSL_version_num();
}

namespace rocket::qwen38::attention {
namespace {

std::array<std::uint8_t, 32> sha256(const std::uint8_t* data,
                                    std::size_t bytes) {
  auto* context = EVP_MD_CTX_new();
  if (!context) throw std::runtime_error("QSA sidecar SHA256 context failed");
  std::array<std::uint8_t, 32> output{};
  unsigned int written = 0;
  const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                  EVP_DigestUpdate(context, data, bytes) == 1 &&
                  EVP_DigestFinal_ex(context, output.data(), &written) == 1;
  EVP_MD_CTX_free(context);
  if (!ok || written != output.size())
    throw std::runtime_error("QSA sidecar SHA256 failed");
  return output;
}

bool nonzero(const std::array<std::uint8_t, 32>& value) {
  return std::any_of(value.begin(), value.end(), [](std::uint8_t byte) {
    return byte != 0;
  });
}

std::array<std::uint8_t, 32> parse_digest(std::string_view text) {
  if (text.size() != 64) throw std::logic_error("QSA sidecar digest constant changed");
  std::array<std::uint8_t, 32> result{};
  const auto nibble = [](char value) -> std::uint8_t {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    throw std::logic_error("QSA sidecar digest constant is invalid");
  };
  for (std::size_t index = 0; index < result.size(); ++index)
    result[index] = static_cast<std::uint8_t>(
        nibble(text[2 * index]) * 16 + nibble(text[2 * index + 1]));
  return result;
}

std::vector<std::uint8_t> read_exact(const std::filesystem::path& path) {
  const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0)
    throw std::runtime_error("QSA sidecar payload open failed");
  std::vector<std::uint8_t> bytes(kQsaSidecarBytes);
  std::size_t consumed = 0;
  while (consumed != bytes.size()) {
    const ssize_t result = pread(descriptor, bytes.data() + consumed,
                                 bytes.size() - consumed, consumed);
    if (result <= 0) {
      close(descriptor);
      throw std::runtime_error("QSA sidecar payload read failed");
    }
    consumed += static_cast<std::size_t>(result);
  }
  std::uint8_t extra = 0;
  const ssize_t tail = pread(descriptor, &extra, 1, bytes.size());
  close(descriptor);
  if (tail != 0) throw std::runtime_error("QSA sidecar payload size changed");
  return bytes;
}

}  // namespace

QsaSidecarIdentity layer3_qsa_sidecar_identity(int rank) {
  return target_qsa_sidecar_identity(rank, 3);
}

std::size_t target_qsa_sidecar_offset(int layer) {
  if (layer < 0 || layer >= 48 || layer % 4 != 3)
    throw std::invalid_argument("QSA sidecar layer changed");
  return static_cast<std::size_t>(layer / 4) * kQsaIndexerLayer3Bytes;
}

QsaSidecarIdentity target_qsa_sidecar_identity(int rank, int layer) {
  if (rank != 0 && rank != 1)
    throw std::invalid_argument("QSA sidecar rank changed");
  if (layer < 0 || layer >= 48 || layer % 4 != 3)
    throw std::invalid_argument("QSA sidecar layer changed");
  constexpr std::array<std::string_view, kQsaSidecarLayers> kLayerDigests{{
      "741d5d2a31a217b922a9d4fb2dee1bdf015ca451d11c668b11b8e6afd00569d2",
      "1cc4dd08119f13234254f4bde24bc578af7d8a4ee41e8eb86ec54678c32c4499",
      "860cb2dfcdd33b14b23deacac9c371e3fbafaf31556cadddad05b3477f65951a",
      "d797dd9eb2039987856c9773b03226044bef1bffff25838278741b44e9ba214c",
      "a69ad1989ae82e6e5e2181ea79c2464fec4052ef36afd8023fc5589655341efd",
      "cea120d2cbad7baab20a7c3a10fe0a2bfe9c2edac8608e005fb566f556088767",
      "6647797d0e39c564cea1594bd1597ee44345a4ab8b5699cc9e3cf6e2dd00f7c0",
      "9d64891ff226ce0df604f133d45534068ad5578b8725da4a5cae7aabe2f27011",
      "68d19d331177d353f1616a6e4aba74738f2bdb0ad0083b1d2a15a1ea00392abc",
      "10cd97aeacafef8153bb8954b5fd2368e1d0f7d71be1ccc277a0a196c4500be1",
      "bdbbb8b15d882cd5885d3a4c2c43ac3bc6b61414c6f03b9cd6c382e5a46e2801",
      "e8b0a895163c92e2d36997ec6110517b81da5aaea286fa8ebfd18008bfa63856",
  }};
  return {kQsaSidecarArtifactKey,
          parse_digest("7752a9b9e0c4bed9f4e20ce4c0500866e98cfd760ff95a6e770aa0563ab03df3"),
          parse_digest(kLayerDigests[static_cast<std::size_t>(layer / 4)]),
          rank, layer};
}

std::vector<std::uint8_t> authenticate_qsa_sidecar_host(
    const std::filesystem::path& path, const QsaSidecarIdentity& identity) {
  const auto expected = target_qsa_sidecar_identity(identity.rank,
                                                     identity.layer);
  if (identity.artifact_key != expected.artifact_key ||
      identity.payload_sha256 != expected.payload_sha256 ||
      identity.layer3_sha256 != expected.layer3_sha256 ||
      identity.layer != expected.layer)
    throw std::invalid_argument("QSA sidecar identity changed");
  if ((OpenSSL_version_num() >> 28) != 3)
    throw std::runtime_error("QSA sidecar requires OpenSSL 3 ABI");
  auto host = read_exact(path);
  if (sha256(host.data(), host.size()) != identity.payload_sha256 ||
      sha256(host.data() + target_qsa_sidecar_offset(identity.layer),
             kQsaIndexerLayer3Bytes) != identity.layer3_sha256)
    throw std::invalid_argument("QSA sidecar payload hash changed");
  return host;
}

QsaSidecarDeviceOwner::QsaSidecarDeviceOwner(
    int device, const std::filesystem::path& path, QsaSidecarIdentity identity)
    : device_(device), identity_(identity) {
  if (device < 0 || !nonzero(identity.payload_sha256) ||
      !nonzero(identity.layer3_sha256))
    throw std::invalid_argument("QSA sidecar identity changed");
  const auto host = authenticate_qsa_sidecar_host(path, identity);
  identity_ = target_qsa_sidecar_identity(identity.rank, identity.layer);
  if (cudaSetDevice(device) != cudaSuccess)
    throw std::runtime_error("QSA sidecar device selection failed");

  cudaStream_t stream = nullptr;
  cudaEvent_t published = nullptr;
  try {
    if (cudaMalloc(reinterpret_cast<void**>(&payload_), host.size()) !=
            cudaSuccess ||
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
            cudaSuccess ||
        cudaEventCreateWithFlags(&published, cudaEventDisableTiming) !=
            cudaSuccess ||
        cudaMemcpyAsync(payload_, host.data(), host.size(),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaEventRecord(published, stream) != cudaSuccess ||
        cudaEventSynchronize(published) != cudaSuccess)
      throw std::runtime_error("QSA sidecar H2D publication failed");
    cudaEventDestroy(published);
    cudaStreamDestroy(stream);
    publication_ = {payload_, host.size(), device_, identity_};
  } catch (...) {
    if (published) cudaEventDestroy(published);
    if (stream) cudaStreamDestroy(stream);
    if (payload_) cudaFree(payload_);
    payload_ = nullptr;
    throw;
  }
}

QsaSidecarDeviceOwner::~QsaSidecarDeviceOwner() {
  if (device_ >= 0) cudaSetDevice(device_);
  if (payload_) cudaFree(payload_);
  publication_ = {};
}

}  // namespace rocket::qwen38::attention

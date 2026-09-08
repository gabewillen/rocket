// SPDX-License-Identifier: Apache-2.0
#include "output/native_token_io.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <stdexcept>
#include <string>
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

namespace rocket::qwen38::output {
namespace {

struct FileIdentity {
  std::string_view name;
  std::uint64_t bytes;
  std::string_view sha256;
};

constexpr std::array kTokenizerFiles{
    FileIdentity{"merges.txt", 3'353'259,
                 "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"},
    FileIdentity{"tokenizer.json", 12'809'320,
                 "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"},
    FileIdentity{"tokenizer_config.json", 17'928,
                 "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"},
    FileIdentity{"vocab.json", 6'722'759,
                 "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"},
};
constexpr std::array kOracleFiles{
    FileIdentity{"manifest.json", 19'533, kTokenIoOracleManifestSha256},
    FileIdentity{"embedding.bin", 179'200,
                 "c358d186071a8c428c4d281f3f5dbabcf3b0c0f9fe915fd39b59636da52f9be1"},
    FileIdentity{"layer-47.bin", 716'800,
                 "40fb8be363118c63262e38d9d94f91e6cd24102d72149acca054446719f61f3b"},
    FileIdentity{"final_norm.bin", 179'200,
                 "0a562c4b0b435443894c95ce8993280f921c9da19eade55ad7da23f549fb01df"},
    FileIdentity{"logits.bin", 496'640,
                 "e31515cfcd98b1fed61f90ae287c9cf97c7a2a52e3fcf8feef956910ff4dc9ab"},
};

struct Fd {
  int value = -1;
  ~Fd() { if (value >= 0) close(value); }
};

std::string hex(const std::array<unsigned char, 32>& digest) {
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result(64, '0');
  for (std::size_t index = 0; index < digest.size(); ++index) {
    result[2 * index] = alphabet[digest[index] >> 4];
    result[2 * index + 1] = alphabet[digest[index] & 15];
  }
  return result;
}

std::string hash_opened(int fd, std::uint64_t bytes) {
  evp_md_ctx_st* context = EVP_MD_CTX_new();
  if (!context) throw std::invalid_argument("token I/O SHA context unavailable");
  std::array<unsigned char, 32> digest{};
  std::array<std::uint8_t, 65'536> buffer{};
  unsigned int digest_bytes = 0;
  std::uint64_t offset = 0;
  bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1;
  while (ok && offset < bytes) {
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(buffer.size(), bytes - offset));
    const ssize_t read_bytes = pread(fd, buffer.data(), count,
                                     static_cast<off_t>(offset));
    ok = read_bytes == static_cast<ssize_t>(count) &&
         EVP_DigestUpdate(context, buffer.data(), count) == 1;
    offset += count;
  }
  ok = ok && EVP_DigestFinal_ex(context, digest.data(), &digest_bytes) == 1 &&
       digest_bytes == digest.size();
  EVP_MD_CTX_free(context);
  if (!ok) throw std::invalid_argument("token I/O file hash/read failed");
  return hex(digest);
}

void authenticate_files(const std::filesystem::path& root,
                        const auto& expected, std::string_view family) {
  Fd directory{open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                      O_NOFOLLOW)};
  if (directory.value < 0)
    throw std::invalid_argument(std::string(family) + " root changed");
  for (const auto& item : expected) {
    const std::string name(item.name);
    Fd file{openat(directory.value, name.c_str(), O_RDONLY | O_CLOEXEC)};
    struct stat before {};
    struct stat after {};
    if (file.value < 0 || fstat(file.value, &before) != 0 ||
        !S_ISREG(before.st_mode) || before.st_size < 0 ||
        static_cast<std::uint64_t>(before.st_size) != item.bytes ||
        hash_opened(file.value, item.bytes) != item.sha256 ||
        fstat(file.value, &after) != 0 || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size ||
        before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
        before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
        before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
        before.st_ctim.tv_nsec != after.st_ctim.tv_nsec)
      throw std::invalid_argument(std::string(family) +
                                  " file identity changed");
  }
}

std::string tokenizer_identity() {
  evp_md_ctx_st* context = EVP_MD_CTX_new();
  if (!context) throw std::invalid_argument("tokenizer SHA context unavailable");
  std::array<unsigned char, 32> digest{};
  unsigned int digest_bytes = 0;
  bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1;
  for (const auto& file : kTokenizerFiles) {
    ok = ok && EVP_DigestUpdate(context, file.name.data(), file.name.size()) == 1 &&
         EVP_DigestUpdate(context, file.sha256.data(), file.sha256.size()) == 1;
  }
  ok = ok && EVP_DigestFinal_ex(context, digest.data(), &digest_bytes) == 1 &&
       digest_bytes == digest.size();
  EVP_MD_CTX_free(context);
  if (!ok) throw std::invalid_argument("tokenizer identity hash failed");
  return hex(digest);
}

}  // namespace

TokenIoArtifactRoots authenticate_token_io_artifact_roots(
    const std::filesystem::path& tokenizer,
    const std::filesystem::path& oracle_capture) {
  if (tokenizer.empty() || oracle_capture.empty())
    throw std::invalid_argument("token I/O artifact roots are required");
  authenticate_files(tokenizer, kTokenizerFiles, "tokenizer");
  authenticate_files(oracle_capture, kOracleFiles, "oracle");
  return {tokenizer, oracle_capture, tokenizer_identity(),
          std::string(kTokenIoOracleManifestSha256)};
}

}  // namespace rocket::qwen38::output

// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer_native_plan.h"

#include <algorithm>
#include <array>
#include <bit>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
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
}

namespace rocket::qwen38::decode {
namespace {

constexpr std::string_view kArtifact =
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4";
constexpr std::string_view kSidecar =
    "bdbebd4f45c398f090a41ab98cd3881b969d958d8ae0bc42f3411844d3262edd";
constexpr std::uint64_t kSlabBytes = 63'212'748'800ULL;

struct Identity {
  int rank;
  int layer;
  std::string_view kind;
  std::string_view descriptor;
  std::string_view inventory;
  std::string_view moe_layout;
  std::string_view publication;
};

constexpr std::array<Identity, 96> kIdentities{{
#include "decode/target_layer_descriptor_identities.inc"
}};

[[noreturn]] void fail(std::string_view reason) {
  throw std::invalid_argument("target layer native plan: " +
                              std::string(reason));
}

std::string sha256(std::string_view input) {
  std::array<unsigned char, 32> digest{};
  unsigned int size = 0;
  evp_md_ctx_st* context = EVP_MD_CTX_new();
  const bool ok = context &&
                  EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                  EVP_DigestUpdate(context, input.data(), input.size()) == 1 &&
                  EVP_DigestFinal_ex(context, digest.data(), &size) == 1 &&
                  size == digest.size();
  if (context) EVP_MD_CTX_free(context);
  if (!ok) fail("SHA256 failed");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto value : digest) output << std::setw(2) << int(value);
  return output.str();
}

bool safe(std::string_view value) {
  return !value.empty() &&
         std::all_of(value.begin(), value.end(), [](const char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '.' || c == '_' ||
                  c == '-';
         });
}

void append_array(std::string& output,
                  const std::vector<std::uint64_t>& values) {
  output.push_back('[');
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) output.push_back(',');
    output += std::to_string(values[i]);
  }
  output.push_back(']');
}

std::string extent_inventory(const TargetLayerNativePlan& plan) {
  std::string output = "[";
  for (std::size_t index = 0; index < plan.extents.size(); ++index) {
    const auto& item = plan.extents[index];
    if (!safe(item.name) || !safe(item.storage) || !safe(item.dtype) ||
        !safe(item.layout) || !safe(item.abi))
      fail("extent text changed");
    if (index) output.push_back(',');
    output += "{\"abi\":\"" + item.abi + "\",\"dtype\":\"" +
              item.dtype + "\",\"layout\":\"" + item.layout +
              "\",\"length_bytes\":" + std::to_string(item.length_bytes) +
              ",\"name\":\"" + item.name + "\",\"offset_bytes\":" +
              std::to_string(item.offset_bytes) + ",\"shape\":";
    append_array(output, item.shape);
    output += ",\"storage\":\"" + item.storage + "\",\"strides\":";
    append_array(output, item.strides);
    output.push_back('}');
  }
  output.push_back(']');
  return output;
}

}  // namespace

void validate_target_layer_native_plan_binding(
    const TargetLayerNativePlan& plan) {
  const bool qsa = plan.layer >= 0 && plan.layer < 48 && plan.layer % 4 == 3;
  const std::string_view kind = qsa ? "qsa" : "gdn";
  const auto expected = std::find_if(
      kIdentities.begin(), kIdentities.end(), [&](const Identity& item) {
        return item.rank == plan.rank && item.layer == plan.layer;
      });
  if (expected == kIdentities.end() || plan.peer_rank != 1 - plan.rank ||
      plan.attention_kind != (qsa ? TargetK0AttentionKind::kQsa
                                 : TargetK0AttentionKind::kGdn) ||
      expected->kind != kind || plan.slab_bytes != kSlabBytes ||
      plan.artifact_key != kArtifact ||
      plan.slab_key != "rank" + std::to_string(plan.rank) + "-target" ||
      plan.indexer_sidecar_key != (qsa ? kSidecar : std::string_view{}) ||
      plan.descriptor_sha256 != expected->descriptor ||
      plan.native_binding_inventory_sha256 != expected->inventory ||
      plan.layout_sha256 != expected->moe_layout ||
      plan.slab_publication_layout_sha256 != expected->publication)
    fail("rank, layer, kind, or publication identity changed");
  const std::size_t expected_extents = qsa ? 3'108 : 3'111;
  if (plan.extents.size() != expected_extents)
    fail("extent inventory cardinality changed");
  std::set<std::string_view> names;
  std::array<std::vector<std::pair<std::uint64_t, std::uint64_t>>, 2> ranges;
  for (const auto& item : plan.extents) {
    const int storage = item.storage == "target_slab" ? 0 :
                        item.storage == "indexer_sidecar" ? 1 : -1;
    const std::uint64_t bound = storage == 0 ? kSlabBytes : 39'321'600ULL;
    if (storage < 0 || !names.insert(item.name).second ||
        item.length_bytes == 0 || item.offset_bytes % 256 != 0 ||
        item.offset_bytes > bound || item.length_bytes > bound - item.offset_bytes)
      fail("extent bounds, alignment, or identity changed");
    ranges[storage].push_back(
        {item.offset_bytes, item.offset_bytes + item.length_bytes});
  }
  for (auto& storage : ranges) {
    std::sort(storage.begin(), storage.end());
    for (std::size_t index = 1; index < storage.size(); ++index)
      if (storage[index].first < storage[index - 1].second)
        fail("extent overlap changed");
  }
  if (sha256(extent_inventory(plan)) != expected->inventory)
    fail("extent inventory digest changed");
}

}  // namespace rocket::qwen38::decode

// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer_native_plan.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
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

extern "C" {
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
int json_object_object_length(const json_object*);
std::size_t json_object_array_length(const json_object*);
json_object* json_object_array_get_idx(const json_object*, std::size_t);
const char* json_object_get_string(const json_object*);
std::uint64_t json_object_get_uint64(const json_object*);
}

namespace rocket::qwen38::decode {
namespace {

constexpr std::string_view kArtifact =
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4";
constexpr std::string_view kSidecar =
    "bdbebd4f45c398f090a41ab98cd3881b969d958d8ae0bc42f3411844d3262edd";
constexpr std::uint64_t kSlabBytes = 63'212'748'800ULL;
constexpr int kJsonObject = 4;
constexpr int kJsonArray = 5;
constexpr int kJsonString = 6;
constexpr int kJsonInt = 3;
constexpr std::string_view kSchema =
    "rocket.qwen38.target-layer-native-descriptor.v1";

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

struct JsonOwner {
  json_object* value;
  ~JsonOwner() { if (value) json_object_put(value); }
};
struct TokenerOwner {
  json_tokener* value;
  ~TokenerOwner() { if (value) json_tokener_free(value); }
};

json_object* field(json_object* object, const char* key, int type) {
  json_object* value = nullptr;
  if (!object || !json_object_object_get_ex(object, key, &value) || !value ||
      json_object_get_type(value) != type)
    fail(std::string("missing or invalid field: ") + key);
  return value;
}

std::string string_field(json_object* object, const char* key) {
  const char* value = json_object_get_string(field(object, key, kJsonString));
  if (!value) fail(std::string("invalid string field: ") + key);
  return value;
}

std::uint64_t uint_field(json_object* object, const char* key) {
  return json_object_get_uint64(field(object, key, kJsonInt));
}

std::vector<std::uint64_t> dimensions(json_object* object, const char* key) {
  auto* array = field(object, key, kJsonArray);
  std::vector<std::uint64_t> result;
  result.reserve(json_object_array_length(array));
  for (std::size_t i = 0; i < json_object_array_length(array); ++i) {
    auto* value = json_object_array_get_idx(array, i);
    if (!value || json_object_get_type(value) != kJsonInt)
      fail("invalid extent dimension");
    result.push_back(json_object_get_uint64(value));
  }
  return result;
}

std::uint8_t hex_byte(char high, char low) {
  const auto nibble = [](char value) -> int {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
  };
  const int a = nibble(high), b = nibble(low);
  if (a < 0 || b < 0) fail("scalar hex changed");
  return static_cast<std::uint8_t>((a << 4) | b);
}

float scalar_value(std::string_view value) {
  if (value.size() != 8) fail("scalar width changed");
  std::array<std::uint8_t, 4> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = hex_byte(value[2 * i], value[2 * i + 1]);
  float result;
  std::memcpy(&result, bytes.data(), sizeof(result));
  if (!std::isfinite(result) || result <= 0.0F)
    fail("scalar value changed");
  return result;
}

std::string scalar_hex(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (int shift = 0; shift < 32; shift += 8)
    output << std::setw(2) << ((bits >> shift) & 0xffU);
  return output.str();
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

std::string descriptor_payload(const TargetLayerNativePlan& plan) {
  const bool qsa = plan.attention_kind == TargetK0AttentionKind::kQsa;
  std::string output = "{\"artifact_key\":\"" + plan.artifact_key +
      "\",\"attention_kind\":\"" + (qsa ? "qsa" : "gdn") +
      "\",\"attention_projection_global_le_hex\":{";
  const std::array<std::string_view, 5> qsa_names{"k", "o", "q", "v", ""};
  const std::array<int, 5> qsa_indices{1, 3, 0, 2, 4};
  const std::array<std::string_view, 5> gdn_names{"a", "b", "out", "qkv", "z"};
  const std::array<int, 5> gdn_indices{3, 2, 4, 0, 1};
  const auto& names = qsa ? qsa_names : gdn_names;
  const auto& indices = qsa ? qsa_indices : gdn_indices;
  const int count = qsa ? 4 : 5;
  for (int i = 0; i < count; ++i) {
    if (i) output.push_back(',');
    output += "\"" + std::string(names[i]) + "\":\"" +
              scalar_hex(plan.attention_projection_globals[indices[i]]) + "\"";
  }
  output += "},\"extents\":" + extent_inventory(plan) +
      ",\"indexer_sidecar_key\":\"" + plan.indexer_sidecar_key +
      "\",\"layer\":" + std::to_string(plan.layer) +
      ",\"moe_layout_sha256\":\"" + plan.layout_sha256 +
      "\",\"native_binding_inventory_sha256\":\"" +
      plan.native_binding_inventory_sha256 + "\",\"peer_rank\":" +
      std::to_string(plan.peer_rank) + ",\"rank\":" +
      std::to_string(plan.rank) + ",\"schema\":\"" + std::string(kSchema) +
      "\",\"slab_bytes\":" + std::to_string(plan.slab_bytes) +
      ",\"slab_key\":\"" + plan.slab_key +
      "\",\"slab_publication_layout_sha256\":\"" +
      plan.slab_publication_layout_sha256 + "\"}";
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
  if (sha256(descriptor_payload(plan)) != expected->descriptor)
    fail("descriptor digest changed");
}

TargetLayerNativePlan load_target_layer_native_plan(
    const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) fail("descriptor is unavailable");
  std::string raw((std::istreambuf_iterator<char>(stream)), {});
  if (raw.empty() || raw.back() != '\n' || raw.find('\0') != std::string::npos ||
      raw.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    fail("descriptor encoding changed");
  raw.pop_back();
  TokenerOwner tokener{json_tokener_new()};
  if (!tokener.value) fail("JSON parser allocation failed");
  JsonOwner root{json_tokener_parse_ex(
      tokener.value, raw.data(), static_cast<int>(raw.size()))};
  if (json_tokener_get_error(tokener.value) != 0 ||
      json_tokener_get_parse_end(tokener.value) != raw.size() ||
      !root.value || json_object_get_type(root.value) != kJsonObject ||
      json_object_object_length(root.value) != 15)
    fail("descriptor JSON shape changed");
  TargetLayerNativePlan result;
  result.rank = static_cast<int>(uint_field(root.value, "rank"));
  result.peer_rank = static_cast<int>(uint_field(root.value, "peer_rank"));
  result.layer = static_cast<int>(uint_field(root.value, "layer"));
  const auto kind = string_field(root.value, "attention_kind");
  if (kind != "qsa" && kind != "gdn") fail("attention kind changed");
  result.attention_kind = kind == "qsa" ? TargetK0AttentionKind::kQsa
                                         : TargetK0AttentionKind::kGdn;
  if (string_field(root.value, "schema") != kSchema)
    fail("descriptor schema changed");
  result.slab_bytes = uint_field(root.value, "slab_bytes");
  result.descriptor_sha256 = string_field(root.value, "descriptor_sha256");
  result.native_binding_inventory_sha256 =
      string_field(root.value, "native_binding_inventory_sha256");
  result.artifact_key = string_field(root.value, "artifact_key");
  result.slab_key = string_field(root.value, "slab_key");
  result.layout_sha256 = string_field(root.value, "moe_layout_sha256");
  result.slab_publication_layout_sha256 =
      string_field(root.value, "slab_publication_layout_sha256");
  result.indexer_sidecar_key = string_field(root.value, "indexer_sidecar_key");
  auto* globals = field(root.value, "attention_projection_global_le_hex",
                        kJsonObject);
  const std::array<std::string_view, 5> qsa_names{"q", "k", "v", "o", ""};
  const std::array<std::string_view, 5> gdn_names{"qkv", "z", "b", "a", "out"};
  const auto& names = kind == "qsa" ? qsa_names : gdn_names;
  const int scalar_count = kind == "qsa" ? 4 : 5;
  if (json_object_object_length(globals) != scalar_count)
    fail("projection scalar inventory changed");
  for (int i = 0; i < scalar_count; ++i)
    result.attention_projection_globals[i] = scalar_value(
        string_field(globals, std::string(names[i]).c_str()));
  auto* extents = field(root.value, "extents", kJsonArray);
  result.extents.reserve(json_object_array_length(extents));
  for (std::size_t i = 0; i < json_object_array_length(extents); ++i) {
    auto* item = json_object_array_get_idx(extents, i);
    if (!item || json_object_get_type(item) != kJsonObject ||
        json_object_object_length(item) != 9)
      fail("extent field inventory changed");
    result.extents.push_back({
        string_field(item, "name"), uint_field(item, "offset_bytes"),
        uint_field(item, "length_bytes"), string_field(item, "storage"),
        string_field(item, "dtype"), string_field(item, "layout"),
        string_field(item, "abi"), dimensions(item, "shape"),
        dimensions(item, "strides")});
  }
  validate_target_layer_native_plan_binding(result);
  return result;
}

}  // namespace rocket::qwen38::decode

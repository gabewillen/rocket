// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_native_plan.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
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
void json_tokener_free(json_tokener* tokener);
json_object* json_tokener_parse_ex(json_tokener* tokener, const char* text,
                                  int length);
int json_tokener_get_error(json_tokener* tokener);
std::size_t json_tokener_get_parse_end(json_tokener* tokener);
int json_object_put(json_object* object);
int json_object_get_type(const json_object* object);
int json_object_object_get_ex(const json_object* object, const char* key,
                              json_object** value);
int json_object_object_length(const json_object* object);
std::size_t json_object_array_length(const json_object* object);
json_object* json_object_array_get_idx(const json_object* object,
                                       std::size_t index);
const char* json_object_get_string(const json_object* object);
int json_object_get_string_len(const json_object* object);
std::int64_t json_object_get_int64(const json_object* object);
}

namespace rocket::qwen38::decode {
namespace {
constexpr int kJsonInt = 3;
constexpr int kJsonObject = 4;
constexpr int kJsonArray = 5;
constexpr int kJsonString = 6;
constexpr std::string_view kSchema = "rocket.qwen38.layer3-native-plan.v1";
constexpr std::string_view kArtifact =
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4";
constexpr std::string_view kManifest =
    "a44a450d9c0b6fe3df904ad1a78ecee959f28d9f055301195181e986bdc7028b";
constexpr std::string_view kOracle =
    "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b";
constexpr std::uint64_t kSidecarBytes = 39'321'600;
constexpr std::uint64_t kTargetSlabBytes = 63'212'748'800;
constexpr std::string_view kSidecar =
    "bdbebd4f45c398f090a41ab98cd3881b969d958d8ae0bc42f3411844d3262edd";
constexpr std::string_view kLayer02 =
    "6503eeeb3c70c9c1c163aca08dcdb4e1df7ca997699f2f20946265a72069d2fe";
constexpr std::string_view kLayer03 =
    "aa2d2a1454f0654ea2082d12e7e3284cad9304a7ede124303f99c52bbf9cddbe";
constexpr std::array<std::string_view, 2> kMoeLayouts{
    "ebf6db24c257c3516f7ff8c94bb2ba70d692c62c4cbf1b2577ebe99a3f56875b",
    "6e20c303b336f980e94e7aa3d897009ac527e1810279c09c8cbab1bd7867f841"};
constexpr std::array<std::string_view, 2> kSlabPublicationLayouts{
    "4f03ccc90c9020ff2e87f044867f2ac9896ac20c0d97c85055beef0b125ce6d6",
    "afd718cf00b6fffa61336ac53e744209750b1c9bd0399622f73cf99cb81de2e9"};
constexpr std::array<std::string_view, 2> kExtentInventories{
    "0d5ad9150c7632ecab86abd320ddea486971db79453c919a4ec8f42ccb6e49b6",
    "9e2a66e4f78d2ac6d13cc2afea164f8c3f4b7329a98f84279015f739c3027cdf"};
constexpr std::string_view kBufferInventory =
    "7e48bcfb21f1f5f1136754a635215318696878f599c75a3a01e6a4dc45f7cc45";
constexpr std::array<std::string_view, 2> kProjectionGlobalInventories{
    "e9196b3e9efed6cc5cdcfc27a2d4aee0c0a81ac088ef2e73ccb8dad04bee5b7d",
    "44ef127172e3c065479e9dbbcf47686a21b9151cdb316cbd5486fca3a6115668"};
constexpr std::array<std::string_view, 4> kProjectionFamilies{"q", "k", "v", "o"};
constexpr std::array<std::string_view, 2> kNativeBindingInventories{
    "0f88201c81e6ace3991969dc397fca345773b7a28f317e08a6727c7903ee30c5",
    "5d6a7b322e1a78059f7abea9d9b6a20a1e7c76263713b5e6dff940d9247e3d93"};
constexpr std::array<std::uint32_t, 4> kProjectionGlobalBits{
    0x39ac30c3U, 0x395e79e7U, 0x399cf3cfU, 0x39d55555U};

[[noreturn]] void fail(std::string_view reason) {
  throw std::invalid_argument("layer-3 native plan: " + std::string(reason));
}

struct JsonOwner {
  json_object* value;
  ~JsonOwner() { if (value) json_object_put(value); }
};

struct JsonTokenerOwner {
  json_tokener* value;
  ~JsonTokenerOwner() { if (value) json_tokener_free(value); }
};

json_object* field(json_object* object, const char* name, int type) {
  json_object* value = nullptr;
  if (!object || json_object_get_type(object) != kJsonObject ||
      !json_object_object_get_ex(object, name, &value) || !value ||
      json_object_get_type(value) != type)
    fail(std::string("missing or typed field: ") + name);
  return value;
}

std::string decoded_string(json_object* item) {
  const char* value = json_object_get_string(item);
  const int length = json_object_get_string_len(item);
  if (!value || length < 0) fail("null or invalid string");
  std::string result(value, static_cast<std::size_t>(length));
  if (result.find('\0') != std::string::npos) fail("decoded string contains NUL");
  return result;
}

std::string string_field(json_object* object, const char* name) {
  return decoded_string(field(object, name, kJsonString));
}

std::uint64_t uint_field(json_object* object, const char* name) {
  const auto value = json_object_get_int64(field(object, name, kJsonInt));
  if (value < 0) fail(std::string("negative field: ") + name);
  return static_cast<std::uint64_t>(value);
}

bool hex64(std::string_view value) {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

int hex_nibble(char value) noexcept {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  return -1;
}

float little_endian_float(std::string_view value) {
  if (value.size() != 8) fail("projection scalar encoding changed");
  std::array<std::uint8_t, 4> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    const int high = hex_nibble(value[2 * index]);
    const int low = hex_nibble(value[2 * index + 1]);
    if (high < 0 || low < 0) fail("projection scalar encoding changed");
    bytes[index] = static_cast<std::uint8_t>((high << 4) | low);
  }
  float result = 0.0F;
  static_assert(sizeof(result) == bytes.size());
  std::memcpy(&result, bytes.data(), bytes.size());
  if (!std::isfinite(result) || result <= 0.0F)
    fail("projection scalar value changed");
  return result;
}

std::string sha256(std::string_view value) {
  std::array<unsigned char, 32> digest{};
  unsigned int length = 0;
  std::unique_ptr<evp_md_ctx_st, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), value.data(), value.size()) != 1 ||
      EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1 ||
      length != 32) {
    fail("SHA256 failed");
  }
  constexpr char digits[] = "0123456789abcdef";
  std::string result(64, '0');
  for (std::size_t i = 0; i < digest.size(); ++i) {
    result[2 * i] = digits[digest[i] >> 4];
    result[2 * i + 1] = digits[digest[i] & 15];
  }
  return result;
}

std::string_view canonical_value(std::string_view raw, std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\":";
  const auto key_at = raw.find(needle);
  if (key_at == std::string_view::npos ||
      raw.find(needle, key_at + needle.size()) != std::string_view::npos)
    fail("canonical field occurrence changed");
  const auto begin = key_at + needle.size();
  if (begin >= raw.size() || (raw[begin] != '[' && raw[begin] != '{'))
    fail("canonical compound field changed");
  int depth = 0;
  bool quoted = false;
  bool escaped = false;
  for (std::size_t index = begin; index < raw.size(); ++index) {
    const char value = raw[index];
    if (quoted) {
      if (escaped) escaped = false;
      else if (value == '\\') escaped = true;
      else if (value == '"') quoted = false;
      continue;
    }
    if (value == '"') quoted = true;
    else if (value == '[' || value == '{') ++depth;
    else if (value == ']' || value == '}') {
      if (--depth == 0) return raw.substr(begin, index - begin + 1);
      if (depth < 0) break;
    }
  }
  fail("canonical compound field is unterminated");
}

std::vector<std::uint64_t> dimensions(json_object* object, const char* name) {
  auto* array = field(object, name, kJsonArray);
  const auto count = json_object_array_length(array);
  std::vector<std::uint64_t> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    auto* value = json_object_array_get_idx(array, index);
    if (!value || json_object_get_type(value) != kJsonInt)
      fail("dimension type changed");
    const auto dimension = json_object_get_int64(value);
    if (dimension <= 0) fail("dimension changed");
    result.push_back(static_cast<std::uint64_t>(dimension));
  }
  return result;
}

std::string array_string(json_object* array, std::size_t index) {
  auto* value = json_object_array_get_idx(array, index);
  if (!value || json_object_get_type(value) != kJsonString)
    fail("string array element changed");
  return decoded_string(value);
}

void validate_shape(json_object* object, std::uint64_t bytes) {
  const auto shape = dimensions(object, "shape");
  const auto strides = dimensions(object, "strides");
  if (shape.size() != strides.size()) fail("shape/stride rank changed");
  std::uint64_t expected_stride = 1;
  for (std::size_t index = shape.size(); index-- > 0;) {
    if (strides[index] != expected_stride ||
        shape[index] > std::numeric_limits<std::uint64_t>::max() / expected_stride)
      fail("non-contiguous or overflowing shape");
    expected_stride *= shape[index];
  }
  const auto dtype = string_field(object, "dtype");
  const std::uint64_t width =
      dtype == "BF16" || dtype == "bfloat16" ? 2 :
      dtype == "F32" || dtype == "float32" || dtype == "int32" ? 4 :
      dtype == "int64" ? 8 :
      dtype == "U8" || dtype == "F8_E4M3" || dtype == "uint8" ? 1 : 0;
  if (!width || expected_stride > std::numeric_limits<std::uint64_t>::max() / width ||
      expected_stride * width != bytes)
    fail("shape/dtype byte extent changed");
}

void validate_buffers(json_object* root) {
  auto* buffers = field(root, "buffers", kJsonObject);
  if (json_object_object_length(buffers) != 4) fail("buffer family inventory changed");
  const std::array<std::pair<const char*, std::size_t>, 4> families{{
      {"qsa_arena", 18}, {"qsa_state", 16},
      {"target_moe_workspace", 24}, {"row", 9}}};
  for (const auto& [family, expected_count] : families) {
    auto* array = field(buffers, family, kJsonArray);
    if (json_object_array_length(array) != expected_count)
      fail("buffer count changed");
    std::set<std::string> names;
    for (std::size_t i = 0; i < json_object_array_length(array); ++i) {
      auto* item = json_object_array_get_idx(array, i);
      if (!item || json_object_get_type(item) != kJsonObject ||
          json_object_object_length(item) != 5)
        fail("buffer field inventory changed");
      if (!names.insert(string_field(item, "name")).second)
        fail("duplicate buffer name");
      (void)string_field(item, "dtype");
      validate_shape(item, uint_field(item, "bytes"));
    }
  }
}

void validate_fixed_contract(json_object* root) {
  if (string_field(root, "indexer_sidecar_key") != kSidecar ||
      string_field(root, "oracle_layer02_sha256") != kLayer02 ||
      string_field(root, "oracle_layer03_sha256") != kLayer03 ||
      uint_field(root, "compare_row") != 34)
    fail("sidecar or oracle boundary identity changed");
  auto* replay = field(root, "replay_rows", kJsonArray);
  if (json_object_array_length(replay) != 35) fail("replay row count changed");
  for (std::size_t index = 0; index < 35; ++index) {
    auto* value = json_object_array_get_idx(replay, index);
    if (!value || json_object_get_type(value) != kJsonInt ||
        json_object_get_int64(value) != static_cast<std::int64_t>(index))
      fail("replay row order changed");
  }
  auto* pair = field(root, "pair_reduce", kJsonObject);
  if (json_object_object_length(pair) != 7 ||
      string_field(pair, "bootstrap_host") != "192.168.100.10" ||
      uint_field(pair, "bootstrap_port") != 18839 ||
      uint_field(pair, "timeout_ms") != 120000 ||
      string_field(pair, "session_sha256") != kOracle ||
      uint_field(pair, "gid_index") != 3 || uint_field(pair, "calls") != 70)
    fail("PairReduce plan changed");
  auto* rails = field(pair, "rails", kJsonArray);
  if (json_object_array_length(rails) != 2 ||
      array_string(rails, 0) != "rocep1s0f1" ||
      array_string(rails, 1) != "roceP2p1s0f1")
    fail("PairReduce rail identity changed");
  auto* abis = field(root, "native_abis", kJsonObject);
  if (json_object_object_length(abis) != 6) fail("native ABI inventory changed");
  for (const auto& [name, version] :
       std::array<std::pair<const char*, std::uint64_t>, 6>{{
           {"target_slab_publication", 1}, {"qsa_c1", 1},
           {"hyperconnection", 1}, {"target_full_moe_c1", 1},
           {"pair_reduce_bootstrap", 3}, {"oracle_comparator", 1}}})
    if (uint_field(abis, name) != version) fail("native ABI version changed");
}

struct ProjectionGlobalEvidence {
  std::string extent_name;
  std::string source_chunk_sha256;
  float value;
  bool extent_matched = false;
};

std::array<ProjectionGlobalEvidence, 4> projection_globals(
    json_object* root, std::string_view raw, std::uint64_t rank) {
  if (rank > 1) fail("projection scalar rank changed");
  const auto claimed = string_field(root, "qsa_projection_globals_sha256");
  if (claimed != kProjectionGlobalInventories[rank] ||
      sha256(canonical_value(raw, "qsa_projection_globals")) != claimed)
    fail("projection scalar inventory changed");
  auto* globals = field(root, "qsa_projection_globals", kJsonObject);
  if (json_object_object_length(globals) != 4)
    fail("projection scalar family inventory changed");
  std::array<ProjectionGlobalEvidence, 4> result;
  for (std::size_t index = 0; index < result.size(); ++index) {
    const std::string family(kProjectionFamilies[index]);
    auto* item = field(globals, family.c_str(), kJsonObject);
    if (json_object_object_length(item) != 4 ||
        string_field(item, "dtype") != "F32")
      fail("projection scalar evidence changed");
    const std::string expected = "model.language_model.layers.3.self_attn." +
                                 family + "_proj.weight_scale_2";
    result[index] = {
        string_field(item, "extent_name"),
        string_field(item, "source_chunk_sha256"),
        little_endian_float(string_field(item, "value_le_hex")), false};
    if (result[index].extent_name != expected ||
        !hex64(result[index].source_chunk_sha256))
      fail("projection scalar source identity changed");
  }
  return result;
}
}  // namespace

void validate_target_layer3_native_plan_binding(
    const TargetLayer3NativePlan& plan) {
  if ((plan.rank != 0 && plan.rank != 1) || plan.peer_rank != 1 - plan.rank ||
      plan.layer != 3 || plan.slab_bytes != kTargetSlabBytes ||
      plan.artifact_key != kArtifact ||
      plan.slab_key != "rank" + std::to_string(plan.rank) + "-target" ||
      plan.layout_sha256 != kMoeLayouts[plan.rank] ||
      plan.slab_publication_layout_sha256 !=
          kSlabPublicationLayouts[plan.rank] ||
      plan.extents.size() != 3'108)
    fail("mutable binding identity changed");
  for (std::size_t index = 0; index < plan.qsa_projection_globals.size(); ++index)
    if (std::bit_cast<std::uint32_t>(plan.qsa_projection_globals[index]) !=
        kProjectionGlobalBits[index])
      fail("mutable projection scalar changed");
  std::string canonical = "[";
  for (std::size_t index = 0; index < plan.extents.size(); ++index) {
    const auto& item = plan.extents[index];
    const auto safe = [](std::string_view text) {
      return std::all_of(text.begin(), text.end(), [](char value) {
        return (value >= 'a' && value <= 'z') ||
               (value >= 'A' && value <= 'Z') ||
               (value >= '0' && value <= '9') || value == '.' ||
               value == '_' || value == '-';
      });
    };
    if (!safe(item.name) || !safe(item.storage))
      fail("mutable extent text changed");
    if (index) canonical.push_back(',');
    canonical += "{\"length_bytes\":" + std::to_string(item.length_bytes) +
                 ",\"name\":\"" + item.name + "\",\"offset_bytes\":" +
                 std::to_string(item.offset_bytes) + ",\"storage\":\"" +
                 item.storage + "\"}";
  }
  canonical.push_back(']');
  const auto expected = kNativeBindingInventories[plan.rank];
  if (plan.native_binding_inventory_sha256 != expected ||
      sha256(canonical) != expected)
    fail("mutable native binding inventory changed");
}

TargetLayer3NativePlan load_target_layer3_native_plan(
    const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) fail("descriptor is unavailable");
  std::string raw((std::istreambuf_iterator<char>(stream)), {});
  if (raw.empty() || raw.back() != '\n' || raw.find('\n') != raw.size() - 1)
    fail("descriptor is not canonical one-line JSON");
  raw.pop_back();
  if (raw.find('\0') != std::string::npos ||
      raw.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    fail("descriptor contains NUL or exceeds parser bounds");
  JsonTokenerOwner tokener{json_tokener_new()};
  if (!tokener.value) fail("JSON parser allocation failed");
  JsonOwner root{json_tokener_parse_ex(
      tokener.value, raw.data(), static_cast<int>(raw.size()))};
  if (json_tokener_get_error(tokener.value) != 0 ||
      json_tokener_get_parse_end(tokener.value) != raw.size())
    fail("descriptor JSON is invalid or has trailing bytes");
  if (!root.value || json_object_get_type(root.value) != kJsonObject ||
      json_object_object_length(root.value) != 26)
    fail("top-level inventory changed");
  const auto claimed = string_field(root.value, "descriptor_sha256");
  const std::string needle = "\"descriptor_sha256\":\"" + claimed + "\",";
  const auto begin = raw.find(needle);
  if (!hex64(claimed) || begin == std::string::npos ||
      raw.find(needle, begin + 1) != std::string::npos)
    fail("descriptor digest field changed");
  std::string authenticated = raw;
  authenticated.erase(begin, needle.size());
  if (sha256(authenticated) != claimed) fail("descriptor digest mismatch");
  if (string_field(root.value, "schema") != kSchema ||
      string_field(root.value, "artifact_key") != kArtifact ||
      string_field(root.value, "manifest_sha256") != kManifest ||
      string_field(root.value, "oracle_manifest_sha256") != kOracle ||
      uint_field(root.value, "layer") != 3)
    fail("descriptor identity changed");
  validate_fixed_contract(root.value);
  TargetLayer3NativePlan result;
  const auto rank = uint_field(root.value, "rank");
  const auto peer_rank = uint_field(root.value, "peer_rank");
  if (rank > 1 || peer_rank > 1 || peer_rank != 1 - rank)
    fail("rank identity changed");
  result.rank = static_cast<int>(rank);
  result.peer_rank = static_cast<int>(peer_rank);
  auto scalar_evidence = projection_globals(root.value, raw, rank);
  for (std::size_t index = 0; index < scalar_evidence.size(); ++index)
    result.qsa_projection_globals[index] = scalar_evidence[index].value;
  if ((result.rank != 0 && result.rank != 1) ||
      result.peer_rank != 1 - result.rank)
    fail("rank identity changed");
  const auto extent_inventory = sha256(canonical_value(raw, "extents"));
  const auto buffer_inventory = sha256(canonical_value(raw, "buffers"));
  result.layer = 3;
  result.slab_bytes = uint_field(root.value, "slab_bytes");
  result.descriptor_sha256 = claimed;
  result.native_binding_inventory_sha256 =
      string_field(root.value, "native_binding_inventory_sha256");
  result.artifact_key = string_field(root.value, "artifact_key");
  result.slab_key = string_field(root.value, "slab_key");
  result.layout_sha256 = string_field(root.value, "layout_sha256");
  result.slab_publication_layout_sha256 =
      string_field(root.value, "slab_publication_layout_sha256");
  if (result.slab_key != "rank" + std::to_string(result.rank) + "-target" ||
      result.slab_bytes != kTargetSlabBytes ||
      result.layout_sha256 != kMoeLayouts[static_cast<std::size_t>(result.rank)] ||
      result.slab_publication_layout_sha256 !=
          kSlabPublicationLayouts[static_cast<std::size_t>(result.rank)] ||
      string_field(root.value, "extent_inventory_sha256") !=
          kExtentInventories[static_cast<std::size_t>(result.rank)] ||
      extent_inventory != kExtentInventories[static_cast<std::size_t>(result.rank)] ||
      string_field(root.value, "buffer_inventory_sha256") != kBufferInventory ||
      buffer_inventory != kBufferInventory)
    fail("slab publication identity changed");
  auto* extents = field(root.value, "extents", kJsonArray);
  if (json_object_array_length(extents) != 3'108)
    fail("extent inventory changed");
  std::set<std::string> names;
  std::array<std::vector<std::pair<std::uint64_t, std::uint64_t>>, 2> ranges;
  for (std::size_t index = 0; index < json_object_array_length(extents); ++index) {
    auto* item = json_object_array_get_idx(extents, index);
    if (!item || json_object_get_type(item) != kJsonObject ||
        json_object_object_length(item) != 10)
      fail("extent field inventory changed");
    TargetLayer3NativeExtent extent{
        string_field(item, "name"), uint_field(item, "offset_bytes"),
        uint_field(item, "length_bytes"), string_field(item, "storage")};
    if (!names.insert(extent.name).second || extent.length_bytes == 0 ||
        extent.offset_bytes % 256 != 0)
      fail("extent name, length, or alignment changed");
    if (string_field(item, "layout").empty() ||
        string_field(item, "abi").empty())
      fail("extent layout or ABI changed");
    const int storage = extent.storage == "target_slab" ? 0 :
                        extent.storage == "indexer-sidecar" ? 1 : -1;
    const std::uint64_t bound = storage == 0 ? result.slab_bytes : kSidecarBytes;
    if (storage < 0 || extent.offset_bytes > bound ||
        extent.length_bytes > bound - extent.offset_bytes)
      fail("extent storage bounds changed");
    validate_shape(item, extent.length_bytes);
    ProjectionGlobalEvidence* scalar = nullptr;
    for (auto& evidence : scalar_evidence)
      if (extent.name == evidence.extent_name) scalar = &evidence;
    if (scalar && (extent.length_bytes != 4 ||
                   string_field(item, "dtype") != "F32" ||
                   extent.storage != "target_slab"))
      fail("projection scalar extent changed");
    auto* chunks = field(item, "source_chunks", kJsonArray);
    if (json_object_array_length(chunks) == 0) fail("extent source is missing");
    std::vector<std::pair<std::uint64_t, std::uint64_t>> source_ranges;
    for (std::size_t c = 0; c < json_object_array_length(chunks); ++c) {
      auto* chunk = json_object_array_get_idx(chunks, c);
      if (!chunk || json_object_get_type(chunk) != kJsonObject ||
          json_object_object_length(chunk) != 3 ||
          !hex64(string_field(chunk, "sha256")))
        fail("source chunk identity changed");
      const auto chunk_sha256 = string_field(chunk, "sha256");
      if (scalar && chunk_sha256 == scalar->source_chunk_sha256)
        scalar->extent_matched = true;
      const auto offset = uint_field(chunk, "offset_bytes");
      const auto length = uint_field(chunk, "length_bytes");
      if (!length || offset > bound || length > bound - offset ||
          extent.offset_bytes >= offset + length ||
          extent.offset_bytes + extent.length_bytes <= offset)
        fail("source chunk does not cover extent");
      source_ranges.push_back({offset, offset + length});
    }
    std::sort(source_ranges.begin(), source_ranges.end());
    std::uint64_t covered = extent.offset_bytes;
    for (const auto& source : source_ranges) {
      if (source.first > covered) fail("source chunk coverage has a gap");
      covered = std::max(covered, source.second);
    }
    if (covered < extent.offset_bytes + extent.length_bytes)
      fail("source chunks do not cover extent");
    ranges[storage].push_back({extent.offset_bytes,
                               extent.offset_bytes + extent.length_bytes});
    result.extents.push_back(std::move(extent));
  }
  if (std::any_of(scalar_evidence.begin(), scalar_evidence.end(),
                  [](const auto& value) { return !value.extent_matched; }))
    fail("projection scalar extent evidence is missing");
  for (auto& storage : ranges) {
    std::sort(storage.begin(), storage.end());
    for (std::size_t i = 1; i < storage.size(); ++i)
      if (storage[i].first < storage[i - 1].second)
        fail("extent overlap changed");
  }
  validate_buffers(root.value);
  validate_target_layer3_native_plan_binding(result);
  return result;
}

}  // namespace rocket::qwen38::decode

#include "model_config.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "json.h"

namespace rocket::fuel {
namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("rocket::fuel::config: " + what);
}

std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
  return s;
}

// --------------------------------------------------------------------------
// A YAML subset, sized to fuels/glm-5.3-flash/attention.yaml: nested block
// maps, scalars, inline flow sequences that may wrap across lines, and folded
// block scalars that are skipped. Sequence-of-mappings ("- name: ...") is
// parsed only far enough to be skipped, because the layer registry does not
// read those blocks. Keys are flattened to dotted paths, so "kda.num_heads"
// and "mla.num_heads" stay distinct where a flat key scan would not.
// --------------------------------------------------------------------------
class YamlFlat {
 public:
  explicit YamlFlat(const std::filesystem::path& p) {
    std::ifstream in(p);
    if (!in) fail("cannot open " + p.string());
    std::vector<std::pair<int, std::string>> stack;  // (indent, key)
    std::string line;
    while (std::getline(in, line)) {
      const std::string_view raw(line);
      std::string_view body = raw;
      // Strip comments that start the line or follow whitespace, but keep '#'
      // inside quotes; attention.yaml has no quoted '#', so a plain scan is
      // enough and a wrong strip would be caught by the cross-check.
      if (const auto hash = body.find('#'); hash != std::string_view::npos) {
        if (hash == 0 || body[hash - 1] == ' ') body = body.substr(0, hash);
      }
      const std::string_view t = trim(body);
      if (t.empty()) continue;

      int indent = 0;
      while (indent < static_cast<int>(raw.size()) && raw[indent] == ' ') ++indent;

      if (pending_skip_indent_ >= 0) {
        if (indent > pending_skip_indent_) continue;
        pending_skip_indent_ = -1;
      }
      if (continuing_) {
        accum_ += ' ';
        accum_.append(t);
        if (balanced(accum_)) {
          values_[accum_key_] = accum_;
          continuing_ = false;
        }
        continue;
      }
      if (t.front() == '-') {           // sequence item: not needed, skip block
        pending_skip_indent_ = indent;
        continue;
      }
      const auto colon = t.find(':');
      if (colon == std::string_view::npos) continue;
      const std::string key(trim(t.substr(0, colon)));
      const std::string_view val = trim(t.substr(colon + 1));

      while (!stack.empty() && stack.back().first >= indent) stack.pop_back();
      std::string path;
      for (const auto& [_, k] : stack) path += k + ".";
      path += key;

      if (val.empty()) {                // nested block opens
        stack.emplace_back(indent, key);
        continue;
      }
      if (val.front() == '>' || val.front() == '|') {  // folded scalar, skip
        pending_skip_indent_ = indent;
        continue;
      }
      std::string v(val);
      if (!balanced(v)) {               // inline sequence wrapping lines
        continuing_ = true;
        accum_ = v;
        accum_key_ = path;
        continue;
      }
      values_[path] = v;
    }
    if (continuing_) fail("unterminated inline sequence for " + accum_key_);
  }

  bool has(const std::string& k) const { return values_.count(k) != 0; }

  const std::string& at(const std::string& k) const {
    const auto it = values_.find(k);
    if (it == values_.end()) fail("attention.yaml is missing " + k);
    return it->second;
  }

  long long integer(const std::string& k) const { return std::stoll(at(k)); }
  double number(const std::string& k) const { return std::stod(at(k)); }

  std::vector<int> int_list(const std::string& k) const {
    std::vector<int> out;
    const std::string& v = at(k);
    std::string digits;
    for (const char c : v) {
      if (std::isdigit(static_cast<unsigned char>(c)) || c == '-') {
        digits.push_back(c);
      } else if (!digits.empty()) {
        out.push_back(std::stoi(digits));
        digits.clear();
      }
    }
    if (!digits.empty()) out.push_back(std::stoi(digits));
    return out;
  }

 private:
  static bool balanced(std::string_view s) {
    int depth = 0;
    for (const char c : s) {
      if (c == '[' || c == '{') ++depth;
      if (c == ']' || c == '}') --depth;
    }
    return depth == 0;
  }

  std::map<std::string, std::string> values_;
  int pending_skip_indent_ = -1;
  bool continuing_ = false;
  std::string accum_, accum_key_;
};

// --------------------------------------------------------------------------

const json::JsonValue& member(const json::JsonValue& v, const char* key) {
  const json::JsonValue* m = v.member(key);
  if (m == nullptr) fail(std::string("config.json is missing ") + key);
  return *m;
}

int json_int(const json::JsonValue& v, const char* key) {
  return static_cast<int>(member(v, key).number);
}
float json_float(const json::JsonValue& v, const char* key) {
  return static_cast<float>(member(v, key).number);
}
bool json_bool(const json::JsonValue& v, const char* key) { return member(v, key).boolean; }

void expect(bool ok, const std::string& what) {
  if (!ok) fail("attention.yaml and config.json disagree on " + what);
}

}  // namespace

int ModelConfig::kda_layer_count() const {
  return static_cast<int>(std::count_if(layers.begin(), layers.end(),
                                        [](const LayerSpec& l) { return l.attn == AttnKind::kKda; }));
}
int ModelConfig::mla_layer_count() const {
  return static_cast<int>(std::count_if(layers.begin(), layers.end(),
                                        [](const LayerSpec& l) { return l.attn == AttnKind::kSparseMla; }));
}

std::filesystem::path default_attention_yaml() {
  if (const char* env = std::getenv("ROCKET_ATTENTION_YAML"); env != nullptr && *env != '\0') {
    return std::filesystem::path(env);
  }
#ifdef ROCKET_REPO_ROOT
  return std::filesystem::path(ROCKET_REPO_ROOT) / "fuels" / "glm-5.3-flash" / "attention.yaml";
#else
  return {};
#endif
}

ModelConfig load_model_config(const std::filesystem::path& attention_yaml,
                              const std::filesystem::path& snapshot_dir) {
  const YamlFlat y(attention_yaml);
  ModelConfig c;

  c.hidden_size = static_cast<int>(y.integer("hidden_size"));
  c.text_layers = static_cast<int>(y.integer("text_layers"));
  c.mtp_layer_id = static_cast<int>(y.integer("layers.mtp_layer_id"));

  c.kda_heads = static_cast<int>(y.integer("kda.num_heads"));
  c.kda_head_dim = static_cast<int>(y.integer("kda.head_dim"));
  c.kda_conv_kernel = static_cast<int>(y.integer("kda.short_conv_kernel_size"));
  c.kda_gate_lower_bound = static_cast<float>(y.number("kda.gate_lower_bound"));

  c.mla_heads = static_cast<int>(y.integer("mla.num_heads"));
  c.q_lora_rank = static_cast<int>(y.integer("mla.q_lora_rank"));
  c.kv_lora_rank = static_cast<int>(y.integer("mla.kv_lora_rank"));
  c.qk_nope_head_dim = static_cast<int>(y.integer("mla.qk_nope_head_dim"));
  c.qk_rope_head_dim = static_cast<int>(y.integer("mla.qk_rope_head_dim"));
  c.v_head_dim = static_cast<int>(y.integer("mla.v_head_dim"));

  c.index_n_heads = static_cast<int>(y.integer("indexer.n_heads"));
  c.index_head_dim = static_cast<int>(y.integer("indexer.head_dim"));
  c.index_topk = static_cast<int>(y.integer("indexer.top_k"));
  c.index_kpool = static_cast<int>(y.integer("indexer.kpool"));

  const std::vector<int> kda_ids = y.int_list("layers.kda_layer_ids");
  const std::vector<int> mla_ids = y.int_list("layers.mla_layer_ids");
  if (static_cast<int>(kda_ids.size()) != y.integer("layers.kda_count"))
    fail("attention.yaml kda_layer_ids does not match kda_count");
  if (static_cast<int>(mla_ids.size()) != y.integer("layers.mla_count"))
    fail("attention.yaml mla_layer_ids does not match mla_count");

  c.layers.resize(c.text_layers);
  for (int i = 0; i < c.text_layers; ++i) c.layers[i].index = i;
  std::vector<int> seen(c.text_layers, 0);
  for (const int i : kda_ids) {
    if (i < 0 || i >= c.text_layers) fail("kda_layer_ids out of range");
    c.layers[i].attn = AttnKind::kKda;
    ++seen[i];
  }
  for (const int i : mla_ids) {
    if (i < 0 || i >= c.text_layers) fail("mla_layer_ids out of range");
    c.layers[i].attn = AttnKind::kSparseMla;
    ++seen[i];
  }
  for (int i = 0; i < c.text_layers; ++i)
    if (seen[i] != 1) fail("layer " + std::to_string(i) + " is not claimed by exactly one attention list");

  // ---- config.json, and the cross-check ----------------------------------
  const std::filesystem::path cfg_path = snapshot_dir / "config.json";
  std::ifstream in(cfg_path, std::ios::binary);
  if (!in) fail("cannot open " + cfg_path.string());
  std::ostringstream buf;
  buf << in.rdbuf();
  const std::string text = buf.str();
  json::JsonParser parser(text.data(), text.data() + text.size());
  const json::JsonValue root = parser.parse_document();
  const json::JsonValue& t = member(root, "text_config");

  expect(json_int(t, "hidden_size") == c.hidden_size, "hidden_size");
  expect(json_int(t, "num_hidden_layers") == c.text_layers, "num_hidden_layers");
  expect(json_int(t, "num_attention_heads") == c.mla_heads, "num_attention_heads");
  expect(json_int(t, "q_lora_rank") == c.q_lora_rank, "q_lora_rank");
  expect(json_int(t, "kv_lora_rank") == c.kv_lora_rank, "kv_lora_rank");
  expect(json_int(t, "qk_nope_head_dim") == c.qk_nope_head_dim, "qk_nope_head_dim");
  expect(json_int(t, "qk_rope_head_dim") == c.qk_rope_head_dim, "qk_rope_head_dim");
  expect(json_int(t, "v_head_dim") == c.v_head_dim, "v_head_dim");
  expect(json_int(t, "index_n_heads") == c.index_n_heads, "index_n_heads");
  expect(json_int(t, "index_head_dim") == c.index_head_dim, "index_head_dim");
  expect(json_int(t, "index_topk") == c.index_topk, "index_topk");
  expect(json_int(t, "index_kpool") == c.index_kpool, "index_kpool");

  const json::JsonValue& lin = member(t, "linear_attn_config");
  expect(json_int(lin, "num_heads") == c.kda_heads, "linear_attn_config.num_heads");
  expect(json_int(lin, "head_dim") == c.kda_head_dim, "linear_attn_config.head_dim");
  expect(json_int(lin, "short_conv_kernel_size") == c.kda_conv_kernel, "short_conv_kernel_size");

  const json::JsonValue& types = member(t, "layer_types");
  if (types.array == nullptr || static_cast<int>(types.array->size()) != c.text_layers)
    fail("config.json layer_types has the wrong length");
  for (int i = 0; i < c.text_layers; ++i) {
    const bool json_is_kda = (*types.array)[i].str == "linear_attention";
    const bool yaml_is_kda = c.layers[i].attn == AttnKind::kKda;
    expect(json_is_kda == yaml_is_kda, "layer_types[" + std::to_string(i) + "]");
  }

  const json::JsonValue& mlps = member(t, "mlp_layer_types");
  if (mlps.array == nullptr || static_cast<int>(mlps.array->size()) != c.text_layers)
    fail("config.json mlp_layer_types has the wrong length");
  for (int i = 0; i < c.text_layers; ++i) {
    c.layers[i].mlp = (*mlps.array)[i].str == "sparse" ? MlpKind::kSparse : MlpKind::kDense;
  }

  // Every indexer in this checkpoint is "full"; the cross-layer top-k sharing
  // path in the reference is not exercised, so refuse a fuel that needs it.
  const json::JsonValue& idx = member(t, "indexer_types");
  for (const json::JsonValue& v : *idx.array)
    if (v.str != "full") fail("indexer_types carries a shared layer; cross-layer top-k sharing is not implemented");

  c.vocab_size = json_int(t, "vocab_size");
  c.rms_norm_eps = json_float(t, "rms_norm_eps");
  c.intermediate_size = json_int(t, "intermediate_size");
  c.moe_intermediate_size = json_int(t, "moe_intermediate_size");
  c.n_routed_experts = json_int(t, "n_routed_experts");
  c.num_experts_per_tok = json_int(t, "num_experts_per_tok");
  c.n_shared_experts = json_int(t, "n_shared_experts");
  c.n_group = json_int(t, "n_group");
  c.topk_group = json_int(t, "topk_group");
  c.norm_topk_prob = json_bool(t, "norm_topk_prob");
  c.routed_scaling_factor = json_float(t, "routed_scaling_factor");
  c.swiglu_limit = json_float(t, "swiglu_limit");
  c.index_kpool_always_select_tail = json_bool(t, "index_kpool_always_select_tail");
  c.index_kpool_compress = json_bool(t, "index_kpool_compress");
  c.hc_mult = json_int(t, "hc_mult");
  c.hc_sinkhorn_iters = json_int(t, "hc_sinkhorn_iters");
  c.hc_eps = json_float(t, "hc_eps");
  c.pad_token_id = json_int(t, "pad_token_id");
  for (const json::JsonValue& v : *member(t, "eos_token_id").array)
    c.eos_token_ids.push_back(static_cast<int>(v.number));

  if (member(t, "scoring_func").str != "sigmoid")
    fail("router scoring_func is not sigmoid; this engine implements only the sigmoid noaux_tc router");
  if (member(t, "topk_method").str != "noaux_tc")
    fail("router topk_method is not noaux_tc");
  if (member(t, "hidden_act").str != "silu") fail("hidden_act is not silu");
  if (!json_bool(t, "mla_use_nope")) fail("mla_use_nope is false; this engine implements only the NoPE path");
  if (!json_bool(t, "mhc")) fail("mhc is false; this engine implements only the hyper-connection residual");
  if (json_int(t, "first_k_dense_replace") != 3) fail("first_k_dense_replace is not 3");

  return c;
}

}  // namespace rocket::fuel

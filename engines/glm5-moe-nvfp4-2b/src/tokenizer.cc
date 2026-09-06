#include "tokenizer.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "json.h"

namespace rocket::fuel {
namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("rocket::fuel::tokenizer: " + what);
}

void utf8_encode(std::uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Returns the codepoint at `i` and advances it.
std::uint32_t utf8_next(const std::string& s, std::size_t& i) {
  const auto c = static_cast<unsigned char>(s[i]);
  if (c < 0x80) { ++i; return c; }
  if ((c >> 5) == 0x6 && i + 1 < s.size()) {
    const std::uint32_t cp = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
    i += 2;
    return cp;
  }
  if ((c >> 4) == 0xE && i + 2 < s.size()) {
    const std::uint32_t cp = ((c & 0x0Fu) << 12) |
                             ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
                             (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
    i += 3;
    return cp;
  }
  if ((c >> 3) == 0x1E && i + 3 < s.size()) {
    const std::uint32_t cp = ((c & 0x07u) << 18) |
                             ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12) |
                             ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6) |
                             (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
    i += 4;
    return cp;
  }
  ++i;
  return c;
}

bool is_ws(std::uint32_t c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B || c == 0x0C;
}
// Exact below 0x80; above it every codepoint is taken to be a letter.
bool is_letter(std::uint32_t c) {
  if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  return true;
}
bool is_number(std::uint32_t c) { return c >= '0' && c <= '9'; }

// The Split pre-tokenizer's pattern, matched leftmost-first exactly as the
// alternation is written:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   [^\r\n\p{L}\p{N}]?\p{L}+
//   \p{N}{1,3}
//    ?[^\s\p{L}\p{N}]+[\r\n]*
//   \s*[\r\n]+
//   \s+(?!\S)
//   \s+
std::vector<std::string> pre_tokenize(const std::string& text) {
  std::vector<std::uint32_t> cp;
  std::vector<std::size_t> off;
  for (std::size_t i = 0; i < text.size();) {
    off.push_back(i);
    cp.push_back(utf8_next(text, i));
  }
  off.push_back(text.size());
  const std::size_t n = cp.size();

  auto lower = [](std::uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };
  auto contraction = [&](std::size_t i) -> std::size_t {
    if (cp[i] != '\'') return 0;
    static const char* subs[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (const char* s : subs) {
      const std::size_t len = std::char_traits<char>::length(s);
      if (i + 1 + len > n) continue;
      bool ok = true;
      for (std::size_t j = 0; j < len; ++j) {
        if (lower(cp[i + 1 + j]) != static_cast<std::uint32_t>(s[j])) {
          ok = false;
          break;
        }
      }
      if (ok) return len + 1;
    }
    return 0;
  };

  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < n) {
    std::size_t take = 0;

    if (const std::size_t c = contraction(i); c > 0) {
      take = c;
    } else {
      // [^\r\n\p{L}\p{N}]? \p{L}+
      std::size_t j = i;
      if (cp[j] != '\r' && cp[j] != '\n' && !is_letter(cp[j]) && !is_number(cp[j])) ++j;
      if (j < n && is_letter(cp[j])) {
        while (j < n && is_letter(cp[j])) ++j;
        take = j - i;
      }
      if (take == 0 && is_number(cp[i])) {  // \p{N}{1,3}
        std::size_t k = 0;
        while (k < 3 && i + k < n && is_number(cp[i + k])) ++k;
        take = k;
      }
      if (take == 0) {  // " ?[^\s\p{L}\p{N}]+[\r\n]*"
        j = i;
        if (cp[j] == ' ') ++j;
        std::size_t start = j;
        while (j < n && !is_ws(cp[j]) && !is_letter(cp[j]) && !is_number(cp[j])) ++j;
        if (j > start) {
          while (j < n && (cp[j] == '\r' || cp[j] == '\n')) ++j;
          take = j - i;
        }
      }
      if (take == 0 && is_ws(cp[i])) {  // "\s*[\r\n]+"
        j = i;
        while (j < n && is_ws(cp[j]) && cp[j] != '\r' && cp[j] != '\n') ++j;
        if (j < n && (cp[j] == '\r' || cp[j] == '\n')) {
          while (j < n && (cp[j] == '\r' || cp[j] == '\n')) ++j;
          take = j - i;
        }
      }
      if (take == 0 && is_ws(cp[i])) {  // "\s+(?!\S)" then "\s+"
        j = i;
        while (j < n && is_ws(cp[j])) ++j;
        // (?!\S) means the run must end at the end of input; otherwise the
        // last whitespace codepoint starts the next piece.
        take = (j == n) ? (j - i) : (j - i - 1);
        if (take == 0) take = j - i;
      }
      if (take == 0) take = 1;
    }

    out.emplace_back(text.substr(off[i], off[i + take] - off[i]));
    i += take;
  }
  return out;
}

}  // namespace

Tokenizer::Tokenizer(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) fail("cannot open " + p.string());
  std::ostringstream buf;
  buf << in.rdbuf();
  const std::string text = buf.str();
  json::JsonParser parser(text.data(), text.data() + text.size());
  const json::JsonValue root = parser.parse_document();

  const json::JsonValue* model = root.member("model");
  if (model == nullptr) fail("tokenizer.json has no model");
  if (const json::JsonValue* ty = model->member("type"); ty == nullptr || ty->str != "BPE")
    fail("tokenizer.json is not a BPE model");

  const json::JsonValue* vocab = model->member("vocab");
  if (vocab == nullptr || vocab->object == nullptr) fail("tokenizer.json has no vocab");
  int max_id = -1;
  for (const auto& [tok, v] : *vocab->object) max_id = std::max(max_id, static_cast<int>(v.number));

  const json::JsonValue* added = root.member("added_tokens");
  if (added != nullptr && added->array != nullptr) {
    for (const json::JsonValue& a : *added->array) {
      const json::JsonValue* id = a.member("id");
      if (id != nullptr) max_id = std::max(max_id, static_cast<int>(id->number));
    }
  }

  id_to_token_.assign(max_id + 1, std::string());
  special_.assign(max_id + 1, false);
  for (const auto& [tok, v] : *vocab->object) {
    const int id = static_cast<int>(v.number);
    id_to_token_[id] = tok;
    token_to_id_[tok] = id;
  }
  if (added != nullptr && added->array != nullptr) {
    for (const json::JsonValue& a : *added->array) {
      const json::JsonValue* id = a.member("id");
      const json::JsonValue* content = a.member("content");
      if (id == nullptr || content == nullptr) continue;
      const int i = static_cast<int>(id->number);
      id_to_token_[i] = content->str;
      special_[i] = true;
      special_to_id_[content->str] = i;
    }
  }

  const json::JsonValue* merges = model->member("merges");
  if (merges == nullptr || merges->array == nullptr) fail("tokenizer.json has no merges");
  int rank = 0;
  for (const json::JsonValue& m : *merges->array) {
    if (m.array == nullptr || m.array->size() != 2) fail("merge entry is not a pair");
    merge_rank_[(*m.array)[0].str + '\x01' + (*m.array)[1].str] = rank++;
  }

  // GPT-2 byte-to-unicode alphabet.
  std::vector<int> bs;
  for (int c = '!'; c <= '~'; ++c) bs.push_back(c);
  for (int c = 0xA1; c <= 0xAC; ++c) bs.push_back(c);
  for (int c = 0xAE; c <= 0xFF; ++c) bs.push_back(c);
  std::vector<int> cs = bs;
  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + extra++);
    }
  }
  for (std::size_t i = 0; i < bs.size(); ++i) {
    std::string u;
    utf8_encode(static_cast<std::uint32_t>(cs[i]), u);
    byte_to_unicode_[bs[i]] = u;
    unicode_to_byte_[u] = static_cast<std::uint8_t>(bs[i]);
  }
}

bool Tokenizer::is_special(int id) const {
  return id >= 0 && id < static_cast<int>(special_.size()) && special_[id];
}

std::vector<int> Tokenizer::encode_piece(const std::string& piece) const {
  std::string mapped;
  for (const char ch : piece) mapped += byte_to_unicode_[static_cast<unsigned char>(ch)];

  // ignore_merges: a pre-token that is already a vocabulary entry is emitted
  // whole, without consulting the merge table.
  if (const auto it = token_to_id_.find(mapped); it != token_to_id_.end()) return {it->second};

  std::vector<std::string> sym;
  for (std::size_t i = 0; i < mapped.size();) {
    const std::size_t start = i;
    utf8_next(mapped, i);
    sym.emplace_back(mapped.substr(start, i - start));
  }

  while (sym.size() > 1) {
    int best = std::numeric_limits<int>::max();
    std::size_t at = 0;
    for (std::size_t i = 0; i + 1 < sym.size(); ++i) {
      const auto it = merge_rank_.find(sym[i] + '\x01' + sym[i + 1]);
      if (it != merge_rank_.end() && it->second < best) {
        best = it->second;
        at = i;
      }
    }
    if (best == std::numeric_limits<int>::max()) break;
    const std::string merged = sym[at] + sym[at + 1];
    std::vector<std::string> next;
    next.reserve(sym.size() - 1);
    for (std::size_t i = 0; i < sym.size();) {
      if (i + 1 < sym.size() && sym[i] + sym[i + 1] == merged && sym[i] == sym[at] &&
          sym[i + 1] == sym[at + 1]) {
        next.push_back(merged);
        i += 2;
      } else {
        next.push_back(sym[i]);
        ++i;
      }
    }
    sym.swap(next);
  }

  std::vector<int> ids;
  ids.reserve(sym.size());
  for (const std::string& s : sym) {
    const auto it = token_to_id_.find(s);
    if (it == token_to_id_.end()) fail("byte-level symbol missing from the vocabulary: " + s);
    ids.push_back(it->second);
  }
  return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
  std::vector<int> ids;
  // Special tokens are matched literally, before the pre-tokenizer sees them.
  std::size_t cursor = 0;
  while (cursor < text.size()) {
    std::size_t best_pos = std::string::npos;
    std::size_t best_len = 0;
    int best_id = -1;
    for (const auto& [content, id] : special_to_id_) {
      const std::size_t at = text.find(content, cursor);
      if (at != std::string::npos && (at < best_pos || (at == best_pos && content.size() > best_len))) {
        best_pos = at;
        best_len = content.size();
        best_id = id;
      }
    }
    const std::size_t end = (best_pos == std::string::npos) ? text.size() : best_pos;
    if (end > cursor) {
      for (const std::string& piece : pre_tokenize(text.substr(cursor, end - cursor))) {
        const std::vector<int> part = encode_piece(piece);
        ids.insert(ids.end(), part.begin(), part.end());
      }
    }
    if (best_pos == std::string::npos) break;
    ids.push_back(best_id);
    cursor = best_pos + best_len;
  }
  return ids;
}

std::string Tokenizer::decode_one(int id) const {
  if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return {};
  if (special_[id]) return id_to_token_[id];
  const std::string& t = id_to_token_[id];
  std::string out;
  for (std::size_t i = 0; i < t.size();) {
    const std::size_t start = i;
    utf8_next(t, i);
    const auto it = unicode_to_byte_.find(t.substr(start, i - start));
    if (it != unicode_to_byte_.end()) out.push_back(static_cast<char>(it->second));
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  for (const int id : ids) out += decode_one(id);
  return out;
}

}  // namespace rocket::fuel

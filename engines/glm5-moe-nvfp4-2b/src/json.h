// Recursive-descent JSON, shared by every reader that must not depend on
// Python: the safetensors header, config.json, and tokenizer.json.
//
// Lifted verbatim out of safetensors.cc, which parsed only its own header.
// config.json and tokenizer.json need the same parser, and a second copy would
// be a second set of escape-handling bugs.
#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocket::fuel::json {

[[noreturn]] inline void fail(const std::string& what) {
  throw std::runtime_error("rocket::fuel::json: " + what);
}

struct JsonValue;
using JsonObject = std::map<std::string, JsonValue, std::less<>>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue {
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject } kind = Kind::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string str;
  std::shared_ptr<JsonArray> array;
  std::shared_ptr<JsonObject> object;

  const JsonValue* member(std::string_view key) const {
    if (kind != Kind::kObject || !object) return nullptr;
    auto it = object->find(key);
    return it == object->end() ? nullptr : &it->second;
  }
};

class JsonParser {
 public:
  JsonParser(const char* begin, const char* end) : p_(begin), end_(end) {}

  JsonValue parse_document() {
    JsonValue v = parse_value();
    skip_space();
    return v;
  }

 private:
  void skip_space() {
    while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
  }
  char peek() {
    skip_space();
    if (p_ >= end_) fail("truncated JSON header");
    return *p_;
  }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "' in JSON header");
    ++p_;
  }

  JsonValue parse_value() {
    switch (peek()) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': {
        JsonValue v;
        v.kind = JsonValue::Kind::kString;
        v.str = parse_string();
        return v;
      }
      case 't': return parse_literal("true", true);
      case 'f': return parse_literal("false", false);
      case 'n': {
        expect_word("null");
        return JsonValue{};
      }
      default: return parse_number();
    }
  }

  JsonValue parse_literal(const char* word, bool value) {
    expect_word(word);
    JsonValue v;
    v.kind = JsonValue::Kind::kBool;
    v.boolean = value;
    return v;
  }

  void expect_word(const char* word) {
    std::size_t n = std::strlen(word);
    if (static_cast<std::size_t>(end_ - p_) < n || std::memcmp(p_, word, n) != 0)
      fail("bad literal in JSON header");
    p_ += n;
  }

  JsonValue parse_number() {
    const char* start = p_;
    if (p_ < end_ && (*p_ == '-' || *p_ == '+')) ++p_;
    while (p_ < end_ && ((*p_ >= '0' && *p_ <= '9') || *p_ == '.' || *p_ == 'e' || *p_ == 'E' ||
                         *p_ == '-' || *p_ == '+'))
      ++p_;
    if (p_ == start) fail("bad number in JSON header");
    JsonValue v;
    v.kind = JsonValue::Kind::kNumber;
    v.number = std::strtod(std::string(start, p_).c_str(), nullptr);
    return v;
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (p_ >= end_) fail("unterminated string in JSON header");
      char c = *p_++;
      if (c == '"') break;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (p_ >= end_) fail("unterminated escape in JSON header");
      char e = *p_++;
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          // Tensor names are ASCII; keep \uXXXX intact rather than guessing an
          // encoding, and only decode the Basic Latin range.
          if (end_ - p_ < 4) fail("truncated \\u escape in JSON header");
          unsigned code = 0;
          for (int i = 0; i < 4; ++i) {
            char h = p_[i];
            code <<= 4;
            if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
            else fail("bad \\u escape in JSON header");
          }
          p_ += 4;
          if (code < 0x80) out.push_back(static_cast<char>(code));
          else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
          } else {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
          }
          break;
        }
        default: fail("bad escape in JSON header");
      }
    }
    return out;
  }

  JsonValue parse_array() {
    expect('[');
    JsonValue v;
    v.kind = JsonValue::Kind::kArray;
    v.array = std::make_shared<JsonArray>();
    if (peek() == ']') { ++p_; return v; }
    while (true) {
      v.array->push_back(parse_value());
      char c = peek();
      ++p_;
      if (c == ']') break;
      if (c != ',') fail("expected ',' or ']' in JSON header");
    }
    return v;
  }

  JsonValue parse_object() {
    expect('{');
    JsonValue v;
    v.kind = JsonValue::Kind::kObject;
    v.object = std::make_shared<JsonObject>();
    if (peek() == '}') { ++p_; return v; }
    while (true) {
      if (peek() != '"') fail("expected key string in JSON header");
      std::string key = parse_string();
      expect(':');
      v.object->emplace(std::move(key), parse_value());
      char c = peek();
      ++p_;
      if (c == '}') break;
      if (c != ',') fail("expected ',' or '}' in JSON header");
    }
    return v;
  }

  const char* p_;
  const char* end_;
};

}  // namespace rocket::fuel::json

// Byte-level BPE for the checkpoint's own tokenizer.json.
//
// No Python at decode time, so the tokenizer is read from tokenizer.json
// directly: a GPT-2 style byte-to-unicode alphabet, the Split pre-tokenizer's
// regex, and 321649 ranked merges with ignore_merges set (a pre-token that is
// already a vocabulary entry is emitted whole).
//
// The pre-tokenizer pattern uses \p{L} and \p{N}. Those are resolved exactly
// for ASCII and approximated above it: every codepoint >= 0x80 is treated as a
// letter. Decoding is exact for all input because it is a pure byte mapping.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocket::fuel {

class Tokenizer {
 public:
  explicit Tokenizer(const std::filesystem::path& tokenizer_json);

  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;
  std::string decode_one(int id) const;

  int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
  bool is_special(int id) const;

 private:
  std::vector<int> encode_piece(const std::string& piece) const;

  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> merge_rank_;  // "a\x01b" -> rank
  std::unordered_map<std::string, int> special_to_id_;
  std::vector<bool> special_;
  std::string byte_to_unicode_[256];
  std::unordered_map<std::string, std::uint8_t> unicode_to_byte_;
};

}  // namespace rocket::fuel

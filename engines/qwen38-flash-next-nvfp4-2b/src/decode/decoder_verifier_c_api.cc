// SPDX-License-Identifier: Apache-2.0
#include "decode/decoder_verifier_c_api.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <string_view>

#include "decode/decoder_verifier.h"

namespace {
thread_local std::string last_error;

bool bounded_string(const char* value) noexcept {
  return value != nullptr && ::strnlen(value, 129) <= 128;
}
}  // namespace

extern "C" int qwen38_decoder_verifier_step(
    void* opaque, std::uint64_t generation, const std::int32_t* token_ids,
    int sequences, int verify_width, const char* trace_id,
    const char* request_id, std::int32_t* output_tokens,
    std::int32_t* accepted_prefixes) noexcept {
  last_error.clear();
  if (!opaque || !token_ids || !output_tokens || !accepted_prefixes ||
      !bounded_string(trace_id) || !bounded_string(request_id)) {
    last_error = "decoder verifier C ABI arguments are invalid";
    return 1;
  }
  try {
    auto& verifier =
        *static_cast<rocket::qwen38::decode::DecoderVerifier*>(opaque);
    const auto output = verifier.step(
        generation, token_ids, {sequences, verify_width}, trace_id, request_id);
    std::copy_n(output.tokens.begin(), output.sequences, output_tokens);
    std::copy_n(output.accepted_prefixes.begin(), output.sequences,
                accepted_prefixes);
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
  } catch (...) {
    last_error = "decoder verifier step failed";
  }
  return 1;
}

extern "C" const char* qwen38_decoder_verifier_last_error() noexcept {
  return last_error.c_str();
}

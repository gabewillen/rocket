// SPDX-License-Identifier: Apache-2.0
#include "mtp/decoder_step_c_api.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>

#include "mtp/decoder_step.h"

namespace {
thread_local std::string last_error;

bool bounded(const char* value) noexcept {
  return value && ::strnlen(value, 129) <= 128;
}
}  // namespace

extern "C" int qwen38_mtp_decoder_step(
    void* opaque, std::uint64_t generation, const char* trace_id,
    const char* request_id, std::int32_t* output_tokens,
    std::int32_t* accepted_prefixes) noexcept {
  last_error.clear();
  if (!opaque || !output_tokens || !accepted_prefixes || !bounded(trace_id) ||
      !bounded(request_id)) {
    last_error = "MTP decoder step C ABI arguments are invalid";
    return 1;
  }
  try {
    auto& step = *static_cast<rocket::qwen38::mtp::DecoderStep*>(opaque);
    const auto output = step.step(generation, trace_id, request_id);
    std::copy_n(output.tokens.begin(), output.sequences, output_tokens);
    std::copy_n(output.accepted_prefixes.begin(), output.sequences,
                accepted_prefixes);
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
  } catch (...) {
    last_error = "MTP decoder step failed";
  }
  return 1;
}

extern "C" const char* qwen38_mtp_decoder_step_last_error() noexcept {
  return last_error.c_str();
}

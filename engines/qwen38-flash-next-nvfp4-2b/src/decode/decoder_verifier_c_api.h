// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Coarse service boundary. The opaque handle is a native
// rocket::qwen38::decode::DecoderVerifier owned by cold initialization.
extern "C" int qwen38_decoder_verifier_step(
    void* verifier, std::uint64_t generation, const std::int32_t* token_ids,
    int sequences, int verify_width, const char* trace_id,
    const char* request_id, std::int32_t* output_tokens,
    std::int32_t* accepted_prefixes) noexcept;

extern "C" const char* qwen38_decoder_verifier_last_error() noexcept;

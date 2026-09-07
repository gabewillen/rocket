// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

extern "C" int qwen38_mtp_decoder_step(
    void* decoder_step, std::uint64_t generation, const char* trace_id,
    const char* request_id, std::int32_t* output_tokens,
    std::int32_t* accepted_prefixes) noexcept;

extern "C" const char* qwen38_mtp_decoder_step_last_error() noexcept;

// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

extern "C" {

struct Qwen38QsaPrefillCounters {
  std::uint64_t union_tokens;
  std::uint64_t selected_tokens;
  std::uint64_t index_loads;
};

// Fixed Qwen3.8 QSA prefill: Hq=12, Hkv=1, D=256, token top-k=2051.
// prepare() is cold-path only. launch() is allocation-free, capture-safe, and
// enqueues exclusively on stream. Indices must be strictly ascending, unique,
// request-relative, causal, and padded with -1 after the valid prefix. The
// caller publishes Q/K/V and indices on this stream before launch. This stage
// treats K/V as immutable and publishes only output after stream completion.
// counters is optional diagnostic telemetry and has no cardinality-bearing
// labels. Production capture passes nullptr to remove its global atomics.
int qwen38_qsa_prefill_prepare(int device);
int qwen38_qsa_prefill_union2(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const std::int32_t* sorted_indices,
    int sequences, int query_tokens, int context_tokens, __nv_bfloat16* output,
    Qwen38QsaPrefillCounters* counters, cudaStream_t stream);
int qwen38_qsa_prefill_control(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const std::int32_t* sorted_indices,
    int sequences, int query_tokens, int context_tokens, __nv_bfloat16* output,
    cudaStream_t stream);
const char* qwen38_qsa_prefill_last_error();

}

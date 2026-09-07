// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

extern "C" {

// Qwen3.8 TP-local QKV postprocessor. qkv is [rows,6656] with 12 repeated
// [q256|gate256] heads followed by K256 and V256. positions is axis-major
// [3,rows]. index_qk halves are the authenticated rank0/rank1 row halves of
// the replicated BF16 [640,2560] matrix.
int qwen38_qsa_preprocess(
    const __nv_bfloat16* hidden, const __nv_bfloat16* qkv,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    const __nv_bfloat16* index_qk_first,
    const __nv_bfloat16* index_qk_second,
    const __nv_bfloat16* index_q_norm,
    const __nv_bfloat16* index_k_norm,
    const std::int64_t* positions, int rows,
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    __nv_bfloat16* gate, __nv_bfloat16* index_query,
    __nv_bfloat16* index_raw_key, __nv_bfloat16* index_projected_scratch,
    cudaStream_t stream);

int qwen38_qsa_format_state_rows(
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    const __nv_bfloat16* index_raw_key,
    const __nv_bfloat16* index_k_norm,
    const std::int64_t* positions, const std::int64_t* logical_positions,
    const std::int32_t* token_to_request,
    const void* active_raw_state, int rows,
    void* main_rows_fp8, void* raw_rows_bf16,
    __nv_bfloat16* compressed_rows, cudaStream_t stream);

int qwen38_qsa_apply_output_gate(__nv_bfloat16* attention,
                                 const __nv_bfloat16* gate, int rows,
                                 cudaStream_t stream);

const char* qwen38_qsa_preprocess_last_error();
}

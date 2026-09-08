#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

extern "C" {

// Fixed Qwen3.8 layer-3 TP2 c16 projection. Create concatenates the three
// authenticated families into one immutable CUTLASS B/SFB allocation. All
// allocation and initialization happens before capture, so launch is safe.
int qwen38_cutlass_qkv_create(const std::uint8_t* q_weight,
                              const std::uint8_t* q_scale, float q_global,
                              const std::uint8_t* k_weight,
                              const std::uint8_t* k_scale, float k_global,
                              const std::uint8_t* v_weight,
                              const std::uint8_t* v_scale, float v_global,
                              int device, void** plan);
int qwen38_cutlass_qkv_launch(void* plan, const void* activations_bf16,
                              cudaStream_t stream);
int qwen38_cutlass_qkv_quantize(void* plan, const void* activations_bf16,
                                cudaStream_t stream);
int qwen38_cutlass_qkv_project(void* plan, cudaStream_t stream);
int qwen38_cutlass_qkv_output(void* plan, void** output_bf16,
                              std::size_t* elements);
int qwen38_cutlass_qkv_destroy(void* plan);
const char* qwen38_cutlass_qkv_last_error();

// Qwen QSA contract: 512 selected compressed blocks expand by four into at
// most 2,048 token ids plus the three-token open causal tail.
int qwen38_qsa_expand_topk(const std::int32_t* block_indices,
                           const std::int64_t* logical_positions,
                           const std::int32_t* sequence_lengths,
                           const std::int32_t* token_to_request,
                           std::int32_t* token_indices, int rows,
                           cudaStream_t stream);

// Fixed Qwen QSA indexer for c16 at the 262,144-token serving cap. The plan
// owns stable query, paged compressed-key, score, selection, and output
// buffers. Input pointers are borrowed so the future pre-indexer can refresh
// their contents without changing captured graph arguments.
int qwen38_qsa_indexer_create(const std::uint8_t* output_weight,
                              const std::uint8_t* output_scale,
                              float output_global, int device, void** plan);
int qwen38_qsa_indexer_launch(void* plan,
                             const std::int64_t* logical_positions,
                             const std::int32_t* sequence_lengths,
                             const std::int32_t* token_to_request,
                             cudaStream_t stream);
int qwen38_qsa_indexer_score(void* plan,
                            const std::int64_t* logical_positions,
                            const std::int32_t* sequence_lengths,
                            const std::int32_t* token_to_request,
                            cudaStream_t stream);
int qwen38_qsa_indexer_select_expand(void* plan,
                                    const std::int64_t* logical_positions,
                                    const std::int32_t* sequence_lengths,
                                    const std::int32_t* token_to_request,
                                    cudaStream_t stream);
int qwen38_qsa_indexer_select_expand_control(
    void* plan, const std::int64_t* logical_positions,
    const std::int32_t* sequence_lengths,
    const std::int32_t* token_to_request, cudaStream_t stream);
int qwen38_qsa_indexer_inputs(void* plan, void** query_bf16,
                             std::size_t* query_bytes, void** key_cache_bf16,
                             std::size_t* key_cache_bytes,
                             void** page_table_i32,
                             std::size_t* page_table_bytes);
int qwen38_qsa_indexer_output(void* plan, void** token_indices,
                             std::size_t* elements);
int qwen38_qsa_indexer_destroy(void* plan);

// Fixed rank-local QSA sparse attention: 12 query heads, one KV head,
// head-dim 256, 32 FP32-LSE splits, and arbitrary logical token ids.
int qwen38_qsa_attention_launch(void* plan, const void* qkv_output_bf16,
                               const std::int64_t* logical_positions,
                               const std::int32_t* token_to_request,
                               cudaStream_t stream);
int qwen38_qsa_sparse_attention(void* plan, const void* qkv_output_bf16,
                               const std::int64_t* logical_positions,
                               const std::int32_t* token_to_request,
                               cudaStream_t stream);
int qwen38_qsa_sparse_attention_control(
    void* plan, const void* qkv_output_bf16,
    const std::int64_t* logical_positions,
    const std::int32_t* token_to_request, cudaStream_t stream);
int qwen38_qsa_output_project(void* plan, cudaStream_t stream);
int qwen38_qsa_attention_output(void* plan, void** output_bf16,
                               std::size_t* elements);
int qwen38_qsa_projected_output(void* plan, void** output_bf16,
                               std::size_t* elements);

// Caller-owned c1 target index selection. Scratch extents are logits[65536],
// visible[1], selected_blocks[512], and selected_tokens[2051].
int qwen38_target_qsa_select_c1(
    const void* index_query_bf16, const void* compressed_cache_bf16,
    const std::int32_t* compressed_block_table,
    const std::int64_t* logical_positions,
    const std::int32_t* sequence_lengths,
    const std::int32_t* token_to_request, float* logits,
    std::int32_t* visible, std::int32_t* selected_blocks,
    std::int32_t* selected_tokens, int compressed_blocks,
    cudaStream_t stream);

// Fixed c1 main Q/K/V and gated output projections. All activation, scale,
// intermediate, and output buffers are caller-owned and graph-stable.
int qwen38_target_qsa_projection_create_c1(
    const std::uint8_t* q_weight, const std::uint8_t* q_scale, float q_global,
    const std::uint8_t* k_weight, const std::uint8_t* k_scale, float k_global,
    const std::uint8_t* v_weight, const std::uint8_t* v_scale, float v_global,
    const std::uint8_t* o_weight, const std::uint8_t* o_scale, float o_global,
    std::uint8_t* qkv_packed, std::uint8_t* qkv_sfa, void* raw_qkv_bf16,
    void* gated_attention_bf16, std::uint8_t* output_packed,
    std::uint8_t* output_sfa, void* projected_output_bf16, int device,
    void** plan);
int qwen38_target_qsa_project_qkv_c1(void* plan, const void* hidden_bf16,
                                    cudaStream_t stream);
int qwen38_target_qsa_project_output_c1(void* plan,
                                       const void* attention_bf16,
                                       const void* gate_bf16,
                                       cudaStream_t stream);
int qwen38_target_qsa_projection_output_c1(void* plan, void** output_bf16,
                                          std::size_t* elements);
int qwen38_target_qsa_projection_destroy_c1(void* plan);

// External-BF16-cache sparse attention. Scratch extents are
// partial_output[32,1,12,256], partial_lse[32,1,12], output[12,256].
int qwen38_target_qsa_attention_c1(
    const void* query_bf16, const void* main_key_cache_bf16,
    const void* main_value_cache_bf16,
    const std::int32_t* selected_tokens,
    const std::int32_t* main_block_table,
    const std::int32_t* token_to_request, float* partial_output,
    float* partial_lse, void* attention_output_bf16, int main_blocks,
    cudaStream_t stream);

}

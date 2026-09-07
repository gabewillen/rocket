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

}

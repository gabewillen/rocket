// Raw-pointer bridge API for the ported exllamav3 EXL3 kernels.
#pragma once
#include <cstdint>

struct Exl3MoeLayerPtrs
{
    const void* gate_trellis[288];
    const void* gate_suh[288];
    const void* gate_svh[288];
    const void* up_trellis[288];
    const void* up_suh[288];
    const void* up_svh[288];
    const void* down_trellis[288];
    const void* down_suh[288];
    const void* down_svh[288];
};

// One launch per MoE layer: gather, gate|up GEMV, silu+clamp (GLM swiglu
// limit), down GEMV, scatter-accumulate into the fp32 output. Pointer arrays
// are device-side, indexed by expert id; unfired entries may be null.
extern "C" void exl3_moe_raw
(
    const void* hidden_state,        // [bsz, hidden] fp16
    void* output_state,              // [bsz, hidden] fp32, zero-initialized
    const void* expert_count,        // [num_experts] int64 bincount
    const void* token_sorted,        // [bsz*topk] int64 token idx, expert-sorted
    const void* weight_sorted,       // [bsz*topk] fp16 routing weights
    void* temp_state_g,
    void* temp_state_u,
    void* temp_intermediate_g,
    void* temp_intermediate_u,
    const Exl3MoeLayerPtrs* ptrs,
    int bsz,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int num_experts_per_tok,
    int max_tokens_per_expert,
    int num_sms,
    float act_limit,
    int bits,
    void* stream
);

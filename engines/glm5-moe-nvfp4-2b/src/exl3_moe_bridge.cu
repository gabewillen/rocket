// Raw-pointer bridge for the ported exllamav3 batched MoE kernel.
//
// Exposes exl3_moe_raw: one launch per MoE layer performs gather, gate|up
#include <cuda_fp16.h>
#include <cstdint>
#include "util.cuh"
// GEMV, swiglu (silu with clamped limit - GLM-5.3 semantics), down GEMV, and
// the scatter-accumulate into an fp32 output, reading expert weights as
// EXL3 trellis blobs via per-expert device pointer arrays.
//
// All pointer arrays are device-side, indexed by expert id; entries for
// unfired experts are never dereferenced (gated by expert_count) and may be
// null. expert_count is the inclusive prefix count over expert-sorted slots.
#include "exl3_kernel_map.cuh"
#include "comp_units/exl3_moe_instances.cuh"
#include "exl3_gemm_inner.cuh"  // EXL3_GEMM_BASE_THREADS, MOE_TILESIZE_K, SMEM_MAX
#include "exl3_devctx.cuh"
#include "torch_check_shim.h"
#include <cuda_fp16.h>

// From exl3_moe_common.cuh (ported verbatim)
#include "exl3_moe_common.cuh"  // MOE_ACT_SILU, MOE_SMS_PER_EXPERT

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

extern "C" void exl3_moe_raw
(
    const void* hidden_state,        // [bsz, hidden] fp16
    void* output_state,              // [bsz, hidden] fp32, zero-initialized
    const void* expert_count,        // [num_experts+1] int64 prefix counts
    const void* token_sorted,        // [bsz*topk] int64 token idx, expert-sorted
    const void* weight_sorted,       // [bsz*topk] fp16 routing weights
    void* temp_state_g,              // [conc, max_tpe, hidden] fp16
    void* temp_state_u,
    void* temp_intermediate_g,       // [conc, max_tpe, inter] fp16
    void* temp_intermediate_u,
    const Exl3MoeLayerPtrs* ptrs,    // device-resident pointer arrays
    int bsz,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int num_experts_per_tok,
    int max_tokens_per_expert,
    int num_sms,
    float act_limit,
    int bits,                        // gate = up = down bit rate
    cudaStream_t stream)
{
    const int act_function_silu = MOE_ACT_SILU;
    int concurrency = num_sms / MOE_SMS_PER_EXPERT;
    if (concurrency < 1) concurrency = 1;
    if (concurrency * MOE_SMS_PER_EXPERT > num_sms)
        concurrency = num_sms / MOE_SMS_PER_EXPERT;

    // kernel instance index: 2*K + N_off, N_off = 0 (n128) or 1 (n256);
    // n256 tiles fit intermediate_dim 2048? intermediate_dim/128 = 16 warps;
    // N_off selects the per-expert tile width in the intermediate dim.
    int n_off = (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0) ? 1 : 0;

    extern fp_exl3_moe_kernel exl3_moe_kernel_instances[];
    fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[2 * bits + n_off];

    dim3 grid_dim(MOE_SMS_PER_EXPERT, 1, concurrency);
    int block_dim = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;

    static bool attr_set[MAX_DEVICES] = {};
    int device;
    cudaGetDevice(&device);
    DevCtx& ctx = DevCtx::instance();
    int* locks = ctx.get_locks(device);
    if (!attr_set[device])
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             90 * 1024);
        attr_set[device] = true;
    }

    const void* _hidden = hidden_state;
    void* _tsg = temp_state_g;
    void* _tsu = temp_state_u;
    void* _tig = temp_intermediate_g;
    void* _tiu = temp_intermediate_u;
    void* _out = output_state;
    const void* _gt = &ptrs->gate_trellis[0];
    const void* _gs = &ptrs->gate_suh[0];
    const void* _gv = &ptrs->gate_svh[0];
    const void* _ut = &ptrs->up_trellis[0];
    const void* _us = &ptrs->up_suh[0];
    const void* _uv = &ptrs->up_svh[0];
    const void* _dt = &ptrs->down_trellis[0];
    const void* _ds = &ptrs->down_suh[0];
    const void* _dv = &ptrs->down_svh[0];
    const void* _ec = expert_count;
    const void* _tok = token_sorted;
    const void* _wt = weight_sorted;

    void* kernelArgs[] =
    {
        (void*)& _hidden,
        (void*)& _tsg,
        (void*)& _tsu,
        (void*)& _tig,
        (void*)& _tiu,
        (void*)& _out,
        (void*)& _gt,
        (void*)& _gs,
        (void*)& _gv,
        (void*)& _ut,
        (void*)& _us,
        (void*)& _uv,
        (void*)& _dt,
        (void*)& _ds,
        (void*)& _dv,
        (void*)& _ec,
        (void*)& _tok,
        (void*)& _wt,
        (void*)& hidden_dim,
        (void*)& intermediate_dim,
        (void*)& num_experts,
        (void*)& num_experts_per_tok,
        (void*)& max_tokens_per_expert,
        (void*)& concurrency,
        (void*)& act_limit,
        // act_function: only MOE_ACT_SILU (0) is implemented in the ported
        // kernel set; GLM-5.3's swiglu is the silu variant
        (void*)& act_function_silu,
        (void*)& bits,
        (void*)& bits,
        (void*)& bits,
        (void*)& locks
    };

    cudaLaunchKernel
    (
        (void*) kernel,
        grid_dim,
        block_dim,
        kernelArgs,
        90 * 1024,
        stream
    );
}

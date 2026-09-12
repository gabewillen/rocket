// Raw-pointer bridge for the ported exllamav3 EXL3 GEMM kernels.
//
// Replicates exl3_gemm_gr's deterministic dispatch (force_shape_idx path)
// without the at::Tensor glue: A [m, k] half, B trellis [k/16, n/16, bits*16]
// int16, C [m, n] half or float, optional suh [k] / svh [n] half scales,
// optional A_had [m, k] scratch for the input hadamard. mcg selects the
// multi-codebook variant this checkpoint uses.
//
// The autotuner is bypassed: shape_idx is chosen by the same
// select_exl3_gemm_kernel heuristic the forced path uses, launched
// cooperatively over num_sms.
#include <cuda_fp16.h>
#include <cstdint>
#include "util.cuh"
#include "exl3_kernel_map.cuh"
#include "exl3_devctx.cuh"
#include "torch_check_shim.h"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "exl3_gemm_inner.cuh"  // SMEM_MAX

fp_exl3_gemm_kernel select_exl3_gemm_kernel
(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* out_num_sms,
    int cb
);

static int dev_num_sms(int device)
{
    static int cached[MAX_DEVICES] = {};
    if (cached[device] == 0)
        cudaDeviceGetAttribute(&cached[device], cudaDevAttrMultiProcessorCount, device);
    return cached[device];
}

extern "C" int exl3_gemm_raw
(
    const void* A_ptr,
    const void* B_ptr,
    void* C_ptr,
    const void* suh_ptr,
    void* A_had_ptr,
    const void* svh_ptr,
    int size_m,
    int size_k,
    int size_n,
    int bits,
    bool mcg,
    bool c_fp32,
    cudaStream_t stream
)
{
    int device;
    cudaGetDevice(&device);
    DevCtx& ctx = DevCtx::instance();
    int num_sms = dev_num_sms(device);
    int cc = ctx.get_cc(device);
    int* locks = ctx.get_locks(device);

    int K = bits;
    int cb = mcg ? 1 : 0;
    int block_dim = 0;
    int shape_idx = 0;
    fp_exl3_gemm_kernel kernel = select_exl3_gemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        0, &block_dim, &shape_idx, &num_sms, cb
    );
    if (!kernel) return 0;

    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr
    };

    static bool attr_set[MAX_DEVICES] = {};
    if (!attr_set[device])
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
        attr_set[device] = true;
    }
    cudaError_t rc = cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        num_sms,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );
    if (rc != cudaSuccess) return 0;
    return shape_idx;
}

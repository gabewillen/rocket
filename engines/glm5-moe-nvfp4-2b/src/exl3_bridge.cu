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
#include <cstdio>
#include <cstdlib>
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

// Exact dynamic shared storage used by exl3_gemm_kernel_inner for each
// upstream template shape. Reserving the 90 KiB compile-time ceiling makes a
// 48-block cooperative launch impossible on GB10 even though the selected
// 4-bit kernels use only 17-50 KiB.
static int gemm_smem_bytes(int shape_idx, int bits)
{
    int tile_k = 0, tile_n = 0, stages = 0;
    switch (shape_idx)
    {
        case 1: tile_k = 16; tile_n = 128; stages = 6; break;
        case 2: tile_k = 32; tile_n = 128; stages = 4; break;
        case 3: tile_k = 32; tile_n = 256; stages = 4; break;
        case 4: tile_k = 16; tile_n = 512; stages = 4; break;
        default: return SMEM_MAX;
    }
    constexpr int tile_m = 16;
    const int a_bytes = 2 * tile_m * tile_k;
    const int b_bytes = tile_k * tile_n * bits / 8;
    const int c_bytes = 4 * tile_m * tile_n;
    return stages * (a_bytes + b_bytes) + c_bytes;
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

    static bool traced = false;
    if (!traced && std::getenv("ROCKET_EXL3_TRACE"))
    {
        traced = true;
        std::fprintf(stderr,
                     "[exl3-gemm] A=%p B=%p C=%p locks=%p suh=%p A_had=%p svh=%p "
                     "m=%d k=%d n=%d shape=%d blocks=%d threads=%d smem=%d\n",
                     A_ptr, B_ptr, C_ptr, locks, suh_ptr, A_had_ptr, svh_ptr,
                     size_m, size_k, size_n, shape_idx, num_sms, block_dim,
                     gemm_smem_bytes(shape_idx, bits));
        std::fflush(stderr);
    }

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

    const int smem = gemm_smem_bytes(shape_idx, bits);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaError_t rc = cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        num_sms,
        block_dim,
        kernelArgs,
        smem,
        stream
    );
    if (rc != cudaSuccess) return 0;
    return shape_idx;
}

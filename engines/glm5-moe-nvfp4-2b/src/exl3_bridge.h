// Raw-pointer bridge API for the ported exllamav3 EXL3 GEMM kernel.
#pragma once
#include <cstdint>

// y[m, n] = A[m, k] @ dequant(B_trellis)^T with per-row (suh) and per-col
// (svh) fp16 scales, optional input hadamard. mcg selects the checkpoint's
// multi-codebook variant. c_fp32 selects fp32 (true) or fp16 (false) output.
// Returns the shape_idx from the kernel selector, or 0 on failure.
extern "C" int exl3_gemm_raw
(
    const void* A,                   // [m, k] fp16
    const void* B,                   // [k/16, n/16, bits*16] int16 trellis
    void* C,                         // [m, n] fp16 or fp32
    const void* suh,                 // [k] fp16 or null
    void* A_had,                     // [m, k] fp16 scratch or null
    const void* svh,                 // [n] fp16 or null
    int m,
    int k,
    int n,
    int bits,                        // 4 for this checkpoint
    bool mcg,                        // true for this checkpoint
    bool c_fp32,
    void* stream
);

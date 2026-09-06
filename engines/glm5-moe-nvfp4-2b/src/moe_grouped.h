// CUTLASS NVFP4 grouped GEMM launcher, stage 2 of the M-stream batching arc
// (kernels.h, model.cu::run_moe). This header is the plug point stage 1 left:
// the block-scaled FP4 tensor core op needs the sm_121a variant of this SM
// (CUDA_ARCHITECTURES "121a"; see bench/nvfp4_grouped_gemm.cu and
// tests/test_loader_swizzle.cu), while every other translation unit in
// rocket_engine builds at plain sm_121. No CUTLASS type crosses this header,
// so model.cu and kernels.cu never put CUTLASS on their include path; only
// moe_grouped.cu does, and it alone gets the "a" arch suffix
// (CMakeLists.txt's rocket_moe_grouped target).
#pragma once

#include <cstdint>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace rocket::engine {

using bf16 = __nv_bfloat16;

// One expert's contribution to a grouped GEMM call:
//   D[m, n] = A[m, k] . B[n, k]^T
// A and B are both NVFP4 (e2m1 data, e4m3 block scale per 16 contiguous K
// elements, CUTLASS SFA/SFB swizzle -- see nvfp4.h::SfLayout and
// kernels.h::nvfp4_quantize_rows for A, weights.cu for B). D is BF16,
// accumulation is FP32, alpha=1, beta=0 (no source read: every group starts
// from nothing, matching the fresh gate/up/down GEMM this replaces). Neither
// weight_scale_2 global nor the per-token router weight is applied here --
// gate/up globals go in swiglu_grouped, the down global and the router
// weight go into moe_scatter_add's weight_of (model.cu::run_moe) -- because
// this launcher shares one epilogue alpha across every group and the two
// halves of a fused w13 do not share a scale (kernels.h::swiglu_grouped).
struct GroupedGemmGroup {
  int m = 0;
  const std::uint8_t* a_packed = nullptr;  // this group's own rows, [m, k/2]
  const std::uint8_t* a_scale = nullptr;   // this group's own SFA slab
  const std::uint8_t* b_packed = nullptr;  // expert weight, [n, k/2]
  const std::uint8_t* b_scale = nullptr;   // expert weight SFB, from the cache
  bf16* d_out = nullptr;                   // this group's output rows, [m, n]
};

// Runs one grouped NVFP4 GEMM over `groups`, all sharing (n, k). Groups with
// m == 0 are skipped. Returns false (nothing written) if CUTLASS cannot
// implement this problem; model.cu treats that as "fall back to the
// per-(stream, expert) GEMV path for this step" rather than a hard error.
bool grouped_gemm_nvfp4(const std::vector<GroupedGemmGroup>& groups, int n, int k, cudaStream_t s);

}  // namespace rocket::engine

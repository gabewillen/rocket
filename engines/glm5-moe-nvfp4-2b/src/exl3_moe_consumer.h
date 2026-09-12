// EXL3 MoE consumer: routes run_moe's readback through the ported
// exllamav3 batched MoE kernel (one launch per layer). The kernel does the
// gather, gate|up GEMV, swiglu with the GLM clamp, down GEMV, and the
// scatter-accumulate into an fp32 per-token output - the router weight and
// the per-output scale (svh) are applied inside the kernel, so no
// scatter_add pass is needed afterward.
#include "exl3_moe_bridge.h"

#include <cuda_fp16.h>

namespace rocket::engine {

void bf16_to_fp16_rows(void* dst, const void* src, long long n, cudaStream_t s);
void fp32_to_bf16_rows(void* dst, const void* src, long long n, cudaStream_t s);

}  // namespace rocket::engine

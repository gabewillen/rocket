// The entry point the engine lane adopts. Parameter list is identical to
// src/kernels.cu kda_recurrent_step(), so adopting it is replacing that
// function's body with this launch and deleting kda_step_kernel.
//
// The sweep picked a different (R, C) per storage width at the engine's dims:
//   FP32 state  R=4  C=1   512 threads, 3584 B shared, 64 regs, 2 blocks/SM
//   BF16 state  R=16 C=4   512 threads, 9728 B shared, 64 regs, 2 blocks/SM
// See kda_step_dropin.cuh for the trade the two knobs make.

#include "kda_step_dropin.cuh"

#include <cstdio>
#include <cstdlib>

namespace rocket_kda {

namespace {
constexpr int kHeads = 64;
constexpr int kDk = 128;

void check_dims(int heads, int head_dim) {
  // The checkpoint fixes both (fuels/glm-5.3-flash/attention.yaml). This engine
  // is built for one chemistry on one architecture, so a mismatch is a build
  // error, not a slow path.
  if (heads != kHeads || head_dim != kDk) {
    std::fprintf(stderr,
                 "rocket_kda::recurrent_step: built for heads=%d head_dim=%d, got %d/%d\n",
                 kHeads, kDk, heads, head_dim);
    std::abort();
  }
}
}  // namespace

void recurrent_step_f32(float* state, bf16* o, const bf16* q, const bf16* k,
                        const bf16* v, const bf16* g, const bf16* beta, int batch,
                        int heads, int head_dim, int row_stride, cudaStream_t s) {
  check_dims(heads, head_dim);
  constexpr int R = 4, C = 1;
  step_split<float, kHeads, kDk, R, C>
      <<<dim3(heads, batch), dim3(kDk / C, R), 0, s>>>(state, o, q, k, v, g, beta,
                                                       row_stride);
}

void recurrent_step_bf16(bf16* state, bf16* o, const bf16* q, const bf16* k,
                         const bf16* v, const bf16* g, const bf16* beta, int batch,
                         int heads, int head_dim, int row_stride, cudaStream_t s) {
  check_dims(heads, head_dim);
  constexpr int R = 16, C = 4;
  step_split<bf16, kHeads, kDk, R, C>
      <<<dim3(heads, batch), dim3(kDk / C, R), 0, s>>>(state, o, q, k, v, g, beta,
                                                       row_stride);
}

}  // namespace rocket_kda

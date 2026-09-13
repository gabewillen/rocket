// DFlash2 CUDA primitives ported from vLLM qwen3_dflash2.py.
#include "dflash2.h"
#include "kernels.h"

#include <stdexcept>

namespace rocket::engine {
namespace {
using bf16 = __nv_bfloat16;
__device__ __forceinline__ float ff(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 bb(float x) { return __float2bfloat16(x); }

// Exact _grouped_conv indexing. Rows are request-major blocks of query
// positions, so row % block_size is the position mask used by upstream.
__global__ void grouped_conv_kernel(bf16* out, const bf16* hidden, const bf16* coeff,
                                    const bf16* base, int rows, int hidden_size,
                                    int block_size, int group_size, int taps, int side) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (c >= hidden_size || row >= rows) return;
  const int groups = hidden_size / group_size;
  const int group = c / group_size;
  const int position = row % block_size;
  float acc = 0.0f;
  for (int tap = 0; tap < taps; ++tap) {
    if (position < tap) continue;
    const long long ci = ((static_cast<long long>(row) * 2 + side) * taps + tap) * groups + group;
    const long long bi = (static_cast<long long>(side) * taps + tap) * hidden_size + c;
    // PyTorch operands and each pointwise result are BF16 in the reference.
    const float coefficient = ff(bb(ff(base[bi]) + ff(coeff[ci])));
    const float term = ff(bb(coefficient *
                             ff(hidden[static_cast<long long>(row - tap) * hidden_size + c])));
    acc = ff(bb(acc + term));
  }
  out[static_cast<long long>(row) * hidden_size + c] = bb(acc);
}
}  // namespace

void dflash2_grouped_conv_prepare(bf16* out, bf16* coeff, const bf16* hidden,
                                  const bf16* kernel_projection, const bf16* base_kernel,
                                  int rows, int hidden_size, int block_size, int group_size,
                                  int taps, cudaStream_t stream) {
  if (hidden_size % group_size != 0) throw std::runtime_error("dflash2 conv group mismatch");
  const int coeff_width = 2 * taps * (hidden_size / group_size);
  gemm_bf16_cublas(coeff, kernel_projection, hidden, rows, coeff_width, hidden_size, stream);
  grouped_conv_kernel<<<dim3((hidden_size + 255) / 256, rows), 256, 0, stream>>>(
      out, hidden, coeff, base_kernel, rows, hidden_size, block_size, group_size, taps, 0);
}

void dflash2_grouped_conv_finish(bf16* out, const bf16* hidden, const bf16* coeff,
                                 const bf16* base_kernel, int rows, int hidden_size,
                                 int block_size, int group_size, int taps, cudaStream_t stream) {
  grouped_conv_kernel<<<dim3((hidden_size + 255) / 256, rows), 256, 0, stream>>>(
      out, hidden, coeff, base_kernel, rows, hidden_size, block_size, group_size, taps, 1);
}

}  // namespace rocket::engine

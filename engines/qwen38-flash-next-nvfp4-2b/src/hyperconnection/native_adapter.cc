// SPDX-License-Identifier: Apache-2.0
#include "hyperconnection/native_adapter.h"

#include <stdexcept>
#include <string>

namespace rocket::qwen38::hyperconnection {

void NativeFullAttentionHyperConnection::mix(
    const __nv_bfloat16* hidden, __nv_bfloat16* block_input,
    __nv_bfloat16* injection, int m, cudaStream_t stream) {
  plan_.mix(hidden, block_input, injection, m, stream);
}

void NativeFullAttentionHyperConnection::combine_and_mix(
    const __nv_bfloat16* hidden, const float* block_output,
    const __nv_bfloat16* injection, __nv_bfloat16* updated_hidden,
    __nv_bfloat16* next_block_input, __nv_bfloat16* next_injection, int m,
    cudaStream_t stream) {
  plan_.combine_and_mix(hidden, block_output, injection, updated_hidden,
                        next_block_input, next_injection, m, stream);
}

void NativeFullAttentionHyperConnection::combine(
    const __nv_bfloat16* hidden, const float* block_output,
    const __nv_bfloat16* injection, __nv_bfloat16* updated_hidden, int m,
    cudaStream_t stream) {
  plan_.combine(hidden, block_output, injection, updated_hidden, m, stream);
}

void NativeFullAttentionHyperConnection::synchronize(cudaStream_t stream) {
  if (!stream)
    throw std::invalid_argument("native Qwen HC fence requires a stream");
  const cudaError_t status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess)
    throw std::runtime_error(
        std::string("native Qwen HC fence: ") + cudaGetErrorString(status));
}

}  // namespace rocket::qwen38::hyperconnection

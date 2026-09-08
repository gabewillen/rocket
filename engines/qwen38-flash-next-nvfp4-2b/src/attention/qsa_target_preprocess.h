// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "attention/qsa_target_state_view.h"

namespace rocket::qwen38::attention {

// Borrowed immutable layer weights. Main Q/K/V/O projection weights retain
// the slab's ModelOpt NVFP4 ABI and are consumed by the graph projection
// plans. This preprocessing slice consumes only the checkpoint-BF16 tensors.
struct TargetQsaPreprocessWeights {
  const __nv_bfloat16* main_q_norm = nullptr;       // [256]
  const __nv_bfloat16* main_k_norm = nullptr;       // [256]
  const __nv_bfloat16* index_qk = nullptr;          // [640,2560], replicated
  const __nv_bfloat16* index_q_norm = nullptr;      // [128]
  const __nv_bfloat16* index_k_norm = nullptr;      // [128]
  const __nv_bfloat16* rope_cos_sin = nullptr;      // [context,64]
};

// Caller-owned graph arena. Every extent is fixed for c1 and stays stable for
// the lifetime of a captured graph.
struct TargetQsaPreprocessBuffers {
  const __nv_bfloat16* raw_main_qkv = nullptr;  // [6656]
  __nv_bfloat16* index_projected_qk = nullptr;  // [640]
  __nv_bfloat16* main_query = nullptr;          // [12,256]
  __nv_bfloat16* attention_gate = nullptr;      // [12,256]
  __nv_bfloat16* index_query = nullptr;         // [4,128]
};

class TargetQsaPreprocessError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Enqueues the pinned vLLM QSA projection/pre-indexer semantics for one target
// token. It performs no allocation, D2H transfer, synchronization, or state
// publication.
void launch_target_qsa_preprocess_c1(
    const __nv_bfloat16* hidden, const TargetQsaPreprocessWeights& weights,
    const TargetQsaPreprocessBuffers& buffers, const TargetQsaStateView& state,
    cudaStream_t stream);

}  // namespace rocket::qwen38::attention

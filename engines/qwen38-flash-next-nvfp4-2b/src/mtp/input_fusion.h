// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace rocket::qwen38::mtp {

inline constexpr int kFusionHidden = 2'560;
inline constexpr int kFusionStreams = 4;
inline constexpr int kFusionHyperHidden = 10'240;
inline constexpr int kFusionLocalHidden = 1'280;

struct InputFusionWeights {
  const __nv_bfloat16* embedding_norm;
  const __nv_bfloat16* hidden_norm;
  const __nv_bfloat16* embedding_projection;
  const __nv_bfloat16* hidden_projection;
};

// Owns fixed c16 scratch and borrows four immutable authenticated MTP slab
// tensors. local_project() ends at an explicit PairReduce boundary. The caller
// reduces embedding_partial as [m,2560] and hidden_partial as [4m,2560], then
// passes both FP32 results to finish().
class InputFusionPlan final {
 public:
  InputFusionPlan(int device, int rank, InputFusionWeights weights);
  ~InputFusionPlan();
  InputFusionPlan(const InputFusionPlan&) = delete;
  InputFusionPlan& operator=(const InputFusionPlan&) = delete;

  void local_project(const __nv_bfloat16* embedding,
                     const __nv_bfloat16* multi_hidden,
                     __nv_bfloat16* embedding_partial,
                     __nv_bfloat16* hidden_partial, int m,
                     cudaStream_t stream);
  void finish(const float* reduced_embedding, const float* reduced_hidden,
              __nv_bfloat16* fused_multi_hidden, int m,
              cudaStream_t stream);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace rocket::qwen38::mtp

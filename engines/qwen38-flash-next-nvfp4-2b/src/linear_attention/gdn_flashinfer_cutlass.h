// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace rocket::qwen38::linear_attention {

inline constexpr char kGdnFlashInferCutlassRevision[] =
    "91bda04c66f7cb851e1ab3b78b9fecea644b9844";
inline constexpr int kGdnFlashInferCutlassTileM = 128;
inline constexpr int kGdnFlashInferCutlassTileN = 128;
inline constexpr int kGdnFlashInferCutlassTileK = 256;
inline constexpr bool kGdnFlashInferCutlassSwapAb = false;
inline constexpr bool kGdnFlashInferCutlassStreamK = false;

class GdnFlashInferCutlassGemm final {
 public:
  GdnFlashInferCutlassGemm();
  ~GdnFlashInferCutlassGemm();
  GdnFlashInferCutlassGemm(const GdnFlashInferCutlassGemm&) = delete;
  GdnFlashInferCutlassGemm& operator=(const GdnFlashInferCutlassGemm&) = delete;

  void init(int m, int n, int k, const std::uint8_t* packed_a,
            const std::uint8_t* sfa, const std::uint8_t* packed_b,
            const std::uint8_t* sfb, const float* alpha,
            __nv_bfloat16* output);
  void run(cudaStream_t stream);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace rocket::qwen38::linear_attention

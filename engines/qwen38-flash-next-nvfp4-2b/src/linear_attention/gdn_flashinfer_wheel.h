// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace rocket::qwen38::linear_attention {

inline constexpr char kGdnFlashInferWheelSha256[] =
    "dfd9f2076fda819e45d169c88774bd26a9fc93bdc714592ab5dfe12b90bbf5ae";
inline constexpr char kGdnFlashInferWheelPackage[] =
    "flashinfer-jit-cache==0.6.17+cu130";
inline constexpr char kGdnFlashInferWheelCudaCompiler[] = "13.0.88";
inline constexpr char kGdnFlashInferWheelArchitecture[] = "sm_120f";
inline constexpr char kGdnFlashInferWheelCutlassRevision[] =
    "b46b16d003484063bca4ed365e44095c4c6ed633";

std::string gdn_sha256_file(std::string_view path);

// Diagnostic-only adapter for the exact FlashInfer 0.6.17 sm_120f wheel
// artifact measured by the standalone comparator. FlashInfer and its runner
// source are Apache-2.0. The binary remains an external input and is not
// redistributed by Rocket. This calls the raw CUTLASS runner and has no Torch
// or TVM-FFI runtime dependency.
class GdnFlashInferWheelGemm final {
 public:
  GdnFlashInferWheelGemm();
  ~GdnFlashInferWheelGemm();
  GdnFlashInferWheelGemm(const GdnFlashInferWheelGemm&) = delete;
  GdnFlashInferWheelGemm& operator=(const GdnFlashInferWheelGemm&) = delete;

  void init(std::string_view shared_object, int m, int n, int k,
            const std::uint8_t* packed_a, const std::uint8_t* sfa,
            const std::uint8_t* packed_b, const std::uint8_t* sfb,
            const float* alpha, __nv_bfloat16* output);
  void run(cudaStream_t stream);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace rocket::qwen38::linear_attention

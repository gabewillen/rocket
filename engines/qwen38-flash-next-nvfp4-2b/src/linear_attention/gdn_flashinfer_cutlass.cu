// SPDX-License-Identifier: Apache-2.0
// FlashInfer 91bda04 SM120 fallback tactic, instantiated from the vendored
// Apache-2.0 template without reconstructing the kernel locally.
#include "linear_attention/gdn_flashinfer_cutlass.h"

#include <dlfcn.h>

#include <stdexcept>
#include <string>

#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/fp4_gemm_template_sm120.h"

namespace flashinfer::gemm {
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 1, 1,
                                     _1SM, false)
}  // namespace flashinfer::gemm

namespace rocket::qwen38::linear_attention {
namespace {

using ImportedLauncher = std::size_t (*)(
    void*, const void*, const void*, const void*, const void*, const float*,
    int, int, int, int, flashinfer::gemm::CutlassGemmConfig, char*,
    std::size_t, cudaStream_t, int*);

constexpr ImportedLauncher kImportedLauncher =
    &flashinfer::gemm::genericFp4GemmKernelLauncher<
        __nv_bfloat16, cute::Int<128>, cute::Int<128>, cute::Int<256>,
        cute::Int<1>, cute::Int<1>, cute::Int<1>, flashinfer::gemm::_1SM,
        false>;

using ImportedRawRunner = std::size_t (*)(
    void*, const void*, const void*, const void*, const void*, const float*,
    int, int, int, int, char*, std::size_t, cudaStream_t, const char*);
constexpr ImportedRawRunner kImportedRawRunner =
    &flashinfer::gemm::runFp4GemmImpl<
        flashinfer::gemm::Fp4Gemm___nv_bfloat16_128_128_256false>;

flashinfer::gemm::CutlassGemmConfig fallback_config() {
  return {flashinfer::gemm::CutlassTileConfigSM120::CtaShape128x128x128B,
          flashinfer::gemm::MainloopScheduleType::AUTO,
          flashinfer::gemm::EpilogueScheduleType::AUTO,
          flashinfer::gemm::ClusterShape::ClusterShape_1x1x1,
          /*swap_ab=*/false, /*use_stream_k=*/false};
}

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

}  // namespace

const char* gdn_flashinfer_cutlass_fallback_symbol() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(kImportedRawRunner), &info) == 0 ||
      !info.dli_sname) {
    throw std::runtime_error(
        "resolve FlashInfer CUTLASS fallback compiler ABI symbol");
  }
  return info.dli_sname;
}

struct GdnFlashInferCutlassGemm::Impl {
  int m = 0;
  int n = 0;
  int k = 0;
  const std::uint8_t* packed_a = nullptr;
  const std::uint8_t* sfa = nullptr;
  const std::uint8_t* packed_b = nullptr;
  const std::uint8_t* sfb = nullptr;
  const float* alpha = nullptr;
  __nv_bfloat16* output = nullptr;
  char* workspace = nullptr;
  std::size_t workspace_bytes = 0;
};

GdnFlashInferCutlassGemm::GdnFlashInferCutlassGemm() : impl_(new Impl) {}

GdnFlashInferCutlassGemm::~GdnFlashInferCutlassGemm() {
  if (impl_) cudaFree(impl_->workspace);
  delete impl_;
}

void GdnFlashInferCutlassGemm::init(
    int m, int n, int k, const std::uint8_t* packed_a,
    const std::uint8_t* sfa, const std::uint8_t* packed_b,
    const std::uint8_t* sfb, const float* alpha, __nv_bfloat16* output) {
  if (!impl_ || m <= 0 || n <= 0 || k <= 0 || !packed_a || !sfa ||
      !packed_b || !sfb || !alpha || !output) {
    throw std::invalid_argument("FlashInfer CUTLASS GDN contract changed");
  }
  impl_->m = m;
  impl_->n = n;
  impl_->k = k;
  impl_->packed_a = packed_a;
  impl_->sfa = sfa;
  impl_->packed_b = packed_b;
  impl_->sfb = sfb;
  impl_->alpha = alpha;
  impl_->output = output;
  impl_->workspace_bytes = kImportedLauncher(
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, 1,
      fallback_config(), nullptr, 0, nullptr, nullptr);
  if (impl_->workspace_bytes != 0) {
    cuda_check(cudaMalloc(&impl_->workspace, impl_->workspace_bytes),
               "malloc FlashInfer CUTLASS GDN workspace");
  }
}

void GdnFlashInferCutlassGemm::run(cudaStream_t stream) {
  if (!impl_ || !stream || !impl_->packed_a) {
    throw std::invalid_argument("FlashInfer CUTLASS GDN launch changed");
  }
  kImportedLauncher(
      impl_->output, impl_->packed_a, impl_->packed_b, impl_->sfa, impl_->sfb,
      impl_->alpha, impl_->m, impl_->n, impl_->k, 1, fallback_config(),
      impl_->workspace, impl_->workspace_bytes, stream, nullptr);
}

}  // namespace rocket::qwen38::linear_attention

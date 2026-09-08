// SPDX-License-Identifier: Apache-2.0
// FlashInfer 91bda04 SM120 fallback tactic, instantiated from the vendored
// Apache-2.0 template without reconstructing the kernel locally.
#include "linear_attention/gdn_flashinfer_cutlass.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/fp4_gemm_template_sm120.h"

namespace flashinfer::gemm {
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 1, 1,
                                     _1SM, false)

// Keep the pinned fallback mainloop, tile, scheduler, and output type. Only
// replace its scalar linear-combination epilogue with CUTLASS 4.5's named
// per-column alpha visitor.
struct GdnQkvzPerColumnDevice {
  using OutElementType = flashinfer::cutlass_dtype<__nv_bfloat16>::type;
  using Arch = cutlass::arch::Sm120;
  using ThreadBlockShape = cute::Shape<cute::_128, cute::_128, cute::_256>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementC = void;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentD =
      128 / cutlass::sizeof_bits<OutElementType>::value;
  using FusionOperation =
      cutlass::epilogue::fusion::PerColLinCombPerColBiasEltAct<
          cutlass::epilogue::thread::Identity, OutElementType, ElementCompute,
          ElementCompute, ElementC, ElementCompute,
          /*AlignmentBias=*/4, /*AlignmentScalar=*/4>;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          Arch, cutlass::arch::OpClassTensorOp, ThreadBlockShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutD, AlignmentD,
          OutElementType, LayoutD, AlignmentD,
          cutlass::epilogue::TmaWarpSpecialized,
          FusionOperation>::CollectiveOp;
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          Arch, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB,
          LayoutB, AlignmentB, ElementAccumulator, ThreadBlockShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop,
      CollectiveEpilogue, cutlass::gemm::StaticPersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
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

struct GdnFlashInferCutlassPerColumnGemm::Impl {
  int m = 0;
  int n = 0;
  int k = 0;
  const std::uint8_t* packed_a = nullptr;
  const std::uint8_t* sfa = nullptr;
  const std::uint8_t* packed_b = nullptr;
  const std::uint8_t* sfb = nullptr;
  float* alpha = nullptr;
  __nv_bfloat16* output = nullptr;
  char* workspace = nullptr;
  std::size_t workspace_bytes = 0;
};

GdnFlashInferCutlassPerColumnGemm::GdnFlashInferCutlassPerColumnGemm()
    : impl_(new Impl) {}

GdnFlashInferCutlassPerColumnGemm::~GdnFlashInferCutlassPerColumnGemm() {
  if (impl_) {
    cudaFree(impl_->workspace);
    cudaFree(impl_->alpha);
  }
  delete impl_;
}

void GdnFlashInferCutlassPerColumnGemm::init(
    int m, int n, int k, int split, float first_scale, float second_scale,
    const std::uint8_t* packed_a, const std::uint8_t* sfa,
    const std::uint8_t* packed_b, const std::uint8_t* sfb,
    __nv_bfloat16* output) {
  if (!impl_ || m <= 0 || n <= 0 || k <= 0 || split <= 0 || split >= n ||
      !std::isfinite(first_scale) || first_scale <= 0.0F ||
      !std::isfinite(second_scale) || second_scale <= 0.0F || !packed_a ||
      !sfa || !packed_b || !sfb || !output || impl_->packed_a) {
    throw std::invalid_argument(
        "FlashInfer per-column QKVZ contract changed");
  }
  std::vector<float> alpha(static_cast<std::size_t>(n), second_scale);
  std::fill_n(alpha.begin(), split, first_scale);
  cuda_check(cudaMalloc(&impl_->alpha, alpha.size() * sizeof(float)),
             "malloc FlashInfer per-column QKVZ alpha");
  cuda_check(cudaMemcpy(impl_->alpha, alpha.data(), alpha.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
             "copy FlashInfer per-column QKVZ alpha");
  impl_->m = m;
  impl_->n = n;
  impl_->k = k;
  impl_->packed_a = packed_a;
  impl_->sfa = sfa;
  impl_->packed_b = packed_b;
  impl_->sfb = sfb;
  impl_->output = output;
  using Gemm = flashinfer::gemm::GdnQkvzPerColumnDevice::Gemm;
  impl_->workspace_bytes = flashinfer::gemm::runFp4GemmImpl<Gemm>(
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, 1,
      nullptr, 0, nullptr, " per-column QKVZ");
  if (impl_->workspace_bytes != 0) {
    cuda_check(cudaMalloc(&impl_->workspace, impl_->workspace_bytes),
               "malloc FlashInfer per-column QKVZ workspace");
  }
}

void GdnFlashInferCutlassPerColumnGemm::run(cudaStream_t stream) {
  if (!impl_ || !stream || !impl_->packed_a) {
    throw std::invalid_argument(
        "FlashInfer per-column QKVZ launch changed");
  }
  using Gemm = flashinfer::gemm::GdnQkvzPerColumnDevice::Gemm;
  flashinfer::gemm::runFp4GemmImpl<Gemm>(
      impl_->output, impl_->packed_a, impl_->packed_b, impl_->sfa, impl_->sfb,
      impl_->alpha, impl_->m, impl_->n, impl_->k, 1, impl_->workspace,
      impl_->workspace_bytes, stream, " per-column QKVZ");
}

}  // namespace rocket::qwen38::linear_attention

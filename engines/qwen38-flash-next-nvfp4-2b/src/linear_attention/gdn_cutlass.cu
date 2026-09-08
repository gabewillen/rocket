// SPDX-License-Identifier: Apache-2.0
// Reuses the fixed non-grouped SM121 block-scaled CUTLASS structure proven in
// projection/cutlass_qkv.cu. It removes runtime shapes and fuses only matrices
// whose row boundaries preserve the pinned SFB tile layout.
#include "linear_attention/gdn_cutlass.h"

#include "linear_attention/gdn_b12x_aot.h"
#include "linear_attention/gdn_flashinfer_cutlass.h"
#include "linear_attention/gdn_flashinfer_wheel.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace rocket::qwen38::linear_attention {
namespace {

thread_local std::string graph_last_error;

constexpr int kM = 16;
constexpr int kInputK = 2'560;
constexpr int kQkvN = 5'120;
constexpr int kZN = 3'072;
constexpr int kQkvzN = kQkvN + kZN;
constexpr int kBN = 24;
constexpr int kAN = 24;
constexpr int kBaN = kBN + kAN;
constexpr int kOutputK = 3'072;
constexpr int kOutputN = 2'560;
constexpr float kOutputActivationGlobal = 1.0F / 256.0F;
constexpr int kInputSfaBytes = 128 * ((kInputK / 16 + 3) / 4) * 4;
constexpr int kOutputSfaBytes = 128 * ((kOutputK / 16 + 3) / 4) * 4;
constexpr int kBaSfbBytes = 128 * (kInputK / 16);

using ElementInput = cutlass::float_e2m1_t;
using ElementA = cutlass::nv_float4_t<ElementInput>;
using ElementB = cutlass::nv_float4_t<ElementInput>;
using ElementD = cutlass::bfloat16_t;
using ElementC = void;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutCTag, 8, ElementD, LayoutCTag, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag, 32, ElementB, LayoutBTag,
        32, ElementAccumulator, ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using ScaleConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;

__constant__ float kE2M1[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                               2.0F, 3.0F, 4.0F, 6.0F};

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

__device__ __forceinline__ float e4m3_to_float(std::uint8_t bits) {
  const std::uint32_t sign = (bits & 0x80U) ? 0x80000000U : 0U;
  const std::uint32_t exp = (bits >> 3) & 0x0fU;
  const std::uint32_t mant = bits & 7U;
  if (exp == 0) {
    if (mant == 0) return __uint_as_float(sign);
    const float value = static_cast<float>(mant) * (1.0F / 8.0F) * 0.015625F;
    return sign ? -value : value;
  }
  return __uint_as_float(sign | ((exp + 120U) << 23) | (mant << 20));
}

__device__ __forceinline__ std::uint8_t float_to_e2m1(float value) {
  const std::uint8_t sign = value < 0.0F ? 8U : 0U;
  const float magnitude = fabsf(value);
  int best = 0;
  float best_error = magnitude;
#pragma unroll
  for (int code = 0; code < 8; ++code) {
    const float error = fabsf(magnitude - kE2M1[code]);
    if (error < best_error || (error == best_error && (code & 1) == 0)) {
      best_error = error;
      best = code;
    }
  }
  return static_cast<std::uint8_t>(sign | best);
}

template <int K>
__global__ void quantize_fixed(std::uint8_t* packed, std::uint8_t* scales,
                               const __nv_bfloat16* input,
                               float activation_global) {
  const int row = blockIdx.x;
  for (int block = threadIdx.x; block < K / 16; block += blockDim.x) {
    float values[16];
    float amax = 0.0F;
#pragma unroll
    for (int item = 0; item < 16; ++item) {
      values[item] = __bfloat162float(input[row * K + block * 16 + item]);
      amax = fmaxf(amax, fabsf(values[item]));
    }
    const std::uint8_t scale =
        amax > 0.0F
            ? __nv_cvt_float_to_fp8(amax / (6.0F * activation_global),
                                    __NV_SATFINITE, __NV_E4M3)
            : 0;
    scales[prefill_sfa_offset(row, block, K / 16)] = scale;
    const float combined = e4m3_to_float(scale) * activation_global;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const std::uint8_t lo = combined > 0.0F
          ? float_to_e2m1(values[item * 2] / combined) : 0;
      const std::uint8_t hi = combined > 0.0F
          ? float_to_e2m1(values[item * 2 + 1] / combined) : 0;
      packed[row * (K / 2) + block * 8 + item] = lo | (hi << 4);
    }
  }
}

struct alignas(32) PackedBf16x16 {
  __nv_bfloat162 values[8];
};

struct PackedE2m1x16 {
  std::uint32_t lo;
  std::uint32_t hi;
};

__device__ __forceinline__ float reciprocal_approximate_ftz(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ void load_bf16x16(PackedBf16x16& value,
                                              const void* address,
                                              bool valid) {
  auto* words = reinterpret_cast<std::uint32_t*>(&value);
  asm volatile(
      "{\n"
      " .reg .pred p;\n"
      " setp.ne.u32 p, %8, 0;\n"
      " mov.u32 %0, 0; mov.u32 %1, 0; mov.u32 %2, 0; mov.u32 %3, 0;\n"
      " mov.u32 %4, 0; mov.u32 %5, 0; mov.u32 %6, 0; mov.u32 %7, 0;\n"
      " @p ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%9];\n"
      "}\n"
      : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3]),
        "=r"(words[4]), "=r"(words[5]), "=r"(words[6]), "=r"(words[7])
      : "r"(static_cast<int>(valid)), "l"(address));
}

__device__ __forceinline__ PackedE2m1x16 pack_e2m1x16(float2 (&values)[8]) {
  PackedE2m1x16 result;
  asm volatile(
      "{\n"
      " .reg .b8 b0; .reg .b8 b1; .reg .b8 b2; .reg .b8 b3;\n"
      " .reg .b8 b4; .reg .b8 b5; .reg .b8 b6; .reg .b8 b7;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
      " mov.b32 %0, {b0,b1,b2,b3}; mov.b32 %1, {b4,b5,b6,b7};\n"
      "}\n"
      : "=r"(result.lo), "=r"(result.hi)
      : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x),
        "f"(values[1].y), "f"(values[2].x), "f"(values[2].y),
        "f"(values[3].x), "f"(values[3].y), "f"(values[4].x),
        "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
        "f"(values[6].x), "f"(values[6].y), "f"(values[7].x),
        "f"(values[7].y));
  return result;
}

// Fixed-K specialization of vLLM scaled_fp4_quant at g8e685d198. GB10 has
// 20 SMs and the source kernel caps occupancy at four blocks per SM.
__global__ __launch_bounds__(512, 3) void quantize_prefill_b12x(
    std::uint8_t* packed, std::uint8_t* scales,
    const __nv_bfloat16* input, int tokens, const float* global_scale_ptr) {
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  constexpr int kValuesPerThread = 16;
  constexpr int kScaleColumns = kInputK / kValuesPerThread;
  const int column = blockIdx.y * blockDim.x + threadIdx.x;
  const int padded_rows = ((tokens + 127) / 128) * 128;
  const float global_scale = global_scale_ptr ? *global_scale_ptr : 1.0F;
  for (int row = blockIdx.x; row < padded_rows; row += gridDim.x) {
    if (column >= kScaleColumns) continue;
    const bool valid = row < tokens;
    PackedBf16x16 source;
    load_bf16x16(source, input + row * kInputK + column * kValuesPerThread,
                 valid);
    auto local_max = __habs2(source.values[0]);
#pragma unroll
    for (int index = 1; index < 8; ++index)
      local_max = __hmax2(local_max, __habs2(source.values[index]));
    const float vector_max = static_cast<float>(__hmax(local_max.x, local_max.y));
    float scale_value =
        global_scale * (vector_max * reciprocal_approximate_ftz(6.0F));
    __nv_fp8_e4m3 scale_fp8(scale_value);
    scales[prefill_sfa_offset(row, column, kScaleColumns)] =
        reinterpret_cast<const std::uint8_t&>(scale_fp8);
    scale_value = static_cast<float>(scale_fp8);
    const float output_scale =
        scale_value != 0.0F
            ? reciprocal_approximate_ftz(
                  scale_value * reciprocal_approximate_ftz(global_scale))
            : 0.0F;
    float2 converted[8];
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      converted[index] = __bfloat1622float2(source.values[index]);
      converted[index].x *= output_scale;
      converted[index].y *= output_scale;
    }
    if (valid) {
      const auto result = pack_e2m1x16(converted);
      reinterpret_cast<std::uint64_t*>(packed)[
          static_cast<std::size_t>(row) * (kInputK / 16) + column] =
          (static_cast<std::uint64_t>(result.hi) << 32) | result.lo;
    }
  }
}

void launch_prefill_input_quant(std::uint8_t* packed, std::uint8_t* scales,
                                const __nv_bfloat16* input, int tokens,
                                const float* global_scale,
                                GdnPrefillInputBackend backend,
                                cudaStream_t stream) {
  if (backend == GdnPrefillInputBackend::kB12x) {
    constexpr int kGb10Sms = 20;
    constexpr int kBlocksPerSm = 4;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kGb10Sms * kBlocksPerSm, 1);
    config.blockDim = dim3(kInputK / 16);
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    cuda_check(cudaLaunchKernelEx(&config, quantize_prefill_b12x, packed,
                                  scales, input, tokens, global_scale),
               "launch vLLM B12X input quantization");
  } else {
    quantize_fixed<kInputK><<<tokens, 256, 0, stream>>>(packed, scales, input,
                                                        1.0F);
  }
}

__global__ void fuse_ba_scales(const std::uint8_t* b,
                               const std::uint8_t* a,
                               std::uint8_t* fused) {
  const int sf = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (row >= kBaN || sf >= kInputK / 16) return;
  const int source_row = row < kBN ? row : row - kBN;
  const auto* source = row < kBN ? b : a;
  fused[prefill_sfa_offset(row, sf, kInputK / 16)] =
      source[prefill_sfa_offset(source_row, sf, kInputK / 16)];
}

__global__ void scale_projection(__nv_bfloat16* output, int n,
                                 int split, float first, float second) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (column >= n) return;
  const int index = row * n + column;
  const float scale = column < split ? first : second;
  output[index] = __float2bfloat16(__bfloat162float(output[index]) * scale);
}

struct FixedGemm {
  cutlass::DeviceAllocation<std::uint8_t> workspace;
  Gemm gemm;

  void init(int m, int n, int k, const std::uint8_t* a, const std::uint8_t* sfa,
            const std::uint8_t* b, const std::uint8_t* sfb,
            __nv_bfloat16* output) {
    StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    LayoutSFA la = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    LayoutSFB lb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
        {reinterpret_cast<const ElementInput*>(a), sa,
         reinterpret_cast<const ElementInput*>(b), sb,
         reinterpret_cast<const ElementSF*>(sfa), la,
         reinterpret_cast<const ElementSF*>(sfb), lb},
        {{1.0F, 0.0F}, nullptr, sd, reinterpret_cast<ElementD*>(output), sd}};
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS fixed GDN shape is unsupported");
    }
    workspace.reset(Gemm::get_workspace_size(args));
    if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS fixed GDN initialization failed");
    }
  }
};

void copy(void* destination, const void* source, std::size_t bytes,
          const char* operation) {
  cuda_check(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice),
             operation);
}

}  // namespace

struct CutlassGdnGraph::Impl {
  Impl(int selected_device, GdnWeights selected_weights)
      : device(selected_device), globals(selected_weights) {}

  int device;
  GdnWeights globals;
  std::uint8_t *input_packed = nullptr, *input_sfa = nullptr;
  std::uint8_t *qkvz_weight = nullptr, *qkvz_scale = nullptr;
  std::uint8_t *ba_weight = nullptr, *ba_scale = nullptr;
  std::uint8_t *output_packed = nullptr, *output_sfa = nullptr;
  std::uint8_t *output_weight = nullptr, *output_scale = nullptr;
  __nv_bfloat16 *qkvz = nullptr, *ba = nullptr, *projected = nullptr;
  FixedGemm qkvz_gemm, ba_gemm, output_gemm;
  std::uint8_t *verify_input_packed = nullptr, *verify_input_sfa = nullptr;
  std::uint8_t *verify_output_packed = nullptr, *verify_output_sfa = nullptr;
  __nv_bfloat16 *verify_qkvz = nullptr, *verify_ba = nullptr,
                 *verify_projected = nullptr;
  FixedGemm verify_qkvz_gemm, verify_ba_gemm, verify_output_gemm;
  std::unique_ptr<CorePlan> core;

  ~Impl() {
    cudaSetDevice(device);
    cudaFree(verify_projected); cudaFree(verify_ba); cudaFree(verify_qkvz);
    cudaFree(verify_output_sfa); cudaFree(verify_output_packed);
    cudaFree(verify_input_sfa); cudaFree(verify_input_packed);
    cudaFree(projected); cudaFree(ba); cudaFree(qkvz);
    cudaFree(output_scale); cudaFree(output_weight);
    cudaFree(output_sfa); cudaFree(output_packed);
    cudaFree(ba_scale); cudaFree(ba_weight);
    cudaFree(qkvz_scale); cudaFree(qkvz_weight);
    cudaFree(input_sfa); cudaFree(input_packed);
  }
};

CutlassGdnGraph::CutlassGdnGraph(int device, GdnWeights weights)
    : impl_(new Impl(device, weights)) {
  const Nvfp4Matrix matrices[] = {weights.qkv, weights.z, weights.b,
                                  weights.a, weights.output};
  if (device < 0 || !weights.conv || !weights.a_log || !weights.dt_bias ||
      !weights.norm) {
    delete impl_; impl_ = nullptr;
    throw std::invalid_argument("authenticated GDN weight pointers are required");
  }
  for (const auto& matrix : matrices) {
    if (!matrix.weight || !matrix.scale || !std::isfinite(matrix.global_scale) ||
        matrix.global_scale <= 0.0F) {
      delete impl_; impl_ = nullptr;
      throw std::invalid_argument("authenticated NVFP4 GDN matrix is invalid");
    }
  }
  try {
    cuda_check(cudaSetDevice(device), "cudaSetDevice");
    cuda_check(cudaMalloc(&impl_->input_packed, kM * kInputK / 2), "malloc input A");
    cuda_check(cudaMalloc(&impl_->input_sfa, kInputSfaBytes), "malloc input SFA");
    cuda_check(cudaMalloc(&impl_->qkvz_weight,
                          static_cast<std::size_t>(kQkvzN) * kInputK / 2),
               "malloc QKVZ B");
    cuda_check(cudaMalloc(&impl_->qkvz_scale,
                          static_cast<std::size_t>(kQkvzN) * kInputK / 16),
               "malloc QKVZ SFB");
    cuda_check(cudaMalloc(&impl_->ba_weight,
                          static_cast<std::size_t>(kBaN) * kInputK / 2),
               "malloc BA B");
    cuda_check(cudaMalloc(&impl_->ba_scale, kBaSfbBytes), "malloc BA SFB");
    cuda_check(cudaMalloc(&impl_->output_packed, kM * kOutputK / 2),
               "malloc output A");
    cuda_check(cudaMalloc(&impl_->output_sfa, kOutputSfaBytes),
               "malloc output SFA");
    cuda_check(cudaMalloc(&impl_->output_weight,
                          static_cast<std::size_t>(kOutputN) * kOutputK / 2),
               "malloc output B");
    cuda_check(cudaMalloc(&impl_->output_scale,
                          static_cast<std::size_t>(kOutputN) * kOutputK / 16),
               "malloc output SFB");
    cuda_check(cudaMalloc(&impl_->qkvz,
                          static_cast<std::size_t>(kM) * kQkvzN * 2),
               "malloc QKVZ output");
    cuda_check(cudaMalloc(&impl_->ba,
                          static_cast<std::size_t>(kM) * kBaN * 2),
               "malloc BA output");
    cuda_check(cudaMalloc(&impl_->projected,
                          static_cast<std::size_t>(kM) * kOutputN * 2),
               "malloc projected output");
    cuda_check(cudaMalloc(&impl_->verify_input_packed,
                          kMaxVerifierRows * kInputK / 2),
               "malloc verifier input A");
    cuda_check(cudaMalloc(&impl_->verify_input_sfa, kInputSfaBytes),
               "malloc verifier input SFA");
    cuda_check(cudaMalloc(&impl_->verify_output_packed,
                          kMaxVerifierRows * kOutputK / 2),
               "malloc verifier output A");
    cuda_check(cudaMalloc(&impl_->verify_output_sfa, kOutputSfaBytes),
               "malloc verifier output SFA");
    cuda_check(cudaMalloc(&impl_->verify_qkvz,
                          static_cast<std::size_t>(kMaxVerifierRows) *
                              kQkvzN * 2),
               "malloc verifier QKVZ output");
    cuda_check(cudaMalloc(&impl_->verify_ba,
                          static_cast<std::size_t>(kMaxVerifierRows) * kBaN *
                              2),
               "malloc verifier BA output");
    cuda_check(cudaMalloc(&impl_->verify_projected,
                          static_cast<std::size_t>(kMaxVerifierRows) *
                              kOutputN * 2),
               "malloc verifier projected output");

    const std::size_t qkv_w = static_cast<std::size_t>(kQkvN) * kInputK / 2;
    const std::size_t qkv_s = static_cast<std::size_t>(kQkvN) * kInputK / 16;
    copy(impl_->qkvz_weight, weights.qkv.weight, qkv_w, "copy QKV weight");
    copy(impl_->qkvz_weight + qkv_w, weights.z.weight,
         static_cast<std::size_t>(kZN) * kInputK / 2, "copy Z weight");
    copy(impl_->qkvz_scale, weights.qkv.scale, qkv_s, "copy QKV scale");
    copy(impl_->qkvz_scale + qkv_s, weights.z.scale,
         static_cast<std::size_t>(kZN) * kInputK / 16, "copy Z scale");
    const std::size_t ba_w = static_cast<std::size_t>(kBN) * kInputK / 2;
    copy(impl_->ba_weight, weights.b.weight, ba_w, "copy B weight");
    copy(impl_->ba_weight + ba_w, weights.a.weight, ba_w, "copy A weight");
    fuse_ba_scales<<<dim3((kInputK / 16 + 255) / 256, kBaN), 256>>>(
        weights.b.scale, weights.a.scale, impl_->ba_scale);
    cuda_check(cudaGetLastError(), "fuse BA scale layout");
    copy(impl_->output_weight, weights.output.weight,
         static_cast<std::size_t>(kOutputN) * kOutputK / 2,
         "copy output weight");
    copy(impl_->output_scale, weights.output.scale,
         static_cast<std::size_t>(kOutputN) * kOutputK / 16,
         "copy output scale");
    cuda_check(cudaDeviceSynchronize(), "synchronize immutable GDN weights");

    impl_->qkvz_gemm.init(kM, kQkvzN, kInputK, impl_->input_packed,
                          impl_->input_sfa, impl_->qkvz_weight,
                          impl_->qkvz_scale, impl_->qkvz);
    impl_->ba_gemm.init(kM, kBaN, kInputK, impl_->input_packed,
                        impl_->input_sfa, impl_->ba_weight,
                        impl_->ba_scale, impl_->ba);
    impl_->core = std::make_unique<CorePlan>(
        device, weights.conv, weights.a_log, weights.dt_bias, weights.norm);
    impl_->output_gemm.init(kM, kOutputN, kOutputK, impl_->output_packed,
                            impl_->output_sfa, impl_->output_weight,
                            impl_->output_scale, impl_->projected);
    impl_->verify_qkvz_gemm.init(
        kMaxVerifierRows, kQkvzN, kInputK, impl_->verify_input_packed,
        impl_->verify_input_sfa, impl_->qkvz_weight, impl_->qkvz_scale,
        impl_->verify_qkvz);
    impl_->verify_ba_gemm.init(
        kMaxVerifierRows, kBaN, kInputK, impl_->verify_input_packed,
        impl_->verify_input_sfa, impl_->ba_weight, impl_->ba_scale,
        impl_->verify_ba);
    impl_->verify_output_gemm.init(
        kMaxVerifierRows, kOutputN, kOutputK, impl_->verify_output_packed,
        impl_->verify_output_sfa, impl_->output_weight, impl_->output_scale,
        impl_->verify_projected);
  } catch (...) {
    delete impl_; impl_ = nullptr;
    throw;
  }
}

CutlassGdnGraph::~CutlassGdnGraph() { delete impl_; }

std::uint64_t CutlassGdnGraph::logical_bytes_per_row(int m) const noexcept {
  if (!allowed_m(m)) return 0;
  constexpr std::uint64_t weight_bytes =
      static_cast<std::uint64_t>(kQkvzN + kBaN) * kInputK * 9 / 16 +
      static_cast<std::uint64_t>(kOutputN) * kOutputK * 9 / 16 +
      static_cast<std::uint64_t>(kQkvWidth) * kConvKernel * 2 +
      static_cast<std::uint64_t>(kValueHeads * 2 + kHeadDim) * 2;
  constexpr std::uint64_t state_bytes =
      2ULL * kValueHeads * kHeadDim * kHeadDim * sizeof(float) +
      6ULL * kQkvWidth * sizeof(__nv_bfloat16);
  return weight_bytes / static_cast<std::uint64_t>(m) + state_bytes +
         2ULL * kInputK + 2ULL * kOutputN;
}

void CutlassGdnGraph::launch(
    const __nv_bfloat16* block_input, __nv_bfloat16* conv_state,
    float* recurrent_state, const std::int32_t* state_indices, int m,
    cudaStream_t stream) {
  if (!block_input || !conv_state || !recurrent_state || !state_indices ||
      !allowed_m(m) || !stream) {
    throw decode::DecodeExecutionContractError(
        "fixed Qwen GDN graph arguments changed");
  }
  quantize_fixed<kInputK><<<kM, 256, 0, stream>>>(
      impl_->input_packed, impl_->input_sfa, block_input, 1.0F);
  if (impl_->qkvz_gemm.gemm.run(stream) != cutlass::Status::kSuccess ||
      impl_->ba_gemm.gemm.run(stream) != cutlass::Status::kSuccess) {
    throw std::runtime_error("fixed Qwen GDN input projection failed");
  }
  scale_projection<<<dim3((kQkvzN + 255) / 256, kM), 256, 0, stream>>>(
      impl_->qkvz, kQkvzN, kQkvN, impl_->globals.qkv.global_scale,
      impl_->globals.z.global_scale);
  scale_projection<<<dim3((kBaN + 255) / 256, kM), 256, 0, stream>>>(
      impl_->ba, kBaN, kBN, impl_->globals.b.global_scale,
      impl_->globals.a.global_scale);
  impl_->core->launch(
      impl_->qkvz, impl_->ba, conv_state,
      static_cast<std::size_t>(kConvStateRows) * kQkvWidth, recurrent_state,
      static_cast<std::size_t>(kValueHeads) * kHeadDim * kHeadDim,
      state_indices, m, stream);
  quantize_fixed<kOutputK><<<kM, 256, 0, stream>>>(
      impl_->output_packed, impl_->output_sfa, impl_->core->output(),
      kOutputActivationGlobal);
  if (impl_->output_gemm.gemm.run(stream) != cutlass::Status::kSuccess) {
    throw std::runtime_error("fixed Qwen GDN output projection failed");
  }
  scale_projection<<<dim3((kOutputN + 255) / 256, kM), 256, 0, stream>>>(
      impl_->projected, kOutputN, kOutputN,
      impl_->globals.output.global_scale * kOutputActivationGlobal,
      impl_->globals.output.global_scale * kOutputActivationGlobal);
  cuda_check(cudaGetLastError(), "fixed Qwen GDN graph launch");
}

const __nv_bfloat16* CutlassGdnGraph::projected_output() const noexcept {
  return impl_ ? impl_->projected : nullptr;
}

void CutlassGdnGraph::launch_verifier(
    const __nv_bfloat16* input, __nv_bfloat16* dense_conv_state,
    float* dense_recurrent_state, __nv_bfloat16* prefix_conv_state,
    float* prefix_recurrent_state, int sequences, int verify_width,
    cudaStream_t stream) {
  const int rows = sequences * verify_width;
  if (!input || !dense_conv_state || !dense_recurrent_state ||
      !prefix_conv_state || !prefix_recurrent_state || !allowed_m(sequences) ||
      verify_width < 1 || verify_width > 8 || rows > kMaxVerifierRows ||
      !stream) {
    throw decode::DecodeExecutionContractError(
        "fixed Qwen GDN verifier arguments changed");
  }
  // The input owner zeroes the inactive tail. One M128 projection is cheaper
  // than verify_width M16 projections and leaves causal order to the core.
  quantize_fixed<kInputK><<<kMaxVerifierRows, 256, 0, stream>>>(
      impl_->verify_input_packed, impl_->verify_input_sfa, input, 1.0F);
  if (impl_->verify_qkvz_gemm.gemm.run(stream) != cutlass::Status::kSuccess ||
      impl_->verify_ba_gemm.gemm.run(stream) != cutlass::Status::kSuccess) {
    throw std::runtime_error("fixed Qwen GDN verifier input projection failed");
  }
  scale_projection<<<dim3((kQkvzN + 255) / 256, kMaxVerifierRows), 256, 0,
                     stream>>>(
      impl_->verify_qkvz, kQkvzN, kQkvN, impl_->globals.qkv.global_scale,
      impl_->globals.z.global_scale);
  scale_projection<<<dim3((kBaN + 255) / 256, kMaxVerifierRows), 256, 0,
                     stream>>>(
      impl_->verify_ba, kBaN, kBN, impl_->globals.b.global_scale,
      impl_->globals.a.global_scale);
  impl_->core->launch_verifier(
      impl_->verify_qkvz, impl_->verify_ba, dense_conv_state,
      dense_recurrent_state, prefix_conv_state, prefix_recurrent_state,
      sequences, verify_width, stream);
  quantize_fixed<kOutputK><<<kMaxVerifierRows, 256, 0, stream>>>(
      impl_->verify_output_packed, impl_->verify_output_sfa,
      impl_->core->output(), kOutputActivationGlobal);
  if (impl_->verify_output_gemm.gemm.run(stream) != cutlass::Status::kSuccess) {
    throw std::runtime_error("fixed Qwen GDN verifier output projection failed");
  }
  scale_projection<<<dim3((kOutputN + 255) / 256, kMaxVerifierRows), 256, 0,
                     stream>>>(
      impl_->verify_projected, kOutputN, kOutputN,
      impl_->globals.output.global_scale * kOutputActivationGlobal,
      impl_->globals.output.global_scale * kOutputActivationGlobal);
  cuda_check(cudaGetLastError(), "fixed Qwen GDN verifier launch");
}

const __nv_bfloat16* CutlassGdnGraph::verifier_output() const noexcept {
  return impl_ ? impl_->verify_projected : nullptr;
}

namespace {

struct PrefillProjectionBucket {
  explicit PrefillProjectionBucket(int selected_tokens) : tokens(selected_tokens) {}
  ~PrefillProjectionBucket() {
    cudaFree(reference_ba);
    cudaFree(reference_qkvz);
    cudaFree(reference_ba_sfa);
    cudaFree(reference_ba_packed);
    cudaFree(reference_qkvz_sfa);
    cudaFree(reference_qkvz_packed);
    cudaFree(projected);
    cudaFree(output_sfa);
    cudaFree(output_packed);
    cudaFree(ba);
    cudaFree(qkvz);
    cudaFree(input_sfa);
    cudaFree(input_packed);
  }

  int tokens;
  std::uint8_t* input_packed = nullptr;
  std::uint8_t* input_sfa = nullptr;
  std::uint8_t* output_packed = nullptr;
  std::uint8_t* output_sfa = nullptr;
  __nv_bfloat16* qkvz = nullptr;
  __nv_bfloat16* ba = nullptr;
  __nv_bfloat16* projected = nullptr;
  std::uint8_t* reference_qkvz_packed = nullptr;
  std::uint8_t* reference_qkvz_sfa = nullptr;
  std::uint8_t* reference_ba_packed = nullptr;
  std::uint8_t* reference_ba_sfa = nullptr;
  __nv_bfloat16* reference_qkvz = nullptr;
  __nv_bfloat16* reference_ba = nullptr;
  GdnFlashInferCutlassGemm qkvz_gemm;
  GdnFlashInferCutlassGemm ba_gemm;
  GdnFlashInferWheelGemm wheel_qkvz_gemm;
  GdnFlashInferWheelGemm wheel_ba_gemm;
  FixedGemm output_gemm;
  FixedGemm reference_qkvz_gemm;
  FixedGemm reference_ba_gemm;
};

}  // namespace

struct CutlassGdnPrefillProjection::Impl {
  Impl(int selected_device, GdnWeights selected_weights,
       bool selected_reference, GdnPrefillInputBackend selected_backend,
       std::string_view selected_wheel_shared_object)
      : device(selected_device), globals(selected_weights),
        reference_enabled(selected_reference), input_backend(selected_backend),
        wheel_shared_object(selected_wheel_shared_object) {}
  ~Impl() {
    cudaSetDevice(device);
    cudaFree(projection_alpha);
    cudaFree(output_scale);
    cudaFree(output_weight);
    cudaFree(ba_scale);
    cudaFree(ba_weight);
    cudaFree(qkvz_scale);
    cudaFree(qkvz_weight);
  }

  PrefillProjectionBucket* bucket(int tokens) const noexcept {
    if (!allowed_prefill_tokens(tokens)) return nullptr;
    for (const auto& candidate : buckets) {
      if (candidate && candidate->tokens == tokens) return candidate.get();
    }
    return nullptr;
  }

  int device;
  GdnWeights globals;
  bool reference_enabled;
  GdnPrefillInputBackend input_backend;
  std::string wheel_shared_object;
  std::unique_ptr<GdnB12xAot> b12x;
  float* projection_alpha = nullptr;
  std::uint8_t* qkvz_weight = nullptr;
  std::uint8_t* qkvz_scale = nullptr;
  std::uint8_t* ba_weight = nullptr;
  std::uint8_t* ba_scale = nullptr;
  std::uint8_t* output_weight = nullptr;
  std::uint8_t* output_scale = nullptr;
  std::array<std::unique_ptr<PrefillProjectionBucket>, 2> buckets;
};

CutlassGdnPrefillProjection::CutlassGdnPrefillProjection(int device,
                                                         GdnWeights weights,
                                                         bool enable_reference,
                                                         GdnPrefillInputBackend input_backend,
                                                         std::string_view wheel_shared_object)
    : impl_(new Impl(device, weights, enable_reference, input_backend,
                     wheel_shared_object)) {
  const Nvfp4Matrix matrices[] = {weights.qkv, weights.z, weights.b,
                                  weights.a, weights.output};
  if (device < 0) {
    delete impl_;
    impl_ = nullptr;
    throw std::invalid_argument("prefill projection device is invalid");
  }
  if (input_backend != GdnPrefillInputBackend::kFlashInferCutlass &&
      input_backend != GdnPrefillInputBackend::kB12x &&
      input_backend != GdnPrefillInputBackend::kFlashInferWheelBenchmark) {
    delete impl_;
    impl_ = nullptr;
    throw std::invalid_argument("prefill projection backend is invalid");
  }
  if ((input_backend == GdnPrefillInputBackend::kFlashInferWheelBenchmark) !=
      !wheel_shared_object.empty()) {
    delete impl_;
    impl_ = nullptr;
    throw std::invalid_argument("FlashInfer wheel benchmark path is invalid");
  }
  for (const auto& matrix : matrices) {
    if (!matrix.weight || !matrix.scale || !std::isfinite(matrix.global_scale) ||
        matrix.global_scale <= 0.0F) {
      delete impl_;
      impl_ = nullptr;
      throw std::invalid_argument(
          "authenticated prefill projection matrix is invalid");
    }
  }
  try {
    cuda_check(cudaSetDevice(device), "set prefill projection device");
    if (input_backend == GdnPrefillInputBackend::kB12x) {
      impl_->b12x = std::make_unique<GdnB12xAot>(device);
    }
    constexpr float one = 1.0F;
    cuda_check(cudaMalloc(&impl_->projection_alpha, sizeof(float)),
               "malloc GDN projection alpha");
    cuda_check(cudaMemcpy(impl_->projection_alpha, &one, sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy GDN projection alpha");
    cuda_check(cudaMalloc(&impl_->qkvz_weight,
                          static_cast<std::size_t>(kQkvzN) * kInputK / 2),
               "malloc prefill QKVZ weight");
    cuda_check(cudaMalloc(&impl_->qkvz_scale,
                          static_cast<std::size_t>(kQkvzN) * kInputK / 16),
               "malloc prefill QKVZ scale");
    cuda_check(cudaMalloc(&impl_->ba_weight,
                          static_cast<std::size_t>(kBaN) * kInputK / 2),
               "malloc prefill BA weight");
    cuda_check(cudaMalloc(&impl_->ba_scale, kBaSfbBytes),
               "malloc prefill BA scale");
    cuda_check(cudaMalloc(&impl_->output_weight,
                          static_cast<std::size_t>(kOutputN) * kOutputK / 2),
               "malloc prefill output weight");
    cuda_check(cudaMalloc(&impl_->output_scale,
                          static_cast<std::size_t>(kOutputN) * kOutputK / 16),
               "malloc prefill output scale");

    const std::size_t qkv_weight_bytes =
        static_cast<std::size_t>(kQkvN) * kInputK / 2;
    const std::size_t qkv_scale_bytes =
        static_cast<std::size_t>(kQkvN) * kInputK / 16;
    copy(impl_->qkvz_weight, weights.qkv.weight, qkv_weight_bytes,
         "copy prefill QKV weight");
    copy(impl_->qkvz_weight + qkv_weight_bytes, weights.z.weight,
         static_cast<std::size_t>(kZN) * kInputK / 2,
         "copy prefill Z weight");
    copy(impl_->qkvz_scale, weights.qkv.scale, qkv_scale_bytes,
         "copy prefill QKV scale");
    copy(impl_->qkvz_scale + qkv_scale_bytes, weights.z.scale,
         static_cast<std::size_t>(kZN) * kInputK / 16,
         "copy prefill Z scale");
    const std::size_t ba_weight_bytes =
        static_cast<std::size_t>(kBN) * kInputK / 2;
    copy(impl_->ba_weight, weights.b.weight, ba_weight_bytes,
         "copy prefill B weight");
    copy(impl_->ba_weight + ba_weight_bytes, weights.a.weight, ba_weight_bytes,
         "copy prefill A weight");
    fuse_ba_scales<<<dim3((kInputK / 16 + 255) / 256, kBaN), 256>>>(
        weights.b.scale, weights.a.scale, impl_->ba_scale);
    copy(impl_->output_weight, weights.output.weight,
         static_cast<std::size_t>(kOutputN) * kOutputK / 2,
         "copy prefill output weight");
    copy(impl_->output_scale, weights.output.scale,
         static_cast<std::size_t>(kOutputN) * kOutputK / 16,
         "copy prefill output scale");

    constexpr std::array<int, 2> token_buckets{300, 8'192};
    for (std::size_t index = 0; index < token_buckets.size(); ++index) {
      const int tokens = token_buckets[index];
      auto bucket = std::make_unique<PrefillProjectionBucket>(tokens);
      cuda_check(cudaMalloc(&bucket->input_packed,
                            static_cast<std::size_t>(tokens) * kInputK / 2),
                 "malloc prefill input A");
      cuda_check(cudaMalloc(&bucket->input_sfa,
                            prefill_sfa_bytes(tokens, kInputK)),
                 "malloc prefill input SFA");
      cuda_check(cudaMalloc(&bucket->output_packed,
                            static_cast<std::size_t>(tokens) * kOutputK / 2),
                 "malloc prefill output A");
      cuda_check(cudaMalloc(&bucket->output_sfa,
                            prefill_sfa_bytes(tokens, kOutputK)),
                 "malloc prefill output SFA");
      cuda_check(cudaMalloc(&bucket->qkvz,
                            static_cast<std::size_t>(tokens) * kQkvzN * 2),
                 "malloc prefill QKVZ output");
      cuda_check(cudaMalloc(&bucket->ba,
                            static_cast<std::size_t>(tokens) * kBaN * 2),
                 "malloc prefill BA output");
      cuda_check(cudaMalloc(&bucket->projected,
                            static_cast<std::size_t>(tokens) * kOutputN * 2),
                 "malloc prefill projected output");
      if (input_backend == GdnPrefillInputBackend::kFlashInferCutlass) {
        bucket->qkvz_gemm.init(tokens, kQkvzN, kInputK, bucket->input_packed,
                               bucket->input_sfa, impl_->qkvz_weight,
                               impl_->qkvz_scale, impl_->projection_alpha,
                               bucket->qkvz);
        bucket->ba_gemm.init(tokens, kBaN, kInputK, bucket->input_packed,
                             bucket->input_sfa, impl_->ba_weight,
                             impl_->ba_scale, impl_->projection_alpha,
                             bucket->ba);
      } else if (input_backend ==
                 GdnPrefillInputBackend::kFlashInferWheelBenchmark) {
        bucket->wheel_qkvz_gemm.init(
            impl_->wheel_shared_object, tokens, kQkvzN, kInputK,
            bucket->input_packed, bucket->input_sfa, impl_->qkvz_weight,
            impl_->qkvz_scale, impl_->projection_alpha, bucket->qkvz);
        bucket->wheel_ba_gemm.init(
            impl_->wheel_shared_object, tokens, kBaN, kInputK,
            bucket->input_packed, bucket->input_sfa, impl_->ba_weight,
            impl_->ba_scale, impl_->projection_alpha, bucket->ba);
      }
      bucket->output_gemm.init(tokens, kOutputN, kOutputK,
                               bucket->output_packed, bucket->output_sfa,
                               impl_->output_weight, impl_->output_scale,
                               bucket->projected);
      if (enable_reference) {
        cuda_check(cudaMalloc(&bucket->reference_qkvz_packed,
                              static_cast<std::size_t>(tokens) * kInputK / 2),
                   "malloc reference QKVZ input A");
        cuda_check(cudaMalloc(&bucket->reference_qkvz_sfa,
                              prefill_sfa_bytes(tokens, kInputK)),
                   "malloc reference QKVZ input SFA");
        cuda_check(cudaMalloc(&bucket->reference_ba_packed,
                              static_cast<std::size_t>(tokens) * kInputK / 2),
                   "malloc reference BA input A");
        cuda_check(cudaMalloc(&bucket->reference_ba_sfa,
                              prefill_sfa_bytes(tokens, kInputK)),
                   "malloc reference BA input SFA");
        cuda_check(cudaMalloc(&bucket->reference_qkvz,
                              static_cast<std::size_t>(tokens) * kQkvzN * 2),
                   "malloc reference QKVZ output");
        cuda_check(cudaMalloc(&bucket->reference_ba,
                              static_cast<std::size_t>(tokens) * kBaN * 2),
                   "malloc reference BA output");
        bucket->reference_qkvz_gemm.init(
            tokens, kQkvzN, kInputK, bucket->reference_qkvz_packed,
            bucket->reference_qkvz_sfa, impl_->qkvz_weight,
            impl_->qkvz_scale, bucket->reference_qkvz);
        bucket->reference_ba_gemm.init(
            tokens, kBaN, kInputK, bucket->reference_ba_packed,
            bucket->reference_ba_sfa, impl_->ba_weight, impl_->ba_scale,
            bucket->reference_ba);
      }
      impl_->buckets[index] = std::move(bucket);
    }
    cuda_check(cudaDeviceSynchronize(),
               "synchronize prefill projection construction");
  } catch (...) {
    delete impl_;
    impl_ = nullptr;
    throw;
  }
}

CutlassGdnPrefillProjection::~CutlassGdnPrefillProjection() { delete impl_; }

void CutlassGdnPrefillProjection::launch_input(const __nv_bfloat16* hidden,
                                               int tokens,
                                               cudaStream_t stream) {
  launch_input_quantize(hidden, tokens, stream);
  launch_qkvz(tokens, stream);
  launch_ba(tokens, stream);
}

void CutlassGdnPrefillProjection::launch_input_quantize(
    const __nv_bfloat16* hidden, int tokens, cudaStream_t stream) {
  auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  if (!bucket || !hidden || !stream)
    throw std::invalid_argument("prefill input quantization contract changed");
  launch_prefill_input_quant(bucket->input_packed, bucket->input_sfa, hidden,
                             tokens, impl_->projection_alpha,
                             impl_->input_backend, stream);
  cuda_check(cudaPeekAtLastError(), "launch prefill input quantization");
}

void CutlassGdnPrefillProjection::launch_qkvz(int tokens,
                                              cudaStream_t stream) {
  auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  if (!bucket || !stream)
    throw std::invalid_argument("prefill QKVZ projection contract changed");
  if (impl_->input_backend == GdnPrefillInputBackend::kB12x) {
    impl_->b12x->launch({bucket->input_packed, bucket->input_sfa,
                         impl_->qkvz_weight, impl_->qkvz_scale, bucket->qkvz,
                         impl_->projection_alpha, tokens, kQkvzN, stream});
  } else if (impl_->input_backend ==
             GdnPrefillInputBackend::kFlashInferWheelBenchmark) {
    bucket->wheel_qkvz_gemm.run(stream);
  } else {
    bucket->qkvz_gemm.run(stream);
  }
  scale_projection<<<dim3((kQkvzN + 255) / 256, tokens), 256, 0, stream>>>(
      bucket->qkvz, kQkvzN, kQkvN, impl_->globals.qkv.global_scale,
      impl_->globals.z.global_scale);
  cuda_check(cudaPeekAtLastError(), "launch prefill QKVZ projection");
}

void CutlassGdnPrefillProjection::launch_ba(int tokens,
                                            cudaStream_t stream) {
  auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  if (!bucket || !stream)
    throw std::invalid_argument("prefill BA projection contract changed");
  if (impl_->input_backend == GdnPrefillInputBackend::kB12x) {
    impl_->b12x->launch({bucket->input_packed, bucket->input_sfa,
                         impl_->ba_weight, impl_->ba_scale, bucket->ba,
                         impl_->projection_alpha, tokens, kBaN, stream});
  } else if (impl_->input_backend ==
             GdnPrefillInputBackend::kFlashInferWheelBenchmark) {
    bucket->wheel_ba_gemm.run(stream);
  } else {
    bucket->ba_gemm.run(stream);
  }
  scale_projection<<<dim3((kBaN + 255) / 256, tokens), 256, 0, stream>>>(
      bucket->ba, kBaN, kBN, impl_->globals.b.global_scale,
      impl_->globals.a.global_scale);
  cuda_check(cudaPeekAtLastError(), "launch prefill BA projection");
}

void CutlassGdnPrefillProjection::launch_reference_input(
    const __nv_bfloat16* hidden, int tokens, cudaStream_t stream) {
  auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  if (!bucket || !impl_->reference_enabled || !hidden || !stream)
    throw std::invalid_argument("prefill reference projection contract changed");
  launch_prefill_input_quant(
      bucket->reference_qkvz_packed, bucket->reference_qkvz_sfa, hidden,
      tokens, impl_->projection_alpha, impl_->input_backend, stream);
  if (bucket->reference_qkvz_gemm.gemm.run(stream) != cutlass::Status::kSuccess)
    throw std::runtime_error("reference QKVZ projection failed");
  scale_projection<<<dim3((kQkvzN + 255) / 256, tokens), 256, 0, stream>>>(
      bucket->reference_qkvz, kQkvzN, kQkvN,
      impl_->globals.qkv.global_scale, impl_->globals.z.global_scale);
  launch_prefill_input_quant(
      bucket->reference_ba_packed, bucket->reference_ba_sfa, hidden, tokens,
      impl_->projection_alpha, impl_->input_backend, stream);
  if (bucket->reference_ba_gemm.gemm.run(stream) != cutlass::Status::kSuccess)
    throw std::runtime_error("reference BA projection failed");
  scale_projection<<<dim3((kBaN + 255) / 256, tokens), 256, 0, stream>>>(
      bucket->reference_ba, kBaN, kBN, impl_->globals.b.global_scale,
      impl_->globals.a.global_scale);
  cuda_check(cudaPeekAtLastError(), "launch prefill reference projection");
}

void CutlassGdnPrefillProjection::launch_output(
    const __nv_bfloat16* normalized, int tokens, cudaStream_t stream) {
  auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  if (!bucket || !normalized || !stream)
    throw std::invalid_argument("prefill output projection contract changed");
  quantize_fixed<kOutputK><<<tokens, 256, 0, stream>>>(
      bucket->output_packed, bucket->output_sfa, normalized,
      kOutputActivationGlobal);
  if (bucket->output_gemm.gemm.run(stream) != cutlass::Status::kSuccess)
    throw std::runtime_error("prefill output projection failed");
  scale_projection<<<dim3((kOutputN + 255) / 256, tokens), 256, 0, stream>>>(
      bucket->projected, kOutputN, kOutputN,
      impl_->globals.output.global_scale * kOutputActivationGlobal,
      impl_->globals.output.global_scale * kOutputActivationGlobal);
  cuda_check(cudaPeekAtLastError(), "launch prefill output projection");
}

const __nv_bfloat16* CutlassGdnPrefillProjection::qkvz(
    int tokens) const noexcept {
  const auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  return bucket ? bucket->qkvz : nullptr;
}

const __nv_bfloat16* CutlassGdnPrefillProjection::ba(
    int tokens) const noexcept {
  const auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  return bucket ? bucket->ba : nullptr;
}

const __nv_bfloat16* CutlassGdnPrefillProjection::output(
    int tokens) const noexcept {
  const auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  return bucket ? bucket->projected : nullptr;
}

const std::uint8_t* CutlassGdnPrefillProjection::input_packed(
    int tokens) const noexcept {
  const auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  return bucket ? bucket->input_packed : nullptr;
}

const std::uint8_t* CutlassGdnPrefillProjection::input_sfa(
    int tokens) const noexcept {
  const auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  return bucket ? bucket->input_sfa : nullptr;
}

const std::uint8_t* CutlassGdnPrefillProjection::reference_qkvz_packed(
    int tokens) const noexcept {
  const auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  return bucket ? bucket->reference_qkvz_packed : nullptr;
}

const std::uint8_t* CutlassGdnPrefillProjection::reference_qkvz_sfa(
    int tokens) const noexcept {
  const auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  return bucket ? bucket->reference_qkvz_sfa : nullptr;
}

const std::uint8_t* CutlassGdnPrefillProjection::reference_ba_packed(
    int tokens) const noexcept {
  const auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  return bucket ? bucket->reference_ba_packed : nullptr;
}

const std::uint8_t* CutlassGdnPrefillProjection::reference_ba_sfa(
    int tokens) const noexcept {
  const auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  return bucket ? bucket->reference_ba_sfa : nullptr;
}

const __nv_bfloat16* CutlassGdnPrefillProjection::reference_qkvz(
    int tokens) const noexcept {
  const auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  return bucket ? bucket->reference_qkvz : nullptr;
}

const __nv_bfloat16* CutlassGdnPrefillProjection::reference_ba(
    int tokens) const noexcept {
  const auto* bucket = impl_ ? impl_->bucket(tokens) : nullptr;
  return bucket ? bucket->reference_ba : nullptr;
}

}  // namespace rocket::qwen38::linear_attention

namespace {
template <typename F>
int graph_wrap(F&& fn) noexcept {
  try {
    fn();
    return 0;
  } catch (const std::exception& error) {
    rocket::qwen38::linear_attention::graph_last_error = error.what();
    return 1;
  } catch (...) {
    rocket::qwen38::linear_attention::graph_last_error = "unknown failure";
    return 1;
  }
}
}  // namespace

extern "C" int qwen38_gdn_graph_create(
    int device,
    const std::uint8_t* qkv_weight, const std::uint8_t* qkv_scale,
    float qkv_global, const std::uint8_t* z_weight,
    const std::uint8_t* z_scale, float z_global,
    const std::uint8_t* b_weight, const std::uint8_t* b_scale,
    float b_global, const std::uint8_t* a_weight,
    const std::uint8_t* a_scale, float a_global,
    const std::uint8_t* output_weight, const std::uint8_t* output_scale,
    float output_global, const __nv_bfloat16* conv,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    const __nv_bfloat16* norm, void** graph) {
  return graph_wrap([&] {
    if (!graph) throw std::invalid_argument("graph output is required");
    *graph = nullptr;
    using namespace rocket::qwen38::linear_attention;
    *graph = new CutlassGdnGraph(
        device, {{qkv_weight, qkv_scale, qkv_global},
                 {z_weight, z_scale, z_global},
                 {b_weight, b_scale, b_global},
                 {a_weight, a_scale, a_global},
                 {output_weight, output_scale, output_global},
                 conv, a_log, dt_bias, norm});
  });
}

extern "C" int qwen38_gdn_graph_launch(
    void* graph, const __nv_bfloat16* block_input,
    __nv_bfloat16* conv_state, float* recurrent_state,
    const std::int32_t* state_indices, int m, cudaStream_t stream) {
  return graph_wrap([&] {
    if (!graph) throw std::invalid_argument("GDN graph is required");
    static_cast<rocket::qwen38::linear_attention::CutlassGdnGraph*>(graph)
        ->launch(block_input, conv_state, recurrent_state, state_indices, m,
                 stream);
  });
}

extern "C" int qwen38_gdn_graph_output(
    void* graph, void** output_bf16, std::size_t* elements) {
  return graph_wrap([&] {
    if (!graph || !output_bf16 || !elements) {
      throw std::invalid_argument("GDN graph output arguments are required");
    }
    auto* typed =
        static_cast<rocket::qwen38::linear_attention::CutlassGdnGraph*>(graph);
    *output_bf16 = const_cast<__nv_bfloat16*>(typed->projected_output());
    *elements = 16U * 2'560U;
  });
}

extern "C" int qwen38_gdn_graph_destroy(void* graph) {
  return graph_wrap([&] {
    delete static_cast<rocket::qwen38::linear_attention::CutlassGdnGraph*>(graph);
  });
}

extern "C" const char* qwen38_gdn_graph_last_error() {
  return rocket::qwen38::linear_attention::graph_last_error.c_str();
}

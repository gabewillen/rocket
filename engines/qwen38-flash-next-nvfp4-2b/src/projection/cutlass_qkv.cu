#include "projection/cutlass_qkv.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <new>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace rocket::qwen {

constexpr int kM = 16;
constexpr int kK = 2560;
constexpr int kNq = 6144;
constexpr int kNk = 256;
constexpr int kNv = 256;
constexpr int kN = kNq + kNk + kNv;
constexpr int kSfaBytes = 128 * ((kK / 16 + 3) / 4) * 4;

thread_local std::string last_error;

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

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
    ElementAccumulator, ElementC, LayoutCTag, 8, ElementD, LayoutCTag, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, 32, ElementB, LayoutBTag, 32,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                       CollectiveMainloop,
                                                       CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;

__constant__ float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                               2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ float e4m3_to_float(std::uint8_t bits) {
  const std::uint32_t sign = (bits & 0x80u) ? 0x80000000u : 0u;
  const std::uint32_t exp = (bits >> 3) & 0x0fu;
  const std::uint32_t mant = bits & 7u;
  if (exp == 0) {
    if (mant == 0) return __uint_as_float(sign);
    const float value = static_cast<float>(mant) * (1.0f / 8.0f) * 0.015625f;
    return sign ? -value : value;
  }
  return __uint_as_float(sign | ((exp + 120u) << 23) | (mant << 20));
}

__device__ __forceinline__ std::uint8_t float_to_e4m3(float value) {
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ std::uint8_t float_to_e2m1(float value) {
  const std::uint8_t sign = value < 0.0f ? 8u : 0u;
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

__device__ __forceinline__ std::size_t sfa_offset(int row, int sf) {
  constexpr int kAtomBytes = 512;
  const int k_tile = sf / 4;
  const int ssub = sf % 4;
  return static_cast<std::size_t>(k_tile) * kAtomBytes +
         static_cast<std::size_t>((row % 32) * 16 + (row / 32) * 4 + ssub);
}

__global__ void quantize_c16(std::uint8_t* packed, std::uint8_t* scales,
                             const __nv_bfloat16* input) {
  const int row = blockIdx.x;
  for (int block = threadIdx.x; block < kK / 16; block += blockDim.x) {
    float values[16];
    float amax = 0.0f;
#pragma unroll
    for (int item = 0; item < 16; ++item) {
      values[item] = __bfloat162float(input[row * kK + block * 16 + item]);
      amax = fmaxf(amax, fabsf(values[item]));
    }
    const std::uint8_t scale = amax > 0.0f ? float_to_e4m3(amax / 6.0f) : 0;
    const float dequant = e4m3_to_float(scale);
    scales[sfa_offset(row, block)] = scale;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const std::uint8_t lo = dequant > 0.0f ? float_to_e2m1(values[item * 2] / dequant) : 0;
      const std::uint8_t hi = dequant > 0.0f ? float_to_e2m1(values[item * 2 + 1] / dequant) : 0;
      packed[row * (kK / 2) + block * 8 + item] = lo | (hi << 4);
    }
  }
}

struct FixedPlan {
  cutlass::DeviceAllocation<std::uint8_t> workspace;
  Gemm gemm;

  bool init(int n, const std::uint8_t* a, const std::uint8_t* sfa,
            const std::uint8_t* b, const std::uint8_t* sfb, float global,
            __nv_bfloat16* d, int device) {
    StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {kM, kK, 1});
    StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {n, kK, 1});
    StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {kM, n, 1});
    LayoutSFA la = ScaleConfig::tile_atom_to_shape_SFA(make_shape(kM, n, kK, 1));
    LayoutSFB lb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(kM, n, kK, 1));
    auto pa = reinterpret_cast<const ElementInput*>(a);
    auto pb = reinterpret_cast<const ElementInput*>(b);
    auto psa = reinterpret_cast<const ElementSF*>(sfa);
    auto psb = reinterpret_cast<const ElementSF*>(sfb);
    auto pd = reinterpret_cast<ElementD*>(d);
    typename Gemm::Arguments args;
    args = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm, {kM, n, kK, 1},
        {pa, sa, pb, sb, psa, la, psb, lb},
        {{global, 0.0f}, nullptr, sd, pd, sd}};
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    workspace.reset(Gemm::get_workspace_size(args));
    return gemm.initialize(args, workspace.get()) == cutlass::Status::kSuccess;
  }
};

struct QkvPlan {
  std::uint8_t* packed = nullptr;
  std::uint8_t* sfa = nullptr;
  __nv_bfloat16* output = nullptr;
  FixedPlan q, k, v;
  ~QkvPlan() { cudaFree(output); cudaFree(sfa); cudaFree(packed); }
};

bool cuda_ok(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  last_error = std::string(operation) + ": " + cudaGetErrorString(status);
  return false;
}

}  // namespace rocket::qwen

using namespace rocket::qwen;

extern "C" int qwen38_cutlass_qkv_create(
    const std::uint8_t* q_weight, const std::uint8_t* q_scale, float q_global,
    const std::uint8_t* k_weight, const std::uint8_t* k_scale, float k_global,
    const std::uint8_t* v_weight, const std::uint8_t* v_scale, float v_global,
    int device, void** result) {
  last_error.clear();
  if (!q_weight || !q_scale || !k_weight || !k_scale || !v_weight || !v_scale ||
      !result || device < 0) {
    last_error = "invalid fixed QKV plan arguments";
    return 1;
  }
  *result = nullptr;
  auto plan = std::make_unique<QkvPlan>();
  if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice") ||
      !cuda_ok(cudaMalloc(&plan->packed, kM * kK / 2), "cudaMalloc packed A") ||
      !cuda_ok(cudaMalloc(&plan->sfa, kSfaBytes), "cudaMalloc SFA") ||
      !cuda_ok(cudaMalloc(&plan->output, kM * kN * sizeof(__nv_bfloat16)), "cudaMalloc output"))
    return 1;
  try {
    if (!plan->q.init(kNq, plan->packed, plan->sfa, q_weight, q_scale, q_global,
                      plan->output, device) ||
        !plan->k.init(kNk, plan->packed, plan->sfa, k_weight, k_scale, k_global,
                      plan->output + kM * kNq, device) ||
        !plan->v.init(kNv, plan->packed, plan->sfa, v_weight, v_scale, v_global,
                      plan->output + kM * (kNq + kNk), device)) {
      last_error = "CUTLASS cannot initialize the fixed QKV shape";
      return 1;
    }
  } catch (const std::exception& error) {
    last_error = error.what();
    return 1;
  }
  *result = plan.release();
  return 0;
}

extern "C" int qwen38_cutlass_qkv_launch(void* opaque, const void* activations,
                                           cudaStream_t stream) {
  if (qwen38_cutlass_qkv_quantize(opaque, activations, stream) != 0) return 1;
  return qwen38_cutlass_qkv_project(opaque, stream);
}

extern "C" int qwen38_cutlass_qkv_quantize(void* opaque, const void* activations,
                                             cudaStream_t stream) {
  last_error.clear();
  if (!opaque || !activations) { last_error = "null QKV launch argument"; return 1; }
  auto* plan = static_cast<QkvPlan*>(opaque);
  quantize_c16<<<kM, 256, 0, stream>>>(
      plan->packed, plan->sfa, static_cast<const __nv_bfloat16*>(activations));
  return cuda_ok(cudaGetLastError(), "quantize_c16") ? 0 : 1;
}

extern "C" int qwen38_cutlass_qkv_project(void* opaque, cudaStream_t stream) {
  last_error.clear();
  if (!opaque) { last_error = "null QKV projection plan"; return 1; }
  auto* plan = static_cast<QkvPlan*>(opaque);
  if (plan->q.gemm.run(stream) != cutlass::Status::kSuccess ||
      plan->k.gemm.run(stream) != cutlass::Status::kSuccess ||
      plan->v.gemm.run(stream) != cutlass::Status::kSuccess) {
    if (last_error.empty()) last_error = "CUTLASS QKV launch failed";
    return 1;
  }
  return 0;
}

extern "C" int qwen38_cutlass_qkv_output(void* opaque, void** output,
                                           std::size_t* elements) {
  if (!opaque || !output || !elements) return 1;
  *output = static_cast<QkvPlan*>(opaque)->output;
  *elements = static_cast<std::size_t>(kM) * kN;
  return 0;
}

extern "C" int qwen38_cutlass_qkv_destroy(void* opaque) {
  delete static_cast<QkvPlan*>(opaque);
  return 0;
}

extern "C" const char* qwen38_cutlass_qkv_last_error() { return last_error.c_str(); }

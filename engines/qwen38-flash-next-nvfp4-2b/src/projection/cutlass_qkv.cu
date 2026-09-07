#include "projection/cutlass_qkv.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/device/device_segmented_radix_sort.cuh>

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
constexpr int kBlockTopk = 512;
constexpr int kCompressRatio = 4;
constexpr int kTokenTopk = 2048;
constexpr int kExpandedWidth = kTokenTopk + kCompressRatio - 1;
constexpr int kQsaHeads = 4;
constexpr int kQsaDim = 128;
constexpr int kQsaPageSize = 64;
constexpr int kQsaColumns = 262144 / kCompressRatio;
constexpr int kQsaPages = kQsaColumns / kQsaPageSize;

// Reference audit (all Apache-2.0): vLLM 8e685d198
// models/qwen3_8_flash_next/nvidia/ops/qsa.py supplies the paged score,
// sqrt(128), causal visibility, and SM120 persistent-top-k contract.
// FlashInfer 91bda04c66f7cb851e1ab3b78b9fecea644b9844
// include/flashinfer/topk.cuh and TensorRT-LLM
// c426264bc4d01930fad01426b81800e96fe19c2b
// cpp/tensorrt_llm/kernels/indexerTopK.cu both use exact radix filtering.
// This first executable slice reuses CUDA 13 CUB's captured segmented radix
// sort as the upstream correctness control. Its full-sort traffic is profiled
// separately so replacing it with the audited fixed K=512 filter is measurable.

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

__global__ void expand_qsa_topk(const std::int32_t* block_indices,
                                const std::int64_t* logical_positions,
                                const std::int32_t* sequence_lengths,
                                const std::int32_t* token_to_request,
                                std::int32_t* token_indices, int rows) {
  const int row = blockIdx.x;
  const int column = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= rows || column >= kExpandedWidth) return;
  const int request = token_to_request[row];
  int token = -1;
  if (request >= 0 && request < kM) {
    const std::int64_t query_end = logical_positions[row] + 1;
    const int sequence_length = sequence_lengths[request];
    const int complete_blocks = min(
        min(static_cast<int>(query_end / kCompressRatio),
            sequence_length / kCompressRatio),
        kBlockTopk);
    const int expanded_count = complete_blocks * kCompressRatio;
    std::int64_t candidate = -1;
    if (column < expanded_count) {
      const int block = block_indices[row * kBlockTopk + column / kCompressRatio];
      candidate = static_cast<std::int64_t>(block) * kCompressRatio +
                  column % kCompressRatio;
    } else {
      const std::int64_t tail_start = (query_end / kCompressRatio) * kCompressRatio;
      const int tail_offset = column - expanded_count;
      const int tail_count = static_cast<int>(query_end - tail_start);
      if (tail_offset < tail_count && tail_offset < kCompressRatio - 1)
        candidate = tail_start + tail_offset;
    }
    if (candidate >= 0 && candidate < sequence_length)
      token = static_cast<int>(candidate);
  }
  token_indices[row * kExpandedWidth + column] = token;
}

__global__ void initialize_qsa_inputs(__nv_bfloat16* query,
                                      __nv_bfloat16* keys,
                                      std::int32_t* page_table,
                                      std::int32_t* indices,
                                      std::int32_t* offsets) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
  const std::size_t query_elements = kM * kQsaHeads * kQsaDim;
  const std::size_t key_elements =
      static_cast<std::size_t>(kQsaColumns) * kQsaDim;
  if (index < query_elements) {
    const int dim = index % kQsaDim;
    query[index] = __float2bfloat16(
        dim == 0 ? 4096.0f : (dim == 1 ? 64.0f : (dim == 2 ? 1.0f : 0.0f)));
  }
  if (index < key_elements) {
    const int physical_page = index / (kQsaPageSize * kQsaDim);
    const int within_page = (index / kQsaDim) % kQsaPageSize;
    const int dim = index % kQsaDim;
    const int logical_page = kQsaPages - 1 - physical_page;
    keys[index] = __float2bfloat16(
        dim == 0 ? static_cast<float>(logical_page / 64)
                 : (dim == 1 ? static_cast<float>(logical_page % 64)
                             : (dim == 2 ? static_cast<float>(within_page) : 0.0f)));
  }
  if (index < static_cast<std::size_t>(kM * kQsaPages)) {
    page_table[index] = kQsaPages - 1 - static_cast<int>(index % kQsaPages);
  }
  if (index < static_cast<std::size_t>(kM * kQsaColumns))
    indices[index] = static_cast<int>(index % kQsaColumns);
  if (index <= kM) offsets[index] = static_cast<int>(index) * kQsaColumns;
}

__global__ void score_qsa_c16(const __nv_bfloat16* __restrict__ query,
                              const __nv_bfloat16* __restrict__ keys,
                              const std::int32_t* __restrict__ page_table,
                              const std::int64_t* logical_positions,
                              const std::int32_t* sequence_lengths,
                              const std::int32_t* token_to_request,
                              float* __restrict__ logits,
                              std::int32_t* __restrict__ visible_blocks) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (column >= kQsaColumns) return;
  const int request = token_to_request[row];
  int visible = 0;
  if (request >= 0 && request < kM) {
    const std::int64_t query_end = logical_positions[row] + 1;
    visible = min(static_cast<int>(query_end / kCompressRatio),
                  sequence_lengths[request] / kCompressRatio);
    visible = max(0, min(visible, kQsaColumns));
  }
  if (column == 0) visible_blocks[row] = visible;
  float score = -INFINITY;
  if (column < visible) {
    const int physical_page = page_table[row * kQsaPages + column / kQsaPageSize];
    if (physical_page >= 0 && physical_page < kQsaPages) {
      const __nv_bfloat16* key =
          keys + (static_cast<std::size_t>(physical_page) * kQsaPageSize +
                  column % kQsaPageSize) * kQsaDim;
      float head_scores[kQsaHeads] = {};
#pragma unroll
      for (int dim = 0; dim < kQsaDim; ++dim) {
        const float key_value = __bfloat162float(key[dim]);
#pragma unroll
        for (int head = 0; head < kQsaHeads; ++head)
          head_scores[head] = fmaf(
              __bfloat162float(query[(row * kQsaHeads + head) * kQsaDim + dim]),
              key_value, head_scores[head]);
      }
      score = 0.0f;
#pragma unroll
      for (int head = 0; head < kQsaHeads; ++head)
        score += fmaxf(head_scores[head], 0.0f);
      score *= 0.08838834764831845f;  // 1 / sqrt(128)
    }
  }
  logits[static_cast<std::size_t>(row) * kQsaColumns + column] = score;
}

__global__ void extract_qsa_topk(const float* sorted_logits,
                                 const std::int32_t* sorted_indices,
                                 const std::int32_t* visible_blocks,
                                 std::int32_t* selected) {
  const int rank = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (rank < kBlockTopk)
    selected[row * kBlockTopk + rank] =
        rank < visible_blocks[row] &&
                isfinite(sorted_logits[static_cast<std::size_t>(row) * kQsaColumns + rank])
            ? sorted_indices[static_cast<std::size_t>(row) * kQsaColumns + rank]
            : -1;
}

__global__ void scale_qkv_families(__nv_bfloat16* output, float q_scale,
                                   float k_scale, float v_scale) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (column >= kN) return;
  const float scale = column < kNq ? q_scale : (column < kNq + kNk ? k_scale : v_scale);
  const int index = row * kN + column;
  output[index] = __float2bfloat16(__bfloat162float(output[index]) * scale);
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
  std::uint8_t* fused_weight = nullptr;
  std::uint8_t* fused_scale = nullptr;
  __nv_bfloat16* output = nullptr;
  float q_global = 1.0f, k_global = 1.0f, v_global = 1.0f;
  FixedPlan fused;
  ~QkvPlan() {
    cudaFree(output);
    cudaFree(fused_scale);
    cudaFree(fused_weight);
    cudaFree(sfa);
    cudaFree(packed);
  }
};

struct QsaPlan {
  __nv_bfloat16* query = nullptr;
  __nv_bfloat16* keys = nullptr;
  std::int32_t* page_table = nullptr;
  float *logits = nullptr, *sorted_logits = nullptr;
  std::int32_t *indices = nullptr, *sorted_indices = nullptr;
  std::int32_t *offsets = nullptr, *visible = nullptr, *selected = nullptr,
               *output = nullptr;
  void* sort_workspace = nullptr;
  std::size_t sort_workspace_bytes = 0;
  ~QsaPlan() {
    cudaFree(sort_workspace); cudaFree(output); cudaFree(selected); cudaFree(visible);
    cudaFree(offsets); cudaFree(sorted_indices); cudaFree(indices);
    cudaFree(sorted_logits); cudaFree(logits); cudaFree(page_table);
    cudaFree(keys); cudaFree(query);
  }
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
      !cuda_ok(cudaMalloc(&plan->fused_weight, kN * kK / 2), "cudaMalloc fused B") ||
      !cuda_ok(cudaMalloc(&plan->fused_scale, kN * kK / 16), "cudaMalloc fused SFB") ||
      !cuda_ok(cudaMalloc(&plan->output, kM * kN * sizeof(__nv_bfloat16)), "cudaMalloc output"))
    return 1;
  try {
    const std::size_t q_weight_bytes = static_cast<std::size_t>(kNq) * kK / 2;
    const std::size_t k_weight_bytes = static_cast<std::size_t>(kNk) * kK / 2;
    const std::size_t q_scale_bytes = static_cast<std::size_t>(kNq) * kK / 16;
    const std::size_t k_scale_bytes = static_cast<std::size_t>(kNk) * kK / 16;
    if (!cuda_ok(cudaMemcpy(plan->fused_weight, q_weight, q_weight_bytes,
                            cudaMemcpyDeviceToDevice), "copy Q weight") ||
        !cuda_ok(cudaMemcpy(plan->fused_weight + q_weight_bytes, k_weight,
                            k_weight_bytes, cudaMemcpyDeviceToDevice), "copy K weight") ||
        !cuda_ok(cudaMemcpy(plan->fused_weight + q_weight_bytes + k_weight_bytes,
                            v_weight, static_cast<std::size_t>(kNv) * kK / 2,
                            cudaMemcpyDeviceToDevice), "copy V weight") ||
        !cuda_ok(cudaMemcpy(plan->fused_scale, q_scale, q_scale_bytes,
                            cudaMemcpyDeviceToDevice), "copy Q scale") ||
        !cuda_ok(cudaMemcpy(plan->fused_scale + q_scale_bytes, k_scale,
                            k_scale_bytes, cudaMemcpyDeviceToDevice), "copy K scale") ||
        !cuda_ok(cudaMemcpy(plan->fused_scale + q_scale_bytes + k_scale_bytes,
                            v_scale, static_cast<std::size_t>(kNv) * kK / 16,
                            cudaMemcpyDeviceToDevice), "copy V scale"))
      return 1;
    plan->q_global = q_global;
    plan->k_global = k_global;
    plan->v_global = v_global;
    if (!plan->fused.init(kN, plan->packed, plan->sfa, plan->fused_weight,
                          plan->fused_scale, 1.0f, plan->output, device)) {
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
  if (plan->fused.gemm.run(stream) != cutlass::Status::kSuccess) {
    if (last_error.empty()) last_error = "CUTLASS QKV launch failed";
    return 1;
  }
  scale_qkv_families<<<dim3((kN + 255) / 256, kM), 256, 0, stream>>>(
      plan->output, plan->q_global, plan->k_global, plan->v_global);
  if (!cuda_ok(cudaGetLastError(), "scale_qkv_families")) return 1;
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

extern "C" int qwen38_qsa_expand_topk(
    const std::int32_t* block_indices, const std::int64_t* logical_positions,
    const std::int32_t* sequence_lengths, const std::int32_t* token_to_request,
    std::int32_t* token_indices, int rows, cudaStream_t stream) {
  last_error.clear();
  if (!block_indices || !logical_positions || !sequence_lengths ||
      !token_to_request || !token_indices || rows < 1 || rows > kM) {
    last_error = "invalid fixed QSA expansion arguments";
    return 1;
  }
  expand_qsa_topk<<<dim3(rows, (kExpandedWidth + 255) / 256), 256, 0, stream>>>(
      block_indices, logical_positions, sequence_lengths, token_to_request,
      token_indices, rows);
  return cuda_ok(cudaGetLastError(), "expand_qsa_topk") ? 0 : 1;
}

extern "C" int qwen38_qsa_indexer_create(int device, void** result) {
  last_error.clear();
  if (!result || device < 0) { last_error = "invalid fixed QSA plan arguments"; return 1; }
  *result = nullptr;
  auto plan = std::make_unique<QsaPlan>();
  const std::size_t entries = static_cast<std::size_t>(kM) * kQsaColumns;
  const std::size_t query_bytes = static_cast<std::size_t>(kM) * kQsaHeads * kQsaDim * 2;
  const std::size_t key_bytes = static_cast<std::size_t>(kQsaColumns) * kQsaDim * 2;
  const std::size_t table_bytes = static_cast<std::size_t>(kM) * kQsaPages * 4;
  if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice") ||
      !cuda_ok(cudaMalloc(&plan->query, query_bytes), "cudaMalloc QSA query") ||
      !cuda_ok(cudaMalloc(&plan->keys, key_bytes), "cudaMalloc QSA keys") ||
      !cuda_ok(cudaMalloc(&plan->page_table, table_bytes), "cudaMalloc QSA page table") ||
      !cuda_ok(cudaMalloc(&plan->logits, entries * 4), "cudaMalloc QSA logits") ||
      !cuda_ok(cudaMalloc(&plan->sorted_logits, entries * 4), "cudaMalloc sorted QSA logits") ||
      !cuda_ok(cudaMalloc(&plan->indices, entries * 4), "cudaMalloc QSA indices") ||
      !cuda_ok(cudaMalloc(&plan->sorted_indices, entries * 4), "cudaMalloc sorted QSA indices") ||
      !cuda_ok(cudaMalloc(&plan->offsets, (kM + 1) * 4), "cudaMalloc QSA offsets") ||
      !cuda_ok(cudaMalloc(&plan->visible, kM * 4), "cudaMalloc QSA visible") ||
      !cuda_ok(cudaMalloc(&plan->selected, kM * kBlockTopk * 4), "cudaMalloc QSA selected") ||
      !cuda_ok(cudaMalloc(&plan->output, kM * kExpandedWidth * 4), "cudaMalloc QSA output"))
    return 1;
  const cudaError_t size_status = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, plan->sort_workspace_bytes, plan->logits, plan->sorted_logits,
      plan->indices, plan->sorted_indices, entries, kM, plan->offsets,
      plan->offsets + 1);
  if (!cuda_ok(size_status, "size QSA segmented sort") ||
      !cuda_ok(cudaMalloc(&plan->sort_workspace, plan->sort_workspace_bytes),
               "cudaMalloc QSA sort workspace"))
    return 1;
  const std::size_t init_elements =
      max(max(query_bytes / 2, key_bytes / 2), max(entries, table_bytes / 4));
  initialize_qsa_inputs<<<(init_elements + 255) / 256, 256>>>(
      plan->query, plan->keys, plan->page_table, plan->indices, plan->offsets);
  if (!cuda_ok(cudaGetLastError(), "initialize QSA inputs") ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize QSA initialization"))
    return 1;
  *result = plan.release();
  return 0;
}

extern "C" int qwen38_qsa_indexer_launch(
    void* opaque, const std::int64_t* logical_positions,
    const std::int32_t* sequence_lengths,
    const std::int32_t* token_to_request, cudaStream_t stream) {
  last_error.clear();
  if (!opaque || !logical_positions || !sequence_lengths || !token_to_request) {
    last_error = "null fixed QSA launch argument"; return 1;
  }
  if (qwen38_qsa_indexer_score(opaque, logical_positions, sequence_lengths,
                               token_to_request, stream) != 0) return 1;
  return qwen38_qsa_indexer_select_expand(
      opaque, logical_positions, sequence_lengths, token_to_request, stream);
}

extern "C" int qwen38_qsa_indexer_score(
    void* opaque, const std::int64_t* logical_positions,
    const std::int32_t* sequence_lengths,
    const std::int32_t* token_to_request, cudaStream_t stream) {
  if (!opaque || !logical_positions || !sequence_lengths || !token_to_request)
    return 1;
  auto* plan = static_cast<QsaPlan*>(opaque);
  score_qsa_c16<<<dim3((kQsaColumns + 255) / 256, kM), 256, 0, stream>>>(
      plan->query, plan->keys, plan->page_table, logical_positions,
      sequence_lengths, token_to_request, plan->logits, plan->visible);
  return cuda_ok(cudaGetLastError(), "score_qsa_c16") ? 0 : 1;
}

extern "C" int qwen38_qsa_indexer_select_expand(
    void* opaque, const std::int64_t* logical_positions,
    const std::int32_t* sequence_lengths,
    const std::int32_t* token_to_request, cudaStream_t stream) {
  if (!opaque || !logical_positions || !sequence_lengths || !token_to_request)
    return 1;
  auto* plan = static_cast<QsaPlan*>(opaque);
  const cudaError_t sort_status = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      plan->sort_workspace, plan->sort_workspace_bytes, plan->logits,
      plan->sorted_logits, plan->indices, plan->sorted_indices,
      static_cast<std::size_t>(kM) * kQsaColumns, kM, plan->offsets,
      plan->offsets + 1, 0, 32, stream);
  if (!cuda_ok(sort_status, "sort QSA scores")) return 1;
  extract_qsa_topk<<<dim3((kBlockTopk + 255) / 256, kM), 256, 0, stream>>>(
      plan->sorted_logits, plan->sorted_indices, plan->visible, plan->selected);
  if (!cuda_ok(cudaGetLastError(), "extract QSA top-k")) return 1;
  expand_qsa_topk<<<dim3(kM, (kExpandedWidth + 255) / 256), 256, 0, stream>>>(
      plan->selected, logical_positions, sequence_lengths, token_to_request,
      plan->output, kM);
  return cuda_ok(cudaGetLastError(), "expand selected QSA blocks") ? 0 : 1;
}

extern "C" int qwen38_qsa_indexer_inputs(
    void* opaque, void** query, std::size_t* query_bytes, void** keys,
    std::size_t* key_bytes, void** page_table, std::size_t* page_table_bytes) {
  if (!opaque || !query || !query_bytes || !keys || !key_bytes || !page_table ||
      !page_table_bytes) return 1;
  auto* plan = static_cast<QsaPlan*>(opaque);
  *query = plan->query; *query_bytes = static_cast<std::size_t>(kM) * kQsaHeads * kQsaDim * 2;
  *keys = plan->keys; *key_bytes = static_cast<std::size_t>(kQsaColumns) * kQsaDim * 2;
  *page_table = plan->page_table; *page_table_bytes = static_cast<std::size_t>(kM) * kQsaPages * 4;
  return 0;
}

extern "C" int qwen38_qsa_indexer_output(void* opaque, void** output,
                                            std::size_t* elements) {
  if (!opaque || !output || !elements) return 1;
  *output = static_cast<QsaPlan*>(opaque)->output;
  *elements = static_cast<std::size_t>(kM) * kExpandedWidth;
  return 0;
}

extern "C" int qwen38_qsa_indexer_destroy(void* opaque) {
  delete static_cast<QsaPlan*>(opaque); return 0;
}

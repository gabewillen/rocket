// NVFP4 grouped GEMM at GLM-5.3-Flash MoE decode shapes, on sm_121 (GB10).
//
// Question: does CUTLASS emit competitive NVFP4 (FP4 data, FP8 block scales)
// grouped GEMM at the shapes MoE decode needs, or does rocket need its own port?
//
// Shapes come from the on-disk GLM-5.3-Flash checkpoint config, not an estimate:
//   ~/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/
//     snapshots/25a44fdbf16862a46b7cc9921142c6c81350af2f/config.json
//   hidden_size=4096, moe_intermediate_size=2048,
//   n_routed_experts=288, num_experts_per_tok=8, hidden_act=silu (SwiGLU)
// so each MoE layer is two grouped GEMMs per expert:
//   w13 (fused gate+up):  N = 2*2048 = 4096, K = 4096
//   w2  (down):           N = 4096,          K = 2048
//
// The FP4 tensor core takes FP4 on both operands, so activations are
// NVFP4-quantized (the standard NVFP4 recipe). Accumulation is FP32 and the
// epilogue writes BF16.
//
// Two paths are timed on identical buffers:
//   cutlass    block-scaled grouped GEMM, ArchTag Sm120, built for sm_121a
//   dequant    NVFP4 -> BF16 dequant kernel + per-group cuBLAS BF16 GEMM
//              (reported both with dequant included and GEMM-only)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err_ = (x);                                                    \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                   cudaGetErrorString(err_), __FILE__, __LINE__);              \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublasStatus_t st_ = (x);                                                  \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__,   \
                   __LINE__);                                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUTLASS_CHECK_STATUS(x)                                                \
  do {                                                                         \
    cutlass::Status s_ = (x);                                                  \
    if (s_ != cutlass::Status::kSuccess) {                                     \
      std::fprintf(stderr, "CUTLASS error %s at %s:%d\n",                      \
                   cutlassGetStatusString(s_), __FILE__, __LINE__);            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// CUTLASS kernel configuration
// ---------------------------------------------------------------------------

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e2m1_t;

using ElementA   = cutlass::nv_float4_t<ElementInput>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<ElementInput>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD   = cutlass::bfloat16_t;
using ElementC   = void;                      // beta = 0, no source read
using LayoutCTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = AlignmentD;

using ElementAccumulator = float;
using ArchTag      = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape     = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag *, AlignmentC,
    ElementD, LayoutCTag *, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag *, AlignmentA,
    ElementB, LayoutBTag *, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;

// NVFP4 scale-factor vector size (elements per FP8 block scale).
constexpr int SFVecSize = Gemm::GemmKernel::CollectiveMainloop::SFVecSize;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float decode_e2m1(uint8_t nib) {
  // e2m1: 1 sign, 2 exponent, 1 mantissa.
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = mag[nib & 0x7];
  return (nib & 0x8) ? -v : v;
}

// Dequantize NVFP4 (packed 2 values/byte, FP8 e4m3 scale per SFVecSize
// contiguous K elements) into BF16. Plain (unswizzled) scale layout.
__global__ void dequant_nvfp4_to_bf16(const uint8_t *__restrict__ packed,
                                      const __nv_fp8_storage_t *__restrict__ scales,
                                      __nv_bfloat16 *__restrict__ out,
                                      long long rows, long long k, int sf_vec) {
  long long total_bytes = rows * k / 2;
  long long stride = (long long)blockDim.x * gridDim.x;
  for (long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       b < total_bytes; b += stride) {
    long long elem = b * 2;
    long long row = elem / k;
    long long col = elem - row * k;
    long long sf_idx = row * (k / sf_vec) + col / sf_vec;
    float s = __half2float(__nv_cvt_fp8_to_halfraw(scales[sf_idx], __NV_E4M3));
    uint8_t byte = packed[b];
    out[elem]     = __float2bfloat16(decode_e2m1(byte & 0xF) * s);
    out[elem + 1] = __float2bfloat16(decode_e2m1(byte >> 4) * s);
  }
}

__global__ void fill_bytes(uint8_t *p, long long n, uint8_t v) {
  long long stride = (long long)blockDim.x * gridDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride)
    p[i] = v;
}

// Deterministic pseudo-random FP4 nibbles in {0,1,2,3} -> {0,0.5,1.0,1.5}.
__global__ void fill_fp4_random(uint8_t *p, long long n, unsigned seed) {
  long long stride = (long long)blockDim.x * gridDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    unsigned h = (unsigned)(i * 2654435761u) ^ seed;
    h ^= h >> 13; h *= 1274126177u; h ^= h >> 16;
    p[i] = (uint8_t)((h & 3) | (((h >> 8) & 3) << 4));
  }
}

// ---------------------------------------------------------------------------
// Bench
// ---------------------------------------------------------------------------

struct ShapeCase {
  const char *name;
  int N;
  int K;
};

struct Options {
  int groups = 288;          // routed experts in GLM-5.3-Flash
  int iterations = 20;
  bool verify = false;
  std::vector<int> m_means = {8, 16, 32, 64};
  bool ragged = true;
};

// Ragged per-group M around a mean, deterministic for a given (mean, groups).
static std::vector<int> make_group_m(int mean, int groups, bool ragged) {
  std::vector<int> m(groups, mean);
  if (!ragged) return m;
  std::mt19937 rng(1234u + (unsigned)mean);
  int lo = std::max(1, mean / 2);
  int hi = mean + mean / 2;
  std::uniform_int_distribution<int> dist(lo, hi);
  for (int i = 0; i < groups; ++i) m[i] = dist(rng);
  return m;
}

struct Result {
  double ms;
  double tflops;
  double gbps;         // effective weight-stream bandwidth
};

int main(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--groups=", 0) == 0) opt.groups = std::atoi(a.c_str() + 9);
    else if (a.rfind("--iterations=", 0) == 0) opt.iterations = std::atoi(a.c_str() + 13);
    else if (a == "--verify") opt.verify = true;
    else if (a == "--uniform-m") opt.ragged = false;
    else if (a == "--help") {
      std::printf("usage: %s [--groups=N] [--iterations=N] [--verify] [--uniform-m]\n", argv[0]);
      return 0;
    }
  }

  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("device        : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("groups        : %d experts\n", opt.groups);
  std::printf("iterations    : %d\n", opt.iterations);
  std::printf("M per expert  : %s\n", opt.ragged ? "ragged around mean" : "uniform");
  std::printf("SF vector size: %d elements per FP8 scale\n\n", SFVecSize);

  cublasHandle_t blas;
  CUBLAS_CHECK(cublasCreate(&blas));

  const ShapeCase shapes[] = {
      {"w13 gate+up", 4096, 4096},
      {"w2 down",     4096, 2048},
  };

  std::printf("%-12s %7s %8s %14s %9s %9s %9s %9s %10s\n",
              "gemm", "M_mean", "tokens", "path", "ms", "TFLOPS", "GB/s",
              "roof%", "vs cutlass");
  std::printf("%s\n", std::string(100, '-').c_str());

  const double kRoofline = 238.0;  // GB/s, measured device read bandwidth

  for (const auto &sc : shapes) {
    for (int mean : opt.m_means) {
      std::vector<int> Ms = make_group_m(mean, opt.groups, opt.ragged);
      long long tokens = 0;
      for (int m : Ms) tokens += m;

      const int G = opt.groups;
      const int N = sc.N, K = sc.K;

      // ---- problem shapes / strides / SF layouts -------------------------
      std::vector<typename ProblemShape::UnderlyingProblemShape> ps_host(G);
      std::vector<StrideA> sA_host(G);
      std::vector<StrideB> sB_host(G);
      std::vector<StrideD> sD_host(G);
      std::vector<LayoutSFA> lSFA_host(G);
      std::vector<LayoutSFB> lSFB_host(G);
      for (int i = 0; i < G; ++i) {
        int M = Ms[i];
        ps_host[i] = {M, N, K};
        sA_host[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        sB_host[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        sD_host[i] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
        lSFA_host[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
        lSFB_host[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
      }

      // ---- device allocations --------------------------------------------
      // A/B packed FP4: 2 values per byte. SF: one FP8 per SFVecSize elements.
      std::vector<cutlass::DeviceAllocation<uint8_t>> dA(G), dB(G), dSFA(G), dSFB(G);
      std::vector<cutlass::DeviceAllocation<ElementD>> dD(G);
      std::vector<const ElementInput *> pA(G), pB(G);
      std::vector<const ElementSF *> pSFA(G), pSFB(G);
      std::vector<ElementD *> pD(G);

      for (int i = 0; i < G; ++i) {
        int M = Ms[i];
        dA[i].reset((size_t)M * K / 2);
        dB[i].reset((size_t)N * K / 2);
        dSFA[i].reset((size_t)size(filter_zeros(lSFA_host[i])));
        dSFB[i].reset((size_t)size(filter_zeros(lSFB_host[i])));
        dD[i].reset((size_t)M * N);

        fill_fp4_random<<<256, 256>>>(dA[i].get(), (long long)M * K / 2, 11u + i);
        fill_fp4_random<<<512, 256>>>(dB[i].get(), (long long)N * K / 2, 77u + i);
        // All block scales = 1.0 in e4m3 (0x38). Constant scales make the
        // swizzled CUTLASS SF layout and the plain fallback layout agree,
        // which is what lets the two paths be compared numerically.
        fill_bytes<<<256, 256>>>(dSFA[i].get(), (long long)dSFA[i].size(), 0x38);
        fill_bytes<<<256, 256>>>(dSFB[i].get(), (long long)dSFB[i].size(), 0x38);

        pA[i] = reinterpret_cast<const ElementInput *>(dA[i].get());
        pB[i] = reinterpret_cast<const ElementInput *>(dB[i].get());
        pSFA[i] = reinterpret_cast<const ElementSF *>(dSFA[i].get());
        pSFB[i] = reinterpret_cast<const ElementSF *>(dSFB[i].get());
        pD[i] = dD[i].get();
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> ps_dev(G);
      ps_dev.copy_from_host(ps_host.data());
      cutlass::DeviceAllocation<StrideA> sA_dev(G); sA_dev.copy_from_host(sA_host.data());
      cutlass::DeviceAllocation<StrideB> sB_dev(G); sB_dev.copy_from_host(sB_host.data());
      cutlass::DeviceAllocation<StrideD> sD_dev(G); sD_dev.copy_from_host(sD_host.data());
      cutlass::DeviceAllocation<LayoutSFA> lSFA_dev(G); lSFA_dev.copy_from_host(lSFA_host.data());
      cutlass::DeviceAllocation<LayoutSFB> lSFB_dev(G); lSFB_dev.copy_from_host(lSFB_host.data());
      cutlass::DeviceAllocation<const ElementInput *> pA_dev(G); pA_dev.copy_from_host(pA.data());
      cutlass::DeviceAllocation<const ElementInput *> pB_dev(G); pB_dev.copy_from_host(pB.data());
      cutlass::DeviceAllocation<const ElementSF *> pSFA_dev(G); pSFA_dev.copy_from_host(pSFA.data());
      cutlass::DeviceAllocation<const ElementSF *> pSFB_dev(G); pSFB_dev.copy_from_host(pSFB.data());
      cutlass::DeviceAllocation<ElementD *> pD_dev(G); pD_dev.copy_from_host(pD.data());

      // ---- CUTLASS arguments ---------------------------------------------
      cutlass::KernelHardwareInfo hw;
      hw.device_id = 0;
      hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

      typename Gemm::Arguments args;
      decltype(args.epilogue.thread) fusion;
      fusion.alpha = 1.0f;
      fusion.beta = 0.0f;
      fusion.alpha_ptr = nullptr;
      fusion.beta_ptr = nullptr;

      args = typename Gemm::Arguments{
          cutlass::gemm::GemmUniversalMode::kGrouped,
          {G, ps_dev.get(), ps_host.data()},
          {reinterpret_cast<const ElementInput **>(pA_dev.get()), sA_dev.get(),
           reinterpret_cast<const ElementInput **>(pB_dev.get()), sB_dev.get(),
           reinterpret_cast<const ElementSF **>(pSFA_dev.get()), lSFA_dev.get(),
           reinterpret_cast<const ElementSF **>(pSFB_dev.get()), lSFB_dev.get()},
          {fusion, nullptr, nullptr, pD_dev.get(), sD_dev.get()},
          hw};

      Gemm gemm;
      size_t ws = Gemm::get_workspace_size(args);
      cutlass::DeviceAllocation<uint8_t> workspace(ws);

      if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "cutlass cannot implement %s M_mean=%d\n", sc.name, mean);
        return 1;
      }
      CUTLASS_CHECK_STATUS(gemm.initialize(args, workspace.get()));

      // ---- time CUTLASS ----------------------------------------------------
      cudaEvent_t e0, e1;
      CUDA_CHECK(cudaEventCreate(&e0));
      CUDA_CHECK(cudaEventCreate(&e1));

      for (int i = 0; i < 5; ++i) CUTLASS_CHECK_STATUS(gemm.run());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(e0));
      for (int i = 0; i < opt.iterations; ++i) CUTLASS_CHECK_STATUS(gemm.run());
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms_total = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms_total, e0, e1));

      double flops = 2.0 * (double)tokens * N * K;
      // Weight stream: FP4 data + FP8 scale per SFVecSize elements.
      double wbytes = (double)G * N * K * (0.5 + 1.0 / SFVecSize);

      Result cut;
      cut.ms = ms_total / opt.iterations;
      cut.tflops = flops / (cut.ms * 1e-3) / 1e12;
      cut.gbps = wbytes / (cut.ms * 1e-3) / 1e9;

      // ---- fallback: dequant to BF16 + cuBLAS ------------------------------
      std::vector<cutlass::DeviceAllocation<__nv_bfloat16>> dBw(G), dAw(G);
      std::vector<cutlass::DeviceAllocation<__nv_bfloat16>> dDref(G);
      for (int i = 0; i < G; ++i) {
        dBw[i].reset((size_t)N * K);
        dAw[i].reset((size_t)Ms[i] * K);
        dDref[i].reset((size_t)Ms[i] * N);
      }

      auto run_dequant = [&]() {
        for (int i = 0; i < G; ++i) {
          dequant_nvfp4_to_bf16<<<1024, 256>>>(
              dB[i].get(), reinterpret_cast<const __nv_fp8_storage_t *>(dSFB[i].get()),
              dBw[i].get(), N, K, SFVecSize);
          dequant_nvfp4_to_bf16<<<64, 256>>>(
              dA[i].get(), reinterpret_cast<const __nv_fp8_storage_t *>(dSFA[i].get()),
              dAw[i].get(), Ms[i], K, SFVecSize);
        }
      };
      auto run_blas = [&]() {
        const float alpha = 1.0f, beta = 0.0f;
        for (int i = 0; i < G; ++i) {
          // D_rm[M][N] = A_rm[M][K] * W_rm[N][K]^T, expressed column-major.
          CUBLAS_CHECK(cublasGemmEx(
              blas, CUBLAS_OP_T, CUBLAS_OP_N, N, Ms[i], K, &alpha,
              dBw[i].get(), CUDA_R_16BF, K,
              dAw[i].get(), CUDA_R_16BF, K, &beta,
              dDref[i].get(), CUDA_R_16BF, N,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
      };

      run_dequant(); run_blas();
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaEventRecord(e0));
      for (int i = 0; i < opt.iterations; ++i) { run_dequant(); run_blas(); }
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms_dq = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms_dq, e0, e1));

      CUDA_CHECK(cudaEventRecord(e0));
      for (int i = 0; i < opt.iterations; ++i) run_blas();
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms_blas = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms_blas, e0, e1));

      Result deq;
      deq.ms = ms_dq / opt.iterations;
      deq.tflops = flops / (deq.ms * 1e-3) / 1e12;
      deq.gbps = wbytes / (deq.ms * 1e-3) / 1e9;

      Result blasonly;
      blasonly.ms = ms_blas / opt.iterations;
      blasonly.tflops = flops / (blasonly.ms * 1e-3) / 1e12;
      // BF16 resident weights, not the NVFP4 stream.
      blasonly.gbps = (double)G * N * K * 2.0 / (blasonly.ms * 1e-3) / 1e9;

      auto row = [&](const char *path, const Result &r, double vs) {
        char vsbuf[16];
        if (vs > 0) std::snprintf(vsbuf, sizeof(vsbuf), "%.2fx", vs);
        else std::snprintf(vsbuf, sizeof(vsbuf), "%s", "-");
        std::printf("%-12s %7d %8lld %14s %9.3f %9.2f %9.1f %9.1f %10s\n",
                    sc.name, mean, tokens, path, r.ms, r.tflops, r.gbps,
                    100.0 * r.gbps / kRoofline, vsbuf);
      };
      row("cutlass", cut, 0);
      row("dequant+blas", deq, deq.ms / cut.ms);
      row("blas-only", blasonly, blasonly.ms / cut.ms);

      // ---- optional numeric cross-check -----------------------------------
      if (opt.verify) {
        int bad = 0;
        for (int i = 0; i < std::min(G, 8); ++i) {
          std::vector<ElementD> h1((size_t)Ms[i] * N);
          std::vector<__nv_bfloat16> h2((size_t)Ms[i] * N);
          CUDA_CHECK(cudaMemcpy(h1.data(), dD[i].get(), h1.size() * sizeof(ElementD),
                                cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(h2.data(), dDref[i].get(), h2.size() * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToHost));
          for (size_t j = 0; j < h1.size(); ++j) {
            float a = (float)h1[j];
            float b = __bfloat162float(h2[j]);
            float tol = 0.02f * std::max(1.f, std::fabs(b));
            if (std::fabs(a - b) > tol) { ++bad; if (bad < 4)
              std::printf("  mismatch g=%d j=%zu cutlass=%g blas=%g\n", i, j, a, b); }
          }
        }
        std::printf("  verify %s: %s (first 8 groups)\n", sc.name,
                    bad == 0 ? "PASS" : "FAIL");
      }

      CUDA_CHECK(cudaEventDestroy(e0));
      CUDA_CHECK(cudaEventDestroy(e1));
    }
  }

  CUBLAS_CHECK(cublasDestroy(blas));
  std::printf("\nroofline = 238 GB/s measured device read bandwidth\n");
  return 0;
}

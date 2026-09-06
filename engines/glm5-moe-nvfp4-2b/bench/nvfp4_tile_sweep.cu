// What tile shapes below 128 in M buy or cost at per-expert M=2..8, on sm_121.
//
// The 128x128x128 tile the grouped-GEMM bench used wastes most of its M extent
// when a decode step routes 2 to 8 tokens to an expert
// (blog/posts/kernels/2026-09-06-cutlass-nvfp4-sm121/). This sweeps the CTA
// tile in M and N at those M and reports whether anything changes. The kernel
// is bandwidth bound on the expert weight stream, so the number to watch is
// GB/s against the 238 GB/s roofline, not TFLOPS.
//
// Tile M below 128 is not in the sweep because it does not exist for a
// block-scaled NVFP4 GEMM in CUTLASS 4.8.0, grouped or not. The scale-factor
// storage block is 128 wide in the M/N direction
// (Sm1xxBlkScaledChunk::Blk_MN = _128, include/cutlass/detail/
// sm100_blockscaled_layout.hpp), and the SM120 mainloop builder forms its
// shared-memory SFA atom from size<0>(TileShape) / Blk_MN
// (sm120_blockscaled_mma_builder.inl:199), which is zero at tile M 64.
// bench/tile_m64_probe.cu reproduces the failure.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

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

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e2m1_t;
using ElementA   = cutlass::nv_float4_t<ElementInput>;
using ElementB   = cutlass::nv_float4_t<ElementInput>;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;

using ElementD   = cutlass::bfloat16_t;
using ElementC   = void;
using LayoutCTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = AlignmentD;

using ElementAccumulator = float;
using ArchTag       = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ClusterShape  = Shape<_1, _1, _1>;

// KernelScheduleAuto picks the cooperative grouped schedule, which carries
//   static_assert(size<0>(TileShape{}) >= 128,
//     "Cooperative kernel requires Tile Size to be greater than or equal to 128
//      along the M-dimension.")
// in sm90_gemm_array_tma_warpspecialized_cooperative.hpp:443. The pingpong
// grouped schedule has no such assert, so sub-128 tile M is reached through it.
template <int TM, int TN, int TK,
          class KernelSched = cutlass::gemm::collective::KernelScheduleAuto,
          class EpiSched = cutlass::epilogue::collective::EpilogueScheduleAuto>
struct Cfg {
  using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag *, AlignmentC,
      ElementD, LayoutCTag *, AlignmentD,
      EpiSched
  >::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag *, AlignmentA,
      ElementB, LayoutBTag *, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      KernelSched
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <int TM, int TN, int TK>
using CoopCfg = Cfg<TM, TN, TK>;

__global__ void fill_bytes(uint8_t *p, long long n, uint8_t v) {
  long long stride = (long long)blockDim.x * gridDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride)
    p[i] = v;
}

__global__ void fill_fp4_random(uint8_t *p, long long n, unsigned seed) {
  long long stride = (long long)blockDim.x * gridDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    unsigned h = (unsigned)(i * 2654435761u) ^ seed;
    h ^= h >> 13; h *= 1274126177u; h ^= h >> 16;
    p[i] = (uint8_t)((h & 3) | (((h >> 8) & 3) << 4));
  }
}

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

static const double kRoofline = 238.0;

template <class C>
static void run_case(const char *tile, const char *gemm_name, int N, int K,
                     const std::vector<int> &Ms, int iterations) {
  using Gemm = typename C::Gemm;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using BlkCfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
  constexpr int SFVecSize = Gemm::GemmKernel::CollectiveMainloop::SFVecSize;

  const int G = (int)Ms.size();
  long long tokens = 0;
  for (int m : Ms) tokens += m;

  std::vector<typename ProblemShape::UnderlyingProblemShape> ps(G);
  std::vector<StrideA> sA(G);
  std::vector<StrideB> sB(G);
  std::vector<StrideD> sD(G);
  std::vector<LayoutSFA> lA(G);
  std::vector<LayoutSFB> lB(G);
  std::vector<size_t> offA(G), offD(G), offSFA(G), offSFB(G);

  size_t cA = 0, cD = 0, cSFA = 0, cSFB = 0;
  for (int i = 0; i < G; ++i) {
    int M = Ms[i];
    ps[i] = {M, N, K};
    sA[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    sB[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    sD[i] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    lA[i] = BlkCfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    lB[i] = BlkCfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    offA[i] = cA; cA += ((size_t)M * K / 2 + 255) / 256 * 256;
    offD[i] = cD; cD += (size_t)M * N;
    offSFA[i] = cSFA; cSFA += ((size_t)size(filter_zeros(lA[i])) + 255) / 256 * 256;
    offSFB[i] = cSFB; cSFB += ((size_t)size(filter_zeros(lB[i])) + 255) / 256 * 256;
  }

  cutlass::DeviceAllocation<uint8_t> A(cA), SFA(cSFA);
  cutlass::DeviceAllocation<uint8_t> B((size_t)G * N * K / 2), SFB(cSFB);
  cutlass::DeviceAllocation<ElementD> D(cD);

  fill_fp4_random<<<512, 256>>>(A.get(), (long long)cA, 11u);
  fill_fp4_random<<<2048, 256>>>(B.get(), (long long)B.size(), 77u);
  fill_bytes<<<256, 256>>>(SFA.get(), (long long)cSFA, 0x38);
  fill_bytes<<<512, 256>>>(SFB.get(), (long long)cSFB, 0x38);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<const ElementInput *> hA(G), hB(G);
  std::vector<const ElementSF *> hSFA(G), hSFB(G);
  std::vector<ElementD *> hD(G);
  for (int i = 0; i < G; ++i) {
    hA[i] = reinterpret_cast<const ElementInput *>(A.get() + offA[i]);
    hB[i] = reinterpret_cast<const ElementInput *>(B.get() + (size_t)i * N * K / 2);
    hSFA[i] = reinterpret_cast<const ElementSF *>(SFA.get() + offSFA[i]);
    hSFB[i] = reinterpret_cast<const ElementSF *>(SFB.get() + offSFB[i]);
    hD[i] = D.get() + offD[i];
  }

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> dps(G);
  dps.copy_from_host(ps.data());
  cutlass::DeviceAllocation<StrideA> dsA(G); dsA.copy_from_host(sA.data());
  cutlass::DeviceAllocation<StrideB> dsB(G); dsB.copy_from_host(sB.data());
  cutlass::DeviceAllocation<StrideD> dsD(G); dsD.copy_from_host(sD.data());
  cutlass::DeviceAllocation<LayoutSFA> dlA(G); dlA.copy_from_host(lA.data());
  cutlass::DeviceAllocation<LayoutSFB> dlB(G); dlB.copy_from_host(lB.data());
  cutlass::DeviceAllocation<const ElementInput *> dpA(G); dpA.copy_from_host(hA.data());
  cutlass::DeviceAllocation<const ElementInput *> dpB(G); dpB.copy_from_host(hB.data());
  cutlass::DeviceAllocation<const ElementSF *> dpSFA(G); dpSFA.copy_from_host(hSFA.data());
  cutlass::DeviceAllocation<const ElementSF *> dpSFB(G); dpSFB.copy_from_host(hSFB.data());
  cutlass::DeviceAllocation<ElementD *> dpD(G); dpD.copy_from_host(hD.data());

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
      {G, dps.get(), ps.data()},
      {reinterpret_cast<const ElementInput **>(dpA.get()), dsA.get(),
       reinterpret_cast<const ElementInput **>(dpB.get()), dsB.get(),
       reinterpret_cast<const ElementSF **>(dpSFA.get()), dlA.get(),
       reinterpret_cast<const ElementSF **>(dpSFB.get()), dlB.get()},
      {fusion, nullptr, nullptr, dpD.get(), dsD.get()},
      hw};

  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    std::printf("%-16s %-6s %6d %8lld %10s %9s %9s %9s\n", tile, gemm_name,
                (int)(tokens / G), tokens, "-", "-", "-", "can_implement=no");
    return;
  }
  cutlass::DeviceAllocation<uint8_t> ws(Gemm::get_workspace_size(args));
  if (gemm.initialize(args, ws.get()) != cutlass::Status::kSuccess) {
    std::printf("%-16s %-6s %6d %8lld %10s %9s %9s %9s\n", tile, gemm_name,
                (int)(tokens / G), tokens, "-", "-", "-", "initialize=no");
    return;
  }

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  for (int i = 0; i < 5; ++i) gemm.run();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(e0));
  for (int i = 0; i < iterations; ++i) gemm.run();
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float total = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&total, e0, e1));
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));

  double ms = total / iterations;
  double flops = 2.0 * (double)tokens * N * K;
  double wbytes = (double)G * N * K * (0.5 + 1.0 / SFVecSize);
  double gbps = wbytes / (ms * 1e-3) / 1e9;
  std::printf("%-16s %-6s %6d %8lld %10.3f %9.2f %9.1f %8.1f%%\n", tile,
              gemm_name, (int)(tokens / G), tokens, ms,
              flops / (ms * 1e-3) / 1e12, gbps, 100.0 * gbps / kRoofline);
}

struct ShapeCase { const char *name; int N; int K; };

int main(int argc, char **argv) {
  int groups = 288;
  int iterations = 20;
  std::vector<int> means = {2, 4, 8};
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--groups=", 0) == 0) groups = std::atoi(a.c_str() + 9);
    else if (a.rfind("--iterations=", 0) == 0) iterations = std::atoi(a.c_str() + 13);
    else if (a == "--help") {
      std::printf("usage: %s [--groups=N] [--iterations=N]\n", argv[0]);
      return 0;
    }
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device      : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("groups      : %d experts, ragged M around mean\n", groups);
  std::printf("iterations  : %d\n\n", iterations);

  const ShapeCase shapes[] = {{"w13", 4096, 4096}, {"w2", 4096, 2048}};

  std::printf("%-16s %-6s %6s %8s %10s %9s %9s %9s\n", "tile MxNxK sched", "gemm",
              "M_mean", "tokens", "ms", "TFLOPS", "GB/s", "roof%");
  std::printf("%s\n", std::string(80, '-').c_str());

  for (int mean : means) {
    std::vector<int> Ms = make_group_m(mean, groups, true);
    for (const auto &sc : shapes) {
      run_case<CoopCfg<128, 128, 128>>("128x128x128", sc.name, sc.N, sc.K, Ms, iterations);
      run_case<CoopCfg<128, 64, 128>>("128x64x128", sc.name, sc.N, sc.K, Ms, iterations);
      run_case<CoopCfg<128, 256, 128>>("128x256x128", sc.name, sc.N, sc.K, Ms, iterations);
      run_case<CoopCfg<128, 128, 256>>("128x128x256", sc.name, sc.N, sc.K, Ms, iterations);
    }
  }

  std::printf("\nroofline = 238 GB/s measured device read bandwidth\n");
  return 0;
}

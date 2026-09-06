// One MoE layer step end to end on sm_121 (GB10): w13 -> SwiGLU+quantize -> w2.
//
// Question: the engine's governing metric is weight bytes streamed per generated
// token. Today the layer is two grouped GEMMs, 14.1 ms (w13) + 7.0 ms (w2)
// = 21.1 ms (blog/posts/kernels/2026-09-06-cutlass-nvfp4-sm121/). Can the two
// be fused so the expert weight stream is issued once per layer step?
//
// Bytes first, because it decides what fusion can possibly buy:
//   w13 weights  G*4096*4096*(0.5 + 1/16) = 2.718 GB
//   w2  weights  G*4096*2048*(0.5 + 1/16) = 1.359 GB
//   total        4.077 GB, and w13 and w2 share no bytes
// So nothing is streamed twice today. The gap is 21.1 ms measured against
// 4.077 GB / 238 GB/s = 17.13 ms of roofline, i.e. achieved bandwidth, not
// redundancy. Every variant below is timed against that 17.13 ms floor.
//
// Variants (total layer ms, same buffers, same iteration count):
//   separate     w13 grouped GEMM, SwiGLU+quantize kernel, w2 grouped GEMM
//   graph        the same three nodes replayed from one CUDA graph
//   chunked      experts split into C chunks, chunks round-robin on 2 streams
//                so chunk c's w2 overlaps chunk c+1's w13 (experts are
//                independent; the w13->w2 dependency is per expert)
//   one-launch   a single grouped GEMM over 2G groups (w13 groups then w2
//                groups). Dependency-violating, so numerically wrong on the
//                second half. It exists only to measure the scheduling ceiling
//                of any single-kernel fusion of the two GEMMs.
//
// Shapes come from the on-disk GLM-5.3-Flash checkpoint config:
//   hidden_size=4096, moe_intermediate_size=2048, n_routed_experts=288,
//   hidden_act=silu.  w13: N=4096 K=4096.  w2: N=4096 K=2048.

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
// CUTLASS kernel configuration (identical to bench/nvfp4_grouped_gemm.cu)
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
using ElementC   = void;
using LayoutCTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = AlignmentD;

using ElementAccumulator = float;
using ArchTag       = cutlass::arch::Sm120;
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
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;

constexpr int SFVecSize = Gemm::GemmKernel::CollectiveMainloop::SFVecSize;

// Model dims.
constexpr int kHidden = 4096;
constexpr int kInter  = 2048;

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

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

// Encode one float to an e2m1 nibble, round to nearest representable.
// Block scales are held at 1.0 everywhere in this bench (see setup notes), so
// this is the whole quantizer.
__device__ __forceinline__ uint8_t encode_e2m1(float v) {
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  uint8_t sign = v < 0.f ? 0x8 : 0x0;
  float a = fabsf(v);
  int best = 0;
  float bd = fabsf(a - mag[0]);
  #pragma unroll
  for (int i = 1; i < 8; ++i) {
    float d = fabsf(a - mag[i]);
    if (d < bd) { bd = d; best = i; }
  }
  return (uint8_t)(sign | (uint8_t)best);
}

// SwiGLU over the fused w13 output plus NVFP4 quantization of the result, which
// is what w2 needs as its A operand. d1 is (rows x 4096) BF16 row major with
// gate in columns [0,2048) and up in [2048,4096). a2 is (rows x 2048) packed
// FP4, two values per byte.
//
// gate_proj and up_proj carry different weight_scale_2 globals on the real
// checkpoint (4.650e-05 vs 3.778e-05 at layer 10, ratio 1.2308), so a fused w13
// GEMM cannot apply one epilogue alpha across both halves. WithAlpha applies the
// two globals right here, where the kernel already holds both halves in
// registers and they cost two fp32 multiplies.
template <bool WithAlpha>
__global__ void swiglu_quant(const ElementD *__restrict__ d1,
                             uint8_t *__restrict__ a2, long long rows,
                             const int *__restrict__ row_group,
                             const float *__restrict__ alpha_gate,
                             const float *__restrict__ alpha_up) {
  const long long bytes_per_row = kInter / 2;
  long long total = rows * bytes_per_row;
  long long stride = (long long)blockDim.x * gridDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += stride) {
    long long row = i / bytes_per_row;
    long long b   = i - row * bytes_per_row;
    long long c   = b * 2;
    const ElementD *r = d1 + row * kHidden;
    float ag = 1.f, au = 1.f;
    if constexpr (WithAlpha) {
      int g = row_group[row];
      ag = alpha_gate[g];
      au = alpha_up[g];
    }
    float g0 = (float)r[c] * ag;
    float u0 = (float)r[c + kInter] * au;
    float g1 = (float)r[c + 1] * ag;
    float u1 = (float)r[c + 1 + kInter] * au;
    float s0 = (g0 / (1.f + __expf(-g0))) * u0;
    float s1 = (g1 / (1.f + __expf(-g1))) * u1;
    a2[i] = (uint8_t)(encode_e2m1(s0) | (encode_e2m1(s1) << 4));
  }
}

__global__ void checksum_bf16(const ElementD *__restrict__ p, long long n,
                              double *out) {
  double acc = 0.0;
  long long stride = (long long)blockDim.x * gridDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride)
    acc += (double)(float)p[i];
  atomicAdd(out, acc);
}

// ---------------------------------------------------------------------------
// Option (b) numerics: fold the gate/up global ratio into one half's e4m3
// block scales at load, so one epilogue alpha covers a fused w13.
// ---------------------------------------------------------------------------

// OCP E4M3: 4 exponent bits (bias 7), 3 mantissa bits, max 448, subnormal
// step 2^-9. Round to nearest.
static float quant_e4m3(float x) {
  if (x == 0.f) return 0.f;
  float sgn = x < 0.f ? -1.f : 1.f;
  float a = std::fabs(x);
  if (a >= 448.f) return sgn * 448.f;
  int e = 0;
  std::frexp(a, &e);
  int exp = e - 1;
  if (exp < -6) exp = -6;
  float step = std::ldexp(1.0f, exp - 3);
  float q = std::round(a / step) * step;
  if (q > 448.f) q = 448.f;
  return sgn * q;
}

static void probe_scale_fold(double ratio) {
  // Every finite positive e4m3 code, which is the full set of block scales a
  // loader could be asked to rescale.
  double max_rel = 0.0, sum_rel = 0.0;
  int n = 0, saturated = 0;
  for (int code = 1; code < 127; ++code) {
    int ec = (code >> 3) & 0xF;
    int mc = code & 0x7;
    if (ec == 0xF && mc == 0x7) continue;  // NaN
    float s;
    if (ec == 0) s = std::ldexp((float)mc / 8.0f, -6);
    else s = std::ldexp(1.0f + (float)mc / 8.0f, ec - 7);
    double exact = (double)s * ratio;
    if (exact > 448.0) { ++saturated; }
    float folded = quant_e4m3((float)exact);
    double rel = std::fabs((double)folded - exact) / exact;
    max_rel = std::max(max_rel, rel);
    sum_rel += rel;
    ++n;
  }
  std::printf("scale fold probe (option b): ratio %.4f folded into e4m3 block scales\n",
              ratio);
  std::printf("  codes tested %d, mean rel err %.4f%%, max rel err %.4f%%, saturated %d\n",
              n, 100.0 * sum_rel / n, 100.0 * max_rel, saturated);
  std::printf("  NVFP4 e2m1 mantissa is 1 bit, so its own step is 50%% near 1.0;\n"
              "  the fold error above is what option (b) adds on top of that.\n\n");
}

// ---------------------------------------------------------------------------
// Host-side grouped-GEMM argument arrays
// ---------------------------------------------------------------------------

static size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Everything a grouped GEMM needs, held as one device array per field so a
// contiguous sub-range of groups can be handed to its own Gemm object.
struct ArgArrays {
  int G = 0;
  std::vector<typename ProblemShape::UnderlyingProblemShape> ps_host;
  std::vector<StrideA> sA_host;
  std::vector<StrideB> sB_host;
  std::vector<StrideD> sD_host;
  std::vector<LayoutSFA> lSFA_host;
  std::vector<LayoutSFB> lSFB_host;
  std::vector<const ElementInput *> pA_host, pB_host;
  std::vector<const ElementSF *> pSFA_host, pSFB_host;
  std::vector<ElementD *> pD_host;

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> ps;
  cutlass::DeviceAllocation<StrideA> sA;
  cutlass::DeviceAllocation<StrideB> sB;
  cutlass::DeviceAllocation<StrideD> sD;
  cutlass::DeviceAllocation<LayoutSFA> lSFA;
  cutlass::DeviceAllocation<LayoutSFB> lSFB;
  cutlass::DeviceAllocation<const ElementInput *> pA, pB;
  cutlass::DeviceAllocation<const ElementSF *> pSFA, pSFB;
  cutlass::DeviceAllocation<ElementD *> pD;

  void upload() {
    G = (int)ps_host.size();
    ps.reset(G);    ps.copy_from_host(ps_host.data());
    sA.reset(G);    sA.copy_from_host(sA_host.data());
    sB.reset(G);    sB.copy_from_host(sB_host.data());
    sD.reset(G);    sD.copy_from_host(sD_host.data());
    lSFA.reset(G);  lSFA.copy_from_host(lSFA_host.data());
    lSFB.reset(G);  lSFB.copy_from_host(lSFB_host.data());
    pA.reset(G);    pA.copy_from_host(pA_host.data());
    pB.reset(G);    pB.copy_from_host(pB_host.data());
    pSFA.reset(G);  pSFA.copy_from_host(pSFA_host.data());
    pSFB.reset(G);  pSFB.copy_from_host(pSFB_host.data());
    pD.reset(G);    pD.copy_from_host(pD_host.data());
  }

  void append(const ArgArrays &o) {
    ps_host.insert(ps_host.end(), o.ps_host.begin(), o.ps_host.end());
    sA_host.insert(sA_host.end(), o.sA_host.begin(), o.sA_host.end());
    sB_host.insert(sB_host.end(), o.sB_host.begin(), o.sB_host.end());
    sD_host.insert(sD_host.end(), o.sD_host.begin(), o.sD_host.end());
    lSFA_host.insert(lSFA_host.end(), o.lSFA_host.begin(), o.lSFA_host.end());
    lSFB_host.insert(lSFB_host.end(), o.lSFB_host.begin(), o.lSFB_host.end());
    pA_host.insert(pA_host.end(), o.pA_host.begin(), o.pA_host.end());
    pB_host.insert(pB_host.end(), o.pB_host.begin(), o.pB_host.end());
    pSFA_host.insert(pSFA_host.end(), o.pSFA_host.begin(), o.pSFA_host.end());
    pSFB_host.insert(pSFB_host.end(), o.pSFB_host.begin(), o.pSFB_host.end());
    pD_host.insert(pD_host.end(), o.pD_host.begin(), o.pD_host.end());
  }
};

// One initialized Gemm over groups [off, off+n) of an ArgArrays.
struct GemmSlice {
  Gemm gemm;
  cutlass::DeviceAllocation<uint8_t> ws;

  void init(ArgArrays &a, int off, int n, const cutlass::KernelHardwareInfo &hw) {
    typename Gemm::Arguments args;
    decltype(args.epilogue.thread) fusion;
    fusion.alpha = 1.0f;
    fusion.beta = 0.0f;
    fusion.alpha_ptr = nullptr;
    fusion.beta_ptr = nullptr;

    args = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {n, a.ps.get() + off, a.ps_host.data() + off},
        {reinterpret_cast<const ElementInput **>(a.pA.get() + off), a.sA.get() + off,
         reinterpret_cast<const ElementInput **>(a.pB.get() + off), a.sB.get() + off,
         reinterpret_cast<const ElementSF **>(a.pSFA.get() + off), a.lSFA.get() + off,
         reinterpret_cast<const ElementSF **>(a.pSFB.get() + off), a.lSFB.get() + off},
        {fusion, nullptr, nullptr, a.pD.get() + off, a.sD.get() + off},
        hw};

    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "cutlass cannot implement slice off=%d n=%d\n", off, n);
      std::exit(1);
    }
    ws.reset(Gemm::get_workspace_size(args));
    CUTLASS_CHECK_STATUS(gemm.initialize(args, ws.get()));
  }

  void run(cudaStream_t s) { CUTLASS_CHECK_STATUS(gemm.run(s)); }
};

// ---------------------------------------------------------------------------
// Bench
// ---------------------------------------------------------------------------

struct Options {
  int groups = 288;
  int iterations = 20;
  std::vector<int> m_means = {8, 32};
  std::vector<int> chunk_counts = {2, 4, 8, 16};
  bool ragged = true;
  bool verify = true;
};

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

int main(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--groups=", 0) == 0) opt.groups = std::atoi(a.c_str() + 9);
    else if (a.rfind("--iterations=", 0) == 0) opt.iterations = std::atoi(a.c_str() + 13);
    else if (a.rfind("--m=", 0) == 0) { opt.m_means.assign(1, std::atoi(a.c_str() + 4)); }
    else if (a == "--uniform-m") opt.ragged = false;
    else if (a == "--no-verify") opt.verify = false;
    else if (a == "--help") {
      std::printf("usage: %s [--groups=N] [--iterations=N] [--m=N] "
                  "[--uniform-m] [--no-verify]\n", argv[0]);
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
  std::printf("SF vector size: %d\n\n", SFVecSize);

  probe_scale_fold(4.650e-05 / 3.778e-05);

  const double kRoofline = 238.0;  // GB/s, measured device read bandwidth
  const int G = opt.groups;

  cutlass::KernelHardwareInfo hw;
  hw.device_id = 0;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  cudaStream_t s0, s1, scap;
  CUDA_CHECK(cudaStreamCreate(&s0));
  CUDA_CHECK(cudaStreamCreate(&s1));
  CUDA_CHECK(cudaStreamCreate(&scap));

  double *d_sum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));

  for (int mean : opt.m_means) {
    std::vector<int> Ms = make_group_m(mean, G, opt.ragged);
    long long tokens = 0;
    for (int m : Ms) tokens += m;

    // ---- offsets ---------------------------------------------------------
    // Activations are one contiguous (tokens x width) block, so a group's rows
    // are just an offset. Weights are one block per GEMM kind.
    std::vector<size_t> offA1(G), offA2(G), offD1(G), offD2(G);
    std::vector<size_t> offSFA1(G), offSFA2(G), offSFB13(G), offSFB2(G);
    std::vector<size_t> szSFA1(G), szSFA2(G), szSFB13(G), szSFB2(G);

    std::vector<LayoutSFA> lSFA13(G), lSFA2(G);
    std::vector<LayoutSFB> lSFB13(G), lSFB2(G);

    size_t curA1 = 0, curA2 = 0, curD1 = 0, curD2 = 0;
    size_t curSFA1 = 0, curSFA2 = 0, curSFB13 = 0, curSFB2 = 0;
    for (int i = 0; i < G; ++i) {
      int M = Ms[i];
      lSFA13[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, kHidden, kHidden, 1));
      lSFB13[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, kHidden, kHidden, 1));
      lSFA2[i]  = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, kHidden, kInter, 1));
      lSFB2[i]  = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, kHidden, kInter, 1));

      offA1[i] = curA1; curA1 += align_up((size_t)M * kHidden / 2, 256);
      offA2[i] = curA2; curA2 += align_up((size_t)M * kInter / 2, 256);
      offD1[i] = curD1; curD1 += (size_t)M * kHidden;
      offD2[i] = curD2; curD2 += (size_t)M * kHidden;

      szSFA1[i]  = (size_t)size(filter_zeros(lSFA13[i]));
      szSFA2[i]  = (size_t)size(filter_zeros(lSFA2[i]));
      szSFB13[i] = (size_t)size(filter_zeros(lSFB13[i]));
      szSFB2[i]  = (size_t)size(filter_zeros(lSFB2[i]));
      offSFA1[i]  = curSFA1;  curSFA1  += align_up(szSFA1[i], 256);
      offSFA2[i]  = curSFA2;  curSFA2  += align_up(szSFA2[i], 256);
      offSFB13[i] = curSFB13; curSFB13 += align_up(szSFB13[i], 256);
      offSFB2[i]  = curSFB2;  curSFB2  += align_up(szSFB2[i], 256);
    }

    // The activation blocks must stay one contiguous (tokens x width) tensor
    // for the SwiGLU kernel, so A1/A2 offsets are recomputed without padding.
    curA1 = 0; curA2 = 0;
    for (int i = 0; i < G; ++i) {
      offA1[i] = curA1; curA1 += (size_t)Ms[i] * kHidden / 2;
      offA2[i] = curA2; curA2 += (size_t)Ms[i] * kInter / 2;
    }

    // ---- allocations -----------------------------------------------------
    cutlass::DeviceAllocation<uint8_t> A1(curA1), A2(curA2);
    cutlass::DeviceAllocation<uint8_t> SFA1(curSFA1), SFA2(curSFA2);
    cutlass::DeviceAllocation<uint8_t> W13((size_t)G * kHidden * kHidden / 2);
    cutlass::DeviceAllocation<uint8_t> W2((size_t)G * kHidden * kInter / 2);
    cutlass::DeviceAllocation<uint8_t> SFB13(curSFB13), SFB2(curSFB2);
    cutlass::DeviceAllocation<ElementD> D1(curD1), D2(curD2);

    // Per-expert, per-half global scales. The absolute checkpoint globals
    // (4.650e-05 / 3.778e-05) would drive every SwiGLU product to zero here
    // because this bench pins block scales at 1.0, so the measured ratio is
    // carried symmetrically instead: gate x sqrt(r), up / sqrt(r). Same two
    // fp32 multiplies, same memory traffic, non-degenerate arithmetic.
    const double kGateGlobal = 4.650e-05, kUpGlobal = 3.778e-05;
    const double kRatio = kGateGlobal / kUpGlobal;
    std::vector<float> h_ag(G), h_au(G);
    for (int i = 0; i < G; ++i) {
      h_ag[i] = (float)std::sqrt(kRatio);
      h_au[i] = (float)(1.0 / std::sqrt(kRatio));
    }
    std::vector<int> h_rg;
    h_rg.reserve((size_t)tokens);
    for (int i = 0; i < G; ++i)
      for (int r = 0; r < Ms[i]; ++r) h_rg.push_back(i);
    cutlass::DeviceAllocation<float> AG(G), AU(G);
    AG.copy_from_host(h_ag.data());
    AU.copy_from_host(h_au.data());
    cutlass::DeviceAllocation<int> RG((size_t)tokens);
    RG.copy_from_host(h_rg.data());

    fill_fp4_random<<<512, 256>>>(A1.get(), (long long)curA1, 11u);
    fill_fp4_random<<<2048, 256>>>(W13.get(), (long long)W13.size(), 77u);
    fill_fp4_random<<<2048, 256>>>(W2.get(), (long long)W2.size(), 99u);
    // All block scales are 1.0 in e4m3 (0x38). Holding them constant is what
    // lets a plain quantizer write into the swizzled CUTLASS SF layout without
    // reproducing the swizzle, and it is the same choice the grouped-GEMM
    // bench made so the two entries stay comparable.
    fill_bytes<<<256, 256>>>(SFA1.get(), (long long)curSFA1, 0x38);
    fill_bytes<<<256, 256>>>(SFA2.get(), (long long)curSFA2, 0x38);
    fill_bytes<<<512, 256>>>(SFB13.get(), (long long)curSFB13, 0x38);
    fill_bytes<<<512, 256>>>(SFB2.get(), (long long)curSFB2, 0x38);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- argument arrays -------------------------------------------------
    ArgArrays a13, a2;
    auto build = [&](ArgArrays &out, int N, int K,
                     cutlass::DeviceAllocation<uint8_t> &Abuf,
                     const std::vector<size_t> &offA,
                     cutlass::DeviceAllocation<uint8_t> &SFAbuf,
                     const std::vector<size_t> &offSFA,
                     cutlass::DeviceAllocation<uint8_t> &Bbuf,
                     cutlass::DeviceAllocation<uint8_t> &SFBbuf,
                     const std::vector<size_t> &offSFB,
                     const std::vector<LayoutSFA> &lA,
                     const std::vector<LayoutSFB> &lB,
                     cutlass::DeviceAllocation<ElementD> &Dbuf,
                     const std::vector<size_t> &offD) {
      out.ps_host.resize(G); out.sA_host.resize(G); out.sB_host.resize(G);
      out.sD_host.resize(G); out.lSFA_host.resize(G); out.lSFB_host.resize(G);
      out.pA_host.resize(G); out.pB_host.resize(G);
      out.pSFA_host.resize(G); out.pSFB_host.resize(G); out.pD_host.resize(G);
      for (int i = 0; i < G; ++i) {
        int M = Ms[i];
        out.ps_host[i] = {M, N, K};
        out.sA_host[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        out.sB_host[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        out.sD_host[i] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
        out.lSFA_host[i] = lA[i];
        out.lSFB_host[i] = lB[i];
        out.pA_host[i] = reinterpret_cast<const ElementInput *>(Abuf.get() + offA[i]);
        out.pB_host[i] = reinterpret_cast<const ElementInput *>(
            Bbuf.get() + (size_t)i * N * K / 2);
        out.pSFA_host[i] = reinterpret_cast<const ElementSF *>(SFAbuf.get() + offSFA[i]);
        out.pSFB_host[i] = reinterpret_cast<const ElementSF *>(SFBbuf.get() + offSFB[i]);
        out.pD_host[i] = Dbuf.get() + offD[i];
      }
      out.upload();
    };

    build(a13, kHidden, kHidden, A1, offA1, SFA1, offSFA1, W13, SFB13, offSFB13,
          lSFA13, lSFB13, D1, offD1);
    build(a2, kHidden, kInter, A2, offA2, SFA2, offSFA2, W2, SFB2, offSFB2,
          lSFA2, lSFB2, D2, offD2);

    // Concatenated 2G-group problem for the single-launch ceiling probe.
    ArgArrays aBoth;
    aBoth.append(a13);
    aBoth.append(a2);
    aBoth.upload();

    GemmSlice g13, g2, gBoth;
    g13.init(a13, 0, G, hw);
    g2.init(a2, 0, G, hw);
    gBoth.init(aBoth, 0, 2 * G, hw);

    // ---- kernels ---------------------------------------------------------
    const int act_blocks = 1024, act_threads = 256;
    auto act = [&](cudaStream_t s) {
      swiglu_quant<true><<<act_blocks, act_threads, 0, s>>>(
          D1.get(), A2.get(), tokens, RG.get(), AG.get(), AU.get());
    };
    auto act_no_alpha = [&](cudaStream_t s) {
      swiglu_quant<false><<<act_blocks, act_threads, 0, s>>>(
          D1.get(), A2.get(), tokens, nullptr, nullptr, nullptr);
    };
    auto act_chunk = [&](cudaStream_t s, int lo, int hi) {
      long long rows = 0;
      for (int i = lo; i < hi; ++i) rows += Ms[i];
      long long row0 = 0;
      for (int i = 0; i < lo; ++i) row0 += Ms[i];
      swiglu_quant<true><<<act_blocks, act_threads, 0, s>>>(
          D1.get() + row0 * kHidden, A2.get() + row0 * kInter / 2, rows,
          RG.get() + row0, AG.get(), AU.get());
    };

    // ---- timing helper ---------------------------------------------------
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    auto time_it = [&](auto &&fn) {
      for (int i = 0; i < 3; ++i) fn();
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(e0));
      for (int i = 0; i < opt.iterations; ++i) fn();
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      return (double)ms / opt.iterations;
    };

    auto checksum = [&]() {
      double h = 0.0;
      CUDA_CHECK(cudaMemcpy(d_sum, &h, sizeof(double), cudaMemcpyHostToDevice));
      checksum_bf16<<<512, 256>>>(D2.get(), (long long)curD2, d_sum);
      CUDA_CHECK(cudaMemcpy(&h, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
      return h;
    };

    // ---- variants --------------------------------------------------------
    auto run_separate = [&]() {
      g13.run(s0);
      act(s0);
      g2.run(s0);
    };

    double ms_w13 = time_it([&]() { g13.run(s0); });
    double ms_act = time_it([&]() { act(s0); });
    double ms_act_noalpha = time_it([&]() { act_no_alpha(s0); });
    double ms_w2  = time_it([&]() { g2.run(s0); });

    double ms_sep = time_it(run_separate);
    CUDA_CHECK(cudaStreamSynchronize(s0));
    double sum_sep = checksum();

    // CUDA graph over the same three nodes.
    cudaGraph_t graph;
    cudaGraphExec_t gexec;
    CUDA_CHECK(cudaStreamBeginCapture(scap, cudaStreamCaptureModeThreadLocal));
    g13.run(scap);
    act(scap);
    g2.run(scap);
    CUDA_CHECK(cudaStreamEndCapture(scap, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
    double ms_graph = time_it([&]() { CUDA_CHECK(cudaGraphLaunch(gexec, s0)); });
    CUDA_CHECK(cudaStreamSynchronize(s0));
    double sum_graph = checksum();

    // Chunked, two streams. Experts are independent, so a chunk's whole
    // w13 -> act -> w2 chain can sit on one stream and overlap another chunk.
    struct ChunkPlan {
      int chunks;
      double ms;
      double sum;
    };
    std::vector<ChunkPlan> chunk_results;
    for (int C : opt.chunk_counts) {
      if (C > G) continue;
      std::vector<GemmSlice> c13(C), c2(C);
      std::vector<int> lo(C), hi(C);
      for (int c = 0; c < C; ++c) {
        lo[c] = (int)((long long)c * G / C);
        hi[c] = (int)((long long)(c + 1) * G / C);
        c13[c].init(a13, lo[c], hi[c] - lo[c], hw);
        c2[c].init(a2, lo[c], hi[c] - lo[c], hw);
      }
      auto run_chunked = [&]() {
        for (int c = 0; c < C; ++c) {
          cudaStream_t s = (c & 1) ? s1 : s0;
          c13[c].run(s);
          act_chunk(s, lo[c], hi[c]);
          c2[c].run(s);
        }
        CUDA_CHECK(cudaStreamSynchronize(s0));
        CUDA_CHECK(cudaStreamSynchronize(s1));
      };
      double ms = time_it(run_chunked);
      chunk_results.push_back({C, ms, checksum()});
    }

    // Single-launch ceiling: 2G groups in one grouped GEMM. Wrong numerically
    // (the w2 half reads a stale A2), timed only as a scheduling upper bound.
    double ms_one = time_it([&]() { gBoth.run(s0); });

    // ---- accounting ------------------------------------------------------
    const double w13_bytes = (double)G * kHidden * kHidden * (0.5 + 1.0 / SFVecSize);
    const double w2_bytes  = (double)G * kHidden * kInter  * (0.5 + 1.0 / SFVecSize);
    const double wbytes    = w13_bytes + w2_bytes;
    // Activation round trip through the SwiGLU kernel: read (tokens x 4096)
    // BF16, write (tokens x 2048) FP4, and w2 reads that back.
    const double act_bytes = (double)tokens * kHidden * 2.0
                           + (double)tokens * kInter * 0.5 * 2.0;
    const double roof_ms = wbytes / (kRoofline * 1e9) * 1e3;

    std::printf("=== M mean %d, %lld tokens, %d experts ===\n", mean, tokens, G);
    std::printf("weight bytes  : w13 %.3f GB + w2 %.3f GB = %.3f GB\n",
                w13_bytes / 1e9, w2_bytes / 1e9, wbytes / 1e9);
    std::printf("activation    : %.3f GB round trip (%.2f%% of weight stream)\n",
                act_bytes / 1e9, 100.0 * act_bytes / wbytes);
    std::printf("roofline      : %.3f ms at %.0f GB/s\n\n", roof_ms, kRoofline);

    std::printf("%-22s %9s %9s %9s %8s %s\n",
                "variant", "ms", "GB/s", "roof%", "vs sep", "correct");
    std::printf("%s\n", std::string(78, '-').c_str());
    auto row = [&](const char *name, double ms, const char *ok) {
      std::printf("%-22s %9.3f %9.1f %9.1f %8.3fx %s\n", name, ms,
                  wbytes / (ms * 1e-3) / 1e9,
                  100.0 * (wbytes / (ms * 1e-3) / 1e9) / kRoofline,
                  ms_sep / ms, ok);
    };
    row("separate", ms_sep, "baseline");
    row("cuda-graph", ms_graph,
        std::fabs(sum_graph - sum_sep) < 1e-6 * std::fabs(sum_sep) ? "match" : "DIFFERS");
    for (const auto &cr : chunk_results) {
      char nm[32];
      std::snprintf(nm, sizeof(nm), "chunked-%d-2stream", cr.chunks);
      row(nm, cr.ms,
          std::fabs(cr.sum - sum_sep) < 1e-6 * std::fabs(sum_sep) ? "match" : "DIFFERS");
    }
    row("one-launch-2G", ms_one, "INVALID (ceiling)");
    std::printf("\n");

    std::printf("%-22s %9s %9s\n", "stage", "ms", "share");
    std::printf("%s\n", std::string(42, '-').c_str());
    std::printf("%-22s %9.3f %8.1f%%\n", "w13 grouped gemm", ms_w13, 100.0 * ms_w13 / ms_sep);
    std::printf("%-22s %9.3f %8.1f%%\n", "swiglu + quantize", ms_act, 100.0 * ms_act / ms_sep);
    std::printf("%-22s %9.3f %8.1f%%\n", "  same, alpha=1", ms_act_noalpha,
                100.0 * ms_act_noalpha / ms_sep);
    std::printf("%-22s %9.3f %8.1f%%\n", "w2 grouped gemm", ms_w2, 100.0 * ms_w2 / ms_sep);
    std::printf("%-22s %9.3f %8.1f%%\n", "sum of stages", ms_w13 + ms_act + ms_w2,
                100.0 * (ms_w13 + ms_act + ms_w2) / ms_sep);
    std::printf("\n");

    CUDA_CHECK(cudaGraphExecDestroy(gexec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
  }

  CUDA_CHECK(cudaFree(d_sum));
  CUDA_CHECK(cudaStreamDestroy(s0));
  CUDA_CHECK(cudaStreamDestroy(s1));
  CUDA_CHECK(cudaStreamDestroy(scap));
  std::printf("roofline = 238 GB/s measured device read bandwidth\n");
  return 0;
}

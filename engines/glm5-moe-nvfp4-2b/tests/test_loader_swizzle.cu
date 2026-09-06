// Real GLM-5.3-Flash NVFP4 expert bytes -> CUTLASS block-scaled GEMM -> FP32
// reference, on sm_121 (GB10).
//
// This closes the gap flagged in fuels/glm-5.3-flash/artifacts.md section 3:
// the earlier grouped-GEMM bench verified with every block scale set to 1.0,
// which is exactly the case where the swizzled CUTLASS scale layout and a
// plain row-major layout agree, so it could not have caught a wrong swizzle.
// Here the scales are the checkpoint's own and are all different.
//
// Three checks, in order of what they can catch:
//   1. rocket::fuel::SfLayout against the instantiated CUTLASS layout, for
//      every (row, scale) coordinate of both shapes. A pure layout check, no
//      GPU needed to be wrong.
//   2. CUTLASS output against an FP32 CPU matmul over fully dequantized
//      operands (fp4 * e4m3 block scale * f32 global).
//   3. Load timing for one expert projection, extrapolated to a full booster.
//
// Usage:
//   test-loader-swizzle [--dir=<snapshot>] [--layer=10] [--expert=0] [--m=8] [--warm]
// Exits 77 (ctest SKIP) when the checkpoint is not on this node.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>

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

#include "nvfp4.h"
#include "safetensors.h"

using namespace cute;
namespace fuel = rocket::fuel;

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
// Kernel configuration. Same collective as bench/nvfp4_grouped_gemm.cu, so the
// layout proved here is the layout that bench and the engine will use. The one
// change is ElementD: this writes FP32 rather than BF16, so the comparison
// measures the kernel and the scale layout instead of epilogue rounding.
// ---------------------------------------------------------------------------

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e2m1_t;

using ElementA   = cutlass::nv_float4_t<ElementInput>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<ElementInput>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD   = float;
using ElementC   = void;
using LayoutCTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = AlignmentD;

using ElementAccumulator = float;
using ArchTag       = cutlass::arch::Sm120;
using OperatorClass  = cutlass::arch::OpClassBlockScaledTensorOp;
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
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;

constexpr int SFVecSize = Gemm::GemmKernel::CollectiveMainloop::SFVecSize;

// ---------------------------------------------------------------------------

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++g_failures;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Ask the kernel to drop this file's clean page cache so a load measurement is
// not just a memcpy from RAM. Needs no privilege; a no-op if it fails.
void drop_page_cache(const std::filesystem::path& file) {
  int fd = ::open(file.c_str(), O_RDONLY);
  if (fd < 0) return;
  ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  ::close(fd);
}

struct QuantizedActivation {
  int m = 0, k = 0;
  std::vector<std::uint8_t> packed;       // [m, k/2]
  std::vector<std::uint8_t> block_scale;  // [m, k/16] linear
  float global_scale = 0.0f;
  std::vector<float> dequantized;         // [m, k] what the kernel actually sees
};

// Standard NVFP4 activation recipe: one FP32 per-tensor global scale
// amax/(448*6), one e4m3 scale per 16 contiguous K elements, e2m1 payload.
QuantizedActivation quantize_activation(const std::vector<float>& a, int m, int k) {
  QuantizedActivation q;
  q.m = m;
  q.k = k;
  q.packed.assign(static_cast<std::size_t>(m) * k / 2, 0);
  q.block_scale.assign(static_cast<std::size_t>(m) * k / 16, 0);
  q.dequantized.assign(static_cast<std::size_t>(m) * k, 0.0f);

  float amax = 0.0f;
  for (float v : a) amax = std::max(amax, std::fabs(v));
  q.global_scale = (amax > 0.0f) ? amax / (448.0f * 6.0f) : 1.0f;

  const int nblk = k / 16;
  for (int row = 0; row < m; ++row) {
    for (int b = 0; b < nblk; ++b) {
      float bmax = 0.0f;
      for (int j = 0; j < 16; ++j)
        bmax = std::max(bmax, std::fabs(a[static_cast<std::size_t>(row) * k + b * 16 + j]));
      const std::uint8_t sf_bits = fuel::float_to_e4m3(bmax / 6.0f / q.global_scale);
      q.block_scale[static_cast<std::size_t>(row) * nblk + b] = sf_bits;
      const float step = fuel::e4m3_to_float(sf_bits) * q.global_scale;
      for (int j = 0; j < 16; ++j) {
        const std::size_t idx = static_cast<std::size_t>(row) * k + b * 16 + j;
        const std::uint8_t nib = (step > 0.0f) ? fuel::float_to_e2m1(a[idx] / step) : 0;
        q.dequantized[idx] = fuel::e2m1_to_float(nib) * step;
        std::uint8_t& byte = q.packed[idx / 2];
        if ((idx & 1) == 0) byte = static_cast<std::uint8_t>((byte & 0xF0) | nib);
        else byte = static_cast<std::uint8_t>((byte & 0x0F) | (nib << 4));
      }
    }
  }
  return q;
}

struct ErrorStats {
  double max_abs = 0.0;
  double max_ref_abs = 0.0;
  double max_rel_significant = 0.0;  // over elements above 1% of max|ref|
  double rms_rel = 0.0;
};

ErrorStats compare(const std::vector<float>& got, const std::vector<float>& ref) {
  ErrorStats s;
  for (float r : ref) s.max_ref_abs = std::max(s.max_ref_abs, static_cast<double>(std::fabs(r)));
  const double floor = 0.01 * s.max_ref_abs;
  double sum_sq = 0.0;
  std::size_t counted = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
    s.max_abs = std::max(s.max_abs, d);
    if (std::fabs(static_cast<double>(ref[i])) > floor) {
      const double rel = d / std::fabs(static_cast<double>(ref[i]));
      s.max_rel_significant = std::max(s.max_rel_significant, rel);
      sum_sq += rel * rel;
      ++counted;
    }
  }
  s.rms_rel = (counted > 0) ? std::sqrt(sum_sq / static_cast<double>(counted)) : 0.0;
  return s;
}

// ---------------------------------------------------------------------------
// Check 1: our closed-form layout against the CUTLASS type itself.
// ---------------------------------------------------------------------------

template <class CutlassLayout>
bool layout_matches(const CutlassLayout& cl, const fuel::SfLayout& ours, const char* tag) {
  const std::size_t cutlass_elems = static_cast<std::size_t>(size(filter_zeros(cl)));
  if (cutlass_elems != ours.bytes()) {
    std::printf("    %s: byte count %zu (ours) vs %zu (cutlass)\n", tag, ours.bytes(), cutlass_elems);
    return false;
  }
  for (std::int64_t row = 0; row < ours.mn; ++row) {
    for (std::int64_t s = 0; s < ours.k_sf; ++s) {
      const std::size_t theirs =
          static_cast<std::size_t>(cl(static_cast<int>(row), static_cast<int>(s * ours.sf_vec), 0));
      const std::size_t mine = ours.offset(row, s);
      if (theirs != mine) {
        std::printf("    %s: first mismatch at row=%lld sf=%lld, ours=%zu cutlass=%zu\n",
                    tag, static_cast<long long>(row), static_cast<long long>(s), mine, theirs);
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------

struct Options {
  std::string dir;
  int layer = 10;
  int expert = 0;
  int m = 8;
  bool cold = true;
};

// Threshold. Both paths consume bit-identical FP4 payloads and bit-identical
// e4m3 block scales, so any difference is accumulation order over K, not
// quantization: CUTLASS accumulates in FP32 tensor cores in tile order, the
// reference accumulates in FP32 in K order. For K=4096 the worst-case relative
// error of FP32 summation is bounded by K*eps = 4096 * 1.19e-7 = 4.9e-4, and
// the realistic random-walk figure is sqrt(K)*eps = 7.6e-6. 1e-4 sits above
// the random-walk bound with margin and far below the O(1) error a wrong
// scale swizzle produces (a misplaced scale changes a whole 16-element block
// by an e4m3 ratio, which is a factor, not a rounding).
constexpr double kMaxRelError = 1e-4;

int run_shape(fuel::Checkpoint& ckpt, const Options& opt, const char* proj_name, const char* label) {
  std::printf("\n== %s (%s, layer %d expert %d) ==\n", label, proj_name, opt.layer, opt.expert);

  // ---- load one expert projection, timed ---------------------------------
  const std::string tensor_name = "model.language_model.layers." + std::to_string(opt.layer) +
                                  ".mlp.experts." + std::to_string(opt.expert) + "." + proj_name +
                                  ".weight";
  auto t0 = std::chrono::steady_clock::now();
  fuel::ExpertProj w = fuel::load_expert_proj(ckpt, opt.layer, opt.expert, proj_name);
  const double t_map = seconds_since(t0);

  const fuel::Shard& shard = ckpt.mapped_shard_for(tensor_name);
  std::printf("  shard         : %s\n", shard.path().filename().string().c_str());
  std::printf("  blob          : %s\n", shard.blob_path().string().c_str());
  std::printf("  shard bytes   : %.2f GiB (header %zu B)\n",
              static_cast<double>(shard.file_bytes()) / (1024.0 * 1024.0 * 1024.0),
              shard.header_bytes());
  std::printf("  weight        : N=%lld K=%lld, packed %.2f MiB, scales %.2f MiB, global=%.9g\n",
              static_cast<long long>(w.n), static_cast<long long>(w.k),
              static_cast<double>(w.packed_bytes()) / (1024.0 * 1024.0),
              static_cast<double>(w.scale_bytes()) / (1024.0 * 1024.0), w.global_scale);

  if (opt.cold) drop_page_cache(shard.blob_path());

  // Touch the bytes this expert needs, which is what actually costs I/O.
  std::vector<std::uint8_t> packed_host(w.packed_bytes());
  std::vector<std::uint8_t> scale_host(w.scale_bytes());
  t0 = std::chrono::steady_clock::now();
  std::memcpy(packed_host.data(), w.packed, packed_host.size());
  std::memcpy(scale_host.data(), w.block_scale, scale_host.size());
  const double t_read = seconds_since(t0);

  // ---- shapes and layouts -------------------------------------------------
  const int M = opt.m;
  const int N = static_cast<int>(w.n);
  const int K = static_cast<int>(w.k);

  auto lSFA_cutlass = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto lSFB_cutlass = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  const fuel::SfLayout lSFA = fuel::sf_layout(M, K, SFVecSize);
  const fuel::SfLayout lSFB = fuel::sf_layout(N, K, SFVecSize);

  std::printf("  SFB layout    : %lldx%lld scales -> %zu B (%lld x %lld atoms of 512 B)\n",
              static_cast<long long>(lSFB.mn), static_cast<long long>(lSFB.k_sf), lSFB.bytes(),
              static_cast<long long>(lSFB.mn_tiles), static_cast<long long>(lSFB.k_tiles));

  check(layout_matches(lSFA_cutlass, lSFA, "SFA"), "SFA layout matches CUTLASS for every coordinate");
  check(layout_matches(lSFB_cutlass, lSFB, "SFB"), "SFB layout matches CUTLASS for every coordinate");

  // ---- swizzle the real scales -------------------------------------------
  std::vector<std::uint8_t> sfb_swizzled(lSFB.bytes());
  t0 = std::chrono::steady_clock::now();
  fuel::swizzle_block_scales(scale_host.data(), lSFB, sfb_swizzled.data());
  const double t_swizzle = seconds_since(t0);

  // A wrong-but-plausible loader would upload the scales as they sit on disk.
  // Confirm that is a different byte string, otherwise this test proves nothing.
  bool swizzle_is_identity = (sfb_swizzled.size() == scale_host.size()) &&
                             std::memcmp(sfb_swizzled.data(), scale_host.data(), scale_host.size()) == 0;
  check(!swizzle_is_identity, "swizzled scales differ from checkpoint-linear scales");

  // ---- activation ---------------------------------------------------------
  std::mt19937 rng(20260906u);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> act(static_cast<std::size_t>(M) * K);
  for (float& v : act) v = fuel::bf16_to_float(fuel::float_to_bf16(dist(rng)));
  QuantizedActivation qa = quantize_activation(act, M, K);

  std::vector<std::uint8_t> sfa_swizzled(lSFA.bytes());
  fuel::swizzle_block_scales(qa.block_scale.data(), lSFA, sfa_swizzled.data());

  // ---- FP32 reference on fully dequantized operands ----------------------
  std::vector<float> wdeq(static_cast<std::size_t>(N) * K);
  double wmax = 0.0;
  for (std::int64_t n = 0; n < N; ++n)
    for (std::int64_t k = 0; k < K; ++k) {
      const float v = w.value(n, k);
      wdeq[static_cast<std::size_t>(n) * K + k] = v;
      wmax = std::max(wmax, static_cast<double>(std::fabs(v)));
    }
  std::printf("  dequant |w|max: %.6g (weight_scale_2 * 448 * 6 = %.6g)\n",
              wmax, static_cast<double>(w.global_scale) * 448.0 * 6.0);

  t0 = std::chrono::steady_clock::now();
  std::vector<float> ref(static_cast<std::size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      const float* arow = qa.dequantized.data() + static_cast<std::size_t>(m) * K;
      const float* brow = wdeq.data() + static_cast<std::size_t>(n) * K;
      for (int k = 0; k < K; ++k) acc += arow[k] * brow[k];
      ref[static_cast<std::size_t>(m) * N + n] = acc;
    }
  const double t_ref = seconds_since(t0);

  // ---- CUTLASS ------------------------------------------------------------
  const int G = 1;
  std::vector<typename ProblemShape::UnderlyingProblemShape> ps_host{{M, N, K}};
  std::vector<StrideA> sA_host{cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1})};
  std::vector<StrideB> sB_host{cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1})};
  std::vector<StrideD> sD_host{cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1})};
  std::vector<LayoutSFA> lSFA_host{lSFA_cutlass};
  std::vector<LayoutSFB> lSFB_host{lSFB_cutlass};

  cutlass::DeviceAllocation<std::uint8_t> dA(qa.packed.size());
  cutlass::DeviceAllocation<std::uint8_t> dB(packed_host.size());
  cutlass::DeviceAllocation<std::uint8_t> dSFA(sfa_swizzled.size());
  cutlass::DeviceAllocation<std::uint8_t> dSFB(sfb_swizzled.size());
  cutlass::DeviceAllocation<ElementD> dD(static_cast<std::size_t>(M) * N);

  dA.copy_from_host(qa.packed.data());
  dSFA.copy_from_host(sfa_swizzled.data());
  auto t_h2d0 = std::chrono::steady_clock::now();
  dB.copy_from_host(packed_host.data());
  dSFB.copy_from_host(sfb_swizzled.data());
  CUDA_CHECK(cudaDeviceSynchronize());
  const double t_h2d = seconds_since(t_h2d0);

  std::vector<const ElementInput*> pA{reinterpret_cast<const ElementInput*>(dA.get())};
  std::vector<const ElementInput*> pB{reinterpret_cast<const ElementInput*>(dB.get())};
  std::vector<const ElementSF*> pSFA{reinterpret_cast<const ElementSF*>(dSFA.get())};
  std::vector<const ElementSF*> pSFB{reinterpret_cast<const ElementSF*>(dSFB.get())};
  std::vector<ElementD*> pD{dD.get()};

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> ps_dev(G);
  ps_dev.copy_from_host(ps_host.data());
  cutlass::DeviceAllocation<StrideA> sA_dev(G); sA_dev.copy_from_host(sA_host.data());
  cutlass::DeviceAllocation<StrideB> sB_dev(G); sB_dev.copy_from_host(sB_host.data());
  cutlass::DeviceAllocation<StrideD> sD_dev(G); sD_dev.copy_from_host(sD_host.data());
  cutlass::DeviceAllocation<LayoutSFA> lSFA_dev(G); lSFA_dev.copy_from_host(lSFA_host.data());
  cutlass::DeviceAllocation<LayoutSFB> lSFB_dev(G); lSFB_dev.copy_from_host(lSFB_host.data());
  cutlass::DeviceAllocation<const ElementInput*> pA_dev(G); pA_dev.copy_from_host(pA.data());
  cutlass::DeviceAllocation<const ElementInput*> pB_dev(G); pB_dev.copy_from_host(pB.data());
  cutlass::DeviceAllocation<const ElementSF*> pSFA_dev(G); pSFA_dev.copy_from_host(pSFA.data());
  cutlass::DeviceAllocation<const ElementSF*> pSFB_dev(G); pSFB_dev.copy_from_host(pSFB.data());
  cutlass::DeviceAllocation<ElementD*> pD_dev(G); pD_dev.copy_from_host(pD.data());

  cutlass::KernelHardwareInfo hw;
  hw.device_id = 0;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  typename Gemm::Arguments args;
  decltype(args.epilogue.thread) fusion;
  // The two F32 per-tensor globals are not block scales; they are one scalar
  // each, so they fold into the epilogue instead of costing a pass over the
  // weights.
  fusion.alpha = qa.global_scale * w.global_scale;
  fusion.beta = 0.0f;
  fusion.alpha_ptr = nullptr;
  fusion.beta_ptr = nullptr;

  args = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {G, ps_dev.get(), ps_host.data()},
      {reinterpret_cast<const ElementInput**>(pA_dev.get()), sA_dev.get(),
       reinterpret_cast<const ElementInput**>(pB_dev.get()), sB_dev.get(),
       reinterpret_cast<const ElementSF**>(pSFA_dev.get()), lSFA_dev.get(),
       reinterpret_cast<const ElementSF**>(pSFB_dev.get()), lSFB_dev.get()},
      {fusion, nullptr, nullptr, pD_dev.get(), sD_dev.get()},
      hw};

  Gemm gemm;
  cutlass::DeviceAllocation<std::uint8_t> workspace(Gemm::get_workspace_size(args));
  check(gemm.can_implement(args) == cutlass::Status::kSuccess, "CUTLASS can implement this shape");
  CUTLASS_CHECK_STATUS(gemm.initialize(args, workspace.get()));
  CUTLASS_CHECK_STATUS(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> got(static_cast<std::size_t>(M) * N);
  dD.copy_to_host(got.data());

  // ---- compare ------------------------------------------------------------
  const ErrorStats err = compare(got, ref);
  std::printf("  |ref|max      : %.6g\n", err.max_ref_abs);
  std::printf("  max abs err   : %.6g\n", err.max_abs);
  std::printf("  max rel err   : %.6g   (threshold %.0e)\n", err.max_rel_significant, kMaxRelError);
  std::printf("  rms rel err   : %.6g\n", err.rms_rel);
  check(err.max_rel_significant < kMaxRelError, "CUTLASS matches FP32 reference on real weights");

  // Control: the same kernel fed checkpoint-linear scales must be wrong. If it
  // is not, this shape cannot distinguish a correct swizzle from no swizzle.
  {
    std::vector<std::uint8_t> unswizzled(lSFB.bytes(), 0);
    std::memcpy(unswizzled.data(), scale_host.data(),
                std::min(unswizzled.size(), scale_host.size()));
    dSFB.copy_from_host(unswizzled.data());
    CUTLASS_CHECK_STATUS(gemm.run());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> wrong(static_cast<std::size_t>(M) * N);
    dD.copy_to_host(wrong.data());
    const ErrorStats werr = compare(wrong, ref);
    std::printf("  control (no swizzle) max rel err: %.6g\n", werr.max_rel_significant);
    check(werr.max_rel_significant > 0.1, "skipping the swizzle produces a visibly wrong result");
    dSFB.copy_from_host(sfb_swizzled.data());
  }

  // ---- load timing --------------------------------------------------------
  const double expert_bytes = static_cast<double>(w.packed_bytes() + w.scale_bytes());
  std::printf("\n  load timing (%s page cache)\n", opt.cold ? "dropped before read" : "warm");
  std::printf("    index+mmap    : %7.3f ms\n", t_map * 1e3);
  std::printf("    read %6.2f MiB: %7.3f ms  (%.2f GB/s)\n",
              expert_bytes / (1024.0 * 1024.0), t_read * 1e3, expert_bytes / t_read / 1e9);
  std::printf("    swizzle scales: %7.3f ms  (%.2f GB/s of scale bytes)\n",
              t_swizzle * 1e3, static_cast<double>(w.scale_bytes()) / t_swizzle / 1e9);
  std::printf("    host->device  : %7.3f ms  (%.2f GB/s)\n",
              t_h2d * 1e3, expert_bytes / t_h2d / 1e9);
  std::printf("    fp32 reference: %7.3f ms (CPU, not part of loading)\n", t_ref * 1e3);

  // Extrapolate to a booster's share. 182 GiB total checkpoint, 2 boosters.
  constexpr double kHalfCheckpointGiB = 91.0;
  constexpr double kNvmeFloorGBs = 6.8;  // blog/posts/hardware measured NVMe floor
  const double half_bytes = kHalfCheckpointGiB * 1024.0 * 1024.0 * 1024.0;
  const double io_seconds = half_bytes / (kNvmeFloorGBs * 1e9);
  // Scale bytes are 1/8 of packed bytes, so the swizzle only has to keep up
  // with 1/9 of the stream.
  const double swizzle_rate = static_cast<double>(w.scale_bytes()) / t_swizzle;
  const double swizzle_seconds = (half_bytes / 9.0) / swizzle_rate;
  std::printf("    full booster  : %.2f GiB at %.1f GB/s NVMe floor = %.1f s I/O\n",
              kHalfCheckpointGiB, kNvmeFloorGBs, io_seconds);
  std::printf("                    swizzle of that share, 1 thread = %.1f s (%s)\n",
              swizzle_seconds,
              swizzle_seconds < io_seconds ? "hides under I/O" : "BECOMES THE BOTTLENECK");

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("--dir=", 0) == 0) opt.dir = a.substr(6);
    else if (a.rfind("--layer=", 0) == 0) opt.layer = std::atoi(a.c_str() + 8);
    else if (a.rfind("--expert=", 0) == 0) opt.expert = std::atoi(a.c_str() + 9);
    else if (a.rfind("--m=", 0) == 0) opt.m = std::atoi(a.c_str() + 4);
    else if (a == "--warm") opt.cold = false;
    else if (a == "--help") {
      std::printf("usage: %s [--dir=<snapshot>] [--layer=N] [--expert=N] [--m=N] [--warm]\n", argv[0]);
      return 0;
    }
  }

  std::filesystem::path dir = opt.dir.empty() ? fuel::default_nvfp4_snapshot_dir()
                                              : std::filesystem::path(opt.dir);
  if (dir.empty() || !std::filesystem::exists(dir / "model.safetensors.index.json")) {
    std::printf("SKIP: no NVFP4 checkpoint at %s\n", dir.string().c_str());
    return 77;  // ctest SKIP_RETURN_CODE
  }

  cudaDeviceProp prop{};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("device        : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("checkpoint    : %s\n", dir.string().c_str());
  std::printf("SF vector size: %d\n", SFVecSize);

  fuel::Checkpoint ckpt(dir);
  std::printf("index tensors : %zu\n", ckpt.tensor_count());

  // Two real shapes, and they are the two the MoE layer needs:
  //   down_proj is w2  (N=4096, K=2048)
  //   gate_proj is one half of w13 (N=2048, K=4096)
  run_shape(ckpt, opt, "down_proj", "w2 down");
  run_shape(ckpt, opt, "gate_proj", "w13 gate half");

  // gate_proj and up_proj carry different weight_scale_2 values, so a fused
  // w13 cannot use one epilogue alpha for both halves. Record it here rather
  // than letting the engine discover it later.
  fuel::ExpertProj g = fuel::load_expert_proj(ckpt, opt.layer, opt.expert, "gate_proj");
  fuel::ExpertProj u = fuel::load_expert_proj(ckpt, opt.layer, opt.expert, "up_proj");
  std::printf("\nfused-w13 note: gate global=%.9g up global=%.9g ratio=%.6f\n",
              g.global_scale, u.global_scale, g.global_scale / u.global_scale);
  std::printf("                a fused w13 needs one alpha per half, or the ratio folded\n"
              "                into one half's block scales, since alpha is per-GEMM.\n");

  std::printf("\n%s: %d check(s) failed\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}

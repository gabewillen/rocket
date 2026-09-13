// CUTLASS NVFP4 grouped GEMM, sm_121a. See moe_grouped.h for the call
// contract and blog/posts/kernels/2026-09-06-cutlass-nvfp4-sm121/ for why
// this is CUTLASS's own GemmUniversalAdapter and not a rocket-owned kernel:
// at these exact shapes (288 experts, N=4096/K=4096 w13, N=4096/K=2048 w2)
// stock CUTLASS 4.8.0 already holds 78-82% of the measured read roofline.
// Type setup mirrors bench/nvfp4_grouped_gemm.cu exactly (same ArchTag,
// same ThreadBlockShape, same collective builders); what's new here is a
// runtime group list from live decode routing instead of synthetic ragged-M,
// and a persistent metadata cache instead of one-shot bench allocations.
#include "moe_grouped.h"

#include <cstdio>

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

namespace rocket::engine {
namespace {

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e2m1_t;

using ElementA = cutlass::nv_float4_t<ElementInput>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<ElementInput>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD = cutlass::bfloat16_t;
using ElementC = void;  // beta = 0, no source read
using LayoutCTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = AlignmentD;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag*, AlignmentC, ElementD, LayoutCTag*, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag*, AlignmentA, ElementB, LayoutBTag*, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;

// Persistent per-call-shape scratch: the decode loop calls this twice a layer
// (w13, w2) at every step, and rebuilding CUTLASS's grouped-launch metadata
// arrays from a cudaMalloc/cudaFree pair each time would put allocator
// traffic on the decode critical path. Keyed by (n, k) rather than a fixed
// pair of slots, so a test that exercises other shapes does not collide with
// the engine's two.
struct Scratch {
  int n = -1, k = -1;
  int capacity = 0;  // groups
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> ps;
  cutlass::DeviceAllocation<StrideA> sA;
  cutlass::DeviceAllocation<StrideB> sB;
  cutlass::DeviceAllocation<StrideD> sD;
  cutlass::DeviceAllocation<LayoutSFA> lSFA;
  cutlass::DeviceAllocation<LayoutSFB> lSFB;
  cutlass::DeviceAllocation<const ElementInput*> pA, pB;
  cutlass::DeviceAllocation<const ElementSF*> pSFA, pSFB;
  cutlass::DeviceAllocation<ElementD*> pD;
  cutlass::DeviceAllocation<std::uint8_t> workspace;
  // Host metadata must outlive the asynchronous H2D copies and kernel launch.
  std::vector<typename ProblemShape::UnderlyingProblemShape> h_ps;
  std::vector<StrideA> h_sA;
  std::vector<StrideB> h_sB;
  std::vector<StrideD> h_sD;
  std::vector<LayoutSFA> h_lSFA;
  std::vector<LayoutSFB> h_lSFB;
  std::vector<const ElementInput*> h_pA, h_pB;
  std::vector<const ElementSF*> h_pSFA, h_pSFB;
  std::vector<ElementD*> h_pD;

  void reserve(int groups) {
    if (groups <= capacity) return;
    ps.reset(groups);
    sA.reset(groups);
    sB.reset(groups);
    sD.reset(groups);
    lSFA.reset(groups);
    lSFB.reset(groups);
    pA.reset(groups);
    pB.reset(groups);
    pSFA.reset(groups);
    pSFB.reset(groups);
    pD.reset(groups);
    h_ps.resize(groups);
    h_sA.resize(groups);
    h_sB.resize(groups);
    h_sD.resize(groups);
    h_lSFA.resize(groups);
    h_lSFB.resize(groups);
    h_pA.resize(groups);
    h_pB.resize(groups);
    h_pSFA.resize(groups);
    h_pSFB.resize(groups);
    h_pD.resize(groups);
    capacity = groups;
  }
};

std::vector<Scratch>& scratch_pool() {
  static std::vector<Scratch> pool;
  return pool;
}

Scratch& scratch_for(int n, int k) {
  auto& pool = scratch_pool();
  for (auto& sc : pool)
    if (sc.n == n && sc.k == k) return sc;
  pool.emplace_back();
  pool.back().n = n;
  pool.back().k = k;
  return pool.back();
}

}  // namespace

namespace {
// Sticky-path state: separate scratch pool so a sticky (n, k) never shares
// DeviceAllocations with the host-driven MoE path, plus the last uploaded
// descriptor set for the skip-when-identical decision.
struct StickyState {
  std::unordered_map<std::uint64_t, Scratch> scratch;
  std::unordered_map<std::uint64_t, std::vector<GroupedGemmGroup>> uploaded;
};
StickyState& sticky_state() {
  static StickyState st;
  return st;
}
// The sticky key folds the full descriptor set: several callers (the three
// dense layers) alternate descriptors on one (n, k), and each needs its own
// scratch + upload state so a capture-time call never re-uploads.
std::uint64_t sticky_key(int n, int k, const std::vector<GroupedGemmGroup>& gs) {
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&h](std::uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  mix(static_cast<std::uint32_t>(n));
  mix(static_cast<std::uint32_t>(k));
  for (const auto& g : gs) {
    mix(static_cast<std::uint64_t>(g.m));
    mix(reinterpret_cast<std::uintptr_t>(g.a_packed));
    mix(reinterpret_cast<std::uintptr_t>(g.a_scale));
    mix(reinterpret_cast<std::uintptr_t>(g.b_packed));
    mix(reinterpret_cast<std::uintptr_t>(g.b_scale));
    mix(reinterpret_cast<std::uintptr_t>(g.d_out));
  }
  return h;
}
}  // namespace

bool run_grouped_impl(const std::vector<GroupedGemmGroup>& groups, int n, int k, cudaStream_t s,
                      Scratch& sc, bool upload) {
  const int G = static_cast<int>(groups.size());
  if (G == 0) return true;
  sc.reserve(G);

  auto& ps_host = sc.h_ps;
  auto& sA_host = sc.h_sA;
  auto& sB_host = sc.h_sB;
  auto& sD_host = sc.h_sD;
  auto& lSFA_host = sc.h_lSFA;
  auto& lSFB_host = sc.h_lSFB;
  auto& pA_host = sc.h_pA;
  auto& pB_host = sc.h_pB;
  auto& pSFA_host = sc.h_pSFA;
  auto& pSFB_host = sc.h_pSFB;
  auto& pD_host = sc.h_pD;

  for (int i = 0; i < G; ++i) {
    const GroupedGemmGroup& g = groups[i];
    const int M = g.m;
    ps_host[i] = {M, n, k};
    sA_host[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, k, 1});
    sB_host[i] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    sD_host[i] = cutlass::make_cute_packed_stride(StrideD{}, {M, n, 1});
    lSFA_host[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, n, k, 1));
    lSFB_host[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, n, k, 1));
    pA_host[i] = reinterpret_cast<const ElementInput*>(g.a_packed);
    pB_host[i] = reinterpret_cast<const ElementInput*>(g.b_packed);
    pSFA_host[i] = reinterpret_cast<const ElementSF*>(g.a_scale);
    pSFB_host[i] = reinterpret_cast<const ElementSF*>(g.b_scale);
    pD_host[i] = reinterpret_cast<ElementD*>(g.d_out);
  }

  // Stream-ordered metadata uploads. Scratch owns the host vectors, so their
  // lifetime extends through the launch; no cudaMemcpy synchronization is
  // needed between descriptor construction and the grouped kernel.
  if (upload) {
#define ROCKET_META_COPY(dst, src) \
    cudaMemcpyAsync((dst).get(), (src).data(), G * sizeof((src)[0]), cudaMemcpyHostToDevice, s)
    ROCKET_META_COPY(sc.ps, ps_host);
    ROCKET_META_COPY(sc.sA, sA_host);
    ROCKET_META_COPY(sc.sB, sB_host);
    ROCKET_META_COPY(sc.sD, sD_host);
    ROCKET_META_COPY(sc.lSFA, lSFA_host);
    ROCKET_META_COPY(sc.lSFB, lSFB_host);
    ROCKET_META_COPY(sc.pA, pA_host);
    ROCKET_META_COPY(sc.pB, pB_host);
    ROCKET_META_COPY(sc.pSFA, pSFA_host);
    ROCKET_META_COPY(sc.pSFB, pSFB_host);
    ROCKET_META_COPY(sc.pD, pD_host);
#undef ROCKET_META_COPY
  }

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
      {G, sc.ps.get(), ps_host.data()},
      {sc.pA.get(), sc.sA.get(), sc.pB.get(), sc.sB.get(), sc.pSFA.get(), sc.lSFA.get(),
       sc.pSFB.get(), sc.lSFB.get()},
      {fusion, nullptr, nullptr, sc.pD.get(), sc.sD.get()},
      hw};

  Gemm gemm;
  const std::size_t ws = Gemm::get_workspace_size(args);
  if (ws > sc.workspace.size()) sc.workspace.reset(ws);

  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "rocket::engine::grouped_gemm_nvfp4: cannot implement n=%d k=%d G=%d\n", n,
                k, G);
    return false;
  }
  if (gemm.initialize(args, sc.workspace.get(), s) != cutlass::Status::kSuccess) return false;
  if (gemm.run(s) != cutlass::Status::kSuccess) return false;
  return true;
}

bool grouped_gemm_nvfp4(const std::vector<GroupedGemmGroup>& groups_in, int n, int k,
                        cudaStream_t s) {
  std::vector<GroupedGemmGroup> groups;
  groups.reserve(groups_in.size());
  for (const auto& g : groups_in)
    if (g.m > 0) groups.push_back(g);
  if (groups.empty()) return true;
  return run_grouped_impl(groups, n, k, s, scratch_for(n, k), /*upload=*/true);
}

bool grouped_gemm_nvfp4_sticky(const std::vector<GroupedGemmGroup>& groups_in, int n, int k,
                               cudaStream_t s) {
  std::vector<GroupedGemmGroup> groups;
  groups.reserve(groups_in.size());
  for (const auto& g : groups_in)
    if (g.m > 0) groups.push_back(g);
  if (groups.empty()) return true;
  const std::uint64_t key = sticky_key(n, k, groups);
  StickyState& st = sticky_state();
  Scratch& sc = st.scratch[key];
  auto& last = st.uploaded[key];
  const bool upload = last.empty();  // fresh slot: first use uploads
  if (!run_grouped_impl(groups, n, k, s, sc, upload)) return false;
  if (upload) last = groups;
  return true;
}

}  // namespace rocket::engine

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

bool grouped_gemm_nvfp4(const std::vector<GroupedGemmGroup>& groups_in, int n, int k,
                        cudaStream_t s) {
  std::vector<GroupedGemmGroup> groups;
  groups.reserve(groups_in.size());
  for (const auto& g : groups_in)
    if (g.m > 0) groups.push_back(g);
  if (groups.empty()) return true;

  const int G = static_cast<int>(groups.size());
  Scratch& sc = scratch_for(n, k);
  sc.reserve(G);

  std::vector<typename ProblemShape::UnderlyingProblemShape> ps_host(G);
  std::vector<StrideA> sA_host(G);
  std::vector<StrideB> sB_host(G);
  std::vector<StrideD> sD_host(G);
  std::vector<LayoutSFA> lSFA_host(G);
  std::vector<LayoutSFB> lSFB_host(G);
  std::vector<const ElementInput*> pA_host(G), pB_host(G);
  std::vector<const ElementSF*> pSFA_host(G), pSFB_host(G);
  std::vector<ElementD*> pD_host(G);

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

  // Synchronous: this metadata is a few hundred bytes per group and the
  // decode loop already syncs once per step to read router indices back to
  // host (model.cu::run_moe), so this is not adding a new sync point class,
  // only more work at an existing one. Making this async would need the host
  // vectors above to outlive an in-flight copy, which the current run_moe
  // call shape (grouped_gemm_nvfp4 returns before the caller reuses them)
  // does not guarantee.
  sc.ps.copy_from_host(ps_host.data());
  sc.sA.copy_from_host(sA_host.data());
  sc.sB.copy_from_host(sB_host.data());
  sc.sD.copy_from_host(sD_host.data());
  sc.lSFA.copy_from_host(lSFA_host.data());
  sc.lSFB.copy_from_host(lSFB_host.data());
  sc.pA.copy_from_host(pA_host.data());
  sc.pB.copy_from_host(pB_host.data());
  sc.pSFA.copy_from_host(pSFA_host.data());
  sc.pSFB.copy_from_host(pSFB_host.data());
  sc.pD.copy_from_host(pD_host.data());

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

}  // namespace rocket::engine

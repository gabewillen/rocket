// Does not compile. That is the result.
//
// Tile M below 128 does not exist for a block-scaled NVFP4 GEMM in CUTLASS
// 4.8.0. Two gates, in order:
//
//   1. The grouped NVFP4 path resolves to the cooperative pointer-array
//      schedule, and
//      include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp:443
//        static_assert(size<0>(TileShape{}) >= 128,
//          "Cooperative kernel requires Tile Size to be greater than or equal
//           to 128 along the M-dimension.");
//      The pingpong pointer-array schedule has no such assert, but
//      KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120 is rejected by
//      the F4 policy check at
//      include/cutlass/gemm/collective/builders/sm1xx_common.inl:630, and a
//      locally composed tag fails detail::find_vector_size
//      (sm1xx_common.inl:139), which matches tags with is_same_v and so returns
//      32 rather than 16 for anything not already enumerated there.
//
//   2. Underneath both of those, the scale-factor storage block is 128 wide in
//      the M/N direction:
//        include/cutlass/detail/sm100_blockscaled_layout.hpp   Blk_MN = _128
//      and the SM120 mainloop builder forms its shared-memory SFA atom from
//        size<0>(TileShape_MNK{}) / Blk_MN{}
//        include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl:199
//      which is zero at tile M 64. This file takes the non-grouped pingpong
//      NVFP4 schedule, which clears gate 1, and still fails on gate 2, so the
//      constraint is the block-scaled layout and not the grouped scheduler.
//      The loader lane confirmed SM120 uses the same Sm1xxBlockScaledConfig
//      swizzle as sm_100, so this is not an SM120-only quirk.
//
// Reproduce with:
//   cmake --build engines/glm5-moe-nvfp4-2b/build --target bench-tile-m64-probe

#include <cstdio>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

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
using TileShape     = Shape<_64, _128, _128>;   // the whole point

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutCTag, AlignmentD,
    cutlass::epilogue::TmaWarpSpecialized
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

int main() {
  Gemm gemm;
  std::printf("built: %zu\n", sizeof(gemm));
  return 0;
}

// Does not compile. That is the result.
//
// SwiGLU needs gate and up for the same channel multiplied together. An
// epilogue can only see one accumulator tile, so the only stock CUTLASS way to
// express it is to split w13 into two grouped GEMMs and have the second one
// read the first one's output as an auxiliary tensor:
//
//   w1 grouped GEMM, epilogue LinCombEltAct<SiLu>          -> T = silu(gate)
//   w3 grouped GEMM, epilogue LinCombDeEltAct<Mul>, aux=T  -> D = up * T
//
// The first half builds. This file is the second half, and it does not, because
// LinCombDeEltAct (the only fusion operation with an auxiliary input) has a
// FusionCallbacks specialization for Sm90TmaWarpSpecialized only. The pointer
// array dispatch policy that grouped GEMM uses,
// Sm90PtrArrayTmaWarpSpecialized / Sm120PtrArrayTmaWarpSpecialized, has
// specializations for LinearCombination, LinCombEltAct,
// LinCombBlockScaleFactor, and LinCombEltActBlockScaleFactor, and for nothing
// with an aux input.
//
//   include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp
//     LinCombDeEltAct specialization takes Sm90TmaWarpSpecialized
//     PtrArray specializations exist for LinearCombination and LinCombEltAct
//   include/cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp
//     Sm120PtrArrayTmaWarpSpecialized: LinCombBlockScaleFactor and
//     LinCombEltActBlockScaleFactor
//
// Reproduce with:
//   cmake --build engines/glm5-moe-nvfp4-2b/build \
//     --target bench-evt-swiglu-aux-probe

#include <cstdio>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/array.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

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
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape     = Shape<_1, _1, _1>;

// D = dY * Z, the binary node SwiGLU needs. Shaped like CUTLASS's own
// de-activation functors (see cutlass/epilogue/thread/activation.h, dReLU_Z).
template <typename T>
struct MulAux {
  CUTLASS_HOST_DEVICE T operator()(T dy, T z) const { return dy * z; }
};

template <typename T, int N>
struct MulAux<cutlass::Array<T, N>> {
  CUTLASS_HOST_DEVICE
  cutlass::Array<T, N> operator()(cutlass::Array<T, N> const &dy,
                                  cutlass::Array<T, N> const &z) const {
    cutlass::Array<T, N> y;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) y[i] = dy[i] * z[i];
    return y;
  }
};

using FusionAuxMul = cutlass::epilogue::fusion::LinCombDeEltAct<
    cutlass::layout::RowMajor, MulAux, ElementD, ElementAccumulator,
    ElementD, ElementC, ElementAccumulator>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag *, AlignmentC,
    ElementD, LayoutCTag *, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionAuxMul
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

int main() {
  Gemm gemm;
  std::printf("built: %zu\n", sizeof(gemm));
  return 0;
}

// SPDX-License-Identifier: Apache-2.0
#include "moe/target_router_shared_c1.h"

#include "moe/target_moe_b12x_aot.h"

#include <cstdlib>

namespace moe = rocket::qwen38::moe;

int main() {
  static_assert(moe::kTargetRouterHidden == 2'560);
  static_assert(moe::kTargetRouterExperts == 512);
  static_assert(moe::kTargetRouterTopK == 10);
  static_assert(moe::kTargetSharedIntermediate == 320);
  static_assert(moe::kTargetSharedRankIntermediate == 160);
  static_assert(moe::kTargetCompositionLayer == 3);

  moe::TargetDenseIdentity identity{};
  if (!moe::parse_target_moe_artifact_key(
          moe::target_moe_artifact_key_ascii(), &identity.artifact_sha256))
    std::abort();
  identity.layout_sha256[0] = 1;
  identity.rank = 0;
  identity.layer = 3;
  const auto raw = reinterpret_cast<void*>(1);
  const auto u8 = static_cast<const std::uint8_t*>(raw);
  const auto f32 = static_cast<float*>(raw);
  const auto bf16 = static_cast<__nv_bfloat16*>(raw);
  const auto u64 = static_cast<std::uint64_t*>(raw);
  const auto i32 = static_cast<std::int32_t*>(raw);
  const auto stream = reinterpret_cast<cudaStream_t>(1);

  const moe::TargetRouterNvfp4Weights router_weights{u8, u8, f32};
  const moe::TargetRouterC1Launch router_launch{
      bf16, f32, i32, f32, u64, u64, stream};
  if (moe::diagnose_target_router_c1(identity, router_weights, router_launch) !=
      moe::TargetDenseFailure::kNone)
    std::abort();

  const moe::TargetSharedBf16Weights shared_weights{bf16, bf16, bf16, bf16};
  const moe::TargetSharedC1Launch shared_launch{
      bf16, bf16, f32, f32, f32, stream};
  if (moe::diagnose_target_shared_c1(identity, shared_weights, shared_launch) !=
      moe::TargetDenseFailure::kNone)
    std::abort();

  auto changed = identity;
  changed.layer = 4;
  if (moe::diagnose_target_router_c1(changed, router_weights, router_launch) !=
      moe::TargetDenseFailure::kLayer ||
      moe::diagnose_target_shared_c1(changed, shared_weights, shared_launch) !=
          moe::TargetDenseFailure::kLayer)
    std::abort();
  return 0;
}

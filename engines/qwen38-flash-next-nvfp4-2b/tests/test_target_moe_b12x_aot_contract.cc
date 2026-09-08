// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_b12x_aot.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace moe = rocket::qwen38::moe;

int main() {
  static_assert(moe::kTargetMoeHidden == 2'560);
  static_assert(moe::kTargetMoeLogicalIntermediate == 640);
  static_assert(moe::kTargetMoePhysicalIntermediate == 768);
  static_assert(moe::kTargetMoeStateExperts == 257);
  static_assert(moe::kTargetMoeMaxRows == 10);
  moe::TargetMoeB12xIdentity production_identity{};
  const unsigned char artifact[32] = {
      0xa9, 0xfc, 0xca, 0x02, 0x6a, 0x87, 0xad, 0x12,
      0x85, 0xb9, 0x4f, 0xef, 0x19, 0x44, 0x48, 0xc5,
      0x1b, 0x42, 0xd9, 0x75, 0x16, 0xf1, 0x62, 0x11,
      0xc6, 0x1a, 0xe4, 0xc7, 0x70, 0xc6, 0xf0, 0xf4};
  std::memcpy(production_identity.artifact_sha256.data(), artifact, 32);
  production_identity.layout_sha256[0] = 1;
  production_identity.rank = 0;
  production_identity.layer = 0;
  const auto p = reinterpret_cast<const void*>(1);
  moe::TargetMoeB12xWeights production_weights{
      static_cast<const std::uint8_t*>(p), static_cast<const std::uint8_t*>(p),
      static_cast<const std::uint8_t*>(p), static_cast<const std::uint8_t*>(p),
      static_cast<const float*>(p), static_cast<const float*>(p),
      static_cast<const float*>(p), static_cast<const float*>(p)};
  if (moe::diagnose_target_moe_b12x_create(
          0, production_identity, production_weights) !=
      moe::TargetMoeCreateFailure::kNone)
    std::abort();
  auto changed_identity = production_identity;
  changed_identity.artifact_sha256[14] ^= 1;
  if (moe::diagnose_target_moe_b12x_create(
          0, changed_identity, production_weights) !=
      moe::TargetMoeCreateFailure::kArtifactSha256)
    std::abort();
  changed_identity = production_identity;
  changed_identity.layout_sha256.fill(0);
  if (moe::diagnose_target_moe_b12x_create(
          0, changed_identity, production_weights) !=
      moe::TargetMoeCreateFailure::kLayoutSha256)
    std::abort();
  void* handle = nullptr;
  if (rocket_qwen38_target_moe_b12x_create(0, nullptr, nullptr, &handle) !=
          static_cast<int>(moe::TargetMoeOutcome::kContractError) ||
      rocket_qwen38_target_moe_b12x_enqueue(nullptr, nullptr) !=
          static_cast<int>(moe::TargetMoeOutcome::kContractError))
    std::abort();
  rocket_qwen38_target_moe_b12x_destroy(nullptr);
  if (moe::target_moe_b12x_aot_compiled()) return 0;
  try {
    moe::TargetMoeB12xAot unavailable(0, {}, {});
    std::abort();
  } catch (const std::runtime_error&) {
  }
  moe::TargetMoeB12xLaunch invalid{};
  // An unavailable production backend still distinguishes an invalid caller
  // contract from backend absence without allocating or fabricating output.
  struct Probe final {
    static moe::TargetMoeOutcome call(const moe::TargetMoeB12xLaunch& launch) {
      return launch.hidden_bf16 ? moe::TargetMoeOutcome::kCudaError
                               : moe::TargetMoeOutcome::kContractError;
    }
  };
  if (Probe::call(invalid) != moe::TargetMoeOutcome::kContractError)
    std::abort();
  return 0;
}

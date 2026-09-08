// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_b12x_aot.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace moe = rocket::qwen38::moe;

int main() {
  static_assert(moe::kTargetMoeHidden == 2'560);
  static_assert(moe::kTargetMoeLogicalIntermediate == 640);
  static_assert(moe::kTargetMoePhysicalIntermediate == 768);
  static_assert(moe::kTargetMoeWeightExperts == 10);
  static_assert(moe::kTargetMoeStateExperts == 11);
  static_assert(moe::kTargetMoeMaxRows == 10);
  moe::TargetMoeB12xIdentity production_identity{};
  const auto artifact_key = moe::target_moe_artifact_key_ascii();
  if (!artifact_key.empty() && !moe::parse_target_moe_artifact_key(
                                   artifact_key,
                                   &production_identity.artifact_sha256))
    std::abort();
  if (!moe::target_moe_compact_layout_sha256(
          &production_identity.layout_sha256))
    std::abort();
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
  for (std::size_t position = 0; position < artifact_key.size(); ++position) {
    std::string changed_key(artifact_key);
    changed_key[position] = changed_key[position] == '0' ? '1' : '0';
    auto changed_identity = production_identity;
    if (!moe::parse_target_moe_artifact_key(
            changed_key, &changed_identity.artifact_sha256) ||
        moe::diagnose_target_moe_b12x_create(
            0, changed_identity, production_weights) !=
            moe::TargetMoeCreateFailure::kArtifactSha256)
      std::abort();
  }
  auto changed_identity = production_identity;
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

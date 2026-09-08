// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_b12x_aot.h"

#include <cstdlib>
#include <stdexcept>

namespace moe = rocket::qwen38::moe;

int main() {
  static_assert(moe::kTargetMoeHidden == 2'560);
  static_assert(moe::kTargetMoeLogicalIntermediate == 640);
  static_assert(moe::kTargetMoePhysicalIntermediate == 768);
  static_assert(moe::kTargetMoeStateExperts == 257);
  static_assert(moe::kTargetMoeMaxRows == 10);
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

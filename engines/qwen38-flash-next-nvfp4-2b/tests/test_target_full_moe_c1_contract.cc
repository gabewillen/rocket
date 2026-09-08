// SPDX-License-Identifier: Apache-2.0
#include "moe/target_full_moe_c1.h"

#include <cstdlib>

namespace moe = rocket::qwen38::moe;

int main() {
  void* handle = nullptr;
  if (rocket_qwen38_target_full_moe_c1_create(0, nullptr, nullptr, &handle) !=
          static_cast<int>(moe::TargetDenseOutcome::kContractError) ||
      rocket_qwen38_target_full_moe_c1_enqueue(nullptr, nullptr) !=
          static_cast<int>(moe::TargetDenseOutcome::kContractError))
    std::abort();
  rocket_qwen38_target_full_moe_c1_destroy(nullptr);
  static_assert(moe::kTargetCompositionLayer == 3);
  static_assert(moe::kTargetRouterTopK == 10);
  static_assert(moe::kTargetSharedRankIntermediate * 2 ==
                moe::kTargetSharedIntermediate);
  return 0;
}

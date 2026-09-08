// SPDX-License-Identifier: Apache-2.0
#include "moe/target_full_moe_c1.h"

#include <type_traits>

namespace moe = rocket::qwen38::moe;

int main() {
  static_assert(std::is_final_v<moe::TargetFullMoeC1>);
  static_assert(std::is_base_of_v<moe::TargetFullMoeC1Port,
                                  moe::TargetFullMoeC1>);
  static_assert(moe::kTargetCompositionLayer == 3);
  static_assert(moe::kTargetRouterTopK == 10);
  static_assert(moe::kTargetSharedRankIntermediate * 2 ==
                moe::kTargetSharedIntermediate);
  return 0;
}

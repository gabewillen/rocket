// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_layer_owner_inventory.h"

#include <type_traits>

namespace decode = rocket::qwen38::decode;

int main() {
  static_assert(!std::is_copy_constructible_v<
                decode::TargetK0LayerOwnerInventory>);
  int qsa = 0;
  int gdn = 0;
  for (int layer = 0; layer < decode::kDecoderLayers; ++layer) {
    if (decode::target_k0_attention_kind(layer) ==
        decode::TargetK0AttentionKind::kQsa)
      ++qsa;
    else
      ++gdn;
  }
  return qsa == 12 && gdn == 36 ? 0 : 1;
}

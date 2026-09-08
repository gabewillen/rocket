// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_layer_native_plan.h"

#include <array>
#include <filesystem>

namespace rocket::qwen38::decode {

// Value-owned startup inventory. Loading authenticates the complete 96-file
// two-rank descriptor set before any CUDA owner or transport is constructed.
class TargetK0NativePlanInventory final {
 public:
  static TargetK0NativePlanInventory load(
      const std::filesystem::path& descriptor_directory);

  const TargetLayerNativePlan& at(int rank, int layer) const;

 private:
  std::array<TargetLayerNativePlan, 96> plans_{};
};

}  // namespace rocket::qwen38::decode

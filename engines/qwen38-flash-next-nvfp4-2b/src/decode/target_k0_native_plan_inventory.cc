// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_native_plan_inventory.h"

#include <set>
#include <stdexcept>
#include <string>

namespace rocket::qwen38::decode {
namespace {

std::string filename(int rank, int layer) {
  return "rank" + std::to_string(rank) + "-layer" +
         std::to_string(layer) + ".json";
}

std::size_t index(int rank, int layer) {
  if ((rank != 0 && rank != 1) || layer < 0 || layer >= 48)
    throw std::out_of_range("K0 native plan identity changed");
  return static_cast<std::size_t>(rank * 48 + layer);
}

}  // namespace

TargetK0NativePlanInventory TargetK0NativePlanInventory::load(
    const std::filesystem::path& descriptor_directory) {
  if (!std::filesystem::is_directory(descriptor_directory))
    throw std::invalid_argument("K0 descriptor directory is unavailable");

  std::set<std::string> expected;
  for (int rank = 0; rank < 2; ++rank)
    for (int layer = 0; layer < 48; ++layer)
      expected.insert(filename(rank, layer));

  std::set<std::string> observed;
  for (const auto& entry : std::filesystem::directory_iterator(
           descriptor_directory)) {
    if (!entry.is_regular_file() ||
        !observed.insert(entry.path().filename().string()).second)
      throw std::invalid_argument("K0 descriptor directory shape changed");
  }
  if (observed != expected)
    throw std::invalid_argument("K0 descriptor inventory changed");

  TargetK0NativePlanInventory result;
  for (int rank = 0; rank < 2; ++rank) {
    for (int layer = 0; layer < 48; ++layer) {
      auto plan = load_target_layer_native_plan(
          descriptor_directory / filename(rank, layer));
      if (plan.rank != rank || plan.layer != layer)
        throw std::invalid_argument("K0 descriptor placement changed");
      result.plans_[index(rank, layer)] = std::move(plan);
    }
  }
  return result;
}

const TargetLayerNativePlan& TargetK0NativePlanInventory::at(
    int rank, int layer) const {
  return plans_[index(rank, layer)];
}

}  // namespace rocket::qwen38::decode

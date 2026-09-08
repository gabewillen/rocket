// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_native_plan_inventory.h"

#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

int main(int argc, char** argv) {
  using rocket::qwen38::decode::TargetK0AttentionKind;
  using rocket::qwen38::decode::TargetK0NativePlanInventory;
  if (argc != 2) return 2;
  try {
    const auto plans = TargetK0NativePlanInventory::load(
        std::filesystem::path(argv[1]));
    for (int rank = 0; rank < 2; ++rank) {
      for (int layer = 0; layer < 48; ++layer) {
        const auto& plan = plans.at(rank, layer);
        const auto kind = layer % 4 == 3 ? TargetK0AttentionKind::kQsa
                                         : TargetK0AttentionKind::kGdn;
        if (plan.rank != rank || plan.layer != layer ||
            plan.attention_kind != kind) return 3;
      }
    }
    bool rejected = false;
    try { (void)plans.at(2, 0); }
    catch (const std::out_of_range&) { rejected = true; }
    if (!rejected) return 4;

    const auto scratch = std::filesystem::temp_directory_path() /
        ("qwen38-k0-plan-inventory-" + std::to_string(
            reinterpret_cast<std::uintptr_t>(&plans)));
    std::filesystem::remove_all(scratch);
    std::filesystem::copy(argv[1], scratch,
                          std::filesystem::copy_options::recursive);
    const auto missing = scratch / "rank1-layer47.json";
    std::filesystem::remove(missing);
    rejected = false;
    try { (void)TargetK0NativePlanInventory::load(scratch); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 5;
    std::filesystem::copy_file(
        std::filesystem::path(argv[1]) / "rank1-layer47.json", missing);
    std::filesystem::copy_file(
        std::filesystem::path(argv[1]) / "rank0-layer0.json",
        scratch / "unknown.json");
    rejected = false;
    try { (void)TargetK0NativePlanInventory::load(scratch); }
    catch (const std::invalid_argument&) { rejected = true; }
    std::filesystem::remove_all(scratch);
    if (!rejected) return 6;
    std::puts("qwen38 K0 native plan inventory: 96/96 authenticated");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

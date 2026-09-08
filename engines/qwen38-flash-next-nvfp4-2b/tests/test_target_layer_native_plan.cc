// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer_native_plan.h"
#include "moe/target_moe_b12x_aot.h"

#include <array>
#include <filesystem>
#include <stdexcept>

int main(int argc, char** argv) {
  using namespace rocket::qwen38::decode;
  if (argc != 7) return 2;
  const std::array<int, 6> expected_ranks{0, 0, 0, 1, 1, 1};
  const std::array<int, 6> expected_layers{0, 3, 47, 0, 3, 47};
  for (int index = 0; index < 6; ++index) {
    const auto plan = load_target_layer_native_plan(
        std::filesystem::path(argv[index + 1]));
    const bool qsa = expected_layers[index] % 4 == 3;
    if (plan.rank != expected_ranks[index] ||
        plan.layer != expected_layers[index] ||
        plan.attention_kind != (qsa ? TargetK0AttentionKind::kQsa
                                    : TargetK0AttentionKind::kGdn) ||
        plan.extents.size() != (qsa ? 3'108U : 3'111U)) return 3;
    const rocket::qwen38::moe::TargetMoeCompactRuntimeIdentity moe_identity{
        plan.rank, plan.layer, plan.descriptor_sha256,
        plan.native_binding_inventory_sha256,
        plan.slab_publication_layout_sha256,
        "modelopt_nvfp4_group16_cutlass_sm121_sfb",
        "rocket.qwen38.target-moe.device-stage.v1",
        "route_position_iota10_unique_positive_remote_zero_v1"};
    if (!rocket::qwen38::moe::authenticate_target_moe_compact_runtime_identity(
            moe_identity)) return 5;
    auto changed_identity = moe_identity;
    changed_identity.descriptor_sha256[0] =
        changed_identity.descriptor_sha256[0] == '0' ? '1' : '0';
    if (rocket::qwen38::moe::authenticate_target_moe_compact_runtime_identity(
            changed_identity)) return 6;
    changed_identity = moe_identity;
    changed_identity.layer = (plan.layer + 1) % 48;
    if (rocket::qwen38::moe::authenticate_target_moe_compact_runtime_identity(
            changed_identity)) return 7;
    auto mutated = plan;
    ++mutated.extents.back().offset_bytes;
    try {
      validate_target_layer_native_plan_binding(mutated);
      return 4;
    } catch (const std::invalid_argument&) {
    }
  }
}

// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_k0_executor.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace rocket::qwen38::decode {

struct TargetLayerNativeExtent {
  std::string name;
  std::uint64_t offset_bytes;
  std::uint64_t length_bytes;
  std::string storage;
  std::string dtype;
  std::string layout;
  std::string abi;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint64_t> strides;
};

struct TargetLayerNativePlan {
  int rank = -1;
  int peer_rank = -1;
  int layer = -1;
  TargetK0AttentionKind attention_kind = TargetK0AttentionKind::kGdn;
  std::uint64_t slab_bytes = 0;
  std::string descriptor_sha256;
  std::string native_binding_inventory_sha256;
  std::string artifact_key;
  std::string slab_key;
  std::string layout_sha256;
  std::string slab_publication_layout_sha256;
  std::string indexer_sidecar_key;
  // QSA order: q,k,v,o,unused. GDN order: qkv,z,b,a,out.
  std::array<float, 5> attention_projection_globals{};
  std::vector<TargetLayerNativeExtent> extents;
};

// Recomputes the complete extent inventory digest and checks the generated
// rank/layer allowlist before any slab address can be formed.
void validate_target_layer_native_plan_binding(
    const TargetLayerNativePlan& plan);

bool authenticate_target_layer_native_plan_identity(
    int rank, int layer, std::string_view descriptor_sha256,
    std::string_view binding_inventory_sha256,
    std::string_view publication_layout_sha256) noexcept;

TargetLayerNativePlan load_target_layer_native_plan(
    const std::filesystem::path& path);

}  // namespace rocket::qwen38::decode

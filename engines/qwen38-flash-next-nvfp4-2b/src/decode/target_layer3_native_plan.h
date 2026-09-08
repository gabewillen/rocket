// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace rocket::qwen38::decode {

struct TargetLayer3NativeExtent {
  std::string name;
  std::uint64_t offset_bytes;
  std::uint64_t length_bytes;
  std::string storage;
};

struct TargetLayer3NativePlan {
  int rank;
  int peer_rank;
  int layer;
  std::uint64_t slab_bytes;
  std::string descriptor_sha256;
  std::string native_binding_inventory_sha256;
  std::string artifact_key;
  std::string slab_key;
  std::string layout_sha256;
  std::string slab_publication_layout_sha256;
  // Host values authenticated from the exact weight_scale_2 slab extents in
  // q,k,v,o order. Native QSA initialization consumes these directly and does
  // not read scalar bytes through Python or D2H.
  std::array<float, 4> qsa_projection_globals;
  std::vector<TargetLayer3NativeExtent> extents;
};

// Reauthenticates every mutable field consumed by the native binding boundary,
// including fixed identities, q/k/v/o scalar bits, and the exact rank-local
// name/offset/length/storage inventory. Throws std::invalid_argument on drift.
void validate_target_layer3_native_plan_binding(
    const TargetLayer3NativePlan& plan);

// Strict, synchronous init-time parser for the canonical CPU handoff. `path`
// is borrowed only for this call; no reference to it or to the file bytes is
// retained. The returned plan owns every string and extent it exposes.
//
// The function is CPU-only, reentrant, and safe for concurrent calls with
// independent or shared immutable paths. It authenticates the complete file,
// rejects unknown/missing fields and trailing bytes, and recomputes extent and
// buffer bounds/alignment/contiguity before any pointer arithmetic, device
// allocation, CUDA work, or publication. It throws std::invalid_argument for
// every unavailable, malformed, unauthenticated, or contract-incompatible
// descriptor and returns no partial plan on failure.
TargetLayer3NativePlan load_target_layer3_native_plan(
    const std::filesystem::path& path);

}  // namespace rocket::qwen38::decode

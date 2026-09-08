// SPDX-License-Identifier: Apache-2.0
#include "model/target_slab_owner.h"

#include <algorithm>
#include <type_traits>

namespace model = rocket::qwen38::model;

int main() {
  static_assert(!std::is_constructible_v<
      model::ProcessLifetimeTargetSlabLease,
      model::TargetSlabPublication>);
  void* first = nullptr;
  constexpr char layout[] =
      "4f03ccc90c9020ff2e87f044867f2ac9896ac20c0d97c85055beef0b125ce6d6";
  constexpr char receipt[] =
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  model::TargetSlabChunkReceipt chunks[model::kTargetSlabChunks]{};
  for (std::size_t index = 0; index < model::kTargetSlabChunks; ++index) {
    const auto offset = index * model::kTargetSlabChunkBytes;
    const auto bytes = std::min(
        model::kTargetSlabChunkBytes, model::kTargetSlabBytes - offset);
    chunks[index] = {index, bytes, 1, 1, 1};
  }
  const model::TargetSlabPublication publication{
      reinterpret_cast<const std::uint8_t*>(0x100000000000ULL),
      reinterpret_cast<cudaEvent_t>(0x2000), model::kTargetSlabBytes, 0, 0,
      model::kTargetSlabArtifactKey, "rank0-target",
      model::kTargetSlabManifestSha256, layout, 1,
      model::kTargetSlabChunks, model::kTargetSlabPeakPinnedBytes};
  const model::TargetSlabCudaProbeResult probe{
      0x100000000000ULL, model::kTargetSlabBytes, 0, true, true};
  if (!model::validate_accepted_loader_publication(
          publication, probe, 1, 2, chunks, model::kTargetSlabChunks))
    return 1;
  auto padded_probe = probe;
  padded_probe.allocation_bytes += 1'703'936;
  if (!model::validate_accepted_loader_publication(
          publication, padded_probe, 1, 2, chunks,
          model::kTargetSlabChunks))
    return 2;
  auto bad_probe = probe;
  --bad_probe.allocation_bytes;
  if (model::validate_accepted_loader_publication(
          publication, bad_probe, 1, 2, chunks, model::kTargetSlabChunks))
    return 3;
  --chunks[7].bytes;
  if (model::validate_accepted_loader_publication(
          publication, probe, 1, 2, chunks, model::kTargetSlabChunks))
    return 4;
  ++chunks[7].bytes;
  const auto retain = [&](std::uintptr_t base, void** output) {
    return model::qwen38_target_slab_retain_accepted_loader(
        base, 0x2000, model::kTargetSlabBytes, 0, 0, "rank0-target",
        layout, receipt, 1, model::kTargetSlabChunks,
        model::kTargetSlabPeakPinnedBytes, 1, 2, chunks,
        model::kTargetSlabChunks, output);
  };
  const auto invalid_probe = retain(0x100000000000ULL, &first);
  if ((invalid_probe < 31 || invalid_probe > 35) || first) return 5;
  if (model::TargetSlabStartupFactory::lease_from_handle(
          reinterpret_cast<void*>(0x1234)))
    return 6;
  return 0;
}

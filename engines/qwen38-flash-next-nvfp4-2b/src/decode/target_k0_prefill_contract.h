// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_k0_executor.h"

#include <array>
#include <cstdint>
#include <string_view>

namespace rocket::qwen38::decode {

inline constexpr int kTargetK0MaxPrefillRows = 8'192;

enum class TargetK0PrefillStateKind : std::uint8_t {
  kGdnChunkRecurrent,
  kQsaKvCache,
};

struct TargetK0PrefillChunk {
  std::uint64_t first_generation = 0;
  int rows = 0;
  // Borrowed device views. The caller retains both allocations and keeps them
  // live through the same-stream call. Each view has rows * 4 * 2,560 BF16
  // elements in row-major order.
  const __nv_bfloat16* replicated_pre_layer = nullptr;
  __nv_bfloat16* replicated_post_layer = nullptr;
};

// Stronger production contract for true chunk-prefill execution. This is a
// separate subtype so a packed-decode-only TargetK0LayerPort cannot be
// relabeled as prefill-capable. The construction-bound state publication must
// authenticate either GDN convolution/recurrent state or QSA KV state and
// remain live for the owner's lifetime. Calls are single-stream and externally
// serialized. Validation failures perform no device work and publish no state.
class TargetK0PrefillLayerPort : public TargetK0LayerPort {
 public:
  virtual TargetK0PrefillStateKind prefill_state_kind() const noexcept = 0;
  virtual bool prefill_state_authenticated() const noexcept = 0;
  virtual int prefill_capacity_rows() const noexcept = 0;

  void execute_prefill_chunk(
      TargetK0PrefillChunk chunk, cudaStream_t stream,
      TargetK0ExecutionProgress* progress = nullptr);

 protected:
  virtual void execute_authenticated_prefill_chunk(
      TargetK0PrefillChunk chunk, cudaStream_t stream,
      TargetK0ExecutionProgress* progress) = 0;
};

// Authenticates and publishes a borrowed all48 true-prefill port inventory.
// The caller owns every port and the PairReduce schedule, and must keep them
// live longer than this inventory. Construction is bounded, performs no CUDA
// work, and emits one bounded OTEL outcome. Failure publishes no ports.
class TargetK0PrefillPortInventory final {
 public:
  TargetK0PrefillPortInventory(
      int rank, std::string_view oracle_manifest_sha256, int rows,
      std::array<TargetK0PrefillLayerPort*, kDecoderLayers> candidate_ports,
      TargetK0PairReduceSchedule& reductions,
      pair_reduce::OtelStageSink& telemetry);

  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int rows() const noexcept { return rows_; }
  [[nodiscard]] bool authenticated() const noexcept { return authenticated_; }
  [[nodiscard]] const std::array<TargetK0PrefillLayerPort*, kDecoderLayers>&
  ports() const noexcept {
    return ports_;
  }

 private:
  int rank_ = -1;
  int rows_ = 0;
  bool authenticated_ = false;
  std::array<TargetK0PrefillLayerPort*, kDecoderLayers> ports_{};
};

}  // namespace rocket::qwen38::decode

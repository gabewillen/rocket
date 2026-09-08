// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_full_layer.h"
#include "moe/target_full_moe_c1.h"

namespace rocket::qwen38::moe {

enum class TargetMoeServingDtype : std::uint8_t { kNvfp4, kBfloat16 };

struct TargetMoeComponentIdentity {
  TargetFullMoeComponent component;
  TargetDenseIdentity identity;
  TargetMoeServingDtype serving_dtype;
};

// Composition adapter for the physically validated router -> localized B12X
// -> shared-expert participant. All storage remains graph-owned by the caller.
// launch() only changes borrowed activation/stream fields and enqueues work.
class NativeTargetMoeGraph final : public decode::TargetMoeGraph {
 public:
  NativeTargetMoeGraph(TargetFullMoeC1Port& participant,
                       const TargetFullMoeC1Workspace& workspace,
                       __nv_bfloat16* rank_local_partial_bf16,
                       TargetFullMoeOtelSink& telemetry);

  int rank() const noexcept override { return rank_; }
  int layer() const noexcept override { return layer_; }
  std::string_view checkpoint_revision() const noexcept override {
    return decode::kFullAttentionCheckpointRevision;
  }
  std::string_view slab_key() const noexcept override {
    return rank_ == 0 ? "rank0-target" : "rank1-target";
  }
  void launch(const __nv_bfloat16* block_input,
              std::uint64_t generation, int m,
              cudaStream_t stream) override;
  void publish_after_fence(std::uint64_t generation) override;
  const __nv_bfloat16* projected_output() const noexcept override {
    return !faulted_ && published_generation_ != 0
               ? rank_local_partial_bf16_
               : nullptr;
  }
  [[nodiscard]] TargetMoeComponentIdentity component_identity(
      TargetFullMoeComponent component) const noexcept;

 private:
  TargetFullMoeC1Port& participant_;
  TargetFullMoeC1Workspace workspace_;
  __nv_bfloat16* rank_local_partial_bf16_;
  TargetFullMoeOtelSink& telemetry_;
  int rank_;
  int layer_;
  std::uint64_t published_generation_ = 0;
  std::uint64_t enqueued_generation_ = 0;
  bool faulted_ = false;
};

}  // namespace rocket::qwen38::moe

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

class NativeTargetMoeCudaApi {
 public:
  virtual ~NativeTargetMoeCudaApi() = default;
  virtual cudaError_t stream_synchronize(cudaStream_t stream) noexcept = 0;
  virtual cudaError_t device_synchronize() noexcept = 0;
};

// Composition adapter for the physically validated router -> localized B12X
// -> shared-expert participant. All storage remains graph-owned by the caller.
// launch() only changes borrowed activation/stream fields and enqueues work.
class NativeTargetMoeGraph final : public decode::TargetMoeGraph {
 public:
  enum class State : std::uint8_t {
    kReady,
    kSourceWaited,
    kPossiblyEnqueued,
    kFenced,
    kPublished,
    kFaultedNoFlight,
    kFaultedPossiblyEnqueued,
    kFaultedFenced,
    kFaultedQuarantined,
  };
  NativeTargetMoeGraph(TargetFullMoeC1Port& participant,
                       const TargetFullMoeC1Workspace& workspace,
                       __nv_bfloat16* rank_local_partial_bf16,
                       TargetFullMoeOtelSink& telemetry,
                       NativeTargetMoeCudaApi* cuda_api = nullptr);
  ~NativeTargetMoeGraph() override;

  int rank() const noexcept override { return rank_; }
  int layer() const noexcept override { return layer_; }
  std::string_view checkpoint_revision() const noexcept override {
    return decode::kFullAttentionCheckpointRevision;
  }
  std::string_view slab_key() const noexcept override {
    return rank_ == 0 ? "rank0-target" : "rank1-target";
  }
  // Initialization-only source dependency. Must precede capture and every
  // launch must use the same borrowed stream.
  void wait_source(cudaStream_t stream);
  void launch(const __nv_bfloat16* block_input,
              std::uint64_t generation, int m,
              cudaStream_t stream) override;
  void publish_after_fence(std::uint64_t generation) override;
  void terminal_fence_succeeded(std::uint64_t generation) override;
  void fault_after_fence(std::uint64_t generation) noexcept override;
  // Destruction gate used by the concrete owner. False means completion could
  // not be proven and every referenced CUDA resource must be quarantined.
  [[nodiscard]] bool drain_for_destruction() noexcept;
  const __nv_bfloat16* projected_output() const noexcept override {
    return state_ == State::kPublished && published_generation_ != 0
               ? rank_local_partial_bf16_
               : nullptr;
  }
  [[nodiscard]] TargetMoeComponentIdentity component_identity(
      TargetFullMoeComponent component) const noexcept;

 private:
  void mark_faulted() noexcept;
  [[nodiscard]] bool drain_faulted_generation() noexcept;

  TargetFullMoeC1Port& participant_;
  TargetFullMoeC1Workspace workspace_;
  __nv_bfloat16* rank_local_partial_bf16_;
  TargetFullMoeOtelSink& telemetry_;
  NativeTargetMoeCudaApi* cuda_api_;
  int rank_;
  int layer_;
  std::uint64_t published_generation_ = 0;
  std::uint64_t enqueued_generation_ = 0;
  State state_ = State::kReady;
  cudaStream_t source_stream_ = nullptr;
};

}  // namespace rocket::qwen38::moe

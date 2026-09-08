// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace rocket::qwen38::decode {

// Closed, allocation-free localization for the oracle-bounded K0 prompt.
// Values are copied to bounded C ABI/OTEL fields. They are never used to
// select execution behavior.
enum class TargetK0ExecutionStage : std::uint8_t {
  kValidation,
  kTokenSourceWait,
  kLayerSourceWait,
  kBeginSequence,
  kBeginRow,
  kEmbeddingReduction,
  kEmbeddingComparison,
  kLayerExecution,
  kLayerComparison,
  kFinalNorm,
  kLmHead,
  kWinnerExchange,
  kTerminalFence,
  kFinalNormComparison,
  kLogitsComparison,
  kTokenComparison,
  kComplete,
};

enum class TargetK0LayerExecutionStage : std::uint8_t {
  kNone,
  kStatePreparation,
  kAttentionHyperconnection,
  kAttention,
  kAttentionFence,
  kAttentionReduction,
  kMlpHyperconnection,
  kMoe,
  kMoeReduction,
  kFinalHyperconnection,
};

enum class TargetK0GdnGraphStage : std::uint8_t {
  kNone,
  kInputQuantize,
  kQkvProjection,
  kBaProjection,
  kInputScale,
  kCore,
  kOutputQuantize,
  kOutputProjection,
  kOutputScale,
  kLaunchCheck,
  kOutputPublication,
};

// Diagnostic-only row0/layer0 boundaries. These values never select forward
// behavior or acceptance. The fixed arrays are safe to publish as bounded
// numeric OTEL measurements; the hash is never used as a metric dimension.
enum class TargetK0LayerBoundary : std::uint8_t {
  kAttentionOutput,
  kAttentionReduction,
  kHyperconnectionCombineMix,
  kMoeOutput,
  kMoeReduction,
  kFinalHyperconnection,
  kCount,
};

enum class TargetK0DiagnosticDtype : std::uint8_t { kBfloat16, kFloat32 };

inline constexpr std::size_t kTargetK0LayerBoundaryCount =
    static_cast<std::size_t>(TargetK0LayerBoundary::kCount);

struct TargetK0LayerBoundaryEvidence {
  std::array<std::uint64_t, kTargetK0LayerBoundaryCount> hashes{};
  std::array<std::uint32_t, kTargetK0LayerBoundaryCount> elements{};
  std::array<std::uint32_t, kTargetK0LayerBoundaryCount> zero_counts{};
  std::array<std::uint32_t, kTargetK0LayerBoundaryCount> nonfinite_counts{};
  std::array<std::uint32_t, kTargetK0LayerBoundaryCount>
      reference_mismatch_counts{};
  std::array<std::uint32_t, kTargetK0LayerBoundaryCount>
      reference_first_mismatches{};
  std::array<std::uint8_t, kTargetK0LayerBoundaryCount> reference_compared{};
  std::array<std::uint8_t, kTargetK0LayerBoundaryCount> reference_exact{};
};

class TargetK0LayerBoundaryObserver {
 public:
  virtual ~TargetK0LayerBoundaryObserver() = default;
  virtual void observe(TargetK0LayerBoundary boundary, const void* device_values,
                       std::size_t elements, TargetK0DiagnosticDtype dtype,
                       cudaStream_t stream,
                       TargetK0LayerBoundaryEvidence& evidence) = 0;
};

struct TargetK0ExecutionProgress {
  TargetK0ExecutionStage stage = TargetK0ExecutionStage::kValidation;
  std::int32_t row = -1;
  std::int32_t layer = -1;
  TargetK0LayerExecutionStage layer_stage =
      TargetK0LayerExecutionStage::kNone;
  TargetK0GdnGraphStage gdn_graph_stage = TargetK0GdnGraphStage::kNone;
  TargetK0LayerBoundaryObserver* boundary_observer = nullptr;
  TargetK0LayerBoundaryEvidence boundary_evidence{};
  std::array<std::uint32_t, 5> oracle_domain_skip_counts{};
};

inline void target_k0_note_oracle_domain_skip(
    TargetK0ExecutionProgress* progress, std::uint8_t boundary) noexcept {
  if (progress && boundary < progress->oracle_domain_skip_counts.size())
    ++progress->oracle_domain_skip_counts[boundary];
}

inline void target_k0_enter(
    TargetK0ExecutionProgress* progress, TargetK0ExecutionStage stage,
    int row = -1, int layer = -1,
    TargetK0LayerExecutionStage layer_stage =
        TargetK0LayerExecutionStage::kNone) noexcept {
  if (!progress) return;
  progress->stage = stage;
  progress->row = row;
  progress->layer = layer;
  progress->layer_stage = layer_stage;
  progress->gdn_graph_stage = TargetK0GdnGraphStage::kNone;
}

inline void target_k0_enter_layer(TargetK0ExecutionProgress* progress,
                                  TargetK0LayerExecutionStage stage) noexcept {
  if (!progress) return;
  progress->stage = TargetK0ExecutionStage::kLayerExecution;
  progress->layer_stage = stage;
}

inline void target_k0_enter_stage(TargetK0ExecutionProgress* progress,
                                  TargetK0ExecutionStage stage) noexcept {
  if (!progress) return;
  progress->stage = stage;
  progress->layer = -1;
  progress->layer_stage = TargetK0LayerExecutionStage::kNone;
  progress->gdn_graph_stage = TargetK0GdnGraphStage::kNone;
}

inline void target_k0_enter_gdn_graph(TargetK0ExecutionProgress* progress,
                                      TargetK0GdnGraphStage stage) noexcept {
  if (progress) progress->gdn_graph_stage = stage;
}

inline void target_k0_observe_layer0(
    TargetK0ExecutionProgress* progress, TargetK0LayerBoundary boundary,
    const void* device_values, std::size_t elements,
    TargetK0DiagnosticDtype dtype, cudaStream_t stream) {
  if (progress && progress->row == 0 && progress->layer == 0 &&
      progress->boundary_observer)
    progress->boundary_observer->observe(
        boundary, device_values, elements, dtype, stream,
        progress->boundary_evidence);
}

}  // namespace rocket::qwen38::decode

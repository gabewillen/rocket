// SPDX-License-Identifier: Apache-2.0
#pragma once

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
  kAttentionReduction,
  kMlpHyperconnection,
  kMoe,
  kMoeReduction,
  kFinalHyperconnection,
};

struct TargetK0ExecutionProgress {
  TargetK0ExecutionStage stage = TargetK0ExecutionStage::kValidation;
  std::int32_t row = -1;
  std::int32_t layer = -1;
  TargetK0LayerExecutionStage layer_stage =
      TargetK0LayerExecutionStage::kNone;
};

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
}

}  // namespace rocket::qwen38::decode

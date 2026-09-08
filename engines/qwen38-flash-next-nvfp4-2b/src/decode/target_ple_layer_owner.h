// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_layer_native_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rocket::qwen38::decode {

inline constexpr int kTargetPleLayer = 1;
inline constexpr int kTargetPleEmbeddingShards = 64;

enum class TargetPleStage : std::uint8_t {
  kEmbeddingLookup = 0,
  kEmbeddingDequantize = 1,
  kKeyProjection = 2,
  kValueProjection = 3,
  kKeyNorm = 4,
  kQueryNorm = 5,
  kGate = 6,
  kConvNorm = 7,
  kShortConv = 8,
  kResidual = 9,
};

inline constexpr std::array<TargetPleStage, 10> kTargetPleStageOrder{
    TargetPleStage::kEmbeddingLookup,
    TargetPleStage::kEmbeddingDequantize,
    TargetPleStage::kKeyProjection,
    TargetPleStage::kValueProjection,
    TargetPleStage::kKeyNorm,
    TargetPleStage::kQueryNorm,
    TargetPleStage::kGate,
    TargetPleStage::kConvNorm,
    TargetPleStage::kShortConv,
    TargetPleStage::kResidual,
};

enum class TargetPleStageOutcome : std::uint8_t {
  kOk = 0,
  kContractError = 1,
  kCudaError = 2,
};

struct TargetPleStageEvent {
  int rank;
  int layer;
  TargetPleStage stage;
  TargetPleStageOutcome outcome;
};

class TargetPleStageSink {
 public:
  virtual ~TargetPleStageSink() = default;
  virtual void emit(const TargetPleStageEvent& event) noexcept = 0;
};

struct TargetPleNvfp4MatrixBinding {
  TargetLayerNativeExtent weight;
  TargetLayerNativeExtent weight_scale;
  TargetLayerNativeExtent weight_scale_2;
  TargetLayerNativeExtent input_scale;
};

// Pointer-free binding. A later device owner may resolve these authenticated
// offsets only after it acquires the process-lifetime target-slab lease.
struct TargetPleLayerBinding {
  int rank = -1;
  int layer = -1;
  std::string artifact_key;
  std::string descriptor_sha256;
  std::string binding_inventory_sha256;
  std::string publication_layout_sha256;
  TargetPleNvfp4MatrixBinding key_projection;
  TargetPleNvfp4MatrixBinding value_projection;
  TargetLayerNativeExtent norm_key;
  TargetLayerNativeExtent norm_query;
  TargetLayerNativeExtent norm_conv;
  TargetLayerNativeExtent convolution;
  TargetLayerNativeExtent layer_multipliers;
  TargetLayerNativeExtent ngram_head_offsets;
  TargetLayerNativeExtent ngram_head_vocab_sizes;
  TargetLayerNativeExtent embedding_scale;
  std::array<TargetLayerNativeExtent, kTargetPleEmbeddingShards>
      embedding_shards;
};

bool validate_target_ple_layer_plan(
    const TargetLayerNativePlan& plan) noexcept;

// Performs no address formation, allocation, synchronization, or CUDA work.
TargetPleLayerBinding bind_target_ple_layer_plan(
    const TargetLayerNativePlan& plan);

// CPU/no-launch owner of the authenticated binding and ordered telemetry seam.
// The telemetry sink is borrowed and must outlive this object.
class TargetPleLayerOwnerContract final {
 public:
  static std::unique_ptr<TargetPleLayerOwnerContract> create(
      const TargetLayerNativePlan& plan, TargetPleStageSink& telemetry);

  int rank() const noexcept { return binding_.rank; }
  int layer() const noexcept { return binding_.layer; }
  bool authenticated() const noexcept { return authenticated_; }
  bool complete() const noexcept {
    return next_stage_ == kTargetPleStageOrder.size() && !faulted_;
  }
  bool faulted() const noexcept { return faulted_; }
  const TargetPleLayerBinding& binding() const noexcept { return binding_; }

  bool record_stage(TargetPleStage stage,
                    TargetPleStageOutcome outcome) noexcept;

 private:
  TargetPleLayerOwnerContract(TargetPleLayerBinding binding,
                              TargetPleStageSink& telemetry);

  TargetPleLayerBinding binding_;
  TargetPleStageSink* telemetry_ = nullptr;
  std::size_t next_stage_ = 0;
  bool authenticated_ = false;
  bool faulted_ = false;
};

}  // namespace rocket::qwen38::decode

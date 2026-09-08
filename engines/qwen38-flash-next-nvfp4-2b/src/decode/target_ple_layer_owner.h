// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_layer_native_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace rocket::qwen38::decode {

inline constexpr int kTargetPleLayer = 1;
inline constexpr int kTargetPleEmbeddingShards = 64;
inline constexpr int kTargetPleNgramContextTokens = 2;
inline constexpr int kTargetPleConvChannels = 10'240;
inline constexpr int kTargetPleConvHistoryTokens = 9;

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

// Device-independent description of the later PLE convolution-state owner.
// The pinned layer uses a width-four convolution with dilation three, hence
// (4 - 1) * 3 == 9 historical positions per one of 10,240 channels.
struct TargetPleConvStateLayout {
  int slots = 0;
  int channels = 0;
  int history_tokens = 0;
};

// Complete owned scheduler input for one packed PLE publication. Token and
// state-index integers retain the scheduler's I32 ABI; a later CUDA owner may
// widen them internally as the pinned implementation does.
struct TargetPleRowContextInput {
  std::uint64_t generation = 0;
  std::vector<std::int32_t> input_ids;
  std::vector<std::int32_t> query_start_loc;
  // Flattened [requests, kTargetPleNgramContextTokens].
  std::vector<std::int32_t> ngram_context;
  std::vector<std::int32_t> conv_state_indices;
  // One canonical byte per request. Values are restricted to {0, 1}.
  std::vector<std::uint8_t> has_initial_state;
};

struct TargetPleRowContextSnapshot {
  std::uint64_t generation = 0;
  int rows = 0;
  int requests = 0;
  std::vector<std::int32_t> input_ids;
  std::vector<std::int32_t> query_start_loc;
  std::vector<std::int32_t> ngram_context;
  std::vector<std::int32_t> conv_state_indices;
  std::vector<std::uint8_t> has_initial_state;
};

enum class TargetPleRowContextOperation : std::uint8_t {
  kPublish = 0,
  kAcquire = 1,
};

struct TargetPleRowContextEvent {
  int rank;
  int layer;
  TargetPleRowContextOperation operation;
  TargetPleStageOutcome outcome;
};

// Bounded dimensions are rank {0,1}, layer {1}, operation, and outcome.
// Generation, token IDs, request counts, and state slots are intentionally
// excluded because they are unbounded or scheduler-dependent.
class TargetPleRowContextOtelSink {
 public:
  virtual ~TargetPleRowContextOtelSink() = default;
  virtual void emit(const TargetPleRowContextEvent& event) noexcept = 0;
};

// Authenticated, generation-bound CPU/no-launch publication owner. It owns
// every published array, so consumers cannot outlive scheduler storage. Any
// invalid publication or stale acquisition permanently faults the instance.
class TargetPleRowContextProvider final {
 public:
  static std::unique_ptr<TargetPleRowContextProvider> create(
      const TargetLayerNativePlan& plan, int max_rows,
      TargetPleConvStateLayout conv_state,
      TargetPleRowContextOtelSink& telemetry);

  int rank() const noexcept { return binding_.rank; }
  int layer() const noexcept { return binding_.layer; }
  bool authenticated() const noexcept { return authenticated_ && !faulted_; }
  bool faulted() const noexcept { return faulted_; }
  int max_rows() const noexcept { return max_rows_; }
  const TargetPleConvStateLayout& conv_state_layout() const noexcept {
    return conv_state_;
  }
  const TargetPleLayerBinding& binding() const noexcept { return binding_; }

  const TargetPleRowContextSnapshot& publish(TargetPleRowContextInput input);
  const TargetPleRowContextSnapshot& acquire(
      std::uint64_t generation) const;

 private:
  TargetPleRowContextProvider(TargetPleLayerBinding binding, int max_rows,
                              TargetPleConvStateLayout conv_state,
                              TargetPleRowContextOtelSink& telemetry);
  [[noreturn]] void fault(TargetPleRowContextOperation operation,
                          std::string_view reason) const;

  TargetPleLayerBinding binding_;
  int max_rows_ = 0;
  TargetPleConvStateLayout conv_state_{};
  bool authenticated_ = false;
  mutable bool faulted_ = false;
  bool published_ = false;
  TargetPleRowContextSnapshot snapshot_{};
  TargetPleRowContextOtelSink* telemetry_ = nullptr;
};

}  // namespace rocket::qwen38::decode

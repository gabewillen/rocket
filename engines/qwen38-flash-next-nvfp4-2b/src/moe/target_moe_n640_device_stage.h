// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "moe/target_moe_b12x_aot.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rocket::qwen38::moe {

inline constexpr int kTargetMoeStagedExperts = kTargetMoeC1TopK;
inline constexpr std::string_view kTargetMoeDeviceStageAbi =
    "rocket.qwen38.target-moe.device-stage.v1";
inline constexpr std::size_t kTargetMoeStagedW13PackedBytes =
    kTargetMoeStagedExperts * 2ULL * kTargetMoePhysicalIntermediate *
    kTargetMoeHidden / 2;
inline constexpr std::size_t kTargetMoeStagedW13ScaleBytes =
    kTargetMoeStagedExperts * 2ULL * kTargetMoePhysicalIntermediate *
    (kTargetMoeHidden / 16);
inline constexpr std::size_t kTargetMoeStagedDownPackedBytes =
    kTargetMoeStagedExperts * 1ULL * kTargetMoeHidden *
    kTargetMoePhysicalIntermediate / 2;
inline constexpr std::size_t kTargetMoeStagedDownScaleBytes =
    kTargetMoeStagedExperts * 1ULL * kTargetMoeHidden *
    (kTargetMoePhysicalIntermediate / 16);

// All pointers alias authenticated immutable N640 extents in the accepted
// target slab. FC1 source ordering remains explicit because FlashInfer's
// serving ABI is [up, gate], while checkpoint names are gate/up.
struct TargetMoeN640DeviceExpert {
  const std::uint8_t* up_packed = nullptr;
  const std::uint8_t* up_scale = nullptr;
  const float* up_input_scale = nullptr;
  const float* up_alpha = nullptr;
  const std::uint8_t* gate_packed = nullptr;
  const std::uint8_t* gate_scale = nullptr;
  const float* gate_input_scale = nullptr;
  const float* gate_alpha = nullptr;
  const std::uint8_t* down_packed = nullptr;
  const std::uint8_t* down_scale = nullptr;
  const float* down_input_scale = nullptr;
  const float* down_alpha = nullptr;
};

struct TargetMoeN640StageEvidence {
  std::uint64_t generation = 0;
  std::int32_t active_experts = 0;
  TargetMoeOutcome outcome = TargetMoeOutcome::kContractError;
};

struct TargetMoeN640StageReference {
  std::array<std::int32_t, kTargetMoeStagedExperts> source_expert_ids{};
  std::array<std::int32_t, kTargetMoeC1TopK> compact_expert_ids{};
  std::array<float, kTargetMoeC1TopK> compact_routing_weights{};
  TargetMoeN640StageEvidence evidence{};
};

TargetMoeN640StageReference target_moe_n640_stage_reference(
    const std::array<std::int32_t, kTargetMoeC1TopK>& local_expert_ids,
    const std::array<float, kTargetMoeC1TopK>& local_routing_weights,
    std::uint64_t source_generation,
    std::uint64_t requested_generation) noexcept;

inline constexpr std::size_t kTargetMoeStageRawPlaneBytes =
    kTargetMoeStagedW13PackedBytes + kTargetMoeStagedW13ScaleBytes +
    kTargetMoeStagedDownPackedBytes + kTargetMoeStagedDownScaleBytes +
    4ULL * kTargetMoeStagedExperts * sizeof(float);
inline constexpr std::size_t kTargetMoeStageScratchBytes =
    kTargetMoeStageRawPlaneBytes +
    2ULL * kTargetMoeStagedExperts * sizeof(std::int32_t) +
    kTargetMoeStagedExperts * sizeof(float) +
    sizeof(TargetMoeN640StageEvidence);

// Caller-owned stable-address scratch. Exactly ten physical expert slots are
// sufficient for c1/top10. The same allocation is reused sequentially across
// rows and can be rebound to each layer's authenticated source table.
struct TargetMoeN768StageScratch {
  std::uint8_t* w13_packed = nullptr;
  std::size_t w13_packed_bytes = 0;
  std::uint8_t* w13_scale = nullptr;
  std::size_t w13_scale_bytes = 0;
  std::uint8_t* down_packed = nullptr;
  std::size_t down_packed_bytes = 0;
  std::uint8_t* down_scale = nullptr;
  std::size_t down_scale_bytes = 0;
  float* input_global_scale = nullptr;
  float* folded_w1_alpha = nullptr;
  float* w2_alpha = nullptr;
  float* down_input_scale = nullptr;
  std::int32_t* source_expert_ids = nullptr;
  std::int32_t* compact_expert_ids = nullptr;
  float* compact_routing_weights = nullptr;
  TargetMoeN640StageEvidence* evidence = nullptr;
  // Host alias of mapped, caller-owned evidence storage. The device pointer
  // above is written in-stream; this alias is read only after the enclosing
  // stream fence, so failure publication needs no D2H copy.
  const TargetMoeN640StageEvidence* host_evidence = nullptr;
  std::size_t scalar_capacity = 0;
  std::size_t route_capacity = 0;
};

struct TargetMoeN640StageLaunch {
  const std::int32_t* local_expert_ids = nullptr;
  const float* local_routing_weights = nullptr;
  const std::uint64_t* source_generation = nullptr;
  const std::uint64_t* requested_generation = nullptr;
  TargetMoeN768StageScratch scratch;
  cudaStream_t stream = nullptr;
};

enum class TargetMoeStageCounter : std::uint8_t {
  kLaunch,
  kActiveExperts,
  kSourceBytes,
  kScratchBytes,
};

struct TargetMoeStageOtelPoint {
  TargetMoeStageCounter counter;
  TargetMoeOutcome outcome;
  int rank;
  int layer;
  std::uint64_t value;
};

class TargetMoeStageOtelSink {
 public:
  virtual ~TargetMoeStageOtelSink() = default;
  virtual void add_counter(const TargetMoeStageOtelPoint& point) noexcept = 0;
};

// Owns only a 256-entry device pointer table. The source planes, slab-ready
// event, launch inputs, stream, and scratch are borrowed; their owners must
// outlive this object and every enqueued use. wait_source() is a one-time
// pre-capture operation. Subsequent enqueue calls are single-owner,
// non-reentrant, and must use that same stream. enqueue() performs fixed-grid
// D2D padding/scale conversion and no allocation, host copy, synchronization,
// D2H, Python, or Torch operation. Any immediate failure publishes no stage
// output; any asynchronous failure is reported through evidence after fence.
class TargetMoeN640DeviceStage final {
 public:
  TargetMoeN640DeviceStage(
      int device, int rank, int layer, cudaEvent_t slab_ready,
      const std::array<TargetMoeN640DeviceExpert,
                       kTargetMoeLocalExperts>& experts);
  ~TargetMoeN640DeviceStage();
  TargetMoeN640DeviceStage(const TargetMoeN640DeviceStage&) = delete;
  TargetMoeN640DeviceStage& operator=(const TargetMoeN640DeviceStage&) = delete;

  void wait_source(cudaStream_t stream);
  // First node of every captured enqueue. It replaces prior-generation mapped
  // evidence with a bounded contract-failure envelope before any router work.
  [[nodiscard]] TargetMoeOutcome enqueue_pending_evidence(
      const std::uint64_t* requested_generation,
      const TargetMoeN768StageScratch& scratch,
      cudaStream_t stream) const noexcept;
  [[nodiscard]] TargetMoeOutcome enqueue(
      const TargetMoeN640StageLaunch& launch) const noexcept;
  int rank() const noexcept { return rank_; }
  int layer() const noexcept { return layer_; }

 private:
  int device_ = -1;
  int rank_ = -1;
  int layer_ = -1;
  cudaEvent_t slab_ready_ = nullptr;
  TargetMoeN640DeviceExpert* device_experts_ = nullptr;
  cudaStream_t source_stream_ = nullptr;
  bool source_wait_enqueued_ = false;
};

TargetMoeB12xWeights target_moe_staged_weights(
    const TargetMoeN768StageScratch& scratch) noexcept;
bool validate_target_moe_stage_scratch(
    const TargetMoeN768StageScratch& scratch) noexcept;
TargetMoeOutcome validate_target_moe_stage_after_fence(
    const TargetMoeN768StageScratch& scratch,
    std::uint64_t requested_generation) noexcept;

// Call only after the borrowed stream's terminal fence. Cardinality is
// counter(4) x outcome(4) x rank(2) x layer(48) = 1,536 bounded series.
// Generation, active count, and bytes are metric values, never attributes.
void export_target_moe_stage_otel_after_fence(
    const TargetMoeN640StageEvidence& evidence,
    std::uint64_t requested_generation, int rank, int layer,
    TargetMoeStageOtelSink& sink) noexcept;

}  // namespace rocket::qwen38::moe

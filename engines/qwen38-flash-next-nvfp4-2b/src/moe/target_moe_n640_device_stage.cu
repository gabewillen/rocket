// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_n640_device_stage.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace rocket::qwen38::moe {
namespace {

constexpr std::size_t kPhysicalW13PerExpert =
    2ULL * kTargetMoePhysicalIntermediate * kTargetMoeHidden / 2;
constexpr std::size_t kPhysicalDownPerExpert =
    1ULL * kTargetMoeHidden * kTargetMoePhysicalIntermediate / 2;
constexpr std::size_t kPhysicalW13ScalePerExpert =
    2ULL * kTargetMoePhysicalIntermediate * (kTargetMoeHidden / 16);
constexpr std::size_t kPhysicalDownScalePerExpert =
    1ULL * kTargetMoeHidden * (kTargetMoePhysicalIntermediate / 16);

bool valid_expert(const TargetMoeN640DeviceExpert& e) noexcept {
  return e.up_packed && e.up_scale && e.up_input_scale && e.up_alpha &&
         e.gate_packed && e.gate_scale && e.gate_input_scale && e.gate_alpha &&
         e.down_packed && e.down_scale && e.down_input_scale && e.down_alpha;
}

bool valid_scratch(const TargetMoeN768StageScratch& s) noexcept {
  return s.w13_packed && s.w13_scale && s.down_packed && s.down_scale &&
         s.input_global_scale && s.folded_w1_alpha && s.w2_alpha &&
         s.down_input_scale && s.source_expert_ids && s.compact_expert_ids &&
         s.compact_routing_weights && s.evidence &&
         s.w13_packed_bytes == kTargetMoeStagedW13PackedBytes &&
         s.w13_scale_bytes == kTargetMoeStagedW13ScaleBytes &&
         s.down_packed_bytes == kTargetMoeStagedDownPackedBytes &&
         s.down_scale_bytes == kTargetMoeStagedDownScaleBytes &&
         s.scalar_capacity == kTargetMoeStagedExperts &&
         s.route_capacity == kTargetMoeC1TopK;
}

__device__ std::size_t swizzled_scale_offset(
    std::size_t row, std::size_t column_block, std::size_t columns) {
  const std::size_t column_tiles = (columns + 3) / 4;
  const std::size_t tile_row = row / 128;
  const std::size_t inner_m = (row % 128) / 32;
  const std::size_t outer_m = row % 32;
  const std::size_t tile_column = column_block / 4;
  const std::size_t inner_column = column_block % 4;
  return ((((tile_row * column_tiles + tile_column) * 32 + outer_m) * 4 +
            inner_m) * 4 + inner_column);
}

__host__ __device__ TargetMoeN640StageEvidence build_route_map(
    const std::int32_t* ids, const float* weights,
    std::uint64_t source_generation, std::uint64_t requested_generation,
    std::int32_t* source_experts, std::int32_t* compact_ids,
    float* compact_weights) {
  for (int slot = 0; slot < kTargetMoeStagedExperts; ++slot) {
    source_experts[slot] = -1;
    compact_ids[slot] = 0;
    compact_weights[slot] = 0.0F;
  }
  if (requested_generation == 0 || source_generation != requested_generation)
    return {0, 0, TargetMoeOutcome::kStaleGeneration};
  int active = 0;
  for (int route = 0; route < kTargetMoeC1TopK; ++route) {
    const int expert = ids[route];
    const float weight = weights[route];
    if (expert < 0 || expert >= kTargetMoeLocalExperts ||
        !isfinite(weight) || weight < 0.0F || weight > 1.0F) {
      for (int slot = 0; slot < kTargetMoeStagedExperts; ++slot) {
        source_experts[slot] = -1;
        compact_ids[slot] = 0;
        compact_weights[slot] = 0.0F;
      }
      return {0, 0, TargetMoeOutcome::kContractError};
    }
    if (weight == 0.0F) continue;
    for (int earlier = 0; earlier < active; ++earlier)
      if (source_experts[earlier] == expert) {
        for (int slot = 0; slot < kTargetMoeStagedExperts; ++slot) {
          source_experts[slot] = -1;
          compact_ids[slot] = 0;
          compact_weights[slot] = 0.0F;
        }
        return {0, 0, TargetMoeOutcome::kContractError};
      }
    if (active >= kTargetMoeStagedExperts)
      return {0, 0, TargetMoeOutcome::kContractError};
    compact_ids[route] = active;
    compact_weights[route] = weight;
    source_experts[active++] = expert;
  }
  return {requested_generation, active, TargetMoeOutcome::kOk};
}

__global__ void select_experts(
    const TargetMoeN640DeviceExpert* experts,
    const std::int32_t* ids, const float* weights,
    const std::uint64_t* source_generation,
    const std::uint64_t* requested_generation,
    TargetMoeN768StageScratch scratch) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const std::uint64_t requested = *requested_generation;
  *scratch.evidence = build_route_map(
      ids, weights, *source_generation, requested,
      scratch.source_expert_ids, scratch.compact_expert_ids,
      scratch.compact_routing_weights);
  if (scratch.evidence->outcome != TargetMoeOutcome::kOk) return;
  for (int compact = 0; compact < scratch.evidence->active_experts;
       ++compact) {
    const int expert = scratch.source_expert_ids[compact];
    const auto& source = experts[expert];
    const float input = *source.gate_input_scale;
    const float alpha = *source.gate_alpha;
    if (input != *source.up_input_scale || alpha != *source.up_alpha ||
        !isfinite(input) || !isfinite(alpha) ||
        !isfinite(*source.down_alpha) ||
        !isfinite(*source.down_input_scale)) {
      for (int slot = 0; slot < kTargetMoeStagedExperts; ++slot) {
        scratch.source_expert_ids[slot] = -1;
        scratch.compact_expert_ids[slot] = 0;
        scratch.compact_routing_weights[slot] = 0.0F;
      }
      *scratch.evidence = {0, 0, TargetMoeOutcome::kContractError};
      return;
    }
  }
}

__global__ void stage_packed(
    const TargetMoeN640DeviceExpert* experts,
    TargetMoeN768StageScratch scratch) {
  const std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t w13_total = kTargetMoeStagedW13PackedBytes;
  const std::size_t down_total = kTargetMoeStagedDownPackedBytes;
  if (linear < w13_total) {
    const int slot = static_cast<int>(linear / kPhysicalW13PerExpert);
    const std::size_t offset = linear % kPhysicalW13PerExpert;
    const std::size_t row = offset / (kTargetMoeHidden / 2);
    const std::size_t column = offset % (kTargetMoeHidden / 2);
    const int expert = scratch.source_expert_ids[slot];
    std::uint8_t value = 0;
    if (scratch.evidence->outcome == TargetMoeOutcome::kOk && expert >= 0) {
      if (row < kTargetMoeLogicalIntermediate)
        value = experts[expert].up_packed[
            row * (kTargetMoeHidden / 2) + column];
      else if (row >= kTargetMoePhysicalIntermediate &&
               row < kTargetMoePhysicalIntermediate +
                         kTargetMoeLogicalIntermediate)
        value = experts[expert].gate_packed[
            (row - kTargetMoePhysicalIntermediate) *
                (kTargetMoeHidden / 2) + column];
    }
    scratch.w13_packed[linear] = value;
  }
  if (linear < down_total) {
    const int slot = static_cast<int>(linear / kPhysicalDownPerExpert);
    const std::size_t offset = linear % kPhysicalDownPerExpert;
    const std::size_t row = offset / (kTargetMoePhysicalIntermediate / 2);
    const std::size_t column = offset % (kTargetMoePhysicalIntermediate / 2);
    const int expert = scratch.source_expert_ids[slot];
    std::uint8_t value = 0;
    if (scratch.evidence->outcome == TargetMoeOutcome::kOk && expert >= 0 &&
        column < kTargetMoeLogicalIntermediate / 2)
      value = experts[expert].down_packed[
          row * (kTargetMoeLogicalIntermediate / 2) + column];
    scratch.down_packed[linear] = value;
  }
}

__global__ void clear_scales(TargetMoeN768StageScratch scratch) {
  const std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear < kTargetMoeStagedW13ScaleBytes)
    scratch.w13_scale[linear] = 0;
  if (linear < kTargetMoeStagedDownScaleBytes)
    scratch.down_scale[linear] = 0;
}

__global__ void stage_scales_and_scalars(
    const TargetMoeN640DeviceExpert* experts,
    TargetMoeN768StageScratch scratch) {
  const std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr std::size_t w13_logical_per_slot =
      2ULL * kTargetMoeLogicalIntermediate * (kTargetMoeHidden / 16);
  constexpr std::size_t down_logical_per_slot =
      1ULL * kTargetMoeHidden * (kTargetMoeLogicalIntermediate / 16);
  if (linear < kTargetMoeStagedExperts * w13_logical_per_slot) {
    const int slot = static_cast<int>(linear / w13_logical_per_slot);
    const std::size_t within = linear % w13_logical_per_slot;
    const bool gate = within >= w13_logical_per_slot / 2;
    const std::size_t half = within % (w13_logical_per_slot / 2);
    const std::size_t row = half / (kTargetMoeHidden / 16);
    const std::size_t column = half % (kTargetMoeHidden / 16);
    const int expert = scratch.source_expert_ids[slot];
    if (scratch.evidence->outcome == TargetMoeOutcome::kOk && expert >= 0) {
      const auto* source = gate ? experts[expert].gate_scale
                                : experts[expert].up_scale;
      const std::size_t destination_row =
          gate ? kTargetMoePhysicalIntermediate + row : row;
      scratch.w13_scale[slot * kPhysicalW13ScalePerExpert +
          swizzled_scale_offset(destination_row, column,
                                kTargetMoeHidden / 16)] =
          source[swizzled_scale_offset(
              row, column, kTargetMoeHidden / 16)];
    }
  }
  if (linear < kTargetMoeStagedExperts * down_logical_per_slot) {
    const int slot = static_cast<int>(linear / down_logical_per_slot);
    const std::size_t within = linear % down_logical_per_slot;
    const std::size_t row = within / (kTargetMoeLogicalIntermediate / 16);
    const std::size_t column = within % (kTargetMoeLogicalIntermediate / 16);
    const int expert = scratch.source_expert_ids[slot];
    if (scratch.evidence->outcome == TargetMoeOutcome::kOk && expert >= 0)
      scratch.down_scale[slot * kPhysicalDownScalePerExpert +
          swizzled_scale_offset(row, column,
                                kTargetMoePhysicalIntermediate / 16)] =
          experts[expert].down_scale[swizzled_scale_offset(
              row, column, kTargetMoeLogicalIntermediate / 16)];
  }
  if (linear < kTargetMoeStagedExperts) {
    const int slot = static_cast<int>(linear);
    const int expert = scratch.source_expert_ids[slot];
    float input = 0.0F, alpha = 0.0F, w2 = 0.0F, down_input = 0.0F;
    if (scratch.evidence->outcome == TargetMoeOutcome::kOk && expert >= 0) {
      const auto& source = experts[expert];
      input = *source.gate_input_scale;
      alpha = *source.gate_alpha;
      w2 = *source.down_alpha;
      down_input = *source.down_input_scale;
    }
    scratch.input_global_scale[slot] = input;
    scratch.folded_w1_alpha[slot] = input * alpha;
    scratch.w2_alpha[slot] = w2;
    scratch.down_input_scale[slot] = down_input;
  }
}

}  // namespace

TargetMoeN640DeviceStage::TargetMoeN640DeviceStage(
    int device, int rank, int layer, cudaEvent_t slab_ready,
    const std::array<TargetMoeN640DeviceExpert,
                     kTargetMoeLocalExperts>& experts)
    : device_(device), rank_(rank), layer_(layer), slab_ready_(slab_ready) {
  if (device < 0 || (rank != 0 && rank != 1) || layer < 0 || layer >= 48 ||
      !slab_ready)
    throw std::invalid_argument("target MoE device stage identity changed");
  for (const auto& expert : experts)
    if (!valid_expert(expert))
      throw std::invalid_argument("target MoE device source extent changed");
  if (cudaSetDevice(device) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&device_experts_),
                 sizeof(experts)) != cudaSuccess ||
      cudaMemcpy(device_experts_, experts.data(), sizeof(experts),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    if (device_experts_) cudaFree(device_experts_);
    device_experts_ = nullptr;
    throw std::runtime_error("target MoE device source table failed");
  }
}

TargetMoeN640DeviceStage::~TargetMoeN640DeviceStage() {
  if (device_ >= 0) cudaSetDevice(device_);
  if (device_experts_) cudaFree(device_experts_);
}

void TargetMoeN640DeviceStage::wait_source(cudaStream_t stream) {
  if (source_wait_enqueued_ || !stream || !slab_ready_ ||
      cudaStreamWaitEvent(stream, slab_ready_, 0) != cudaSuccess)
    throw std::runtime_error("target MoE device slab wait failed");
  source_stream_ = stream;
  source_wait_enqueued_ = true;
}

TargetMoeOutcome TargetMoeN640DeviceStage::enqueue(
    const TargetMoeN640StageLaunch& launch) const noexcept {
  if (!device_experts_ || !source_wait_enqueued_ ||
      launch.stream != source_stream_ || !launch.local_expert_ids ||
      !launch.local_routing_weights || !launch.source_generation ||
      !launch.requested_generation || !valid_scratch(launch.scratch) ||
      !launch.stream)
    return TargetMoeOutcome::kContractError;
  select_experts<<<1, 1, 0, launch.stream>>>(
      device_experts_, launch.local_expert_ids, launch.local_routing_weights,
      launch.source_generation, launch.requested_generation, launch.scratch);
  constexpr int threads = 256;
  constexpr std::size_t packed_work =
      kTargetMoeStagedW13PackedBytes > kTargetMoeStagedDownPackedBytes
          ? kTargetMoeStagedW13PackedBytes
          : kTargetMoeStagedDownPackedBytes;
  stage_packed<<<(packed_work + threads - 1) / threads, threads, 0,
                 launch.stream>>>(device_experts_, launch.scratch);
  constexpr std::size_t scale_work =
      kTargetMoeStagedW13ScaleBytes > kTargetMoeStagedDownScaleBytes
          ? kTargetMoeStagedW13ScaleBytes
          : kTargetMoeStagedDownScaleBytes;
  clear_scales<<<(scale_work + threads - 1) / threads, threads, 0,
                 launch.stream>>>(launch.scratch);
  constexpr std::size_t logical_scale_work =
      kTargetMoeStagedExperts * 2ULL * kTargetMoeLogicalIntermediate *
      (kTargetMoeHidden / 16);
  stage_scales_and_scalars<<<
      (logical_scale_work + threads - 1) / threads, threads, 0,
      launch.stream>>>(device_experts_, launch.scratch);
  return cudaPeekAtLastError() == cudaSuccess ? TargetMoeOutcome::kOk
                                              : TargetMoeOutcome::kCudaError;
}

TargetMoeB12xWeights target_moe_staged_weights(
    const TargetMoeN768StageScratch& s) noexcept {
  return {s.w13_packed, s.w13_scale, s.down_packed, s.down_scale,
          s.input_global_scale, s.folded_w1_alpha, s.w2_alpha,
          s.down_input_scale};
}

bool validate_target_moe_stage_scratch(
    const TargetMoeN768StageScratch& scratch) noexcept {
  return valid_scratch(scratch);
}

TargetMoeN640StageReference target_moe_n640_stage_reference(
    const std::array<std::int32_t, kTargetMoeC1TopK>& ids,
    const std::array<float, kTargetMoeC1TopK>& weights,
    std::uint64_t source_generation,
    std::uint64_t requested_generation) noexcept {
  TargetMoeN640StageReference result{};
  result.evidence = build_route_map(
      ids.data(), weights.data(), source_generation, requested_generation,
      result.source_expert_ids.data(), result.compact_expert_ids.data(),
      result.compact_routing_weights.data());
  return result;
}

void export_target_moe_stage_otel_after_fence(
    const TargetMoeN640StageEvidence& evidence,
    std::uint64_t requested_generation, int rank, int layer,
    TargetMoeStageOtelSink& sink) noexcept {
  TargetMoeOutcome outcome = evidence.outcome;
  if ((rank != 0 && rank != 1) || layer < 0 || layer >= 48 ||
      requested_generation == 0 || evidence.generation != requested_generation ||
      evidence.active_experts < 0 ||
      evidence.active_experts > kTargetMoeStagedExperts)
    outcome = TargetMoeOutcome::kContractError;
  const std::uint64_t active =
      evidence.active_experts >= 0 &&
              evidence.active_experts <= kTargetMoeStagedExperts
          ? static_cast<std::uint64_t>(evidence.active_experts)
          : 0;
  constexpr std::uint64_t logical_bytes_per_expert =
      2ULL * kTargetMoeLogicalIntermediate * kTargetMoeHidden / 2 +
      2ULL * kTargetMoeLogicalIntermediate * (kTargetMoeHidden / 16) +
      1ULL * kTargetMoeHidden * kTargetMoeLogicalIntermediate / 2 +
      1ULL * kTargetMoeHidden * (kTargetMoeLogicalIntermediate / 16) +
      4ULL * sizeof(float);
  sink.add_counter({TargetMoeStageCounter::kLaunch, outcome, rank, layer, 1});
  sink.add_counter(
      {TargetMoeStageCounter::kActiveExperts, outcome, rank, layer, active});
  sink.add_counter({TargetMoeStageCounter::kSourceBytes, outcome, rank, layer,
                    active * logical_bytes_per_expert});
  sink.add_counter({TargetMoeStageCounter::kScratchBytes, outcome, rank, layer,
                    kTargetMoeStageScratchBytes});
}

}  // namespace rocket::qwen38::moe

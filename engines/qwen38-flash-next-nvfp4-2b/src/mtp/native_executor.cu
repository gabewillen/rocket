// SPDX-License-Identifier: Apache-2.0
#include "mtp/native_executor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

namespace rocket::qwen38::mtp {
namespace {

constexpr int kPhaseCount = static_cast<int>(Phase::kCount);

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw NativeExecutorError(std::string(operation) + ": " +
                              cudaGetErrorString(status));
}

bool valid_digest(const std::array<std::uint8_t, 32>& digest) noexcept {
  return std::any_of(digest.begin(), digest.end(),
                     [](std::uint8_t byte) { return byte != 0; });
}

__global__ void aggregate_experts(const std::int32_t* ids, int count,
                                  std::uint32_t* mask) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int expert = ids[index];
  if (expert >= 0 && expert < kLocalExperts)
    atomicOr(mask + expert / 32, std::uint32_t{1} << (expert % 32));
}

int popcount(const std::array<std::uint32_t, 8>& words) noexcept {
  int count = 0;
  for (std::uint32_t word : words) count += __builtin_popcount(word);
  return count;
}

}  // namespace

NativeExecutor::NativeExecutor(ImmutableSlabs slabs, BoundGraph graph,
                               StateArena& state,
                               TelemetrySink& telemetry, cudaStream_t stream)
    : slabs_(slabs), graph_(graph), state_(state), telemetry_(telemetry), stream_(stream) {
  if (!stream_ || !allowed_graph_key(graph_.key) ||
      !slabs_.target_rank_slab || slabs_.target_rank_slab_bytes == 0 ||
      !slabs_.mtp_rank_slab ||
      slabs_.mtp_rank_slab_bytes != kNativeRankSlabBytes ||
      !valid_digest(slabs_.source_contract_digest) ||
      !graph_.verification_tokens || state_.depth() != graph_.key.depth ||
      state_.sequences() != graph_.key.sequences)
    throw NativeExecutorError("native MTP binding contract changed");
  for (int step = 0; step < graph_.key.depth; ++step) {
    if (!graph_.router_expert_ids[step])
      throw NativeExecutorError("native MTP step buffers are incomplete");
    for (int phase = 0; phase < kPhaseCount; ++phase)
      if (!graph_.phase_graphs[step][phase])
        throw NativeExecutorError("native MTP phase graph is missing");
  }
  try {
    for (auto& step_events : events_)
      for (cudaEvent_t& event : step_events)
        check(cudaEventCreate(&event), "cudaEventCreate");
    check(cudaMalloc(&expert_masks_device_,
                     kMaxDepth * 8 * sizeof(std::uint32_t)),
          "cudaMalloc expert masks");
  } catch (...) {
    for (const auto& step_events : events_)
      for (cudaEvent_t event : step_events)
        if (event) cudaEventDestroy(event);
    if (expert_masks_device_) cudaFree(expert_masks_device_);
    throw;
  }
}

NativeExecutor::~NativeExecutor() {
  for (const auto& step_events : events_)
    for (cudaEvent_t event : step_events)
      if (event) cudaEventDestroy(event);
  if (expert_masks_device_) cudaFree(expert_masks_device_);
}

DeviceDraftView NativeExecutor::draft(std::uint64_t generation) {
  if (phase_ != ExecutorPhase::kReady || generation != active_generation_ + 1) {
    telemetry_.record_phase({Phase::kInputFusion, Outcome::kContractError,
                             graph_.key.depth, graph_.key.sequences, 0});
    throw NativeExecutorError("native MTP generation or phase changed");
  }
  try {
    check(cudaMemsetAsync(expert_masks_device_, 0,
                          graph_.key.depth * 8 * sizeof(std::uint32_t), stream_),
          "clear expert masks");
    for (int step = 0; step < graph_.key.depth; ++step) {
      check(cudaEventRecord(events_[step][0], stream_), "record phase begin");
      for (int phase = 0; phase < kPhaseCount; ++phase) {
        check(cudaGraphLaunch(graph_.phase_graphs[step][phase], stream_),
              "launch MTP phase graph");
        if (phase == static_cast<int>(Phase::kRoutedAndSharedMoe)) {
          const int routes = graph_.key.sequences * kRouterTopK;
          aggregate_experts<<<(routes + 127) / 128, 128, 0, stream_>>>(
              graph_.router_expert_ids[step], routes,
              expert_masks_device_ + step * 8);
          check(cudaGetLastError(), "aggregate actual router experts");
        }
        check(cudaEventRecord(events_[step][phase + 1], stream_),
              "record phase end");
      }
    }
    pending_generation_ = generation;
    phase_ = ExecutorPhase::kDrafted;
    return {graph_.verification_tokens, graph_.key.depth,
            graph_.key.sequences, generation};
  } catch (...) {
    phase_ = ExecutorPhase::kFaulted;
    telemetry_.record_phase({Phase::kInputFusion, Outcome::kCudaError,
                             graph_.key.depth, graph_.key.sequences, 0});
    throw;
  }
}

void NativeExecutor::stage_accept(
    std::uint64_t generation, std::byte* inactive_state,
    const std::int32_t* accepted_widths_device,
    decode::DecoderVerifierShape shape, cudaStream_t stream) {
  if (phase_ != ExecutorPhase::kDrafted || generation != pending_generation_ ||
      !inactive_state || !accepted_widths_device || stream != stream_ ||
      shape.sequences != graph_.key.sequences ||
      shape.verify_width != graph_.key.depth + 1)
    throw NativeExecutorError("accepted-prefix publication contract changed");
  try {
    state_.select(accepted_widths_device, generation, stream_, inactive_state);
  } catch (...) {
    phase_ = ExecutorPhase::kFaulted;
    throw;
  }
}

void NativeExecutor::commit(std::uint64_t generation) noexcept {
  if (phase_ == ExecutorPhase::kDrafted && generation == pending_generation_) {
    active_generation_ = generation;
    state_.commit(generation);
    pending_generation_ = 0;
    phase_ = ExecutorPhase::kReady;
  } else {
    phase_ = ExecutorPhase::kFaulted;
  }
}

void NativeExecutor::export_telemetry_after_fence(
    std::uint64_t generation) noexcept {
  if (phase_ != ExecutorPhase::kReady || generation != active_generation_) return;
  if (cudaMemcpy(expert_masks_host_.data(), expert_masks_device_,
                   graph_.key.depth * 8 * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
    telemetry_.record_phase({Phase::kInputFusion, Outcome::kCudaError,
                             graph_.key.depth, graph_.key.sequences, 0});
    return;
  }
  std::array<float, kPhaseCount> phase_ms{};
  for (int step = 0; step < graph_.key.depth; ++step)
    for (int phase = 0; phase < kPhaseCount; ++phase) {
      float elapsed = 0.0F;
      if (cudaEventElapsedTime(&elapsed, events_[step][phase],
                               events_[step][phase + 1]) != cudaSuccess)
        return;
      phase_ms[phase] += elapsed;
    }
  for (int phase = 0; phase < kPhaseCount; ++phase)
    telemetry_.record_phase(
        {static_cast<Phase>(phase), Outcome::kOk, graph_.key.depth,
         graph_.key.sequences,
         static_cast<std::uint64_t>(phase_ms[phase] * 1'000'000.0F)});
  for (int step = 0; step < graph_.key.depth; ++step) {
    const int unique = popcount(expert_masks_host_[step]);
    telemetry_.record_expert_usage(
        {Outcome::kOk, graph_.key.depth, graph_.key.sequences, step, unique,
         static_cast<std::uint64_t>(unique) *
             kNvidiaFp8BytesPerLocalExpert});
  }
}

void NativeExecutor::discard(std::uint64_t generation) noexcept {
  if (phase_ == ExecutorPhase::kDrafted && generation == pending_generation_) {
    state_.discard(generation);
    pending_generation_ = 0;
    phase_ = ExecutorPhase::kReady;
  }
}

}  // namespace rocket::qwen38::mtp

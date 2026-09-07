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

__global__ void publish_accepted_snapshots(
    std::byte* active, const std::byte* const* snapshots,
    const std::int32_t* accepted_widths, int sequences, int depth,
    std::size_t state_bytes) {
  const std::size_t total = static_cast<std::size_t>(sequences) * state_bytes;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total; index += blockDim.x * gridDim.x) {
    const int sequence = static_cast<int>(index / state_bytes);
    const std::size_t offset = index % state_bytes;
    const int step = min(accepted_widths[sequence], depth) - 1;
    active[index] = snapshots[step][static_cast<std::size_t>(sequence) *
                                      state_bytes + offset];
  }
}

int popcount(const std::array<std::uint32_t, 8>& words) noexcept {
  int count = 0;
  for (std::uint32_t word : words) count += __builtin_popcount(word);
  return count;
}

}  // namespace

NativeExecutor::NativeExecutor(ImmutableSlabs slabs, BoundGraph graph,
                               TelemetrySink& telemetry, cudaStream_t stream)
    : slabs_(slabs), graph_(graph), telemetry_(telemetry), stream_(stream) {
  if (!stream_ || !allowed_graph_key(graph_.key) ||
      !slabs_.target_rank_slab || slabs_.target_rank_slab_bytes == 0 ||
      !slabs_.mtp_rank_slab ||
      slabs_.mtp_rank_slab_bytes != kNativeRankSlabBytes ||
      !valid_digest(slabs_.source_contract_digest) || !graph_.proposal_tokens ||
      !graph_.active_causal_state || graph_.state_bytes_per_sequence == 0)
    throw NativeExecutorError("native MTP binding contract changed");
  for (int step = 0; step < graph_.key.depth; ++step) {
    if (!graph_.router_expert_ids[step] || !graph_.causal_snapshots[step])
      throw NativeExecutorError("native MTP step buffers are incomplete");
    for (int phase = 0; phase < kPhaseCount; ++phase)
      if (!graph_.phase_graphs[step][phase])
        throw NativeExecutorError("native MTP phase graph is missing");
  }
  try {
    for (cudaEvent_t& event : events_)
      check(cudaEventCreate(&event), "cudaEventCreate");
    check(cudaMalloc(&expert_masks_device_,
                     kMaxDepth * 8 * sizeof(std::uint32_t)),
          "cudaMalloc expert masks");
    check(cudaMalloc(&accepted_widths_device_,
                     kMaxSequences * sizeof(std::int32_t)),
          "cudaMalloc accepted widths");
    check(cudaMalloc(&snapshots_device_, kMaxDepth * sizeof(std::byte*)),
          "cudaMalloc snapshot table");
    check(cudaMemcpy(snapshots_device_, graph_.causal_snapshots.data(),
                     graph_.key.depth * sizeof(std::byte*),
                     cudaMemcpyHostToDevice),
          "bind immutable snapshot table");
  } catch (...) {
    for (cudaEvent_t event : events_)
      if (event) cudaEventDestroy(event);
    if (expert_masks_device_) cudaFree(expert_masks_device_);
    if (accepted_widths_device_) cudaFree(accepted_widths_device_);
    if (snapshots_device_) cudaFree(snapshots_device_);
    throw;
  }
}

NativeExecutor::~NativeExecutor() {
  for (cudaEvent_t event : events_)
    if (event) cudaEventDestroy(event);
  if (expert_masks_device_) cudaFree(expert_masks_device_);
  if (accepted_widths_device_) cudaFree(accepted_widths_device_);
  if (snapshots_device_) cudaFree(snapshots_device_);
}

DraftResult NativeExecutor::draft(std::uint64_t generation) {
  if (phase_ != ExecutorPhase::kReady || generation != active_generation_ + 1) {
    telemetry_.record_phase({Phase::kInputFusion, Outcome::kContractError,
                             graph_.key.depth, graph_.key.sequences, 0});
    throw NativeExecutorError("native MTP generation or phase changed");
  }
  try {
    check(cudaMemsetAsync(expert_masks_device_, 0,
                          graph_.key.depth * 8 * sizeof(std::uint32_t), stream_),
          "clear expert masks");
    std::array<float, kPhaseCount> phase_ms{};
    for (int step = 0; step < graph_.key.depth; ++step) {
      check(cudaEventRecord(events_[0], stream_), "record phase begin");
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
        check(cudaEventRecord(events_[phase + 1], stream_),
              "record phase end");
      }
      check(cudaEventSynchronize(events_[kPhaseCount]),
            "synchronize MTP draft step");
      for (int phase = 0; phase < kPhaseCount; ++phase) {
        float elapsed = 0.0F;
        check(cudaEventElapsedTime(&elapsed, events_[phase], events_[phase + 1]),
              "read MTP phase time");
        phase_ms[phase] += elapsed;
      }
    }
    check(cudaMemcpy(expert_masks_host_.data(), expert_masks_device_,
                     graph_.key.depth * 8 * sizeof(std::uint32_t),
                     cudaMemcpyDeviceToHost),
          "copy expert usage");
    DraftResult result;
    result.depth = graph_.key.depth;
    result.sequences = graph_.key.sequences;
    result.generation = generation;
    std::array<std::int32_t, kMaxSequences * kMaxDepth> flat{};
    check(cudaMemcpy(flat.data(), graph_.proposal_tokens,
                     graph_.key.sequences * graph_.key.depth *
                         sizeof(std::int32_t),
                     cudaMemcpyDeviceToHost),
          "copy proposal tokens");
    for (int sequence = 0; sequence < graph_.key.sequences; ++sequence)
      for (int step = 0; step < graph_.key.depth; ++step) {
        const std::int32_t token = flat[sequence * graph_.key.depth + step];
        if (token < 0 || token >= 248'320)
          throw NativeExecutorError("proposal token is outside Qwen vocabulary");
        result.tokens[sequence][step] = token;
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
    pending_generation_ = generation;
    phase_ = ExecutorPhase::kDrafted;
    return result;
  } catch (...) {
    phase_ = ExecutorPhase::kFaulted;
    telemetry_.record_phase({Phase::kInputFusion, Outcome::kCudaError,
                             graph_.key.depth, graph_.key.sequences, 0});
    throw;
  }
}

void NativeExecutor::publish(std::uint64_t generation,
                             const std::int32_t* accepted_widths_host) {
  if (phase_ != ExecutorPhase::kDrafted || generation != pending_generation_ ||
      !accepted_widths_host)
    throw NativeExecutorError("accepted-prefix publication contract changed");
  for (int sequence = 0; sequence < graph_.key.sequences; ++sequence) {
    const int width = accepted_widths_host[sequence];
    if (width < 1 || width > graph_.key.depth + 1)
      throw NativeExecutorError("accepted width is outside verifier window");
    accepted_widths_[sequence] = width;
  }
  try {
    check(cudaMemcpyAsync(accepted_widths_device_, accepted_widths_.data(),
                          graph_.key.sequences * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice, stream_),
          "copy accepted widths");
    const std::size_t bytes = static_cast<std::size_t>(graph_.key.sequences) *
                              graph_.state_bytes_per_sequence;
    const int blocks = static_cast<int>(std::min<std::size_t>(
        4096, (bytes + 255) / 256));
    publish_accepted_snapshots<<<blocks, 256, 0, stream_>>>(
        graph_.active_causal_state, snapshots_device_, accepted_widths_device_,
        graph_.key.sequences, graph_.key.depth,
        graph_.state_bytes_per_sequence);
    check(cudaGetLastError(), "publish accepted MTP snapshot");
    check(cudaStreamSynchronize(stream_), "synchronize MTP publication");
    active_generation_ = generation;
    pending_generation_ = 0;
    phase_ = ExecutorPhase::kReady;
  } catch (...) {
    phase_ = ExecutorPhase::kFaulted;
    throw;
  }
}

void NativeExecutor::discard(std::uint64_t generation) noexcept {
  if (phase_ == ExecutorPhase::kDrafted && generation == pending_generation_) {
    pending_generation_ = 0;
    phase_ = ExecutorPhase::kReady;
  }
}

}  // namespace rocket::qwen38::mtp

// SPDX-License-Identifier: Apache-2.0
#include "mtp/native_executor.h"
#include <cuda_runtime.h>
#include <string>

namespace rocket::qwen38::mtp {
namespace {
constexpr int kPhaseCount = static_cast<int>(Phase::kCount);
void check(cudaError_t status, const char* op) {
  if (status != cudaSuccess)
    throw NativeExecutorError(std::string(op) + ": " + cudaGetErrorString(status));
}
__global__ void aggregate_experts(const std::int32_t* ids, int count,
                                  std::uint32_t* mask) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count && ids[i] >= 0 && ids[i] < kLocalExperts)
    atomicOr(mask + ids[i] / 32, std::uint32_t{1} << (ids[i] % 32));
}
int popcount(const std::array<std::uint32_t, 8>& words) noexcept {
  int count = 0;
  for (const auto word : words) count += __builtin_popcount(word);
  return count;
}
}  // namespace

NativeExecutor::NativeExecutor(GraphKey key, MtpGraphRuntime& runtime,
                               MtpMiddleStagePort& middle,
                               WinnerExchangePort& exchange, StateArena& state,
                               TelemetrySink& telemetry, cudaStream_t stream)
    : key_(key), runtime_(runtime), middle_(middle), exchange_(exchange),
      state_(state), telemetry_(telemetry), stream_(stream) {
  if (!stream || !allowed_graph_key(key) || state.depth() != key.depth ||
      state.sequences() != key.sequences)
    throw NativeExecutorError("native MTP binding contract changed");
  try {
    for (auto& row : events_)
      for (auto& event : row) check(cudaEventCreate(&event), "create event");
    check(cudaMalloc(&expert_masks_device_, 7 * 8 * sizeof(std::uint32_t)),
          "allocate expert masks");
  } catch (...) {
    for (const auto& row : events_)
      for (const auto event : row) if (event) cudaEventDestroy(event);
    cudaFree(expert_masks_device_);
    throw;
  }
}
NativeExecutor::~NativeExecutor() {
  for (const auto& row : events_)
    for (const auto event : row) if (event) cudaEventDestroy(event);
  cudaFree(expert_masks_device_);
}

DeviceDraftView NativeExecutor::draft(std::uint64_t generation) {
  if (phase_ != ExecutorPhase::kReady || generation != active_generation_ + 1) {
    telemetry_.record_phase({Phase::kInputFusion, Outcome::kContractError,
                             key_.depth, key_.sequences, 0});
    throw NativeExecutorError("native MTP generation or phase changed");
  }
  try {
    check(cudaMemsetAsync(expert_masks_device_, 0,
                          key_.depth * 8 * sizeof(std::uint32_t), stream_),
          "clear expert masks");
    const auto arena = runtime_.arena();
    verification_tokens_ = middle_.prepare(arena, state_, key_, stream_);
    if (!verification_tokens_)
      throw NativeExecutorError("MTP verification tokens are unavailable");
    for (int step = 0; step < key_.depth; ++step) {
      check(cudaEventRecord(events_[step][0], stream_), "record input begin");
      runtime_.launch_input_local(key_.sequences, stream_);
      middle_.reduce_input(arena, step, key_, stream_);
      runtime_.launch_input_finish(key_.sequences, stream_);
      check(cudaEventRecord(events_[step][1], stream_), "record input end");
      middle_.stage_attention(arena, state_.prefix(step), step, key_, stream_);
      check(cudaEventRecord(events_[step][2], stream_), "record attention end");
      middle_.reduce_attention(arena, step, key_, stream_);
      check(cudaEventRecord(events_[step][3], stream_), "record attention reduce end");
      middle_.stage_moe(arena, step, key_, stream_);
      const auto* routes = middle_.router_expert_ids(step);
      if (!routes) throw NativeExecutorError("MTP router output is unavailable");
      const int route_count = key_.sequences * kRouterTopK;
      aggregate_experts<<<(route_count + 127) / 128, 128, 0, stream_>>>(
          routes, route_count, expert_masks_device_ + step * 8);
      check(cudaPeekAtLastError(), "aggregate router experts");
      check(cudaEventRecord(events_[step][4], stream_), "record MoE end");
      middle_.reduce_moe(arena, step, key_, stream_);
      check(cudaEventRecord(events_[step][5], stream_), "record MoE reduce end");
      runtime_.launch_final_local(key_.sequences, stream_);
      check(cudaEventRecord(events_[step][6], stream_), "record final HC end");
      runtime_.launch_logits_local(key_.sequences, stream_);
      check(cudaEventRecord(events_[step][7], stream_), "record logits end");
      const auto* proposals = runtime_.enqueue_winner_exchange_and_greedy(
          exchange_, key_.sequences, stream_);
      middle_.advance(arena, state_, step, key_, proposals, stream_);
      check(cudaEventRecord(events_[step][8], stream_), "record proposal end");
    }
    pending_generation_ = generation;
    phase_ = ExecutorPhase::kDrafted;
    return {verification_tokens_, key_.depth, key_.sequences, generation};
  } catch (...) {
    phase_ = ExecutorPhase::kFaulted;
    telemetry_.record_phase({Phase::kInputFusion, Outcome::kCudaError,
                             key_.depth, key_.sequences, 0});
    throw;
  }
}

void NativeExecutor::stage_accept(
    std::uint64_t generation, std::byte* inactive,
    const std::int32_t* widths, decode::DecoderVerifierShape shape,
    cudaStream_t stream) {
  if (phase_ != ExecutorPhase::kDrafted || generation != pending_generation_ ||
      !inactive || !widths || stream != stream_ ||
      shape.sequences != key_.sequences || shape.verify_width != key_.depth + 1)
    throw NativeExecutorError("accepted-prefix publication contract changed");
  try { state_.select(widths, generation, stream_, inactive); }
  catch (...) { phase_ = ExecutorPhase::kFaulted; throw; }
}
void NativeExecutor::commit(std::uint64_t generation) noexcept {
  if (phase_ == ExecutorPhase::kDrafted && generation == pending_generation_) {
    active_generation_ = generation;
    state_.commit(generation);
    pending_generation_ = 0;
    phase_ = ExecutorPhase::kReady;
  } else phase_ = ExecutorPhase::kFaulted;
}
void NativeExecutor::export_telemetry_after_fence(
    std::uint64_t generation) noexcept {
  if (phase_ != ExecutorPhase::kReady || generation != active_generation_) return;
  if (cudaMemcpy(expert_masks_host_.data(), expert_masks_device_,
                 key_.depth * 8 * sizeof(std::uint32_t),
                 cudaMemcpyDeviceToHost) != cudaSuccess) return;
  std::array<float, kPhaseCount> elapsed_ms{};
  for (int step = 0; step < key_.depth; ++step)
    for (int phase = 0; phase < kPhaseCount; ++phase)
    {
      float step_ms = 0.0F;
      if (cudaEventElapsedTime(&step_ms, events_[step][phase],
                               events_[step][phase + 1]) != cudaSuccess) return;
      elapsed_ms[phase] += step_ms;
    }
  for (int phase = 0; phase < kPhaseCount; ++phase)
    telemetry_.record_phase({static_cast<Phase>(phase), Outcome::kOk,
      key_.depth, key_.sequences,
      static_cast<std::uint64_t>(elapsed_ms[phase] * 1'000'000.0F)});
  for (int step = 0; step < key_.depth; ++step) {
    const int unique = popcount(expert_masks_host_[step]);
    telemetry_.record_expert_usage({Outcome::kOk, key_.depth, key_.sequences,
      step, unique, static_cast<std::uint64_t>(unique) *
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

// SPDX-License-Identifier: Apache-2.0
#include "moe/native_target_moe_graph.h"

#include <stdexcept>

namespace rocket::qwen38::moe {
namespace {
class RuntimeCudaApi final : public NativeTargetMoeCudaApi {
 public:
  cudaError_t stream_synchronize(cudaStream_t stream) noexcept override {
    return cudaStreamSynchronize(stream);
  }
  cudaError_t device_synchronize() noexcept override {
    return cudaDeviceSynchronize();
  }
};
RuntimeCudaApi& runtime_cuda_api() {
  static RuntimeCudaApi api;
  return api;
}
}  // namespace

NativeTargetMoeGraph::NativeTargetMoeGraph(
    TargetFullMoeC1Port& participant, const TargetFullMoeC1Workspace& workspace,
    __nv_bfloat16* rank_local_partial_bf16, TargetFullMoeOtelSink& telemetry,
    NativeTargetMoeCudaApi* cuda_api)
    : participant_(participant),
      workspace_(workspace),
      rank_local_partial_bf16_(rank_local_partial_bf16),
      telemetry_(telemetry),
      cuda_api_(cuda_api ? cuda_api : &runtime_cuda_api()),
      rank_(participant.identity().rank),
      layer_(participant.identity().layer) {
  if ((rank_ != 0 && rank_ != 1) || layer_ != kTargetCompositionLayer ||
      !rank_local_partial_bf16_)
    throw std::invalid_argument("native target MoE graph identity changed");
}

NativeTargetMoeGraph::~NativeTargetMoeGraph() {
  (void)drain_faulted_generation();
}

bool NativeTargetMoeGraph::drain_faulted_generation() noexcept {
  if (state_ == State::kFaultedQuarantined) return false;
  const bool possibly_enqueued =
      state_ == State::kPossiblyEnqueued ||
      state_ == State::kFaultedPossiblyEnqueued;
  const bool fenced =
      state_ == State::kFenced || state_ == State::kFaultedFenced;
  if (!possibly_enqueued && !fenced) return true;
  if (possibly_enqueued && source_stream_ &&
      cuda_api_->stream_synchronize(source_stream_) != cudaSuccess &&
      cuda_api_->device_synchronize() != cudaSuccess) {
    telemetry_.emit({TargetFullMoeComponent::kStaging,
                     TargetDenseOutcome::kCudaError, rank_, layer_});
    state_ = State::kFaultedQuarantined;
    return false;
  }
  (void)participant_.publish_after_fence(enqueued_generation_, telemetry_);
  state_ = State::kFaultedNoFlight;
  return true;
}

bool NativeTargetMoeGraph::drain_for_destruction() noexcept {
  return drain_faulted_generation();
}

void NativeTargetMoeGraph::mark_faulted() noexcept {
  if (state_ == State::kFaultedQuarantined) return;
  if (state_ == State::kPossiblyEnqueued ||
      state_ == State::kFaultedPossiblyEnqueued)
    state_ = State::kFaultedPossiblyEnqueued;
  else if (state_ == State::kFenced || state_ == State::kFaultedFenced)
    state_ = State::kFaultedFenced;
  else
    state_ = State::kFaultedNoFlight;
}

void NativeTargetMoeGraph::wait_source(cudaStream_t stream) {
  if (state_ != State::kReady || source_stream_ || !stream) {
    mark_faulted();
    throw std::invalid_argument("native target MoE source wait changed");
  }
  try {
    participant_.wait_source(stream);
    source_stream_ = stream;
    state_ = State::kSourceWaited;
  } catch (...) {
    mark_faulted();
    throw;
  }
}

void NativeTargetMoeGraph::launch(
    const __nv_bfloat16* block_input, std::uint64_t generation, int m,
    cudaStream_t stream) {
  if ((state_ != State::kSourceWaited && state_ != State::kPublished) ||
      enqueued_generation_ != published_generation_ ||
      !block_input || !stream || stream != source_stream_ || m != 1 ||
      generation == 0 ||
      generation != published_generation_ + 1) {
    mark_faulted();
    throw std::invalid_argument("native target MoE graph launch changed");
  }
  const TargetFullMoeC1Launch launch{
      block_input, rank_local_partial_bf16_, workspace_, stream};
  enqueued_generation_ = generation;
  state_ = State::kPossiblyEnqueued;
  const auto outcome = participant_.enqueue_with_telemetry(launch, telemetry_);
  if (outcome != TargetDenseOutcome::kOk) {
    mark_faulted();
    if (outcome == TargetDenseOutcome::kContractError)
      throw decode::DecodeExecutionContractError(
          "native target MoE graph enqueue failed");
    throw decode::DecodeExecutionCudaError(
        "native target MoE graph enqueue failed");
  }
}

void NativeTargetMoeGraph::publish_after_fence(std::uint64_t generation) {
  if (state_ != State::kFenced || generation == 0 ||
      generation != enqueued_generation_ ||
      published_generation_ + 1 != generation) {
    mark_faulted();
    throw std::invalid_argument("native target MoE fence generation changed");
  }
  const auto outcome = participant_.publish_after_fence(generation, telemetry_);
  if (outcome != TargetDenseOutcome::kOk) {
    state_ = State::kFaultedNoFlight;
    if (outcome == TargetDenseOutcome::kContractError)
      throw decode::DecodeExecutionContractError(
          "native target MoE stage publication failed");
    throw decode::DecodeExecutionCudaError(
        "native target MoE stage publication failed");
  }
  published_generation_ = generation;
  state_ = State::kPublished;
}

void NativeTargetMoeGraph::terminal_fence_succeeded(
    std::uint64_t generation) {
  if (state_ != State::kPossiblyEnqueued || generation == 0 ||
      generation != enqueued_generation_) {
    mark_faulted();
    throw std::invalid_argument("native target MoE terminal fence changed");
  }
  state_ = State::kFenced;
}

void NativeTargetMoeGraph::fault_after_fence(
    std::uint64_t generation) noexcept {
  const bool cleanup_bearing =
      state_ == State::kPossiblyEnqueued || state_ == State::kFenced ||
      state_ == State::kFaultedPossiblyEnqueued ||
      state_ == State::kFaultedFenced;
  if (cleanup_bearing && generation != 0 &&
      generation == enqueued_generation_)
    (void)drain_faulted_generation();
  else
    mark_faulted();
}

TargetMoeComponentIdentity NativeTargetMoeGraph::component_identity(
    TargetFullMoeComponent component) const noexcept {
  const auto dtype = component == TargetFullMoeComponent::kSharedExpert
                         ? TargetMoeServingDtype::kBfloat16
                         : TargetMoeServingDtype::kNvfp4;
  return {component, participant_.identity(), dtype};
}

}  // namespace rocket::qwen38::moe

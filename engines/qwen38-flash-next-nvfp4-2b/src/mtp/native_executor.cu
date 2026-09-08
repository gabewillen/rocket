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
void require_route_enqueue(moe::RouteCompactionOutcome outcome) {
  if (outcome == moe::RouteCompactionOutcome::kOk) return;
  throw NativeExecutorError("MTP route compaction enqueue contract changed");
}
void require_expert_enqueue(moe::RoutedExpertOutcome outcome) {
  if (outcome == moe::RoutedExpertOutcome::kOk) return;
  throw NativeExecutorError("MTP routed expert enqueue contract changed");
}
__global__ void clear_routed_expert_publication(
    moe::RoutedExpertDeviceSummary* summary) {
  if (threadIdx.x == 0) {
    *summary = {
        .generation = 0,
        .active_weight_bytes = 0,
        .fc1_tiles = 0,
        .fc2_tiles = 0,
        .active_experts = 0,
        .active_routes = 0,
        .outcome = moe::RoutedExpertOutcome::kContractError,
    };
  }
}
}  // namespace

void NativeExecutor::CudaDeleter::operator()(void* pointer) const noexcept {
  if (pointer) static_cast<void>(cudaFree(pointer));
}

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
    std::uint64_t* requested_generation = nullptr;
    check(cudaMalloc(&requested_generation, sizeof(std::uint64_t)),
          "allocate requested generation");
    requested_generation_device_.reset(requested_generation);
    moe::RouteCompactionDeviceSummary* route_summaries = nullptr;
    check(cudaMalloc(&route_summaries,
                     kMaxDepth * sizeof(moe::RouteCompactionDeviceSummary)),
          "allocate route summaries");
    route_summaries_device_.reset(route_summaries);
    moe::RoutedExpertDeviceSummary* expert_summaries = nullptr;
    check(cudaMalloc(&expert_summaries,
                     kMaxDepth * sizeof(moe::RoutedExpertDeviceSummary)),
          "allocate routed expert summaries");
    expert_summaries_device_.reset(expert_summaries);
  } catch (...) {
    for (const auto& row : events_)
      for (const auto event : row) if (event) cudaEventDestroy(event);
    throw;
  }
}
NativeExecutor::~NativeExecutor() {
  for (const auto& row : events_)
    for (const auto event : row) if (event) cudaEventDestroy(event);
}

DeviceDraftView NativeExecutor::draft(std::uint64_t generation) {
  if (phase_ != ExecutorPhase::kReady || generation != active_generation_ + 1) {
    telemetry_.record_phase({Phase::kInputFusion, Outcome::kContractError,
                             key_.depth, key_.sequences, 0});
    throw NativeExecutorError("native MTP generation or phase changed");
  }
  try {
    check(cudaMemcpyAsync(requested_generation_device_.get(), &generation,
                          sizeof(generation), cudaMemcpyHostToDevice, stream_),
          "stage requested generation");
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
      const auto qsa_state = attention::bind_mtp_qsa_write_view(
          state_.prefix(step),
          {key_.sequences, key_.depth, key_.query_tokens, step,
           state_.uses_mrope(), generation, active_generation_ + 1});
      middle_.stage_attention(arena, qsa_state, step, key_, stream_);
      check(cudaEventRecord(events_[step][2], stream_), "record attention end");
      middle_.reduce_attention(arena, step, key_, stream_);
      check(cudaEventRecord(events_[step][3], stream_), "record attention reduce end");
      auto router = middle_.stage_router(arena, step, key_, generation, stream_);
      router.compacted.summary = route_summaries_device_.get() + step;
      const moe::RouteCompactionShape route_shape{
          runtime_.rank(), key_.sequences, key_.sequences};
      require_route_enqueue(moe::enqueue_route_compaction({
          .shape = route_shape,
          .capacity = router.capacity,
          .input = {
              .global_expert_ids = router.global_expert_ids,
              .routing_weights = router.routing_weights,
              .source_generation = router.source_generation,
              .requested_generation = requested_generation_device_.get(),
          },
          .output = router.compacted,
          .stream = stream_,
      }));
      clear_routed_expert_publication<<<1, 1, 0, stream_>>>(
          expert_summaries_device_.get() + step);
      check(cudaPeekAtLastError(), "clear routed expert publication");
      require_expert_enqueue(middle_.stage_moe(
          arena, router.compacted, expert_summaries_device_.get() + step, step,
          key_, stream_));
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
  if (phase_ == ExecutorPhase::kDrafted && generation == pending_generation_ &&
      generation == routes_validated_generation_) {
    active_generation_ = generation;
    state_.commit(generation);
    pending_generation_ = 0;
    routes_validated_generation_ = 0;
    phase_ = ExecutorPhase::kReady;
  } else phase_ = ExecutorPhase::kFaulted;
}
void NativeExecutor::validate_after_fence(std::uint64_t generation) {
  if (phase_ != ExecutorPhase::kDrafted || generation != pending_generation_)
    throw NativeExecutorError("MTP post-fence generation changed");
  exchange_.validate_after_fence();
  check(cudaMemcpy(route_summaries_host_.data(), route_summaries_device_.get(),
                   key_.depth * sizeof(moe::RouteCompactionDeviceSummary),
                   cudaMemcpyDeviceToHost),
        "read route summaries after fence");
  check(cudaMemcpy(expert_summaries_host_.data(), expert_summaries_device_.get(),
                   key_.depth * sizeof(moe::RoutedExpertDeviceSummary),
                   cudaMemcpyDeviceToHost),
        "read routed expert summaries after fence");
  const moe::RouteCompactionShape shape{
      runtime_.rank(), key_.sequences, key_.sequences};
  for (int step = 0; step < key_.depth; ++step) {
    moe::export_route_compaction_otel_after_fence({
        .snapshot = route_summaries_host_[step],
        .requested_generation = generation,
        .shape = shape,
        .sink = telemetry_,
    });
    const auto outcome = moe::validate_route_compaction_summary(
        {.summary = route_summaries_host_[step],
         .requested_generation = generation,
         .shape = shape});
    if (outcome != moe::RouteCompactionOutcome::kOk)
      throw NativeExecutorError("MTP route compaction publication rejected");
    moe::export_routed_expert_otel_after_fence(
        expert_summaries_host_[step], generation, shape, telemetry_);
    if (moe::validate_routed_expert_summary(
            {.summary = expert_summaries_host_[step],
             .requested_generation = generation,
             .shape = shape}) != moe::RoutedExpertOutcome::kOk)
      throw NativeExecutorError("MTP routed expert publication rejected");
  }
  routes_validated_generation_ = generation;
}
void NativeExecutor::export_telemetry_after_fence(
    std::uint64_t generation) noexcept {
  if (phase_ != ExecutorPhase::kReady || generation != active_generation_) return;
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
    const auto& summary = route_summaries_host_[step];
    telemetry_.record_expert_usage({Outcome::kOk, key_.depth, key_.sequences,
      step, summary.active_experts, summary.active_weight_bytes});
  }
}
void NativeExecutor::discard(std::uint64_t generation) noexcept {
  if (phase_ == ExecutorPhase::kDrafted && generation == pending_generation_) {
    state_.discard(generation);
    pending_generation_ = 0;
    routes_validated_generation_ = 0;
    phase_ = ExecutorPhase::kReady;
  }
}
}  // namespace rocket::qwen38::mtp

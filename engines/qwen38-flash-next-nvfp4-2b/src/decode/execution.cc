#include "decode/execution.h"

#include <chrono>
#include <cstddef>
#include <limits>
#include <string>

namespace rocket::qwen38::decode {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point start) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

[[noreturn]] void contract_fail(const std::string& reason) {
  throw DecodeExecutionContractError("qwen38 decode TP2 contract: " + reason);
}

}  // namespace

Tp2DecodeExecution::Tp2DecodeExecution(HiddenPartialReducer& reducer,
                                       pair_reduce::OtelStageSink& telemetry,
                                       std::uint64_t initial_completed_generation)
    : reducer_(reducer), telemetry_(telemetry),
      last_completed_generation_(initial_completed_generation) {
  if (reducer_.world_size() != pair_reduce::kWorldSize ||
      (reducer_.rank() != 0 && reducer_.rank() != 1)) {
    emit("rocket.qwen38.decode.tp2.lifecycle",
         pair_reduce::Outcome::kContractError, 0, {}, {}, 0);
    contract_fail("topology must be exactly ranks 0 and 1");
  }
  if (initial_completed_generation == std::numeric_limits<std::uint64_t>::max()) {
    emit("rocket.qwen38.decode.tp2.lifecycle",
         pair_reduce::Outcome::kContractError, 0, {}, {}, 0);
    contract_fail("completed device generation cannot advance");
  }
}

ReductionPoint Tp2DecodeExecution::point_for_ordinal(int ordinal) {
  if (ordinal < 0 || ordinal >= kReductionPoints)
    contract_fail("reduction ordinal must be within 0 through 95");
  return {ordinal / kReductionsPerLayer,
          ordinal % kReductionsPerLayer == 0
              ? ReductionKind::kAttentionOutput
              : ReductionKind::kMoeOutput};
}

void Tp2DecodeExecution::begin_step(std::uint64_t device_generation, int m,
                                    std::string_view trace_id,
                                    std::string_view request_id) {
  const auto start = Clock::now();
  try {
    if (phase_ != StepPhase::kIdle) contract_fail("begin requires idle phase");
    if (device_generation == 0 ||
        device_generation != last_completed_generation_ + 1)
      contract_fail("device generation must increase by one");
    if (!pair_reduce::allowed_m(m))
      contract_fail("M must be a sequences*(K+1) verifier row bucket");
  } catch (const DecodeExecutionContractError&) {
    emit("rocket.qwen38.decode.tp2.lifecycle",
         pair_reduce::Outcome::kContractError, m, trace_id, request_id,
         elapsed_ns(start));
    throw;
  }
  active_generation_ = device_generation;
  m_ = m;
  next_ordinal_ = 0;
  phase_ = StepPhase::kActive;
  emit("rocket.qwen38.decode.tp2.lifecycle", pair_reduce::Outcome::kOk, m_,
       trace_id, request_id, elapsed_ns(start));
}

void Tp2DecodeExecution::reduce_at(
    std::uint64_t device_generation, ReductionPoint point,
    const __nv_bfloat16* local_partial, float* reduced_hidden,
    std::string_view trace_id, std::string_view request_id, cudaStream_t stream) {
  const auto start = Clock::now();
  const std::string_view stage = stage_for(point);
  try {
    if (phase_ != StepPhase::kActive) contract_fail("reduce requires active phase");
    if (device_generation != active_generation_)
      contract_fail("active device generation drift");
    if (!valid_point(point)) contract_fail("reduction point is outside the Qwen graph");
    const ReductionPoint expected = point_for_ordinal(next_ordinal_);
    if (point.layer != expected.layer || point.kind != expected.kind)
      contract_fail("reduction point is out of layer-major order");
    if (local_partial == nullptr || reduced_hidden == nullptr)
      contract_fail("borrowed input and output device buffers are required");
    if (stream == nullptr) contract_fail("one non-default transaction stream is required");
    if (active_stream_ != nullptr && active_stream_ != stream)
      contract_fail("all reductions must use the transaction stream");
  } catch (const DecodeExecutionContractError&) {
    emit(stage, pair_reduce::Outcome::kContractError, m_, trace_id, request_id,
         elapsed_ns(start));
    throw;
  }

  try {
    active_stream_ = stream;
    reducer_.reduce(local_partial, reduced_hidden, m_, trace_id, request_id, stream);
  } catch (const pair_reduce::PairReduceContractError& error) {
    phase_ = StepPhase::kFaulted;
    emit(stage, pair_reduce::Outcome::kContractError, m_, trace_id, request_id,
         elapsed_ns(start));
    throw DecodeExecutionTransportError(
        std::string("qwen38 decode TP2 protocol failed: ") + error.what());
  } catch (const pair_reduce::PairReduceCudaError& error) {
    phase_ = StepPhase::kFaulted;
    emit(stage, pair_reduce::Outcome::kCudaError, m_, trace_id, request_id,
         elapsed_ns(start));
    throw DecodeExecutionCudaError(
        std::string("qwen38 decode TP2 CUDA failed: ") + error.what());
  } catch (const std::exception& error) {
    phase_ = StepPhase::kFaulted;
    emit(stage, pair_reduce::Outcome::kTransportError, m_, trace_id, request_id,
         elapsed_ns(start));
    throw DecodeExecutionTransportError(
        std::string("qwen38 decode TP2 reduction failed: ") + error.what());
  } catch (...) {
    phase_ = StepPhase::kFaulted;
    emit(stage, pair_reduce::Outcome::kTransportError, m_, trace_id, request_id,
         elapsed_ns(start));
    throw DecodeExecutionTransportError("qwen38 decode TP2 reduction failed");
  }
  ++next_ordinal_;
  emit(stage, pair_reduce::Outcome::kOk, m_, trace_id, request_id,
       elapsed_ns(start));
}

void Tp2DecodeExecution::finish_step(std::uint64_t device_generation,
                                     std::string_view trace_id,
                                     std::string_view request_id) {
  const auto start = Clock::now();
  try {
    if (phase_ != StepPhase::kActive) contract_fail("finish requires active phase");
    if (device_generation != active_generation_)
      contract_fail("active device generation drift");
    if (next_ordinal_ != kReductionPoints)
      contract_fail("finish requires all 96 reduction points");
  } catch (const DecodeExecutionContractError&) {
    emit("rocket.qwen38.decode.tp2.lifecycle",
         pair_reduce::Outcome::kContractError, m_, trace_id, request_id,
         elapsed_ns(start));
    throw;
  }
  try {
    reducer_.complete(active_stream_, kReductionPoints, trace_id, request_id);
  } catch (const pair_reduce::PairReduceCudaError& error) {
    phase_ = StepPhase::kFaulted;
    emit("rocket.qwen38.decode.tp2.lifecycle",
         pair_reduce::Outcome::kCudaError, m_, trace_id, request_id,
         elapsed_ns(start));
    throw DecodeExecutionCudaError(
        std::string("qwen38 decode TP2 completion failed: ") + error.what());
  } catch (const std::exception& error) {
    phase_ = StepPhase::kFaulted;
    emit("rocket.qwen38.decode.tp2.lifecycle",
         pair_reduce::Outcome::kTransportError, m_, trace_id, request_id,
         elapsed_ns(start));
    throw DecodeExecutionTransportError(
        std::string("qwen38 decode TP2 completion failed: ") + error.what());
  }
  const int completed_m = m_;
  last_completed_generation_ = active_generation_;
  active_generation_ = 0;
  m_ = 0;
  next_ordinal_ = 0;
  active_stream_ = nullptr;
  phase_ = StepPhase::kIdle;
  emit("rocket.qwen38.decode.tp2.lifecycle", pair_reduce::Outcome::kOk, completed_m,
       trace_id, request_id, elapsed_ns(start));
}

void Tp2DecodeExecution::emit(std::string_view stage, pair_reduce::Outcome outcome,
                              int m, std::string_view trace_id,
                              std::string_view request_id,
                              std::uint64_t duration_ns) noexcept {
  const int rank = reducer_.rank() == 0 || reducer_.rank() == 1 ? reducer_.rank() : -1;
  const int m_bucket = pair_reduce::allowed_m(m) ? m : 0;
  const std::uint64_t bytes = static_cast<std::uint64_t>(m_bucket) *
                              pair_reduce::kHidden * sizeof(__nv_bfloat16);
  telemetry_.emit_span_and_log({stage, trace_id, request_id, rank, m_bucket,
                                pair_reduce::kDtype, outcome, duration_ns, bytes});
}

bool Tp2DecodeExecution::valid_point(ReductionPoint point) noexcept {
  return point.layer >= 0 && point.layer < kLayers &&
         (point.kind == ReductionKind::kAttentionOutput ||
          point.kind == ReductionKind::kMoeOutput);
}

std::string_view Tp2DecodeExecution::stage_for(ReductionPoint point) noexcept {
  if (point.kind == ReductionKind::kAttentionOutput)
    return "rocket.qwen38.decode.tp2.attention";
  if (point.kind == ReductionKind::kMoeOutput)
    return "rocket.qwen38.decode.tp2.moe";
  return "rocket.qwen38.decode.tp2.lifecycle";
}

}  // namespace rocket::qwen38::decode

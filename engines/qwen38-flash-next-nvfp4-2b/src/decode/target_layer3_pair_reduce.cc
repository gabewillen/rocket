// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_pair_reduce.h"

#include <algorithm>
#include <stdexcept>

namespace rocket::qwen38::decode {
namespace {

constexpr std::array<std::uint8_t, 32> kOracleSession{
    0x05, 0xea, 0x3a, 0xf1, 0xc4, 0x69, 0x4a, 0x9c,
    0x03, 0x5c, 0xe2, 0xfe, 0x9c, 0xe0, 0x06, 0xac,
    0xc5, 0x88, 0x81, 0xdf, 0x0f, 0xe8, 0x67, 0x71,
    0xb1, 0x84, 0x6f, 0x4b, 0xd8, 0xe5, 0xf4, 0x8b};

pair_reduce::RdmaConfig make_config(
    const TargetLayer3PairReduceBootstrap& bootstrap) {
  TargetLayer3PairReduceOwner::validate_bootstrap(bootstrap);
  pair_reduce::RdmaConfig config;
  config.rank = bootstrap.rank;
  config.bootstrap_host = bootstrap.bootstrap_host;
  config.bootstrap_port = bootstrap.bootstrap_port;
  config.operation_timeout_ms = bootstrap.timeout_ms;
  config.session_sha256 = bootstrap.session_sha256;
  return config;
}

}  // namespace

TargetLayer3PairReduceSchedule::TargetLayer3PairReduceSchedule(
    HiddenPartialReducer& reducer, pair_reduce::OtelStageSink& telemetry)
    : reducer_(reducer), telemetry_(telemetry),
      attention_(*this, TargetLayer3ReductionStage::kAttention),
      moe_(*this, TargetLayer3ReductionStage::kMoe) {
  if ((reducer_.rank() != 0 && reducer_.rank() != 1) ||
      reducer_.world_size() != 2) {
    emit(pair_reduce::Outcome::kContractError, "layer3-pairreduce-init",
         "oracle-05ea3af");
    throw std::invalid_argument("layer-3 PairReduce topology changed");
  }
}

void TargetLayer3PairReduceSchedule::StagePort::reduce(
    const __nv_bfloat16* input, float* output, int m,
    std::string_view trace_id, std::string_view request_id,
    cudaStream_t stream) {
  owner_.reduce(stage_, input, output, m, trace_id, request_id, stream);
}

void TargetLayer3PairReduceSchedule::begin_row(
    int row, std::uint64_t generation) {
  try {
    transition(TargetLayer3RowEvent{row, generation});
  } catch (...) {
    emit(pair_reduce::Outcome::kContractError, "layer3-pairreduce-row",
         "oracle-05ea3af");
    throw;
  }
}

void TargetLayer3PairReduceSchedule::transition(TargetLayer3RowEvent event) {
  if (phase_ != TargetLayer3PairReducePhase::kAwaitRow ||
      next_ordinal_ % 2 != 0 || event.row != next_ordinal_ / 2 ||
      event.generation != next_generation()) {
    if (!complete()) phase_ = TargetLayer3PairReducePhase::kFaulted;
    throw std::invalid_argument("layer-3 PairReduce generation changed");
  }
  phase_ = TargetLayer3PairReducePhase::kAwaitAttention;
}

void TargetLayer3PairReduceSchedule::guard(
    TargetLayer3ReductionEvent event) const {
  const bool accepted =
      (phase_ == TargetLayer3PairReducePhase::kAwaitAttention &&
       event.stage == TargetLayer3ReductionStage::kAttention) ||
      (phase_ == TargetLayer3PairReducePhase::kAwaitMoe &&
       event.stage == TargetLayer3ReductionStage::kMoe);
  if (!accepted)
    throw std::invalid_argument("layer-3 PairReduce stage changed");
}

void TargetLayer3PairReduceSchedule::transition(
    TargetLayer3ReductionEvent event) {
  ++next_ordinal_;
  if (event.stage == TargetLayer3ReductionStage::kAttention) {
    phase_ = TargetLayer3PairReducePhase::kAwaitMoe;
  } else if (next_ordinal_ == kTargetLayer3PairReduceCalls) {
    phase_ = TargetLayer3PairReducePhase::kCompleted;
  } else {
    phase_ = TargetLayer3PairReducePhase::kAwaitRow;
  }
}

void TargetLayer3PairReduceSchedule::reduce(
    TargetLayer3ReductionStage stage, const __nv_bfloat16* input,
    float* output, int m, std::string_view trace_id,
    std::string_view request_id, cudaStream_t stream) {
  const bool was_complete = complete();
  try {
    guard(TargetLayer3ReductionEvent{stage});
  } catch (...) {
    if (!was_complete) phase_ = TargetLayer3PairReducePhase::kFaulted;
    emit(pair_reduce::Outcome::kContractError, trace_id, request_id);
    throw;
  }
  if (m != 1 || input == nullptr || output == nullptr || stream == nullptr) {
    phase_ = TargetLayer3PairReducePhase::kFaulted;
    emit(pair_reduce::Outcome::kContractError, trace_id, request_id);
    throw std::invalid_argument("layer-3 PairReduce buffers changed");
  }
  try {
    reducer_.reduce(input, output, 1, trace_id, request_id, stream);
  } catch (const pair_reduce::PairReduceCudaError&) {
    phase_ = TargetLayer3PairReducePhase::kFaulted;
    emit(pair_reduce::Outcome::kCudaError, trace_id, request_id);
    throw;
  } catch (const pair_reduce::PairReduceContractError&) {
    phase_ = TargetLayer3PairReducePhase::kFaulted;
    emit(pair_reduce::Outcome::kContractError, trace_id, request_id);
    throw;
  } catch (...) {
    phase_ = TargetLayer3PairReducePhase::kFaulted;
    emit(pair_reduce::Outcome::kTransportError, trace_id, request_id);
    throw;
  }
  transition(TargetLayer3ReductionEvent{stage});
  if (complete())
    emit(pair_reduce::Outcome::kOk, trace_id, request_id);
}

void TargetLayer3PairReduceSchedule::emit(
    pair_reduce::Outcome outcome, std::string_view trace_id,
    std::string_view request_id) noexcept {
  const int rank =
      (reducer_.rank() == 0 || reducer_.rank() == 1) ? reducer_.rank() : -1;
  telemetry_.emit_span_and_log({
      "rocket.qwen38.layer3_pair_reduce.lifecycle", trace_id, request_id,
      rank, 1, pair_reduce::kDtype, outcome, 0,
      static_cast<std::uint64_t>(next_ordinal_) * pair_reduce::kHidden *
          sizeof(__nv_bfloat16)});
  if (rank >= 0)
    telemetry_.record_duration(
        {rank, 1, pair_reduce::kDtype, outcome, 0});
}

std::array<std::uint8_t, 32>
TargetLayer3PairReduceOwner::oracle_session_sha256() noexcept {
  return kOracleSession;
}

void TargetLayer3PairReduceOwner::validate_bootstrap(
    const TargetLayer3PairReduceBootstrap& bootstrap) {
  if ((bootstrap.rank != 0 && bootstrap.rank != 1) ||
      bootstrap.peer_rank != 1 - bootstrap.rank ||
      bootstrap.bootstrap_host.empty() || bootstrap.bootstrap_port <= 0 ||
      bootstrap.bootstrap_port > 65'535 ||
      !pair_reduce::valid_operation_timeout_ms(bootstrap.timeout_ms) ||
      bootstrap.session_sha256 != kOracleSession)
    throw std::invalid_argument("layer-3 PairReduce bootstrap identity changed");
}

TargetLayer3PairReduceOwner::TargetLayer3PairReduceOwner(
    const TargetLayer3PairReduceBootstrap& bootstrap,
    pair_reduce::OtelStageSink& telemetry) {
  const auto emit_failure = [&](pair_reduce::Outcome outcome) noexcept {
    const int rank =
        (bootstrap.rank == 0 || bootstrap.rank == 1) ? bootstrap.rank : -1;
    telemetry.emit_span_and_log({
        "rocket.qwen38.layer3_pair_reduce.bootstrap", "layer3-pairreduce-init",
        "oracle-05ea3af", rank, 1, pair_reduce::kDtype, outcome, 0, 0});
    if (rank >= 0)
      telemetry.record_duration(
          {rank, 1, pair_reduce::kDtype, outcome, 0});
  };
  try {
    auto config = make_config(bootstrap);
    transport_ = std::make_unique<pair_reduce::RdmaTransport>(config);
    reduction_ =
        std::make_unique<pair_reduce::PairReduce>(*transport_, telemetry);
    adapter_ = std::make_unique<PairReduceAdapter>(*reduction_);
    schedule_ = std::make_unique<TargetLayer3PairReduceSchedule>(*adapter_,
                                                                 telemetry);
  } catch (const std::invalid_argument&) {
    emit_failure(pair_reduce::Outcome::kContractError);
    throw;
  } catch (...) {
    emit_failure(pair_reduce::Outcome::kTransportError);
    throw;
  }
}

}  // namespace rocket::qwen38::decode

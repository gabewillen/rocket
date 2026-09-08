// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_pair_reduce.h"

#include <algorithm>
#include <stdexcept>

namespace rocket::qwen38::decode {
namespace {

pair_reduce::RdmaConfig make_config(
    const TargetK0PairReduceBootstrap& bootstrap) {
  TargetK0PairReduceOwner::validate_bootstrap(bootstrap);
  pair_reduce::RdmaConfig config;
  config.rank = bootstrap.rank;
  config.bootstrap_host = bootstrap.bootstrap_host;
  config.bootstrap_port = bootstrap.bootstrap_port;
  config.operation_timeout_ms = bootstrap.timeout_ms;
  config.session_sha256 = bootstrap.session_sha256;
  return config;
}

bool nonzero(const std::array<std::uint8_t, 32>& digest) noexcept {
  return std::any_of(digest.begin(), digest.end(),
                     [](std::uint8_t value) { return value != 0; });
}

}  // namespace

TargetK0PairReduceSchedule::TargetK0PairReduceSchedule(
    HiddenPartialReducer& reducer, pair_reduce::OtelStageSink& telemetry)
    : reducer_(reducer), telemetry_(telemetry) {
  if ((reducer_.rank() != 0 && reducer_.rank() != 1) ||
      reducer_.world_size() != 2)
    throw std::invalid_argument("K0 PairReduce topology changed");
  for (int layer = 0; layer < kLayers; ++layer) {
    attention_[layer] = std::make_unique<StagePort>(
        *this, layer, TargetK0ReductionStage::kAttention);
    moe_[layer] =
        std::make_unique<StagePort>(*this, layer, TargetK0ReductionStage::kMoe);
  }
}

HiddenPartialReducer& TargetK0PairReduceSchedule::attention_port(int layer) {
  if (layer < 0 || layer >= kLayers)
    throw std::invalid_argument("K0 attention PairReduce layer changed");
  return *attention_[layer];
}

HiddenPartialReducer& TargetK0PairReduceSchedule::moe_port(int layer) {
  if (layer < 0 || layer >= kLayers)
    throw std::invalid_argument("K0 MoE PairReduce layer changed");
  return *moe_[layer];
}

void TargetK0PairReduceSchedule::begin_sequence(
    std::uint64_t first_generation, int rows) {
  if (phase_ != TargetK0PairReducePhase::kAwaitSequence ||
      first_generation == 0 || rows <= 0 || rows > kTargetK0MaxRows) {
    emit(pair_reduce::Outcome::kContractError, "k0-pairreduce-sequence",
         "k0");
    throw std::invalid_argument("K0 PairReduce sequence changed");
  }
  first_generation_ = first_generation;
  rows_ = rows;
  current_row_ = -1;
  phase_ = TargetK0PairReducePhase::kAwaitRow;
}

void TargetK0PairReduceSchedule::begin_row(int row,
                                           std::uint64_t generation) {
  if (phase_ != TargetK0PairReducePhase::kAwaitRow ||
      row != current_row_ + 1 || row < 0 || row >= rows_ ||
      generation != first_generation_ + static_cast<std::uint64_t>(row)) {
    if (!complete()) phase_ = TargetK0PairReducePhase::kFaulted;
    emit(pair_reduce::Outcome::kContractError, "k0-pairreduce-row", "k0");
    throw std::invalid_argument("K0 PairReduce row/generation changed");
  }
  current_row_ = row;
  row_calls_ = 0;
  phase_ = TargetK0PairReducePhase::kAwaitReduction;
}

void TargetK0PairReduceSchedule::StagePort::reduce(
    const __nv_bfloat16* input, float* output, int m,
    std::string_view trace_id, std::string_view request_id,
    cudaStream_t stream) {
  owner_.reduce(layer_, stage_, input, output, m, trace_id, request_id,
                stream);
}

int TargetK0PairReduceSchedule::expected_layer() const noexcept {
  return row_calls_ / 2;
}

TargetK0ReductionStage TargetK0PairReduceSchedule::expected_stage() const
    noexcept {
  return row_calls_ % 2 == 0 ? TargetK0ReductionStage::kAttention
                             : TargetK0ReductionStage::kMoe;
}

void TargetK0PairReduceSchedule::reduce(
    int layer, TargetK0ReductionStage stage, const __nv_bfloat16* input,
    float* output, int m, std::string_view trace_id,
    std::string_view request_id, cudaStream_t stream) {
  if (phase_ != TargetK0PairReducePhase::kAwaitReduction ||
      layer != expected_layer() || stage != expected_stage() || m != 1 ||
      !input || !output || !stream) {
    if (!complete()) phase_ = TargetK0PairReducePhase::kFaulted;
    emit(pair_reduce::Outcome::kContractError, trace_id, request_id);
    throw std::invalid_argument("K0 PairReduce order or buffers changed");
  }
  try {
    reducer_.reduce(input, output, 1, trace_id, request_id, stream);
  } catch (const pair_reduce::PairReduceCudaError&) {
    phase_ = TargetK0PairReducePhase::kFaulted;
    emit(pair_reduce::Outcome::kCudaError, trace_id, request_id);
    throw;
  } catch (const pair_reduce::PairReduceContractError&) {
    phase_ = TargetK0PairReducePhase::kFaulted;
    emit(pair_reduce::Outcome::kContractError, trace_id, request_id);
    throw;
  } catch (...) {
    phase_ = TargetK0PairReducePhase::kFaulted;
    emit(pair_reduce::Outcome::kTransportError, trace_id, request_id);
    throw;
  }
  ++row_calls_;
  ++completed_calls_;
  if (row_calls_ == kTargetK0ReductionsPerRow) {
    if (current_row_ + 1 == rows_) {
      phase_ = TargetK0PairReducePhase::kCompleted;
      emit(pair_reduce::Outcome::kOk, trace_id, request_id);
    } else {
      phase_ = TargetK0PairReducePhase::kAwaitRow;
    }
  }
}

void TargetK0PairReduceSchedule::emit(
    pair_reduce::Outcome outcome, std::string_view trace_id,
    std::string_view request_id) noexcept {
  const int diagnostic_rank =
      reducer_.rank() == 0 || reducer_.rank() == 1 ? reducer_.rank() : -1;
  telemetry_.emit_span_and_log({
      "rocket.qwen38.k0_pair_reduce.lifecycle", trace_id, request_id,
      diagnostic_rank, 1, pair_reduce::kDtype, outcome, 0,
      static_cast<std::uint64_t>(completed_calls_) * pair_reduce::kHidden *
          sizeof(__nv_bfloat16)});
  if (diagnostic_rank >= 0)
    telemetry_.record_duration(
        {diagnostic_rank, 1, pair_reduce::kDtype, outcome, 0});
}

void TargetK0PairReduceOwner::validate_bootstrap(
    const TargetK0PairReduceBootstrap& bootstrap) {
  if ((bootstrap.rank != 0 && bootstrap.rank != 1) ||
      bootstrap.peer_rank != 1 - bootstrap.rank ||
      bootstrap.bootstrap_host.empty() || bootstrap.bootstrap_port <= 0 ||
      bootstrap.bootstrap_port > 65'535 ||
      !pair_reduce::valid_operation_timeout_ms(bootstrap.timeout_ms) ||
      !nonzero(bootstrap.session_sha256))
    throw std::invalid_argument("K0 PairReduce bootstrap identity changed");
}

TargetK0PairReduceOwner::TargetK0PairReduceOwner(
    const TargetK0PairReduceBootstrap& bootstrap,
    pair_reduce::OtelStageSink& telemetry) {
  try {
    auto config = make_config(bootstrap);
    transport_ = std::make_unique<pair_reduce::RdmaTransport>(config);
    reduction_ =
        std::make_unique<pair_reduce::PairReduce>(*transport_, telemetry);
    adapter_ = std::make_unique<PairReduceAdapter>(*reduction_);
    schedule_ =
        std::make_unique<TargetK0PairReduceSchedule>(*adapter_, telemetry);
  } catch (const std::invalid_argument&) {
    telemetry.emit_span_and_log({
        "rocket.qwen38.k0_pair_reduce.bootstrap", "k0-pairreduce-init", "k0",
        (bootstrap.rank == 0 || bootstrap.rank == 1) ? bootstrap.rank : -1,
        1, pair_reduce::kDtype, pair_reduce::Outcome::kContractError, 0, 0});
    throw;
  } catch (...) {
    telemetry.emit_span_and_log({
        "rocket.qwen38.k0_pair_reduce.bootstrap", "k0-pairreduce-init", "k0",
        (bootstrap.rank == 0 || bootstrap.rank == 1) ? bootstrap.rank : -1,
        1, pair_reduce::kDtype, pair_reduce::Outcome::kTransportError, 0, 0});
    throw;
  }
}

}  // namespace rocket::qwen38::decode

// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_prefill_contract.h"

#include <limits>
#include <stdexcept>

namespace rocket::qwen38::decode {

namespace {

[[nodiscard]] constexpr TargetK0PrefillStateKind expected_state_kind(
    int layer) noexcept {
  return is_qsa_layer(layer) ? TargetK0PrefillStateKind::kQsaKvCache
                             : TargetK0PrefillStateKind::kGdnChunkRecurrent;
}

[[nodiscard]] constexpr TargetK0AttentionKind expected_attention_kind(
    int layer) noexcept {
  return is_qsa_layer(layer) ? TargetK0AttentionKind::kQsa
                             : TargetK0AttentionKind::kGdn;
}

}  // namespace

void TargetK0PrefillLayerPort::execute_prefill_chunk(
    TargetK0PrefillChunk chunk, cudaStream_t stream,
    TargetK0ExecutionProgress* progress) {
  if (!authenticated() || !prefill_state_authenticated() || chunk.rows <= 0 ||
      chunk.rows > kTargetK0MaxPrefillRows ||
      chunk.rows > prefill_capacity_rows() ||
      chunk.first_generation >
          std::numeric_limits<std::uint64_t>::max() -
              static_cast<std::uint64_t>(chunk.rows - 1) ||
      chunk.replicated_pre_layer == nullptr ||
      chunk.replicated_post_layer == nullptr ||
      chunk.replicated_pre_layer == chunk.replicated_post_layer)
    throw std::invalid_argument("target K0 prefill chunk contract changed");
  execute_authenticated_prefill_chunk(chunk, stream, progress);
}

TargetK0PrefillPortInventory::TargetK0PrefillPortInventory(
    int rank, std::string_view oracle_manifest_sha256, int rows,
    std::array<TargetK0PrefillLayerPort*, kDecoderLayers> candidate_ports,
    TargetK0PairReduceSchedule& reductions,
    pair_reduce::OtelStageSink& telemetry)
    : rank_(rank), rows_(rows) {
  const auto emit = [&](pair_reduce::Outcome outcome) noexcept {
    telemetry.emit_span_and_log({
        "rocket.qwen38.k0.prefill_port_inventory", "owner-init",
        "all48-prefill-inventory", (rank == 0 || rank == 1) ? rank : -1, 1,
        pair_reduce::kDtype, outcome, 0, 0});
    telemetry.record_duration({(rank == 0 || rank == 1) ? rank : -1, 1,
                               pair_reduce::kDtype, outcome, 0});
  };
  try {
    if ((rank != 0 && rank != 1) ||
        !accepted_target_k0_oracle(oracle_manifest_sha256, rows) ||
        rows <= 0 || rows > kTargetK0MaxPrefillRows)
      throw std::invalid_argument("target K0 prefill inventory root changed");

    std::array<const TargetK0LayerPort*, kDecoderLayers> seen_ports{};
    std::array<const HiddenPartialReducer*, 2 * kDecoderLayers>
        seen_reducers{};
    int seen_port_count = 0;
    int seen_reducer_count = 0;
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      auto* port = candidate_ports[static_cast<std::size_t>(layer)];
      auto* attention = port ? port->attention_reducer_identity() : nullptr;
      auto* moe = port ? port->moe_reducer_identity() : nullptr;

      bool duplicate_port = false;
      for (int i = 0; i < seen_port_count; ++i)
        duplicate_port |= seen_ports[static_cast<std::size_t>(i)] == port;
      bool duplicate_reducer = false;
      for (int i = 0; i < seen_reducer_count; ++i) {
        duplicate_reducer |=
            seen_reducers[static_cast<std::size_t>(i)] == attention ||
            seen_reducers[static_cast<std::size_t>(i)] == moe;
      }

      if (port == nullptr || port->rank() != rank || port->layer() != layer ||
          port->attention_kind() != expected_attention_kind(layer) ||
          port->prefill_state_kind() != expected_state_kind(layer) ||
          !port->authenticated() || !port->prefill_state_authenticated() ||
          port->prefill_capacity_rows() < rows ||
          attention != &reductions.attention_port(layer) ||
          moe != &reductions.moe_port(layer) || attention == moe ||
          duplicate_port || duplicate_reducer)
        throw std::invalid_argument(
            "target K0 prefill port inventory binding changed");

      seen_ports[static_cast<std::size_t>(seen_port_count++)] = port;
      seen_reducers[static_cast<std::size_t>(seen_reducer_count++)] = attention;
      seen_reducers[static_cast<std::size_t>(seen_reducer_count++)] = moe;
      ports_[static_cast<std::size_t>(layer)] = port;
    }
  } catch (...) {
    ports_.fill(nullptr);
    emit(pair_reduce::Outcome::kContractError);
    throw;
  }
  authenticated_ = true;
  emit(pair_reduce::Outcome::kOk);
}

}  // namespace rocket::qwen38::decode

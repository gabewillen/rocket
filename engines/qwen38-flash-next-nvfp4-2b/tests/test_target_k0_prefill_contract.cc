// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_prefill_contract.h"

#include <array>
#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace decode = rocket::qwen38::decode;
namespace pr = rocket::qwen38::pair_reduce;

namespace {

void check(bool value) {
  if (!value) throw std::runtime_error("K0 prefill contract proof failed");
}

template <class F>
void expect_rejected(F&& operation) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected);
}

struct Sink final : pr::OtelStageSink {
  void emit_span_and_log(const pr::SpanRecord& value) noexcept override {
    ++spans;
    last = value.outcome;
  }
  void record_duration(const pr::MetricPoint&) noexcept override { ++metrics; }
  int spans = 0;
  int metrics = 0;
  pr::Outcome last = pr::Outcome::kContractError;
};

struct Reducer final : decode::HiddenPartialReducer {
  int rank() const noexcept override { return 0; }
  int world_size() const noexcept override { return 2; }
  void reduce(const __nv_bfloat16*, float*, int, std::string_view,
              std::string_view, cudaStream_t) override {}
};

struct RowOnlyLayer final : decode::TargetK0LayerPort {
  RowOnlyLayer(int layer, decode::HiddenPartialReducer& attention,
               decode::HiddenPartialReducer& moe)
      : layer_(layer), attention_(attention), moe_(moe) {}
  int rank() const noexcept override { return 0; }
  int layer() const noexcept override { return layer_; }
  decode::TargetK0AttentionKind attention_kind() const noexcept override {
    return decode::is_qsa_layer(layer_)
               ? decode::TargetK0AttentionKind::kQsa
               : decode::TargetK0AttentionKind::kGdn;
  }
  bool authenticated() const noexcept override { return true; }
  const decode::HiddenPartialReducer* attention_reducer_identity()
      const noexcept override {
    return &attention_;
  }
  const decode::HiddenPartialReducer* moe_reducer_identity()
      const noexcept override {
    return &moe_;
  }
  void wait_source(cudaStream_t) override {}
  void execute_row(std::uint64_t, const __nv_bfloat16*, __nv_bfloat16*,
                   cudaStream_t,
                   decode::TargetK0ExecutionProgress*) override {}
  int layer_;
  decode::HiddenPartialReducer& attention_;
  decode::HiddenPartialReducer& moe_;
};

struct PrefillLayer final : decode::TargetK0PrefillLayerPort {
  PrefillLayer(int layer, decode::HiddenPartialReducer& attention,
               decode::HiddenPartialReducer& moe)
      : layer_(layer), attention_(attention), moe_(moe) {}
  int rank() const noexcept override { return 0; }
  int layer() const noexcept override { return layer_; }
  decode::TargetK0AttentionKind attention_kind() const noexcept override {
    return decode::is_qsa_layer(layer_)
               ? decode::TargetK0AttentionKind::kQsa
               : decode::TargetK0AttentionKind::kGdn;
  }
  bool authenticated() const noexcept override { return authenticated_; }
  const decode::HiddenPartialReducer* attention_reducer_identity()
      const noexcept override {
    return &attention_;
  }
  const decode::HiddenPartialReducer* moe_reducer_identity()
      const noexcept override {
    return &moe_;
  }
  void wait_source(cudaStream_t) override {}
  void execute_row(std::uint64_t, const __nv_bfloat16*, __nv_bfloat16*,
                   cudaStream_t,
                   decode::TargetK0ExecutionProgress*) override {}
  decode::TargetK0PrefillStateKind prefill_state_kind()
      const noexcept override {
    if (wrong_state_kind_)
      return decode::is_qsa_layer(layer_)
                 ? decode::TargetK0PrefillStateKind::kGdnChunkRecurrent
                 : decode::TargetK0PrefillStateKind::kQsaKvCache;
    return decode::is_qsa_layer(layer_)
               ? decode::TargetK0PrefillStateKind::kQsaKvCache
               : decode::TargetK0PrefillStateKind::kGdnChunkRecurrent;
  }
  bool prefill_state_authenticated() const noexcept override {
    return state_authenticated_;
  }
  int prefill_capacity_rows() const noexcept override { return capacity_; }
  void execute_authenticated_prefill_chunk(
      decode::TargetK0PrefillChunk, cudaStream_t,
      decode::TargetK0ExecutionProgress*) override {
    ++chunks;
  }

  int layer_;
  decode::HiddenPartialReducer& attention_;
  decode::HiddenPartialReducer& moe_;
  bool authenticated_ = true;
  bool state_authenticated_ = true;
  bool wrong_state_kind_ = false;
  int capacity_ = 35;
  int chunks = 0;
};

using PrefillOwners =
    std::array<std::unique_ptr<PrefillLayer>, decode::kDecoderLayers>;

PrefillOwners make_owners(decode::TargetK0PairReduceSchedule& schedule) {
  PrefillOwners owners{};
  for (int layer = 0; layer < decode::kDecoderLayers; ++layer) {
    owners[static_cast<std::size_t>(layer)] = std::make_unique<PrefillLayer>(
        layer, schedule.attention_port(layer), schedule.moe_port(layer));
  }
  return owners;
}

std::array<decode::TargetK0PrefillLayerPort*, decode::kDecoderLayers> borrowed(
    const PrefillOwners& owners) {
  std::array<decode::TargetK0PrefillLayerPort*, decode::kDecoderLayers>
      result{};
  for (int layer = 0; layer < decode::kDecoderLayers; ++layer)
    result[static_cast<std::size_t>(layer)] =
        owners[static_cast<std::size_t>(layer)].get();
  return result;
}

}  // namespace

int main() {
  static_assert(!std::is_convertible_v<RowOnlyLayer*,
                                       decode::TargetK0PrefillLayerPort*>);
  Reducer reducer;
  Sink schedule_sink;
  decode::TargetK0PairReduceSchedule schedule(reducer, schedule_sink);

  auto owners = make_owners(schedule);
  Sink accepted_sink;
  decode::TargetK0PrefillPortInventory accepted(
      0, decode::kTargetK0OracleManifestSha256, 35, borrowed(owners), schedule,
      accepted_sink);
  check(accepted.authenticated() && accepted.rank() == 0 &&
        accepted.rows() == 35 && accepted_sink.spans == 1 &&
        accepted_sink.metrics == 1 && accepted_sink.last == pr::Outcome::kOk);

  __nv_bfloat16 input{};
  __nv_bfloat16 output{};
  accepted.ports()[0]->execute_prefill_chunk(
      {17, 1, &input, &output}, nullptr);
  check(owners[0]->chunks == 1);
  expect_rejected([&] {
    accepted.ports()[0]->execute_prefill_chunk(
        {17, 36, &input, &output}, nullptr);
  });
  expect_rejected([&] {
    accepted.ports()[0]->execute_prefill_chunk(
        {17, 1, &input, &input}, nullptr);
  });
  expect_rejected([&] {
    accepted.ports()[0]->execute_prefill_chunk(
        {std::numeric_limits<std::uint64_t>::max(), 2, &input, &output},
        nullptr);
  });
  check(owners[0]->chunks == 1);

  auto missing_prefill_ports = borrowed(owners);
  missing_prefill_ports[0] = nullptr;
  Sink row_only_sink;
  expect_rejected([&] {
    decode::TargetK0PrefillPortInventory inventory(
        0, decode::kTargetK0OracleManifestSha256, 35, missing_prefill_ports,
        schedule, row_only_sink);
  });
  check(row_only_sink.spans == 1 && row_only_sink.metrics == 1 &&
        row_only_sink.last == pr::Outcome::kContractError);

  owners[0]->state_authenticated_ = false;
  Sink missing_state_sink;
  expect_rejected([&] {
    decode::TargetK0PrefillPortInventory inventory(
        0, decode::kTargetK0OracleManifestSha256, 35, borrowed(owners),
        schedule, missing_state_sink);
  });
  owners[0]->state_authenticated_ = true;
  check(missing_state_sink.spans == 1 && missing_state_sink.metrics == 1 &&
        missing_state_sink.last == pr::Outcome::kContractError);

  owners[3]->wrong_state_kind_ = true;
  Sink wrong_kind_sink;
  expect_rejected([&] {
    decode::TargetK0PrefillPortInventory inventory(
        0, decode::kTargetK0OracleManifestSha256, 35, borrowed(owners),
        schedule, wrong_kind_sink);
  });
  owners[3]->wrong_state_kind_ = false;
  check(wrong_kind_sink.spans == 1 && wrong_kind_sink.metrics == 1 &&
        wrong_kind_sink.last == pr::Outcome::kContractError);

  Sink short_capacity_sink;
  expect_rejected([&] {
    decode::TargetK0PrefillPortInventory inventory(
        0, decode::kTargetK0ShortOracleManifestSha256, 87, borrowed(owners),
        schedule, short_capacity_sink);
  });
  check(short_capacity_sink.spans == 1 && short_capacity_sink.metrics == 1 &&
        short_capacity_sink.last == pr::Outcome::kContractError);

  Sink wrong_manifest_sink;
  expect_rejected([&] {
    decode::TargetK0PrefillPortInventory inventory(
        0, "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        35, borrowed(owners), schedule, wrong_manifest_sink);
  });
  check(wrong_manifest_sink.spans == 1 && wrong_manifest_sink.metrics == 1 &&
        wrong_manifest_sink.last == pr::Outcome::kContractError);
  return 0;
}

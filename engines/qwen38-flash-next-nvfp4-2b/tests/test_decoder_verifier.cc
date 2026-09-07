#include "decode/decoder_verifier.h"
#include "decode/decoder_verifier_c_api.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace decode = rocket::qwen38::decode;
namespace pr = rocket::qwen38::pair_reduce;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class T>
T* pointer(std::uintptr_t value) {
  return reinterpret_cast<T*>(value);
}

class Telemetry final : public pr::OtelStageSink {
 public:
  void emit_span_and_log(const pr::SpanRecord& record) noexcept override {
    outcomes.push_back(record.outcome);
  }
  void record_duration(const pr::MetricPoint&) noexcept override {}
  std::vector<pr::Outcome> outcomes;
};

class State final : public decode::DecoderStateTransaction {
 public:
  std::uint64_t active_generation() const noexcept override { return active; }
  void* begin(std::uint64_t generation,
              decode::DecoderVerifierShape) override {
    pending = generation;
    return this;
  }
  decode::GdnInactiveState gdn_state(void* transaction, int layer) override {
    check(transaction == this, "transaction identity drift");
    return {pointer<__nv_bfloat16>(0x1000 + layer),
            pointer<float>(0x2000 + layer),
            pointer<const std::int32_t>(0x3000 + layer)};
  }
  std::byte* mtp_state(void* transaction, std::size_t bytes) override {
    check(transaction == this && bytes == 4, "MTP inactive state ABI drift");
    return fail_mtp_state ? nullptr : mtp.data();
  }
  void publish(void* transaction) noexcept override {
    if (transaction == this) {
      active = pending;
      ++publications;
    }
  }
  void discard(void* transaction) noexcept override {
    if (transaction == this) ++discards;
  }
  std::uint64_t active = 7;
  std::uint64_t pending = 0;
  int publications = 0;
  int discards = 0;
  bool fail_mtp_state = false;
  std::array<std::byte, 4> mtp{};
};

class MtpParticipant final : public decode::AcceptedStateParticipant {
 public:
  std::size_t state_bytes_per_sequence() const noexcept override { return 4; }
  void stage_accept(std::uint64_t generation, std::byte* inactive,
                    const std::int32_t* widths,
                    decode::DecoderVerifierShape, cudaStream_t) override {
    check(generation == 8 && inactive && widths, "MTP stage inputs changed");
    ++stages;
    if (fail_stage) throw std::runtime_error("injected MTP accept failure");
  }
  void commit(std::uint64_t generation) noexcept override {
    active = generation;
    ++commits;
  }
  void validate_after_fence(std::uint64_t generation) override {
    check(generation == 8, "MTP post-fence generation changed");
    if (fail_validate) throw std::runtime_error("injected MTP fabric failure");
  }
  void discard(std::uint64_t) noexcept override { ++discards; }
  std::uint64_t active = 7;
  int stages = 0;
  int commits = 0;
  int discards = 0;
  bool fail_stage = false;
  bool fail_validate = false;
};

class Gdn final : public decode::GdnVerifierPort {
 public:
  Gdn(int layer, std::vector<std::string>& events)
      : layer(layer), events(events) {}
  void stage(const __nv_bfloat16* input, decode::GdnInactiveState inactive,
             decode::DecoderVerifierShape shape, std::string_view,
             std::string_view, cudaStream_t) override {
    check(input && inactive.convolution && inactive.recurrent &&
              inactive.authenticated_slots,
          "GDN stage ABI lost a required pointer");
    check(shape.token_rows() <= 128, "GDN stage exceeded 128 rows");
    staged = true;
    events.push_back("gdn:" + std::to_string(layer));
  }
  const __nv_bfloat16* staged_output() const noexcept override {
    return staged ? pointer<const __nv_bfloat16>(0x4000 + layer) : nullptr;
  }
  void accept(const std::int32_t* prefixes, std::string_view,
              std::string_view, cudaStream_t) override {
    check(staged && prefixes, "GDN accept without stage");
    if (fail_accept) throw std::runtime_error("injected GDN accept failure");
    staged = false;
    ++accepts;
    events.push_back("accept:" + std::to_string(layer));
  }
  void reset(std::string_view, std::string_view) noexcept override {
    staged = false;
    ++resets;
  }
  int layer;
  std::vector<std::string>& events;
  bool staged = false;
  int accepts = 0;
  int resets = 0;
  bool fail_accept = false;
};

class Runtime final : public decode::DecoderStepRuntime {
 public:
  explicit Runtime(std::vector<std::string>& events) : events(events) {}
  const __nv_bfloat16* embed(const std::int32_t*,
                             decode::DecoderVerifierShape,
                             cudaStream_t) override {
    events.push_back("embed");
    return pointer<const __nv_bfloat16>(0x5000);
  }
  const __nv_bfloat16* gdn_input(int layer, const __nv_bfloat16*,
                                decode::DecoderVerifierShape,
                                cudaStream_t) override {
    return pointer<const __nv_bfloat16>(0x6000 + layer);
  }
  const __nv_bfloat16* consume_attention(
      int layer, const __nv_bfloat16* output, decode::DecoderVerifierShape,
      cudaStream_t) override {
    check(output != nullptr, "attention output missing");
    events.push_back("attention:" + std::to_string(layer));
    return pointer<const __nv_bfloat16>(0x7000 + layer);
  }
  const __nv_bfloat16* stage_qsa(
      int layer, const __nv_bfloat16*, void* inactive,
      decode::DecoderVerifierShape shape, cudaStream_t) override {
    check(inactive && shape.token_rows() <= 128, "QSA inactive ABI drift");
    events.push_back("qsa:" + std::to_string(layer));
    return pointer<const __nv_bfloat16>(0x8000 + layer);
  }
  const __nv_bfloat16* pair_reduce(
      int layer, decode::ReductionKind kind, const __nv_bfloat16* partial,
      decode::DecoderVerifierShape, cudaStream_t) override {
    check(partial != nullptr, "PairReduce partial missing");
    events.push_back(std::string("reduce:") + std::to_string(layer) + ":" +
                     (kind == decode::ReductionKind::kAttentionOutput ? "a"
                                                                      : "m"));
    ++reductions;
    return pointer<const __nv_bfloat16>(0x9000 + reductions);
  }
  const __nv_bfloat16* stage_moe(
      int layer, const __nv_bfloat16*, decode::DecoderVerifierShape,
      cudaStream_t) override {
    events.push_back("moe:" + std::to_string(layer));
    if (layer == fail_moe_layer) throw std::runtime_error("injected MoE failure");
    ++moes;
    return pointer<const __nv_bfloat16>(0xa000 + layer);
  }
  void produce_logits(const __nv_bfloat16*, decode::DecoderVerifierShape,
                      cudaStream_t) override {
    events.push_back("logits");
  }
  decode::VerificationOutput sample_and_verify(
      decode::DecoderVerifierShape shape, cudaStream_t) override {
    events.push_back("sample");
    decode::VerificationOutput output;
    output.sequences = shape.sequences;
    output.accepted_prefixes_device = pointer<const std::int32_t>(0xb000);
    for (int i = 0; i < shape.sequences; ++i) {
      output.tokens[i] = 100 + i;
      output.accepted_prefixes[i] = shape.verify_width;
    }
    return output;
  }
  void accept_qsa(void* inactive, const std::int32_t*,
                  decode::DecoderVerifierShape, cudaStream_t) override {
    check(inactive != nullptr, "QSA accept lost inactive transaction");
    events.push_back("accept_qsa");
  }
  void reset_qsa(void* inactive) noexcept override {
    if (inactive) ++qsa_resets;
  }
  void synchronize(cudaStream_t) override { events.push_back("sync"); }

  std::vector<std::string>& events;
  int fail_moe_layer = -1;
  int reductions = 0;
  int moes = 0;
  int qsa_resets = 0;
};

struct Fixture {
  Fixture() : runtime(events) {
    owned.reserve(decode::kDecoderGdnLayers);
    for (int layer = 0; layer < decode::kDecoderLayers; ++layer) {
      if (!decode::is_qsa_layer(layer)) {
        owned.emplace_back(layer, events);
        ports[layer] = &owned.back();
      }
    }
  }
  std::vector<std::string> events;
  std::vector<Gdn> owned;
  std::array<decode::GdnVerifierPort*, decode::kDecoderLayers> ports{};
  Runtime runtime;
  State state;
  Telemetry telemetry;
};

void test_success_publishes_after_all_layers_and_accepts() {
  Fixture fixture;
  decode::DecoderVerifier verifier(
      fixture.ports, fixture.runtime, fixture.state, fixture.telemetry,
      reinterpret_cast<cudaStream_t>(0xb000));
  std::int32_t tokens[2] = {1, 2};
  const auto output = verifier.step(8, tokens, {2, 8}, "trace", "request");
  const int gdn = std::count_if(
      fixture.events.begin(), fixture.events.end(),
      [](const std::string& event) { return event.starts_with("gdn:"); });
  const int qsa = std::count_if(
      fixture.events.begin(), fixture.events.end(),
      [](const std::string& event) { return event.starts_with("qsa:"); });
  check(output.sequences == 2 && fixture.runtime.moes == 48 &&
            fixture.runtime.reductions == 96 && gdn == 36 && qsa == 12,
        "whole decoder topology counts changed");
  int accepts = 0;
  for (const auto& port : fixture.owned) accepts += port.accepts;
  check(accepts == 36 && fixture.state.publications == 1 &&
            fixture.state.active == 8 &&
            fixture.events[fixture.events.size() - 2] == "accept_qsa" &&
            fixture.events.back() == "sync",
        "state published before every verifier accepted and fenced");
}

void test_partial_accept_keeps_active_state_unchanged() {
  Fixture fixture;
  fixture.owned[10].fail_accept = true;
  decode::DecoderVerifier verifier(
      fixture.ports, fixture.runtime, fixture.state, fixture.telemetry,
      reinterpret_cast<cudaStream_t>(0xb000));
  std::int32_t token = 1;
  bool threw = false;
  try {
    (void)verifier.step(8, &token, {1, 4}, "trace", "request");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw && verifier.phase() == decode::DecoderVerifierPhase::kFaulted &&
            fixture.state.active == 7 && fixture.state.publications == 0 &&
            fixture.state.discards == 1 && fixture.runtime.qsa_resets == 1 &&
            fixture.runtime.moes == 48 && fixture.runtime.reductions == 96,
        "partial failure exposed inactive layer state");
  int accepted_inactive = 0;
  for (const auto& port : fixture.owned) accepted_inactive += port.accepts;
  check(accepted_inactive == 10,
        "accept failure did not occur after ten private layer writes");
}

void test_mtp_accept_faults_leave_every_generation_unpublished() {
  for (int fault = 0; fault < 3; ++fault) {
    Fixture fixture;
    MtpParticipant mtp;
    fixture.state.fail_mtp_state = fault == 0;
    mtp.fail_stage = fault == 1;
    mtp.fail_validate = fault == 2;
    decode::DecoderVerifier verifier(
        fixture.ports, fixture.runtime, fixture.state, fixture.telemetry,
        reinterpret_cast<cudaStream_t>(0xb000), &mtp);
    std::int32_t token = 1;
    bool threw = false;
    try {
      (void)verifier.step(8, &token, {1, 4}, "trace", "request");
    } catch (const std::runtime_error&) {
      threw = true;
    }
    check(threw && fixture.state.active == 7 && mtp.active == 7 &&
              fixture.state.publications == 0 && fixture.state.discards == 1 &&
              mtp.commits == 0 && mtp.discards == 1,
          "MTP accept fault exposed target or MTP generation");
  }
}

}  // namespace

int main() {
  try {
    check(qwen38_decoder_verifier_step(nullptr, 0, nullptr, 0, 0, nullptr,
                                       nullptr, nullptr, nullptr) == 1,
          "coarse C ABI accepted null construction state");
    test_success_publishes_after_all_layers_and_accepts();
    test_partial_accept_keeps_active_state_unchanged();
    test_mtp_accept_faults_leave_every_generation_unpublished();
    std::puts("decoder verifier transaction: 48 layers, 96 reductions, atomic state passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

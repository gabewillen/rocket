// SPDX-License-Identifier: Apache-2.0
#include "decode/target_gdn_layer.h"
#include "decode/target_k0_executor.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace decode = rocket::qwen38::decode;
namespace pr = rocket::qwen38::pair_reduce;

namespace {
void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

struct Trace final : pr::OtelStageSink {
  void emit_span_and_log(const pr::SpanRecord& r) noexcept override {
    stages.emplace_back(r.stage);
    outcomes.push_back(r.outcome);
  }
  void record_duration(const pr::MetricPoint&) noexcept override {}
  std::vector<std::string> stages;
  std::vector<pr::Outcome> outcomes;
};

struct Reducer final : decode::HiddenPartialReducer {
  Reducer(int rank, std::string_view name, std::vector<std::string>& order)
      : selected_rank(rank), name(name), order(order) {}
  int rank() const noexcept override { return selected_rank; }
  int world_size() const noexcept override { return 2; }
  void reduce(const __nv_bfloat16* input, float* output, int m,
              std::string_view, std::string_view, cudaStream_t stream) override {
    check(input && output && m == 1 && stream, "reducer arguments changed");
    order.emplace_back(name);
  }
  int selected_rank;
  std::string name;
  std::vector<std::string>& order;
};

struct Graph final : decode::LinearAttentionGraph {
  Graph(int rank, int layer, std::vector<std::string>& order)
      : selected_rank(rank), selected_layer(layer), order(order) {}
  int rank() const noexcept override { return selected_rank; }
  int layer() const noexcept override { return selected_layer; }
  std::string_view checkpoint_revision() const noexcept override {
    return decode::kQwen38CheckpointRevision;
  }
  std::string_view slab_key() const noexcept override {
    return selected_rank == 0 ? "rank0-target" : "rank1-target";
  }
  std::string_view conv_state_family() const noexcept override {
    return "target_gdn_conv";
  }
  std::string_view recurrent_state_family() const noexcept override {
    return "target_gdn_recurrent";
  }
  bool has_captured_bucket(int m) const noexcept override {
    return decode::allowed_linear_m(m);
  }
  std::uint64_t logical_bytes_per_row(int m) const noexcept override {
    return decode::allowed_linear_m(m) ? 1 : 0;
  }
  void launch(const __nv_bfloat16* input, __nv_bfloat16* convolution,
              float* recurrent, const std::int32_t* state_index, int m,
              cudaStream_t stream,
              decode::TargetK0ExecutionProgress* = nullptr) override {
    check(input && convolution && recurrent && state_index && m == 1 && stream,
          "GDN graph arguments changed");
    order.emplace_back("gdn");
  }
  const __nv_bfloat16* projected_output() const noexcept override {
    return &output;
  }
  int selected_rank;
  int selected_layer;
  std::vector<std::string>& order;
  __nv_bfloat16 output{};
};

struct Moe final : decode::TargetMoeGraph {
  Moe(int rank, int layer, std::vector<std::string>& order)
      : selected_rank(rank), selected_layer(layer), order(order) {}
  int rank() const noexcept override { return selected_rank; }
  int layer() const noexcept override { return selected_layer; }
  std::string_view checkpoint_revision() const noexcept override {
    return decode::kQwen38CheckpointRevision;
  }
  std::string_view slab_key() const noexcept override {
    return selected_rank == 0 ? "rank0-target" : "rank1-target";
  }
  void launch(const __nv_bfloat16* input, std::uint64_t generation, int m,
              cudaStream_t stream) override {
    check(input && generation == 1 && m == 1 && stream,
          "MoE arguments changed");
    order.emplace_back("moe");
  }
  void publish_after_fence(std::uint64_t generation) override {
    check(generation == 1, "MoE publication changed");
    order.emplace_back("moe_publish");
  }
  void terminal_fence_succeeded(std::uint64_t generation) override {
    check(generation == 1, "MoE fence changed");
    order.emplace_back("moe_fence");
  }
  void fault_after_fence(std::uint64_t) noexcept override {
    order.emplace_back("moe_fault");
  }
  const __nv_bfloat16* projected_output() const noexcept override {
    return &output;
  }
  int selected_rank;
  int selected_layer;
  std::vector<std::string>& order;
  __nv_bfloat16 output{};
};

struct Hyper final : decode::LinearAttentionHyperConnection {
  explicit Hyper(std::vector<std::string>& order) : order(order) {}
  void mix(const __nv_bfloat16* h, __nv_bfloat16* input,
           __nv_bfloat16* injection, int m, cudaStream_t stream) override {
    check(h && input && injection && m == 1 && stream, "HC mix changed");
    order.emplace_back("mix");
  }
  void combine_and_mix(const __nv_bfloat16* h, const float* partial,
                       const __nv_bfloat16* injection, __nv_bfloat16* updated,
                       __nv_bfloat16* next_input,
                       __nv_bfloat16* next_injection, int m,
                       cudaStream_t stream) override {
    check(h && partial && injection && updated && next_input && next_injection &&
              m == 1 && stream,
          "HC combine-and-mix changed");
    order.emplace_back("combine_and_mix");
  }
  void combine(const __nv_bfloat16* h, const float* partial,
               const __nv_bfloat16* injection, __nv_bfloat16* updated, int m,
               cudaStream_t stream) override {
    check(h && partial && injection && updated && m == 1 && stream,
          "HC combine changed");
    order.emplace_back("combine");
  }
  void synchronize(cudaStream_t stream) override {
    check(stream, "HC stream changed");
    order.emplace_back("fence");
  }
  std::vector<std::string>& order;
};

struct Generation final : decode::TargetGdnMoeGenerationPort {
  Generation(int rank, int layer, std::vector<std::string>& order)
      : selected_rank(rank), selected_layer(layer), order(order) {}
  int rank() const noexcept override { return selected_rank; }
  int layer() const noexcept override { return selected_layer; }
  bool authenticated() const noexcept override { return true; }
  void enqueue(std::uint64_t generation, cudaStream_t stream) override {
    check(generation == 1 && stream, "MoE generation changed");
    order.emplace_back("generation");
  }
  int selected_rank;
  int selected_layer;
  std::vector<std::string>& order;
};

struct Storage {
  std::vector<__nv_bfloat16> convolution = std::vector<__nv_bfloat16>(
      decode::kTargetGdnStateSlots * decode::kTargetGdnConvSlotElements);
  std::vector<float> recurrent = std::vector<float>(
      decode::kTargetGdnStateSlots * decode::kTargetGdnRecurrentSlotElements);
  std::int32_t state_index = 1;
  __nv_bfloat16 attention_input{}, attention_injection{}, post_attention{},
      moe_input{}, moe_injection{};
  float reduced_attention{}, reduced_moe{};
  decode::TargetGdnC1State state() {
    return {convolution.data(), convolution.size(), recurrent.data(),
            recurrent.size(), &state_index, 1};
  }
  decode::TargetGdnC1Buffers buffers() {
    return {&attention_input, &attention_injection, &reduced_attention,
            &post_attention, &moe_input, &moe_injection, &reduced_moe};
  }
};

struct BoundaryObserver final : decode::TargetK0LayerBoundaryObserver {
  void observe(decode::TargetK0LayerBoundary boundary, const void*,
               std::size_t elements, decode::TargetK0DiagnosticDtype dtype,
               cudaStream_t stream,
               decode::TargetK0LayerBoundaryEvidence& evidence) override {
    check(stream && elements > 0, "boundary diagnostic extent changed");
    const auto index = static_cast<std::size_t>(boundary);
    order.push_back(boundary);
    evidence.elements[index] = static_cast<std::uint32_t>(elements);
    evidence.hashes[index] = 1 + index;
    if (boundary == decode::TargetK0LayerBoundary::kAttentionReduction ||
        boundary == decode::TargetK0LayerBoundary::kMoeReduction)
      check(dtype == decode::TargetK0DiagnosticDtype::kFloat32,
            "reduction diagnostic dtype changed");
    else
      check(dtype == decode::TargetK0DiagnosticDtype::kBfloat16,
            "BF16 diagnostic dtype changed");
  }
  std::vector<decode::TargetK0LayerBoundary> order;
};
}  // namespace

int main() {
  try {
    for (int rank = 0; rank < 2; ++rank) {
      for (int layer = 0; layer < decode::kDecoderLayers; ++layer) {
        if (!decode::is_linear_attention_layer(layer)) continue;
        std::vector<std::string> order;
        Graph graph(rank, layer, order);
        Moe moe(rank, layer, order);
        Reducer attention(rank, "attention_reduce", order);
        Reducer moe_reduce(rank, "moe_reduce", order);
        Hyper hyper(order);
        Generation generation(rank, layer, order);
        Trace trace;
        Storage storage;
        decode::TargetGdnLayer owner(graph, moe, attention, moe_reduce, hyper,
                                     generation, trace, storage.state(),
                                     storage.buffers());
        check(owner.rank() == rank && owner.layer() == layer &&
                  owner.attention_reducer_identity() == &attention &&
                  owner.moe_reducer_identity() == &moe_reduce,
              "GDN owner identity changed");
      }
    }

    std::vector<std::string> order;
    Graph graph(1, 46, order);
    Moe moe(1, 46, order);
    Reducer attention(1, "attention_reduce", order);
    Reducer moe_reduce(1, "moe_reduce", order);
    Hyper hyper(order);
    Generation generation(1, 46, order);
    Trace trace;
    Storage storage;
    decode::TargetGdnLayer owner(graph, moe, attention, moe_reduce, hyper,
                                 generation, trace, storage.state(),
                                 storage.buffers());
    __nv_bfloat16 input{}, output{};
    const auto result = owner.execute(
        1, &input, &output, "gdn-contract", "row-0",
        reinterpret_cast<cudaStream_t>(&storage));
    check(result.generation == 1 && result.rank == 1 && result.layer == 46 &&
              result.post_layer == &output,
          "GDN publication changed");
    check(order == std::vector<std::string>{
                       "mix", "fence", "gdn", "fence", "attention_reduce",
                       "combine_and_mix", "fence", "generation", "moe",
                       "fence", "moe_fence", "moe_publish", "moe_reduce",
                       "combine", "fence"},
          "GDN execution order changed");
    check(trace.stages.back() == "rocket.qwen38.k0.gdn_layer.lifecycle" &&
              trace.outcomes.back() == pr::Outcome::kOk,
          "GDN lifecycle telemetry changed");

    std::vector<std::string> diagnostic_order;
    Graph diagnostic_graph(0, 0, diagnostic_order);
    Moe diagnostic_moe(0, 0, diagnostic_order);
    Reducer diagnostic_attention(0, "attention_reduce", diagnostic_order);
    Reducer diagnostic_moe_reduce(0, "moe_reduce", diagnostic_order);
    Hyper diagnostic_hyper(diagnostic_order);
    Generation diagnostic_generation(0, 0, diagnostic_order);
    Trace diagnostic_trace;
    Storage diagnostic_storage;
    decode::TargetGdnLayer diagnostic_owner(
        diagnostic_graph, diagnostic_moe, diagnostic_attention,
        diagnostic_moe_reduce, diagnostic_hyper, diagnostic_generation,
        diagnostic_trace, diagnostic_storage.state(),
        diagnostic_storage.buffers());
    BoundaryObserver observer;
    decode::TargetK0ExecutionProgress progress;
    progress.row = 0;
    progress.layer = 0;
    progress.boundary_observer = &observer;
    diagnostic_owner.execute(
        1, &input, &output, "gdn-diagnostic", "row-0",
        reinterpret_cast<cudaStream_t>(&diagnostic_storage), &progress);
    check(observer.order == std::vector<decode::TargetK0LayerBoundary>{
                                decode::TargetK0LayerBoundary::kAttentionOutput,
                                decode::TargetK0LayerBoundary::kAttentionReduction,
                                decode::TargetK0LayerBoundary::kHyperconnectionCombineMix,
                                decode::TargetK0LayerBoundary::kMoeOutput,
                                decode::TargetK0LayerBoundary::kMoeReduction,
                                decode::TargetK0LayerBoundary::kFinalHyperconnection},
          "layer0 boundary diagnostic order changed");

    auto short_state = storage.state();
    --short_state.recurrent_elements;
    bool rejected = false;
    try {
      decode::TargetGdnLayer invalid(graph, moe, attention, moe_reduce, hyper,
                                     generation, trace, short_state,
                                     storage.buffers());
    } catch (const decode::DecodeExecutionContractError&) {
      rejected = true;
    }
    check(rejected, "short GDN state was accepted");
    std::puts("qwen38 target GDN layer: 36 owners and c1 order passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

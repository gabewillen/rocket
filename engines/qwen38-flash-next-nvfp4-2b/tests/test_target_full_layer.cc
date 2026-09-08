#include "decode/target_full_layer.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace decode = rocket::qwen38::decode;
namespace attention = rocket::qwen38::attention;
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

struct Graph final : decode::FullAttentionGraph {
  explicit Graph(std::vector<std::string>& order, int selected_layer = 3)
      : order(order), selected_layer(selected_layer) {}
  int rank() const noexcept override { return 0; }
  int layer() const noexcept override { return selected_layer; }
  std::string_view checkpoint_revision() const noexcept override {
    return decode::kFullAttentionCheckpointRevision;
  }
  std::string_view slab_key() const noexcept override { return "rank0-target"; }
  void launch(const __nv_bfloat16*, const attention::TargetQsaStateView&,
              std::uint64_t, int, cudaStream_t) override {
    order.emplace_back("qsa");
  }
  const __nv_bfloat16* projected_output() const noexcept override {
    return &partial;
  }
  std::vector<std::string>& order;
  int selected_layer;
  __nv_bfloat16 partial{};
};

struct Moe final : decode::TargetMoeGraph {
  explicit Moe(std::vector<std::string>& order, int selected_layer = 3)
      : order(order), selected_layer(selected_layer) {}
  int rank() const noexcept override { return 0; }
  int layer() const noexcept override { return selected_layer; }
  std::string_view checkpoint_revision() const noexcept override {
    return decode::kFullAttentionCheckpointRevision;
  }
  std::string_view slab_key() const noexcept override { return "rank0-target"; }
  void launch(const __nv_bfloat16*, std::uint64_t, int,
              cudaStream_t) override {
    order.emplace_back("moe");
    if (fail) throw std::runtime_error("injected target MoE failure");
  }
  void publish_after_fence(std::uint64_t) override {
    order.emplace_back("moe_publish");
  }
  void terminal_fence_succeeded(std::uint64_t) override {
    order.emplace_back("moe_fenced");
  }
  void fault_after_fence(std::uint64_t) noexcept override {
    order.emplace_back("moe_fault");
  }
  const __nv_bfloat16* projected_output() const noexcept override {
    return &partial;
  }
  std::vector<std::string>& order;
  int selected_layer;
  __nv_bfloat16 partial{};
  bool fail = false;
};

struct Reducer final : decode::HiddenPartialReducer {
  Reducer(std::vector<std::string>& order, std::string name)
      : order(order), name(std::move(name)) {}
  int rank() const noexcept override { return 0; }
  int world_size() const noexcept override { return 2; }
  void reduce(const __nv_bfloat16*, float*, int, std::string_view,
              std::string_view, cudaStream_t) override {
    order.push_back(name);
  }
  std::vector<std::string>& order;
  std::string name;
};

struct Hc final : decode::FullAttentionHyperConnection {
  explicit Hc(std::vector<std::string>& order) : order(order) {}
  void mix(const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int,
           cudaStream_t) override { order.emplace_back("mix"); }
  void combine_and_mix(const __nv_bfloat16*, const float*,
                       const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
                       __nv_bfloat16*, int, cudaStream_t) override {
    order.emplace_back("combine_and_mix");
  }
  void combine(const __nv_bfloat16*, const float*, const __nv_bfloat16*,
               __nv_bfloat16*, int, cudaStream_t) override {
    order.emplace_back("combine");
  }
  void synchronize(cudaStream_t) override { order.emplace_back("fence"); }
  std::vector<std::string>& order;
};

attention::TargetQsaStateView state(std::uint64_t generation) {
  static __nv_bfloat16 b{};
  static std::int32_t i32{};
  static std::int64_t i64{};
  return {.main_key_cache=&b, .main_value_cache=&b, .raw_key_cache=&b,
          .compressed_key_cache=&b, .positions=&i64, .main_slot_mapping=&i32,
          .main_block_table=&i32, .raw_slot_mapping=&i32,
          .raw_block_table=&i32, .compressed_slot_mapping=&i32,
          .compressed_block_table=&i32, .query_start_locations=&i32,
          .logical_positions=&i64, .sequence_lengths=&i32,
          .token_to_request=&i32, .compression_work=&i32, .main_blocks=1,
          .compressed_blocks=1, .compression_work_items=1, .rows=1, .rank=0,
          .layer=3, .uses_mrope=false,
          .main_kv_dtype=attention::TargetQsaServingDtype::kBfloat16,
          .side_cache_dtype=attention::TargetQsaServingDtype::kBfloat16,
          .generation=generation, .expected_generation=generation};
}
}  // namespace

int main() {
  try {
    std::vector<std::string> order;
    Graph qsa(order); Moe moe(order); Reducer ar(order, "attention_reduce");
    Reducer mr(order, "moe_reduce"); Hc hc(order); Trace trace;
    decode::TargetFullLayer layer(qsa, moe, ar, mr, hc, trace);
    __nv_bfloat16 b{}; float f{};
    auto result = layer.execute(1, state(1), &b, &b, &b, &f, &b, &b, &b, &f,
                                &b, "trace", "request",
                                reinterpret_cast<cudaStream_t>(0x1230));
    check(result.post_layer == &b && result.layer == 3 && result.rank == 0,
          "post-layer publication changed");
    check(order == std::vector<std::string>{
        "mix", "fence", "qsa", "fence", "attention_reduce",
        "combine_and_mix", "fence", "moe", "fence", "moe_fenced",
        "moe_publish", "moe_reduce",
        "combine", "fence"}, "layer-3 composition order changed");
    check(trace.stages.size() == 8 && trace.outcomes.back() == pr::Outcome::kOk,
          "bounded success telemetry changed");

    for (int selected_layer = 3; selected_layer < 48; selected_layer += 4) {
      std::vector<std::string> layer_order;
      Graph layer_qsa(layer_order, selected_layer);
      Moe layer_moe(layer_order, selected_layer);
      Reducer layer_ar(layer_order, "attention_reduce");
      Reducer layer_mr(layer_order, "moe_reduce");
      Hc layer_hc(layer_order);
      Trace layer_trace;
      decode::TargetFullLayer qsa_layer(layer_qsa, layer_moe, layer_ar,
                                        layer_mr, layer_hc, layer_trace);
      check(qsa_layer.layer() == selected_layer,
            "all-layer QSA composition identity changed");
    }

    std::vector<std::string> failed_order;
    Graph failed_qsa(failed_order); Moe failed_moe(failed_order);
    failed_moe.fail = true;
    Reducer failed_ar(failed_order, "attention_reduce");
    Reducer failed_mr(failed_order, "moe_reduce"); Hc failed_hc(failed_order);
    Trace failed_trace;
    decode::TargetFullLayer failed(failed_qsa, failed_moe, failed_ar,
                                   failed_mr, failed_hc, failed_trace);
    bool rejected = false;
    try {
      failed.execute(1, state(1), &b, &b, &b, &f, &b, &b, &b, &f, &b,
                     "trace", "request",
                     reinterpret_cast<cudaStream_t>(0x1230));
    } catch (const std::runtime_error&) { rejected = true; }
    check(rejected && failed_trace.outcomes.back() != pr::Outcome::kOk,
          "target MoE failure was published");
    bool retry_rejected = false;
    try {
      failed.execute(1, state(1), &b, &b, &b, &f, &b, &b, &b, &f, &b,
                     "trace", "request",
                     reinterpret_cast<cudaStream_t>(0x1230));
    } catch (const decode::DecodeExecutionContractError&) {
      retry_rejected = true;
    }
    check(retry_rejected, "faulted layer-3 transition replayed");
    std::puts("qwen38 target layer-3 composition contract passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

// SPDX-License-Identifier: Apache-2.0
#include "attention/native_qsa_graph.h"

#include <stdexcept>
#include <string>
#include <memory>

#include "projection/cutlass_qkv.h"

namespace rocket::qwen38::attention {
namespace {

void require(bool condition, const char* message) {
  if (!condition) throw TargetQsaPreprocessError(message);
}

void require_cuda_call(int status, const char* phase) {
  if (status == 0) return;
  const char* detail = qwen38_cutlass_qkv_last_error();
  throw TargetQsaPreprocessError(
      std::string(phase) + ": " + (detail && *detail ? detail : "failed"));
}

}  // namespace

NativeQsaFullAttentionGraph::NativeQsaFullAttentionGraph(
    int device, int rank, int layer, std::string_view sidecar_key,
    const TargetQsaProjectionWeights& weights,
    const TargetQsaPreprocessWeights& preprocess_weights,
    const TargetQsaGraphArena& arena)
    : rank_(rank), layer_(layer), preprocess_weights_(preprocess_weights),
      arena_(arena) {
  require(device >= 0 && (rank == 0 || rank == 1) &&
              is_target_qsa_layer(layer),
          "native QSA graph device/rank/layer identity is invalid");
  require(sidecar_key == kTargetQsaIndexerSidecarKey,
          "native QSA graph indexer sidecar identity drift");
  require(arena.index_projected_qk && arena.main_query &&
              arena.attention_gate && arena.index_query &&
              arena.index_logits && arena.visible_blocks &&
              arena.selected_blocks && arena.selected_tokens &&
              arena.attention_partial && arena.attention_lse &&
              arena.attention_output,
          "native QSA graph caller-owned arena is incomplete");
  require_cuda_call(qwen38_target_qsa_projection_create_c1(
      weights.q_weight, weights.q_scale, weights.q_global,
      weights.k_weight, weights.k_scale, weights.k_global,
      weights.v_weight, weights.v_scale, weights.v_global,
      weights.o_weight, weights.o_scale, weights.o_global,
      arena.qkv_packed, arena.qkv_sfa, arena.raw_main_qkv,
      arena.gated_attention, arena.output_packed, arena.output_sfa,
      arena.projected_output, device, &projection_plan_),
      "initialize c1 QSA projections");
}

NativeQsaFullAttentionGraph::~NativeQsaFullAttentionGraph() {
  qwen38_target_qsa_projection_destroy_c1(projection_plan_);
}

void NativeQsaFullAttentionGraph::launch(
    const __nv_bfloat16* block_input, const TargetQsaStateView& state,
    std::uint64_t generation, int m, cudaStream_t stream) {
  require(!faulted_, "faulted native QSA graph cannot be retried");
  try {
    require(m == 1 && block_input && stream,
            "native QSA graph requires one row and a CUDA stream");
    validate_target_qsa_state_view(state, rank_, layer_, generation);
    require_cuda_call(qwen38_target_qsa_project_qkv_c1(
                          projection_plan_, block_input, stream),
                      "project c1 QKV");
    launch_target_qsa_preprocess_c1(
        block_input, preprocess_weights_,
        {arena_.raw_main_qkv, arena_.index_projected_qk, arena_.main_query,
         arena_.attention_gate, arena_.index_query},
        state, stream);
    require_cuda_call(qwen38_target_qsa_select_c1(
        arena_.index_query, state.compressed_key_cache,
        state.compressed_block_table, state.logical_positions,
        state.sequence_lengths, state.token_to_request, arena_.index_logits,
        arena_.visible_blocks, arena_.selected_blocks, arena_.selected_tokens,
        state.compressed_blocks, stream), "select c1 QSA tokens");
    require_cuda_call(qwen38_target_qsa_attention_c1(
        arena_.main_query, state.main_key_cache, state.main_value_cache,
        arena_.selected_tokens, state.main_block_table,
        state.token_to_request, arena_.attention_partial, arena_.attention_lse,
        arena_.attention_output, state.main_blocks, stream),
        "attend c1 QSA tokens");
    require_cuda_call(qwen38_target_qsa_project_output_c1(
        projection_plan_, arena_.attention_output, arena_.attention_gate,
        stream), "project c1 QSA output");
  } catch (...) {
    faulted_ = true;
    throw;
  }
}

}  // namespace rocket::qwen38::attention

namespace {
thread_local std::string target_qsa_last_error;

struct TargetQsaCHandle {
  std::unique_ptr<rocket::qwen38::attention::NativeQsaFullAttentionGraph> graph;
  std::uint64_t published_generation = 0;
};

template <class Operation>
int qsa_call(Operation operation) noexcept {
  target_qsa_last_error.clear();
  try {
    operation();
    return 0;
  } catch (const std::exception& error) {
    target_qsa_last_error = error.what();
  } catch (...) {
    target_qsa_last_error = "unknown native target QSA failure";
  }
  return 1;
}

int qsa_reject(const char* reason) noexcept {
  target_qsa_last_error = reason;
  return 1;
}
}  // namespace

extern "C" int rocket_qwen38_target_qsa_create_c1(
    int device, int rank, int layer, const char* sidecar_key,
    const rocket::qwen38::attention::TargetQsaProjectionWeights* projection,
    const rocket::qwen38::attention::TargetQsaPreprocessWeights* preprocess,
    const rocket::qwen38::attention::TargetQsaGraphArena* arena,
    void** handle) noexcept {
  if (!sidecar_key || !projection || !preprocess || !arena || !handle ||
      *handle)
    return qsa_reject("native target QSA create contract changed");
  return qsa_call([&] {
    auto owned = std::make_unique<TargetQsaCHandle>();
    owned->graph = std::make_unique<
        rocket::qwen38::attention::NativeQsaFullAttentionGraph>(
        device, rank, layer, sidecar_key, *projection, *preprocess, *arena);
    *handle = owned.release();
  });
}

extern "C" int rocket_qwen38_target_qsa_launch_c1(
    void* handle, const __nv_bfloat16* block_input,
    const rocket::qwen38::attention::TargetQsaStateView* state,
    std::uint64_t generation, int m, cudaStream_t stream) noexcept {
  if (!handle || !state)
    return qsa_reject("native target QSA launch contract changed");
  return qsa_call([&] {
    auto* owned = static_cast<TargetQsaCHandle*>(handle);
    if (generation == 0 || generation != owned->published_generation + 1)
      throw std::invalid_argument("native target QSA generation changed");
    owned->graph->launch(block_input, *state, generation, m, stream);
    owned->published_generation = generation;
  });
}

extern "C" int rocket_qwen38_target_qsa_projected_output_c1(
    void* handle, const __nv_bfloat16** output) noexcept {
  if (!handle || !output)
    return qsa_reject("native target QSA output contract changed");
  return qsa_call([&] {
    auto* owned = static_cast<TargetQsaCHandle*>(handle);
    if (owned->published_generation == 0)
      throw std::runtime_error("native target QSA has no published generation");
    *output = owned->graph->projected_output();
    if (!*output)
      throw std::runtime_error("native target QSA output is absent");
  });
}

extern "C" void rocket_qwen38_target_qsa_destroy_c1(void* handle) noexcept {
  delete static_cast<TargetQsaCHandle*>(handle);
}

extern "C" const char* rocket_qwen38_target_qsa_last_error() noexcept {
  return target_qsa_last_error.c_str();
}

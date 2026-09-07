// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "decode/linear_attention_layer.h"
#include "linear_attention/gdn_core.h"

namespace rocket::qwen38::linear_attention {

struct Nvfp4Matrix {
  const std::uint8_t* weight;
  const std::uint8_t* scale;
  float global_scale;
};

struct GdnWeights {
  Nvfp4Matrix qkv;
  Nvfp4Matrix z;
  Nvfp4Matrix b;
  Nvfp4Matrix a;
  Nvfp4Matrix output;
  const __nv_bfloat16* conv;
  const __nv_bfloat16* a_log;
  const __nv_bfloat16* dt_bias;
  const __nv_bfloat16* norm;
};

// Exact layer-0 rank-0 graph adapter. Construction copies authenticated weight
// extents into graph-owned immutable storage before capture. No allocations or
// pointer changes occur in launch().
class CutlassGdnGraph final : public decode::LinearAttentionGraph {
 public:
  CutlassGdnGraph(int device, GdnWeights weights);
  ~CutlassGdnGraph();
  CutlassGdnGraph(const CutlassGdnGraph&) = delete;
  CutlassGdnGraph& operator=(const CutlassGdnGraph&) = delete;

  int rank() const noexcept override { return 0; }
  int layer() const noexcept override { return 0; }
  std::string_view checkpoint_revision() const noexcept override {
    return decode::kQwen38CheckpointRevision;
  }
  std::string_view slab_key() const noexcept override { return "rank0-target"; }
  std::string_view conv_state_family() const noexcept override {
    return "target_gdn_conv";
  }
  std::string_view recurrent_state_family() const noexcept override {
    return "target_gdn_recurrent";
  }
  bool has_captured_bucket(int m) const noexcept override {
    return allowed_m(m);
  }
  std::uint64_t logical_bytes_per_row(int m) const noexcept override;
  void launch(const __nv_bfloat16* block_input,
              __nv_bfloat16* conv_state, float* recurrent_state,
              const std::int32_t* state_indices, int m,
              cudaStream_t stream) override;
  const __nv_bfloat16* projected_output() const noexcept override;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace rocket::qwen38::linear_attention

extern "C" {
int qwen38_gdn_graph_create(
    int device,
    const std::uint8_t* qkv_weight, const std::uint8_t* qkv_scale,
    float qkv_global, const std::uint8_t* z_weight,
    const std::uint8_t* z_scale, float z_global,
    const std::uint8_t* b_weight, const std::uint8_t* b_scale,
    float b_global, const std::uint8_t* a_weight,
    const std::uint8_t* a_scale, float a_global,
    const std::uint8_t* output_weight, const std::uint8_t* output_scale,
    float output_global, const __nv_bfloat16* conv,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    const __nv_bfloat16* norm, void** graph);
int qwen38_gdn_graph_launch(
    void* graph, const __nv_bfloat16* block_input,
    __nv_bfloat16* conv_state, float* recurrent_state,
    const std::int32_t* state_indices, int m, cudaStream_t stream);
int qwen38_gdn_graph_output(void* graph, void** output_bf16,
                            std::size_t* elements);
int qwen38_gdn_graph_destroy(void* graph);
const char* qwen38_gdn_graph_last_error();
}

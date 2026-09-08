// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

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

[[nodiscard]] constexpr bool allowed_prefill_tokens(int tokens) noexcept {
  return tokens == 300 || tokens == 8'192;
}

[[nodiscard]] constexpr std::size_t prefill_sfa_bytes(int tokens,
                                                       int width) noexcept {
  return allowed_prefill_tokens(tokens) && width > 0 && width % 16 == 0
             ? static_cast<std::size_t>(((tokens + 127) / 128) * 128) *
                   (width / 16)
             : 0;
}

inline constexpr int kPrefillInputQuantizationsPerLaunch = 1;
inline constexpr int kPrefillReferenceInputQuantizationsPerLaunch = 2;
inline constexpr char kPrefillB12xQuantSourceRevision[] = "8e685d198";

enum class GdnPrefillInputBackend {
  kFlashInferCutlass,
  kB12x,
  kFlashInferWheelBenchmark,
};

#if defined(__CUDACC__)
#define ROCKET_QWEN38_GDN_HOST_DEVICE __host__ __device__
#else
#define ROCKET_QWEN38_GDN_HOST_DEVICE
#endif
ROCKET_QWEN38_GDN_HOST_DEVICE constexpr std::size_t prefill_sfa_offset(
    int row, int scale_column, int scale_columns) noexcept {
  const int row_tile = row / 128;
  const int tile_row = row % 128;
  return static_cast<std::size_t>(row_tile) * 128 * scale_columns +
         static_cast<std::size_t>(scale_column / 4) * 512 +
         static_cast<std::size_t>((tile_row % 32) * 16 +
                                  (tile_row / 32) * 4 + scale_column % 4);
}
#undef ROCKET_QWEN38_GDN_HOST_DEVICE

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
  void launch_verifier(
      const __nv_bfloat16* position_major_input,
      __nv_bfloat16* dense_conv_state, float* dense_recurrent_state,
      __nv_bfloat16* prefix_conv_state, float* prefix_recurrent_state,
      int sequences, int verify_width, cudaStream_t stream);
  const __nv_bfloat16* verifier_output() const noexcept;
  const __nv_bfloat16* projected_output() const noexcept override;

 private:
  struct Impl;
  Impl* impl_;
};

// Prefill-only projection owner for the workload anchors. Input quantization
// is shared by QKVZ and BA, unlike the two independent general-engine linear
// calls. The plan owns immutable copies of authenticated projection slabs and
// fixed arenas for both graph shapes.
class CutlassGdnPrefillProjection final {
 public:
  CutlassGdnPrefillProjection(int device, GdnWeights weights,
                              bool enable_reference = false,
                              GdnPrefillInputBackend input_backend =
                                  GdnPrefillInputBackend::kFlashInferCutlass,
                              std::string_view wheel_shared_object = {});
  ~CutlassGdnPrefillProjection();
  CutlassGdnPrefillProjection(const CutlassGdnPrefillProjection&) = delete;
  CutlassGdnPrefillProjection& operator=(
      const CutlassGdnPrefillProjection&) = delete;

  void launch_input(const __nv_bfloat16* hidden, int tokens,
                    cudaStream_t stream);
  void launch_input_quantize(const __nv_bfloat16* hidden, int tokens,
                             cudaStream_t stream);
  // Diagnostic phase boundaries. The raw launch writes unscaled GEMM output;
  // the scale launch applies the two family-specific weight_scale_2 values.
  // Production callers use launch_qkvz()/launch_ba(), which compose both.
  void launch_qkvz_raw(int tokens, cudaStream_t stream);
  void launch_qkvz_scale(int tokens, cudaStream_t stream);
  void launch_qkvz(int tokens, cudaStream_t stream);
  void launch_ba_raw(int tokens, cudaStream_t stream);
  void launch_ba_scale(int tokens, cudaStream_t stream);
  void launch_ba(int tokens, cudaStream_t stream);
  // Matched two-quant control over the same authenticated immutable weights.
  // It exists for parity and phase attribution, not production dispatch.
  void launch_reference_input(const __nv_bfloat16* hidden, int tokens,
                              cudaStream_t stream);
  void launch_output(const __nv_bfloat16* normalized, int tokens,
                     cudaStream_t stream);
  [[nodiscard]] const __nv_bfloat16* qkvz(int tokens) const noexcept;
  [[nodiscard]] const __nv_bfloat16* ba(int tokens) const noexcept;
  [[nodiscard]] const __nv_bfloat16* output(int tokens) const noexcept;
  [[nodiscard]] const std::uint8_t* input_packed(int tokens) const noexcept;
  [[nodiscard]] const std::uint8_t* input_sfa(int tokens) const noexcept;
  [[nodiscard]] const std::uint8_t* qkvz_weight() const noexcept;
  [[nodiscard]] const std::uint8_t* qkvz_sfb() const noexcept;
  [[nodiscard]] const std::uint8_t* ba_weight() const noexcept;
  [[nodiscard]] const std::uint8_t* ba_sfb() const noexcept;
  [[nodiscard]] const float* projection_alpha() const noexcept;
  [[nodiscard]] const std::uint8_t* reference_qkvz_packed(
      int tokens) const noexcept;
  [[nodiscard]] const std::uint8_t* reference_qkvz_sfa(
      int tokens) const noexcept;
  [[nodiscard]] const std::uint8_t* reference_ba_packed(
      int tokens) const noexcept;
  [[nodiscard]] const std::uint8_t* reference_ba_sfa(
      int tokens) const noexcept;
  [[nodiscard]] const __nv_bfloat16* reference_qkvz(
      int tokens) const noexcept;
  [[nodiscard]] const __nv_bfloat16* reference_ba(int tokens) const noexcept;
  [[nodiscard]] static constexpr int input_quantizations_per_launch() noexcept {
    return kPrefillInputQuantizationsPerLaunch;
  }
  [[nodiscard]] static constexpr int
  reference_input_quantizations_per_launch() noexcept {
    return kPrefillReferenceInputQuantizationsPerLaunch;
  }

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

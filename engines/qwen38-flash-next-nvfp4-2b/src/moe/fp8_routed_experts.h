// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

#include "moe/route_compaction.h"

namespace rocket::qwen38::moe {

// Authenticated by the named qwen38-rank-slab.v1 artifact and model revision.
// Source and serving weights remain FP8 E4M3 with BF16
// inverse scales over 128x128 blocks. The 768 dimension is scratch padding,
// while every matrix and scale addresses the logical 640 dimension only.
inline constexpr char kMtpExpertSourceAbi[] = "fp8_e4m3_block_128x128";
inline constexpr char kMtpExpertServingAbi[] = "fp8_e4m3_block_128x128";
inline constexpr char kMtpExpertSlabSchema[] = "rocket.qwen38-rank-slab.v1";
inline constexpr char kMtpExpertSlabArtifact[] =
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4";
inline constexpr char kMtpExpertModelRevision[] =
    "fc694b54fb0174e0913e6adf86691ef85a4ead47";
inline constexpr char kMtpExpertPrimitive[] =
    "vllm-triton-fp8-block128-active-prefix";

enum class RoutedExpertOutcome : std::uint32_t {
  kOk,
  kContractError,
  kStaleGeneration,
  kCudaError,
};

// Each device pointer table has kLocalExperts entries in owner-local order.
// Matrix elements are row-major: gate/up [640,2560], down [2560,640]. Scale
// elements are BF16 row-major: gate/up [5,20], down [20,5]. The table storage,
// pointed-to tensors, and all launch buffers are borrowed from the caller.
struct Fp8ExpertTables {
  const std::uint8_t* const* gate_weights;
  const __nv_bfloat16* const* gate_scale_inv;
  const std::uint8_t* const* up_weights;
  const __nv_bfloat16* const* up_scale_inv;
  const std::uint8_t* const* down_weights;
  const __nv_bfloat16* const* down_scale_inv;
};

struct RoutedExpertDeviceSummary {
  std::uint64_t generation;
  std::uint64_t active_weight_bytes;
  std::uint32_t fc1_tiles;
  std::uint32_t fc2_tiles;
  std::int32_t active_experts;
  std::int32_t active_routes;
  RoutedExpertOutcome outcome;
};

struct RoutedExpertBuffers {
  // All scratch is caller-owned and fixed at graph construction. hidden is
  // [capacity.rows,2560] BF16; quantized_hidden is FP8 with FP32 inverse scales
  // [capacity.rows,20]. gate_up is BF16 [capacity.routes,2,768]. activated is
  // FP8 [capacity.routes,768] with FP32 inverse scales [capacity.routes,5].
  // rank_output is [capacity.rows,2560] FP32 and is overwritten by each launch.
  const __nv_bfloat16* hidden;
  std::uint8_t* quantized_hidden;
  float* hidden_scale_inv;
  __nv_bfloat16* gate_up;
  std::uint8_t* activated;
  float* activated_scale_inv;
  float* rank_output;
  RoutedExpertDeviceSummary* summary;
};

struct RoutedExpertLaunch {
  RouteCompactionShape shape;
  RouteCompactionCapacity capacity;
  RouteCompactionBuffers routes;
  Fp8ExpertTables experts;
  RoutedExpertBuffers buffers;
  cudaStream_t stream;
};

// Fixed-grid CUDA port of the selected vLLM Triton fallback dataflow:
// grouped gate/up, fused SiLU*up, grouped down, route-weighted reduction.
// Route compaction has already provided the stable grouped permutation, so
// this path skips vLLM's second sort/pad pass. Inactive grid positions exit
// before weight or activation reads. There is no allocation, synchronization,
// device-to-host transfer, or host dependence on device counts.
class Fp8RoutedExperts final {
 public:
  Fp8RoutedExperts(int device, int rank,
                   const char* module_directory = nullptr);
  ~Fp8RoutedExperts();
  Fp8RoutedExperts(const Fp8RoutedExperts&) = delete;
  Fp8RoutedExperts& operator=(const Fp8RoutedExperts&) = delete;

  [[nodiscard]] RoutedExpertOutcome enqueue(
      const RoutedExpertLaunch& launch) const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct RoutedExpertValidation {
  const RoutedExpertDeviceSummary& summary;
  std::uint64_t requested_generation;
  RouteCompactionShape shape;
};

[[nodiscard]] RoutedExpertOutcome validate_routed_expert_summary(
    const RoutedExpertValidation& validation) noexcept;

enum class RoutedExpertCounter : std::uint8_t {
  kActiveExperts,
  kActiveRoutes,
  kActiveWeightBytes,
  kFc1Tiles,
  kFc2Tiles,
};

struct RoutedExpertOtelPoint {
  RoutedExpertCounter counter;
  RoutedExpertOutcome outcome;
  int rank;
  int sequence_bucket;
  std::uint64_t value;
};

class RoutedExpertOtelSink {
 public:
  virtual ~RoutedExpertOtelSink() = default;
  virtual void add_routed_expert_counter(
      const RoutedExpertOtelPoint& point) noexcept = 0;
};

// Counter identity is bounded to 5 counters * 4 outcomes * 2 ranks * 5
// sequence buckets = 200 series. Generation, expert ids, and measured values
// never appear as attributes.
void export_routed_expert_otel_after_fence(
    const RoutedExpertDeviceSummary& snapshot,
    std::uint64_t requested_generation, RouteCompactionShape shape,
    RoutedExpertOtelSink& sink) noexcept;

}  // namespace rocket::qwen38::moe

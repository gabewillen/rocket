// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>

namespace rocket::qwen38::moe {

inline constexpr int kTargetMoeC1TopK = 10;
inline constexpr int kTargetMoeGlobalExperts = 512;
inline constexpr int kTargetMoeLocalExperts = 256;

enum class TargetMoeOutcome : std::uint32_t {
  kOk,
  kContractError,
  kStaleGeneration,
  kCudaError,
};

struct TargetMoeC1Summary {
  std::uint64_t generation;
  std::int32_t local_routes;
  TargetMoeOutcome outcome;
};

struct TargetMoeC1RouteLaunch {
  int rank;
  int layer;
  const std::int32_t* global_expert_ids;
  const float* routing_weights;
  const std::uint64_t* source_generation;
  const std::uint64_t* requested_generation;
  std::int32_t* local_expert_ids;
  float* local_routing_weights;
  TargetMoeC1Summary* summary;
  cudaStream_t stream;
};

// One fixed block localizes the ten global routes in their original slot
// order. Remote routes retain a valid local ID 0 with an exact zero weight.
// Invalid or stale input publishes ten zero weights and a failed summary.
// Every pointer is caller-owned. The launch allocates, copies, synchronizes,
// and reads back nothing.
[[nodiscard]] TargetMoeOutcome enqueue_target_moe_c1_routes(
    const TargetMoeC1RouteLaunch& launch) noexcept;

[[nodiscard]] TargetMoeOutcome validate_target_moe_c1_summary(
    const TargetMoeC1Summary& summary, std::uint64_t requested_generation,
    int rank, int layer) noexcept;

struct TargetMoeC1RouteReference {
  std::array<std::int32_t, kTargetMoeC1TopK> local_expert_ids{};
  std::array<float, kTargetMoeC1TopK> local_routing_weights{};
  TargetMoeC1Summary summary{};
};

[[nodiscard]] TargetMoeC1RouteReference target_moe_c1_route_reference(
    int rank, int layer,
    const std::array<std::int32_t, kTargetMoeC1TopK>& global_expert_ids,
    const std::array<float, kTargetMoeC1TopK>& routing_weights,
    std::uint64_t source_generation,
    std::uint64_t requested_generation) noexcept;

enum class TargetMoeCounter : std::uint8_t { kLaunch, kLocalRoutes };

struct TargetMoeOtelPoint {
  TargetMoeCounter counter;
  TargetMoeOutcome outcome;
  int rank;
  int layer;
  std::uint64_t value;
};

class TargetMoeOtelSink {
 public:
  virtual ~TargetMoeOtelSink() = default;
  virtual void add_counter(const TargetMoeOtelPoint& point) noexcept = 0;
};

// Called only after the enclosing graph fence. Attribute cardinality is
// counter(2) x outcome(4) x rank(2) x layer(48) = 768 bounded series.
// Generations and route counts are values, never attributes.
void export_target_moe_c1_otel_after_fence(
    const TargetMoeC1Summary& summary, std::uint64_t requested_generation,
    int rank, int layer, TargetMoeOtelSink& sink) noexcept;

}  // namespace rocket::qwen38::moe

// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rocket::qwen38::moe {

inline constexpr int kGlobalExperts = 512;
inline constexpr int kLocalExperts = 256;
inline constexpr int kTopK = 10;
inline constexpr int kMaxSequences = 16;
inline constexpr int kMaxVerifyWidth = 8;
inline constexpr int kMaxRows = kMaxSequences * kMaxVerifyWidth;
inline constexpr int kMaxRoutes = kMaxRows * kTopK;
inline constexpr std::uint64_t kHidden = 2'560;
inline constexpr std::uint64_t kLogicalIntermediate = 640;
inline constexpr std::uint64_t kPhysicalIntermediate = 768;
inline constexpr std::uint64_t kFp8Block = 128;
// The authenticated MTP slab keeps three independent FP8 E4M3 matrices and
// one BF16 inverse scale per 128x128 block. Physical intermediate padding is
// workspace only and is not present in the slab payload.
inline constexpr std::uint64_t kFp8BytesPerExpert =
    3 * kHidden * kLogicalIntermediate +
    3 * (kHidden / kFp8Block) * (kLogicalIntermediate / kFp8Block) *
        sizeof(std::uint16_t);
static_assert(kFp8BytesPerExpert == 4'915'800);

enum class RouteCompactionOutcome : std::uint32_t {
  kOk,
  kContractError,
  kOverflow,
  kStaleGeneration,
  kCudaError,
};
static_assert(static_cast<std::uint32_t>(RouteCompactionOutcome::kCudaError) +
                  1 ==
              5);

struct RouteCompactionShape {
  int rank;
  int sequences;
  int rows;
};

[[nodiscard]] constexpr bool allowed_shape(RouteCompactionShape shape) noexcept {
  const bool allowed_sequences =
      shape.sequences == 1 || shape.sequences == 2 || shape.sequences == 4 ||
      shape.sequences == 8 || shape.sequences == 16;
  return (shape.rank == 0 || shape.rank == 1) && allowed_sequences &&
         shape.rows >= shape.sequences &&
         shape.rows <= shape.sequences * kMaxVerifyWidth;
}

struct RouteCompactionCapacity {
  int experts = kLocalExperts;
  int rows = kMaxRows;
  int routes = kMaxRoutes;
};

[[nodiscard]] constexpr bool allowed_capacity(
    RouteCompactionCapacity capacity) noexcept {
  return capacity.experts >= 1 && capacity.experts <= kLocalExperts &&
         capacity.rows >= 1 && capacity.rows <= kMaxRows &&
         capacity.routes >= 1 && capacity.routes <= kMaxRoutes;
}

// Both route arrays contain shape.rows * kTopK elements in input row-major
// order. All pointers are borrowed device storage owned by the caller and must
// remain valid through stream completion. Generation pointers each address one
// value. They are read on device, which keeps graph replay independent of host
// state. The caller is the sole writer until this launch completes.
struct RouteCompactionInput {
  const std::int32_t* global_expert_ids;
  const float* routing_weights;
  const std::uint64_t* source_generation;
  const std::uint64_t* requested_generation;
};

struct RouteCompactionDeviceSummary {
  std::uint64_t generation;
  std::uint64_t active_weight_bytes;
  std::int32_t active_experts;
  std::int32_t active_rows;
  std::int32_t active_routes;
  RouteCompactionOutcome outcome;
};

// active_global_expert_ids, expert_row_counts, and expert_route_cursors have
// capacity.experts elements. local_to_active has kLocalExperts elements.
// expert_route_offsets has capacity.experts + 1 elements. Every owner-route
// and expert-route array has capacity.routes elements; summary has one element.
//
// The owner-route prefix retains input row-major order and partitions the
// global top-k10 pairs across the two rank owners without changing id or weight
// bits. expert_route_indices is a permutation over that prefix, grouped by
// first-seen active expert. Within every expert group, row and top-k slot order
// match the input. Downstream GEMM must use summary.active_experts,
// expert_row_counts, and expert_route_offsets as its work bounds, never the
// storage capacity.
struct RouteCompactionBuffers {
  std::int32_t* active_global_expert_ids;
  std::int32_t* local_to_active;
  std::int32_t* expert_row_counts;
  std::int32_t* expert_route_offsets;
  std::int32_t* expert_route_cursors;
  std::int32_t* owner_route_global_expert_ids;
  float* owner_route_weights;
  std::int32_t* owner_route_rows;
  std::uint8_t* owner_route_slots;
  std::int32_t* expert_route_indices;
  RouteCompactionDeviceSummary* summary;
};

struct RouteCompactionLaunch {
  RouteCompactionShape shape;
  RouteCompactionCapacity capacity;
  RouteCompactionInput input;
  RouteCompactionBuffers output;
  cudaStream_t stream;
};

// Enqueues one fixed-grid kernel on launch.stream and performs no allocation,
// synchronization, or device-to-host transfer. Calls that share buffers need
// external stream ordering and are not concurrently reentrant. A successful
// return reports enqueue status only; consumers must check summary outcome and
// generation after the enclosing stream fence. A failed device result
// publishes zero active counts and generation, so partial payload writes are
// unreachable. Invalid host bindings return kContractError without enqueueing.
[[nodiscard]] RouteCompactionOutcome enqueue_route_compaction(
    const RouteCompactionLaunch& launch) noexcept;

// Validates the device result after the enclosing stream fence. Generation is
// compared with the transaction requested by NativeExecutor. Counts and bytes
// must describe one reachable prefix for the fixed launch shape.
struct RouteCompactionValidation {
  const RouteCompactionDeviceSummary& summary;
  std::uint64_t requested_generation;
  RouteCompactionShape shape;
};

[[nodiscard]] RouteCompactionOutcome validate_route_compaction_summary(
    const RouteCompactionValidation& validation) noexcept;

struct CpuRouteCompactionInput {
  RouteCompactionShape shape;
  RouteCompactionCapacity capacity;
  std::span<const std::int32_t> global_expert_ids;
  std::span<const float> routing_weights;
  std::uint64_t source_generation;
  std::uint64_t requested_generation;
};

struct CpuRouteCompactionResult {
  RouteCompactionDeviceSummary summary{};
  std::vector<std::int32_t> active_global_expert_ids;
  std::vector<std::int32_t> expert_row_counts;
  std::vector<std::int32_t> expert_route_offsets;
  std::vector<std::int32_t> owner_route_global_expert_ids;
  std::vector<float> owner_route_weights;
  std::vector<std::int32_t> owner_route_rows;
  std::vector<std::uint8_t> owner_route_slots;
  std::vector<std::int32_t> expert_route_indices;
};

// Deterministic host reference for contract tests. It preserves every selected
// owner pair bit-for-bit and returns failures without a valid active prefix.
[[nodiscard]] CpuRouteCompactionResult compact_owner_routes_reference(
    const CpuRouteCompactionInput& input);

enum class RouteCompactionCounter : std::uint8_t {
  kActiveExperts,
  kActiveRows,
  kActiveWeightBytes,
};

struct RouteCompactionOtelAttributes {
  RouteCompactionOutcome outcome;
  int rank;
  int sequence_bucket;
};

struct RouteCompactionOtelPoint {
  RouteCompactionCounter counter;
  RouteCompactionOtelAttributes attributes;
  std::uint64_t value;
};

class RouteCompactionOtelSink {
 public:
  virtual ~RouteCompactionOtelSink() = default;
  virtual void add_counter(const RouteCompactionOtelPoint& point) noexcept = 0;
};

struct RouteCompactionOtelExport {
  const RouteCompactionDeviceSummary& snapshot;
  std::uint64_t requested_generation;
  RouteCompactionShape shape;
  RouteCompactionOtelSink& sink;
};

// active_weight_bytes is the authenticated FP8 block-128 payload of active experts,
// not an HBM counter. Called after the enclosing stream fence with a
// caller-owned host snapshot. requested_generation is the executor-owned
// transaction identity used to classify a stale producer.
// Counter identity has at most 3 * 5 * 2 * 5 = 150 series. Active values and
// generation are measurements and never OTEL attributes.
void export_route_compaction_otel_after_fence(
    const RouteCompactionOtelExport& export_request) noexcept;

}  // namespace rocket::qwen38::moe

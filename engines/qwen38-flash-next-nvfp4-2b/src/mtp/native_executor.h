// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace rocket::qwen38::mtp {

inline constexpr int kMaxDepth = 7;
inline constexpr int kMaxSequences = 16;
inline constexpr int kLocalExperts = 256;
inline constexpr int kRouterTopK = 8;
inline constexpr std::size_t kNativeRankSlabBytes = 1'370'161'152;
// Three TP2 projections: 3 * 1,638,400 FP8 bytes plus three 200-byte
// block-128 inverse-scale tensors in the authenticated rank slab.
inline constexpr std::uint64_t kNvidiaFp8BytesPerLocalExpert = 4'915'800;

enum class Phase : std::uint8_t {
  kInputFusion,
  kAttention,
  kAttentionReduce,
  kRoutedAndSharedMoe,
  kMoeReduce,
  kFinalHyperconnection,
  kLogits,
  kProposalSample,
  kCount,
};

enum class Outcome : std::uint8_t { kOk, kContractError, kCudaError };
enum class ExecutorPhase : std::uint8_t { kReady, kDrafted, kFaulted };

struct GraphKey {
  int depth;
  int sequences;
};

[[nodiscard]] constexpr bool allowed_graph_key(GraphKey key) noexcept {
  const bool batch = key.sequences == 1 || key.sequences == 2 ||
                     key.sequences == 4 || key.sequences == 8 ||
                     key.sequences == 16;
  return key.depth >= 1 && key.depth <= kMaxDepth && batch &&
         (key.depth <= 4 || key.sequences <= 4);
}

struct PhaseMetric {
  Phase phase;
  Outcome outcome;
  int depth;
  int sequences;
  std::uint64_t duration_ns;
};

struct ExpertUsageMetric {
  Outcome outcome;
  int depth;
  int sequences;
  int draft_step;
  int unique_local_experts;
  std::uint64_t resident_expert_bytes;
};

// All metric dimensions are bounded enums or graph buckets. Correlation IDs,
// sequence slots, token IDs, and expert IDs are intentionally absent.
class TelemetrySink {
 public:
  virtual ~TelemetrySink() = default;
  virtual void record_phase(const PhaseMetric& metric) noexcept = 0;
  virtual void record_expert_usage(const ExpertUsageMetric& metric) noexcept = 0;
};

struct ImmutableSlabs {
  const void* target_rank_slab;
  std::size_t target_rank_slab_bytes;
  const void* mtp_rank_slab;
  std::size_t mtp_rank_slab_bytes;
  // SHA-256 bytes returned by mtp_source.inspect_native_mtp_source().
  std::array<std::uint8_t, 32> source_contract_digest;
};

struct BoundGraph {
  GraphKey key;
  // Exact captured graphs for every phase and draft position. The native
  // executor owns no graph nodes and never mutates an exec after binding.
  std::array<std::array<cudaGraphExec_t, static_cast<int>(Phase::kCount)>,
             kMaxDepth>
      phase_graphs{};
  // Actual post-router expert IDs, laid out [sequences, top_k], per step.
  std::array<const std::int32_t*, kMaxDepth> router_expert_ids{};
  // Captured target+proposal output, position-major [depth + 1, sequences].
  const std::int32_t* verification_tokens = nullptr;
  // Private causal snapshots, one [sequences, state_bytes] array per step.
  std::array<const std::byte*, kMaxDepth> causal_snapshots{};
  std::byte* active_causal_state = nullptr;
  std::size_t state_bytes_per_sequence = 0;
};

struct DeviceDraftView {
  const std::int32_t* verification_tokens = nullptr;
  int depth = 0;
  int sequences = 0;
  std::uint64_t generation = 0;
};

class NativeExecutorError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// One logical writer. draft() performs every MTP layer and proposal step in
// native code. publish() is the only accepted-state mutation and selects the
// private snapshot corresponding to DecoderVerifier's accepted target width.
class NativeExecutor final {
 public:
  NativeExecutor(ImmutableSlabs slabs, BoundGraph graph,
                 TelemetrySink& telemetry, cudaStream_t stream);
  ~NativeExecutor();

  NativeExecutor(const NativeExecutor&) = delete;
  NativeExecutor& operator=(const NativeExecutor&) = delete;

  DeviceDraftView draft(std::uint64_t generation);
  void publish(std::uint64_t generation,
               const std::int32_t* accepted_widths_device);
  // Invoke only after DecoderVerifier's terminal fence. This method performs
  // no CUDA synchronization and launches no device work.
  void export_telemetry_after_fence(std::uint64_t generation);
  void discard(std::uint64_t generation) noexcept;

  [[nodiscard]] ExecutorPhase phase() const noexcept { return phase_; }
  [[nodiscard]] GraphKey key() const noexcept { return graph_.key; }

 private:
  ImmutableSlabs slabs_;
  BoundGraph graph_;
  TelemetrySink& telemetry_;
  cudaStream_t stream_;
  std::array<std::array<cudaEvent_t, static_cast<int>(Phase::kCount) + 1>,
             kMaxDepth>
      events_{};
  std::uint32_t* expert_masks_device_ = nullptr;
  const std::byte** snapshots_device_ = nullptr;
  std::array<std::array<std::uint32_t, 8>, kMaxDepth> expert_masks_host_{};
  ExecutorPhase phase_ = ExecutorPhase::kReady;
  std::uint64_t active_generation_ = 0;
  std::uint64_t pending_generation_ = 0;
};

}  // namespace rocket::qwen38::mtp

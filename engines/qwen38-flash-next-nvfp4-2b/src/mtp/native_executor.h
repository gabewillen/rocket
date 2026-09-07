// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "decode/decoder_verifier.h"
#include "mtp/graph_runtime.h"
#include "mtp/state_arena.h"

namespace rocket::qwen38::mtp {
inline constexpr int kMaxDepth = 7, kMaxSequences = 16, kLocalExperts = 256,
                     kRouterTopK = 8;
inline constexpr std::uint64_t kNvidiaFp8BytesPerLocalExpert = 4'915'800;
enum class Phase : std::uint8_t { kInputFusion, kAttention, kAttentionReduce,
  kRoutedAndSharedMoe, kMoeReduce, kFinalHyperconnection, kLogits,
  kProposalSample, kCount };
enum class Outcome : std::uint8_t { kOk, kContractError, kCudaError };
enum class ExecutorPhase : std::uint8_t { kReady, kDrafted, kFaulted };
struct GraphKey { int depth; int sequences; };
constexpr bool allowed_graph_key(GraphKey k) noexcept {
  const bool b = k.sequences == 1 || k.sequences == 2 || k.sequences == 4 ||
                 k.sequences == 8 || k.sequences == 16;
  return k.depth >= 1 && k.depth <= 7 && b &&
         (k.depth <= 4 || k.sequences <= 4);
}
struct PhaseMetric { Phase phase; Outcome outcome; int depth; int sequences;
  std::uint64_t duration_ns; };
struct ExpertUsageMetric { Outcome outcome; int depth; int sequences;
  int draft_step; int unique_local_experts; std::uint64_t resident_expert_bytes; };
class TelemetrySink { public: virtual ~TelemetrySink() = default;
  virtual void record_phase(const PhaseMetric&) noexcept = 0;
  virtual void record_expert_usage(const ExpertUsageMetric&) noexcept = 0; };
struct DeviceDraftView { const std::int32_t* verification_tokens = nullptr;
  int depth = 0; int sequences = 0; std::uint64_t generation = 0; };

// Required native QSA/MoE and PairReduce boundary. Methods enqueue only on the
// caller stream and may write fixed runtime/StateArena storage, never active state.
class MtpMiddleStagePort { public: virtual ~MtpMiddleStagePort() = default;
  virtual const std::int32_t* prepare(GraphArenaView, StateArena&, GraphKey,
                                      cudaStream_t) = 0;
  virtual void reduce_input(GraphArenaView, int, GraphKey, cudaStream_t) = 0;
  virtual void stage_attention(GraphArenaView, PrefixStateView, int, GraphKey,
                               cudaStream_t) = 0;
  virtual void reduce_attention(GraphArenaView, int, GraphKey, cudaStream_t) = 0;
  virtual void stage_moe(GraphArenaView, int, GraphKey, cudaStream_t) = 0;
  virtual void reduce_moe(GraphArenaView, int, GraphKey, cudaStream_t) = 0;
  virtual const std::int32_t* router_expert_ids(int) const noexcept = 0;
  virtual void advance(GraphArenaView, StateArena&, int, GraphKey,
                       const std::int32_t*, cudaStream_t) = 0; };
class NativeExecutorError : public std::runtime_error { public:
  using std::runtime_error::runtime_error; };
class NativeExecutor final : public decode::AcceptedStateParticipant {
 public:
  NativeExecutor(GraphKey, MtpGraphRuntime&, MtpMiddleStagePort&,
                 WinnerExchangePort&, StateArena&, TelemetrySink&, cudaStream_t);
  ~NativeExecutor();
  NativeExecutor(const NativeExecutor&) = delete;
  NativeExecutor& operator=(const NativeExecutor&) = delete;
  DeviceDraftView draft(std::uint64_t generation);
  std::size_t state_bytes_per_sequence() const noexcept override {
    return state_.transaction_bytes(); }
  void stage_accept(std::uint64_t, std::byte*, const std::int32_t*,
                    decode::DecoderVerifierShape, cudaStream_t) override;
  void commit(std::uint64_t) noexcept override;
  void export_telemetry_after_fence(std::uint64_t) noexcept;
  void discard(std::uint64_t) noexcept override;
  ExecutorPhase phase() const noexcept { return phase_; }
  GraphKey key() const noexcept { return key_; }
 private:
  GraphKey key_; MtpGraphRuntime& runtime_; MtpMiddleStagePort& middle_;
  WinnerExchangePort& exchange_; StateArena& state_; TelemetrySink& telemetry_;
  cudaStream_t stream_;
  std::array<std::array<cudaEvent_t, static_cast<int>(Phase::kCount) + 1>, 7> events_{};
  std::uint32_t* expert_masks_device_ = nullptr;
  std::array<std::array<std::uint32_t, 8>, 7> expert_masks_host_{};
  ExecutorPhase phase_ = ExecutorPhase::kReady;
  std::uint64_t active_generation_ = 0, pending_generation_ = 0;
  const std::int32_t* verification_tokens_ = nullptr;
};
}  // namespace rocket::qwen38::mtp

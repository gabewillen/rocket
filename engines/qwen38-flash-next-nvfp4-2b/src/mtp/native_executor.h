// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include "attention/qsa_mtp_state_view.h"
#include "decode/decoder_verifier.h"
#include "moe/route_compaction.h"
#include "mtp/graph_runtime.h"
#include "mtp/state_arena.h"

namespace rocket::qwen38::mtp {
inline constexpr int kMaxDepth = 7, kMaxSequences = 16;
enum class Phase : std::uint8_t { kInputFusion, kAttention, kAttentionReduce,
  kRoutedAndSharedMoe, kMoeReduce, kFinalHyperconnection, kLogits,
  kProposalSample, kCount };
enum class Outcome : std::uint8_t { kOk, kContractError, kCudaError };
enum class ExecutorPhase : std::uint8_t { kReady, kDrafted, kFaulted };
// Complete immutable identity for a captured MTP graph. query_tokens is the
// fixed prefill shape that produced the QSA cache consumed by this graph.
// Generation is transaction state owned by NativeExecutor, never a cache key.
struct GraphKey { int depth; int sequences; int query_tokens; };
constexpr bool allowed_graph_key(GraphKey k) noexcept {
  const bool b = k.sequences == 1 || k.sequences == 2 || k.sequences == 4 ||
                 k.sequences == 8 || k.sequences == 16;
  return k.depth >= 1 && k.depth <= 7 && b &&
         attention::allowed_mtp_qsa_query_tokens(k.query_tokens) &&
         (k.depth <= 4 || k.sequences <= 4);
}
struct PhaseMetric { Phase phase; Outcome outcome; int depth; int sequences;
  std::uint64_t duration_ns; };
struct ExpertUsageMetric { Outcome outcome; int depth; int sequences;
  int draft_step; int unique_local_experts; std::uint64_t resident_expert_bytes; };
class TelemetrySink : public moe::RouteCompactionOtelSink { public:
  ~TelemetrySink() override = default;
  virtual void record_phase(const PhaseMetric&) noexcept = 0;
  virtual void record_expert_usage(const ExpertUsageMetric&) noexcept = 0; };
struct DeviceDraftView { const std::int32_t* verification_tokens = nullptr;
  int depth = 0; int sequences = 0; std::uint64_t generation = 0; };

// Borrowed output of one router stage. The producer writes source_generation
// on the executor stream. NativeExecutor supplies the requested-generation
// pointer and summary, so the middle stage cannot publish either identity.
struct MtpRouterOutput {
  const std::int32_t* global_expert_ids = nullptr;
  const float* routing_weights = nullptr;
  const std::uint64_t* source_generation = nullptr;
  moe::RouteCompactionCapacity capacity{};
  moe::RouteCompactionBuffers compacted{};
};

// Required native QSA/MoE and PairReduce boundary. Calls are single-writer and
// enqueue only on the borrowed caller stream. QSA receives a borrowed write
// view which aliases StateArena for the duration of the call. Implementations
// may write fixed runtime/StateArena storage, never active accepted state.
// stage_router publishes borrowed route storage plus its device generation;
// stage_moe consumes the compacted buffers later on the same stream. A throw or
// incomplete binding faults the enclosing transaction before publication.
class MtpMiddleStagePort { public: virtual ~MtpMiddleStagePort() = default;
  virtual const std::int32_t* prepare(GraphArenaView, StateArena&, GraphKey,
                                      cudaStream_t) = 0;
  virtual void reduce_input(GraphArenaView, int, GraphKey, cudaStream_t) = 0;
  virtual void stage_attention(GraphArenaView, attention::MtpQsaWriteView, int,
                               GraphKey, cudaStream_t) = 0;
  virtual void reduce_attention(GraphArenaView, int, GraphKey, cudaStream_t) = 0;
  virtual MtpRouterOutput stage_router(GraphArenaView, int, GraphKey,
                                       std::uint64_t, cudaStream_t) = 0;
  virtual void stage_moe(GraphArenaView, const moe::RouteCompactionBuffers&,
                         int, GraphKey, cudaStream_t) = 0;
  virtual void reduce_moe(GraphArenaView, int, GraphKey, cudaStream_t) = 0;
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
  void validate_after_fence(std::uint64_t generation) override;
  void commit(std::uint64_t) noexcept override;
  void export_telemetry_after_fence(std::uint64_t) noexcept;
  void discard(std::uint64_t) noexcept override;
  ExecutorPhase phase() const noexcept { return phase_; }
 GraphKey key() const noexcept { return key_; }
 private:
  struct CudaDeleter {
    void operator()(void* pointer) const noexcept;
  };
  template <typename T>
  using DeviceOwner = std::unique_ptr<T, CudaDeleter>;
  GraphKey key_; MtpGraphRuntime& runtime_; MtpMiddleStagePort& middle_;
  WinnerExchangePort& exchange_; StateArena& state_; TelemetrySink& telemetry_;
  cudaStream_t stream_;
  std::array<std::array<cudaEvent_t, static_cast<int>(Phase::kCount) + 1>, 7> events_{};
  DeviceOwner<std::uint64_t> requested_generation_device_;
  DeviceOwner<moe::RouteCompactionDeviceSummary> route_summaries_device_;
  std::array<moe::RouteCompactionDeviceSummary, kMaxDepth>
      route_summaries_host_{};
  ExecutorPhase phase_ = ExecutorPhase::kReady;
  std::uint64_t active_generation_ = 0, pending_generation_ = 0;
  std::uint64_t routes_validated_generation_ = 0;
  const std::int32_t* verification_tokens_ = nullptr;
};
}  // namespace rocket::qwen38::mtp

// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "decode/decoder_verifier.h"
#include "decode/target_k0_execution_progress.h"
#include "decode/target_k0_pair_reduce.h"

namespace rocket::qwen38::decode {

inline constexpr std::string_view kTargetK0OracleManifestSha256 =
    "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b";
inline constexpr std::string_view kTargetK0ShortOracleManifestSha256 =
    "04c5d4310f68aee54106940b916e8214d2836f8aa9c75a15c23b3960225c0465";
inline constexpr int kTargetK0HiddenStreams = 4;
inline constexpr int kTargetK0Hidden = 2'560;
inline constexpr int kTargetK0HyperHidden =
    kTargetK0HiddenStreams * kTargetK0Hidden;
inline constexpr int kTargetK0LocalVocab = 124'160;

enum class TargetK0AttentionKind : std::uint8_t { kGdn, kQsa };
enum class TargetK0Boundary : std::uint8_t {
  kEmbedding,
  kLayer,
  kFinalNorm,
  kLocalLogits,
  kToken,
};
enum class TargetK0ExecutionDomain : std::uint8_t {
  kChunkPrefill,
  kPackedDecodeRows,
};
enum class TargetK0ExecutorPhase : std::uint8_t {
  kReady,
  kActive,
  kCompleted,
  kFaulted,
};

struct TargetK0ExecutorArena {
  __nv_bfloat16* hidden_a = nullptr;  // [4,2560]
  __nv_bfloat16* hidden_b = nullptr;  // [4,2560]
};

struct TargetK0TokenOutput {
  // Post-final-mixer collapsed BF16 [2560], named `final_norm` by the oracle
  // and `token_hidden` by hyperconnection::FinalPlan.
  const __nv_bfloat16* final_hidden_bf16 = nullptr;
  const float* local_logits = nullptr;          // [124160]
  std::int32_t global_token = -1;
};

[[nodiscard]] constexpr bool accepted_target_k0_oracle(
    std::string_view manifest_sha256, int rows) noexcept {
  return (manifest_sha256 == kTargetK0OracleManifestSha256 && rows == 35) ||
         (manifest_sha256 == kTargetK0ShortOracleManifestSha256 && rows == 87);
}

class TargetK0LayerPort {
 public:
  virtual ~TargetK0LayerPort() = default;
  virtual int rank() const noexcept = 0;
  virtual int layer() const noexcept = 0;
  virtual TargetK0AttentionKind attention_kind() const noexcept = 0;
  virtual bool authenticated() const noexcept = 0;
  virtual const HiddenPartialReducer* attention_reducer_identity()
      const noexcept = 0;
  virtual const HiddenPartialReducer* moe_reducer_identity() const noexcept = 0;
  virtual void wait_source(cudaStream_t stream) = 0;
  virtual void execute_row(std::uint64_t generation,
                           const __nv_bfloat16* replicated_pre_layer,
                           __nv_bfloat16* replicated_post_layer,
                           cudaStream_t stream,
                           TargetK0ExecutionProgress* progress = nullptr) = 0;
};

class TargetK0TokenIoPort {
 public:
  virtual ~TargetK0TokenIoPort() = default;
  virtual int rank() const noexcept = 0;
  virtual bool authenticated() const noexcept = 0;
  virtual void wait_source(cudaStream_t stream) = 0;
  // Performs rank-local lookup, its dedicated TP2 embedding reduction, BF16
  // rounding, and four-stream replication into replicated_hidden.
  virtual void embed_row(std::int32_t token, std::uint64_t generation,
                         __nv_bfloat16* replicated_hidden,
                         cudaStream_t stream) = 0;
  // Performs final HC collapse, final norm, local lm_head/argmax, the distinct
  // TP2 winner exchange, and global greedy. The returned views remain valid
  // until the next call or owner destruction.
  virtual TargetK0TokenOutput finish_prefill(
      const __nv_bfloat16* replicated_post_layer,
      std::uint64_t generation, cudaStream_t stream,
      TargetK0ExecutionProgress* progress = nullptr) = 0;
};

class TargetK0OracleComparator {
 public:
  virtual ~TargetK0OracleComparator() = default;
  virtual int rank() const noexcept = 0;
  virtual int rows() const noexcept = 0;
  virtual std::string_view manifest_sha256() const noexcept = 0;
  virtual bool authenticated() const noexcept = 0;
  virtual std::int32_t expected_input_token(int row) const = 0;
  virtual bool supports_strict_comparison(
      TargetK0Boundary boundary,
      TargetK0ExecutionDomain execution_domain) const noexcept = 0;
  // Synchronous comparison boundary. The comparator owns any D2H staging and
  // fence. `layer` is 0..47 only for kLayer and -1 otherwise.
  virtual void compare(TargetK0Boundary boundary, int row, int layer,
                       const void* device_values, std::size_t elements,
                       cudaStream_t stream) = 0;
  virtual void compare_token(std::int32_t token) = 0;
};

struct TargetK0ExecutionResult {
  std::int32_t token;
  int rows;
  std::uint64_t final_generation;
};

// Concrete one-rank K0 production execution order. Layer ports own all native
// attention/HC/MoE state. The executor owns no weights and accepts no raw
// weight pointers. Every layer port must expose the exact stage-fixed reducers
// minted by `reductions`; cross-wired or duplicate reducers are rejected at
// construction. The peer rank runs the same rows/generations concurrently.
class TargetK0Executor final {
 public:
  TargetK0Executor(int rank,
                   std::array<TargetK0LayerPort*, kDecoderLayers> layers,
                   TargetK0TokenIoPort& token_io,
                   TargetK0PairReduceSchedule& reductions,
                   TargetK0OracleComparator& comparator,
                   pair_reduce::OtelStageSink& telemetry,
                   TargetK0ExecutorArena arena, cudaStream_t stream);

  TargetK0ExecutionResult execute_prefill(
      std::uint64_t first_generation,
      std::span<const std::int32_t> prompt_tokens,
      std::string_view trace_id, std::string_view request_id,
      TargetK0ExecutionProgress* progress = nullptr);

  [[nodiscard]] TargetK0ExecutorPhase phase() const noexcept { return phase_; }

 private:
  void emit(pair_reduce::Outcome outcome, std::string_view trace_id,
            std::string_view request_id, std::uint64_t bytes) noexcept;

  int rank_;
  std::array<TargetK0LayerPort*, kDecoderLayers> layers_;
  TargetK0TokenIoPort& token_io_;
  TargetK0PairReduceSchedule& reductions_;
  TargetK0OracleComparator& comparator_;
  pair_reduce::OtelStageSink& telemetry_;
  TargetK0ExecutorArena arena_;
  cudaStream_t stream_;
  TargetK0ExecutorPhase phase_ = TargetK0ExecutorPhase::kReady;
};

}  // namespace rocket::qwen38::decode

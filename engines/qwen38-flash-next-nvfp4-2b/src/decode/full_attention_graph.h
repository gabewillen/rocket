// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "decode/full_attention_layer.h"

namespace rocket::qwen38::decode {

constexpr int kFullAttentionLayerCount = 12;
constexpr std::array<int, kFullAttentionLayerCount> kFullAttentionLayers = {
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47};
constexpr int kFullAttentionMaxSequences = 16;
constexpr int kFullAttentionMaxVerifyWidth = 8;
constexpr int kFullAttentionMaxTokenRows = 128;

constexpr std::uint64_t kFullAttentionMainStateBytes = 2147483648ULL;
constexpr std::uint64_t kFullAttentionRawStateBytes = 35840ULL;
constexpr std::uint64_t kFullAttentionCompressedStateBytes = 268435456ULL;

bool is_full_attention_layer(int layer) noexcept;

class FullAttentionGraphContractError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct FullAttentionIdentity {
  std::string revision;
  std::string artifact_key;
  std::string weight_chunk_sha256;
  std::string peer_index_chunk_sha256;
  std::string state_commit_sha256;
  int rank = -1;
  int layer = -1;
  std::uint64_t state_generation = 0;
};

struct FullAttentionDeviceExtent {
  void* pointer = nullptr;
  std::uint64_t bytes = 0;
};

enum class FullAttentionGraphStage : std::uint8_t {
  kValidate,
  kStage,
  kAccept,
  kReset,
};

enum class FullAttentionGraphOutcome : std::uint8_t { kOk, kError };

struct FullAttentionGraphOtelRecord {
  int rank;
  int layer;
  int m;
  FullAttentionGraphStage stage;
  FullAttentionGraphOutcome outcome;
};

class FullAttentionGraphOtelSink {
 public:
  virtual ~FullAttentionGraphOtelSink() = default;
  virtual void emit(const FullAttentionGraphOtelRecord& record) noexcept = 0;
};

// The native program owns captured CUDA graphs and private per-step state
// deltas. It may read active state while staging, but cannot modify the three
// active state extents until accept() is called. Every callback is synchronous
// with respect to contract validation; accept/reset must also fence their CUDA
// writes before returning.
struct FullAttentionCudaProgram {
  void* context = nullptr;
  int (*stage)(void*, const __nv_bfloat16*, FullAttentionLaunchShape,
               cudaStream_t) = nullptr;
  int (*accept)(void*, const std::int32_t*, int, std::uint64_t,
                cudaStream_t) = nullptr;
  int (*reset)(void*, cudaStream_t) = nullptr;
  const __nv_bfloat16* (*projected_output)(void*) = nullptr;
  const char* (*last_error)(void*) = nullptr;
};

struct FullAttentionDeviceBindings {
  FullAttentionDeviceExtent q_weight;
  FullAttentionDeviceExtent q_scale;
  FullAttentionDeviceExtent k_weight;
  FullAttentionDeviceExtent k_scale;
  FullAttentionDeviceExtent v_weight;
  FullAttentionDeviceExtent v_scale;
  FullAttentionDeviceExtent o_weight;
  FullAttentionDeviceExtent o_scale;
  FullAttentionDeviceExtent q_norm;
  FullAttentionDeviceExtent k_norm;
  FullAttentionDeviceExtent index_qk_weight_first;
  FullAttentionDeviceExtent index_qk_weight_second;
  FullAttentionDeviceExtent index_q_norm;
  FullAttentionDeviceExtent index_k_norm;

  FullAttentionDeviceExtent main_state;
  FullAttentionDeviceExtent raw_state;
  FullAttentionDeviceExtent compressed_state;
  FullAttentionDeviceExtent projected_output;

  FullAttentionDeviceExtent rope_positions;
  FullAttentionDeviceExtent logical_positions;
  FullAttentionDeviceExtent sequence_lengths;
  FullAttentionDeviceExtent token_to_request;
  FullAttentionDeviceExtent query_start_offsets;

  const std::uint64_t* active_state_generation = nullptr;
  FullAttentionCudaProgram program;
};

// Concrete fail-closed owner for one rank-local layer graph. Stage output can
// feed PairReduce immediately, while QSA/main-cache writes remain private.
// accept() publishes only per-sequence accepted prefixes. reset() discards all
// speculative rows. Any callback failure after stage begins is terminal.
class RankLocalFullAttentionGraph final : public FullAttentionGraph {
 public:
  RankLocalFullAttentionGraph(FullAttentionIdentity identity,
                              FullAttentionDeviceBindings bindings,
                              FullAttentionGraphOtelSink* telemetry);

  int rank() const noexcept override { return identity_.rank; }
  int layer() const noexcept override { return identity_.layer; }
  void launch(const __nv_bfloat16* block_input,
              FullAttentionLaunchShape shape,
              cudaStream_t stream) override;
  const __nv_bfloat16* projected_output() const noexcept override;

  void accept(const std::int32_t* accepted_lengths, int count,
              std::uint64_t next_state_generation, cudaStream_t stream);
  void reset(cudaStream_t stream);
  void fault() noexcept override { faulted_ = true; }

  bool faulted() const noexcept { return faulted_; }
  bool staged() const noexcept { return staged_; }
  std::uint64_t state_generation() const noexcept { return state_generation_; }

 private:
  void emit(FullAttentionGraphStage stage, FullAttentionGraphOutcome outcome,
            int m) noexcept;
  [[noreturn]] void fail_callback(const char* operation);

  FullAttentionIdentity identity_;
  FullAttentionDeviceBindings bindings_;
  FullAttentionGraphOtelSink* telemetry_;
  FullAttentionLaunchShape staged_shape_{};
  cudaStream_t staged_stream_ = nullptr;
  std::uint64_t state_generation_ = 0;
  bool staged_ = false;
  bool faulted_ = false;
};

}  // namespace rocket::qwen38::decode

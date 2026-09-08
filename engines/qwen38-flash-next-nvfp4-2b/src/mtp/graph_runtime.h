// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "mtp/input_fusion.h"
#include "output/token_output.h"

namespace rocket::qwen38::mtp {

struct TensorExtent {
  std::uint64_t offset_bytes;
  std::uint64_t length_bytes;
};

struct NonexpertLayout {
  TensorExtent pre_fc_norm_embedding;
  TensorExtent pre_fc_norm_hidden;
  TensorExtent fc_embedding;
  TensorExtent fc_hidden;
  TensorExtent final_hc_norm;
  TensorExtent final_hc_down;
  TensorExtent final_hc_up;
};

struct GraphRuntimeBinding {
  int device;
  int rank;
  const std::byte* target_rank_slab;
  std::size_t target_rank_slab_bytes;
  const std::byte* mtp_rank_slab;
  std::size_t mtp_rank_slab_bytes;
  std::array<std::uint8_t, 32> source_contract_digest;
  NonexpertLayout layout;
};

struct GraphArenaView {
  __nv_bfloat16* embedding;
  __nv_bfloat16* multi_hidden;
  __nv_bfloat16* embedding_partial;
  __nv_bfloat16* hidden_partial;
  float* reduced_embedding;
  float* reduced_hidden;
  __nv_bfloat16* fused_multi_hidden;
  float* reduced_moe_output;
  __nv_bfloat16* final_injection;
  __nv_bfloat16* updated_multi_hidden;
  __nv_bfloat16* token_hidden;
  float* rank_logits;
  output::Winner* local_winners;
  output::Winner* rank_winners;
  std::int32_t* proposal_tokens;
};

// A fabric implementation must enqueue a same-stream device-to-device
// exchange and write [m,2] in rank0,rank1 order. It may not synchronize the
// stream or expose host winner values. Throwing leaves proposal_tokens
// unpublished and faults the enclosing decoder transaction.
class WinnerExchangePort {
 public:
  virtual ~WinnerExchangePort() = default;
  virtual void enqueue(const output::Winner* local,
                       output::Winner* rank_ordered, int m, int rank,
                       cudaStream_t stream) = 0;
  // Called after DecoderStepRuntime's sole terminal fence and before the
  // common transaction publishes. Async fabric failure must throw here.
  virtual void validate_after_fence() = 0;
};

// Owns fixed c16 storage and immutable CUDA graphs for the native non-QSA MTP
// regions. Python selects one bucket and makes one launch per explicit native
// boundary. It never supplies raw graph handles.
class MtpGraphRuntime final {
 public:
  explicit MtpGraphRuntime(GraphRuntimeBinding binding);
  ~MtpGraphRuntime();
  MtpGraphRuntime(const MtpGraphRuntime&) = delete;
  MtpGraphRuntime& operator=(const MtpGraphRuntime&) = delete;

  [[nodiscard]] GraphArenaView arena() const noexcept;
  [[nodiscard]] int rank() const noexcept;
  void launch_input_local(int m, cudaStream_t stream);
  void launch_input_finish(int m, cudaStream_t stream);
  void launch_final_local(int m, cudaStream_t stream);
  void launch_logits_local(int m, cudaStream_t stream);
  const std::int32_t* enqueue_winner_exchange_and_greedy(
      WinnerExchangePort& exchange, int m, cudaStream_t stream);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace rocket::qwen38::mtp

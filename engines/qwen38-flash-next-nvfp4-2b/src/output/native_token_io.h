// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include "decode/target_k0_executor.h"
#include "decode/target_k0_startup_progress.h"
#include "model/target_slab_owner.h"
#include "mtp/graph_runtime.h"
#include "pair_reduce/pair_reduce.h"

namespace rocket::qwen38::output {

inline constexpr std::string_view kTokenIoOracleManifestSha256 =
    "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b";
inline constexpr std::string_view kTokenIoTokenizerClass = "Qwen2Tokenizer";

struct TokenIoArtifactRoots {
  std::filesystem::path tokenizer;
  std::filesystem::path oracle_capture;
  std::string tokenizer_identity_sha256;
  std::string oracle_manifest_sha256;
};

// Startup filesystem boundary. It hashes the four pinned tokenizer files and
// the exact K0 oracle manifest plus token-I/O boundary artifacts. Inputs are
// borrowed paths. The returned paths name the authenticated roots; consumers
// must retain this value and must not reopen a different root under its
// identity. Validation or I/O failure throws std::invalid_argument.
TokenIoArtifactRoots authenticate_token_io_artifact_roots(
    const std::filesystem::path& tokenizer,
    const std::filesystem::path& oracle_capture);

struct TokenIoArenaView {
  const __nv_bfloat16* final_hidden = nullptr;  // [2560], final_norm oracle
  const float* local_logits = nullptr;          // [124160]
  const Winner* local_winner = nullptr;         // [1], device-only
  const Winner* rank_winners = nullptr;         // [2], rank0 then rank1
  const std::int32_t* global_token = nullptr;   // [1], device-only
};

// Single-writer production owner for K0 c1 token ingress and egress. The
// accepted-loader handle and Transport are borrowed for this owner's lifetime;
// the owner retains the process-lifetime slab lease. Construction re-probes the
// CUDA allocation and ready event, rejects non-factory/fake leases, allocates
// fixed c1 storage, and launches no kernel. Runtime calls borrow all device
// pointers through stream completion and are not thread-safe or reentrant.
class NativeTokenIoOwner final : public decode::TargetK0TokenIoPort {
 public:
  static std::unique_ptr<NativeTokenIoOwner> create(
      int device, int rank, void* accepted_loader_lease_handle,
      pair_reduce::Transport& embedding_transport,
      mtp::WinnerExchangePort& winner_exchange,
      pair_reduce::OtelStageSink& telemetry,
      TokenIoArtifactRoots roots,
      decode::TargetK0StartupConstructionStage* construction_progress =
          nullptr);
  ~NativeTokenIoOwner();
  NativeTokenIoOwner(const NativeTokenIoOwner&) = delete;
  NativeTokenIoOwner& operator=(const NativeTokenIoOwner&) = delete;

  [[nodiscard]] int rank() const noexcept override;
  [[nodiscard]] bool authenticated() const noexcept override;
  [[nodiscard]] TokenIoArenaView arena() const noexcept;

  // Enqueues an ordering dependency on the authenticated slab publication.
  void wait_source(cudaStream_t stream) override;

  // Performs local shard lookup, a dedicated TP2 reduction, BF16 rounding,
  // and exact four-stream replication. The host token is validated before H2D.
  void embed_row(std::int32_t token, std::uint64_t generation,
                 __nv_bfloat16* replicated_hidden,
                 cudaStream_t stream) override;

  // The input is the materialized post-layer47 [4,2560] state. This method
  // performs the final HC grouped norm/mix/collapse, rank-local lm_head and
  // argmax, device winner exchange, global greedy, and one terminal D2H fence.
  // Returned device views remain valid until the next call or destruction.
  decode::TargetK0TokenOutput finish_prefill(
      const __nv_bfloat16* replicated_post_layer,
      std::uint64_t generation, cudaStream_t stream,
      decode::TargetK0ExecutionProgress* progress = nullptr) override;

 private:
  struct Impl;
  explicit NativeTokenIoOwner(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rocket::qwen38::output

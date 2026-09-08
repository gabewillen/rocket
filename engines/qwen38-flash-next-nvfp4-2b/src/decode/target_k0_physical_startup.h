// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_k0_physical_layer_owners.h"
#include "decode/target_k0_startup.h"
#include "mtp/graph_runtime.h"

#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

namespace rocket::qwen38::decode {

enum class TargetK0PhysicalStartupStage : std::uint8_t {
  kValidation,
  kLayerPairReduceBootstrap,
  kEmbeddingPairReduceBootstrap,
  kTokenIoConstruction,
  kPhysicalLayerConstruction,
  kComparatorStartupConstruction,
};

struct TargetK0PhysicalStartupConfig {
  int device = -1;
  int rank = -1;
  void* accepted_loader_lease_handle = nullptr;
  std::filesystem::path descriptor_directory;
  std::filesystem::path qsa_sidecar_payload;
  output::TokenIoArtifactRoots token_io_roots;
  TargetK0PairReduceBootstrap layer_reductions;
  pair_reduce::RdmaConfig embedding_reduction;
};

// Pure contract gate. This performs no CUDA, filesystem, or network work.
// The two transports must use distinct ports and distinct authenticated
// session identities, with each identity identical on both ranks. The injected
// winner exchange owns its
// communicator and must outlive Token I/O.
void validate_target_k0_physical_startup_config(
    const TargetK0PhysicalStartupConfig& config);

// Oracle35-only one-rank production ownership root. Construction consumes the
// accepted-loader capability and creates two independent TP2 transport
// sessions. Both ranks must construct concurrently in the same order: layer
// reductions first, embedding reduction second. The winner-exchange owner is
// injected by the dedicated NCCL bootstrap component.
class TargetK0PhysicalStartupOwner final {
 public:
  static std::unique_ptr<TargetK0PhysicalStartupOwner> create(
      TargetK0PhysicalStartupConfig config,
      std::unique_ptr<mtp::WinnerExchangePort> winner_exchange,
      std::shared_ptr<pair_reduce::OtelStageSink> lifecycle_telemetry,
      std::shared_ptr<moe::TargetFullMoeOtelSink> moe_telemetry,
      std::shared_ptr<moe::TargetMoeStageOtelSink> stage_telemetry,
      std::shared_ptr<attention::TargetK0OracleQsaStateOtelSink>
          state_telemetry,
      TargetK0PhysicalStartupStage* failure_stage = nullptr,
      TargetK0PhysicalLayerConstructionProgress* layer_progress = nullptr,
      TargetK0StartupConstructionStage* construction_progress = nullptr);
  ~TargetK0PhysicalStartupOwner();

  TargetK0PhysicalStartupOwner(const TargetK0PhysicalStartupOwner&) = delete;
  TargetK0PhysicalStartupOwner& operator=(
      const TargetK0PhysicalStartupOwner&) = delete;

  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] bool authenticated() const noexcept { return authenticated_; }
  TargetK0GeneratedToken execute_oracle35(std::uint64_t first_generation,
                                           std::string_view trace_id,
                                           std::string_view request_id,
                                           TargetK0ExecutionProgress* progress =
                                               nullptr);

 private:
  TargetK0PhysicalStartupOwner() = default;
  int device_ = -1;
  int rank_ = -1;
  bool authenticated_ = false;
  std::vector<std::int32_t> prompt_tokens_;

  // Reverse destruction is intentional. startup_ goes first. Its borrowed
  // stream, arenas, telemetry, embedding transport, and winner exchange remain
  // alive until every child has been destroyed.
  std::shared_ptr<pair_reduce::OtelStageSink> lifecycle_telemetry_;
  std::shared_ptr<moe::TargetFullMoeOtelSink> moe_telemetry_;
  std::shared_ptr<moe::TargetMoeStageOtelSink> stage_telemetry_;
  std::shared_ptr<attention::TargetK0OracleQsaStateOtelSink> state_telemetry_;
  cudaStream_t stream_ = nullptr;
  __nv_bfloat16* hidden_a_ = nullptr;
  __nv_bfloat16* hidden_b_ = nullptr;
  std::unique_ptr<pair_reduce::RdmaTransport> embedding_transport_;
  std::unique_ptr<mtp::WinnerExchangePort> winner_exchange_;
  std::unique_ptr<TargetK0StartupOwner> startup_;
};

}  // namespace rocket::qwen38::decode

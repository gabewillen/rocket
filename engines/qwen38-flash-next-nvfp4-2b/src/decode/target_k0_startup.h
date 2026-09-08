// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <span>
#include <string>

#include "decode/target_k0_physical_layers.h"
#include "decode/target_k0_oracle_comparator.h"
#include "output/native_token_io.h"

namespace rocket::qwen38::decode {

struct TargetK0GeneratedToken {
  TargetK0ExecutionResult execution;
  std::string text;
};

std::string detokenize_target_k0_token(
    const output::TokenIoArtifactRoots& tokenizer, std::int32_t token);

// Value-owned production root. Member order preserves the borrowed reducer and
// slab lifetimes: destruction runs executor, inventory, token I/O, comparator,
// then PairReduce. Telemetry and the CUDA stream remain caller-owned.
class TargetK0StartupOwner final {
 public:
  static std::unique_ptr<TargetK0StartupOwner> create(
      int rank, std::unique_ptr<TargetK0PairReduceOwner> reductions,
      std::unique_ptr<TargetK0OracleComparator> comparator,
      std::unique_ptr<TargetK0TokenIoPort> token_io,
      std::unique_ptr<TargetK0PhysicalLayers> layers,
      pair_reduce::OtelStageSink& telemetry,
      output::TokenIoArtifactRoots tokenizer,
      TargetK0ExecutorArena arena, cudaStream_t stream);

  TargetK0GeneratedToken execute_prefill(
      std::uint64_t first_generation,
      std::span<const std::int32_t> prompt_tokens,
      std::string_view trace_id, std::string_view request_id,
      TargetK0ExecutionProgress* progress = nullptr);

 private:
  TargetK0StartupOwner() = default;
  std::unique_ptr<TargetK0PairReduceOwner> reductions_;
  std::unique_ptr<TargetK0OracleComparator> comparator_;
  std::unique_ptr<TargetK0TokenIoPort> token_io_;
  output::TokenIoArtifactRoots tokenizer_;
  std::unique_ptr<TargetK0PhysicalLayers> layers_;
  std::unique_ptr<TargetK0Executor> executor_;
};

}  // namespace rocket::qwen38::decode

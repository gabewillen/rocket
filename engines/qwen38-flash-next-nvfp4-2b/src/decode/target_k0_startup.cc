// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_startup.h"

#include <stdexcept>

namespace rocket::qwen38::decode {

std::string detokenize_target_k0_token(
    const output::TokenIoArtifactRoots& tokenizer, std::int32_t token) {
  if (tokenizer.tokenizer.empty() || tokenizer.oracle_capture.empty() ||
      tokenizer.tokenizer_identity_sha256 !=
          "c8b5202c5bd8b6c5f136965a864eabd43105ffa20a5ce30e8765ab16e65f9d34" ||
      tokenizer.oracle_manifest_sha256 != kTargetK0OracleManifestSha256)
    throw std::invalid_argument("K0 tokenizer identity changed");
  // The authenticated tokenizer maps 248046 to the EOS special token
  // `<|im_end|>`. Generation decoding skips special tokens, so its text is
  // deterministically empty for this accepted first-token oracle.
  if (token != 248'046)
    throw std::invalid_argument("K0 first-oracle token changed");
  return {};
}

std::unique_ptr<TargetK0StartupOwner> TargetK0StartupOwner::create(
    int rank, std::unique_ptr<TargetK0PairReduceOwner> reductions,
    std::unique_ptr<TargetK0OracleComparator> comparator,
    std::unique_ptr<TargetK0TokenIoPort> token_io,
    std::unique_ptr<TargetK0PhysicalLayers> layers,
    pair_reduce::OtelStageSink& telemetry,
    output::TokenIoArtifactRoots tokenizer, TargetK0ExecutorArena arena,
    cudaStream_t stream) {
  if ((rank != 0 && rank != 1) || !reductions || !comparator || !token_io ||
      !layers || layers->rank() != rank || !layers->authenticated() ||
      token_io->rank() != rank || !token_io->authenticated() ||
      comparator->rank() != rank || !comparator->authenticated())
    throw std::invalid_argument("K0 startup dependencies are incomplete");
  tokenizer = output::authenticate_token_io_artifact_roots(
      tokenizer.tokenizer, tokenizer.oracle_capture);
  auto owner = std::unique_ptr<TargetK0StartupOwner>(new TargetK0StartupOwner);
  owner->reductions_ = std::move(reductions);
  owner->comparator_ = std::move(comparator);
  owner->token_io_ = std::move(token_io);
  owner->tokenizer_ = std::move(tokenizer);
  owner->layers_ = std::move(layers);
  owner->executor_ = std::make_unique<TargetK0Executor>(
      rank, owner->layers_->inventory().ports(), *owner->token_io_,
      owner->reductions_->schedule(), *owner->comparator_, telemetry, arena,
      stream);
  return owner;
}

TargetK0GeneratedToken TargetK0StartupOwner::execute_prefill(
    std::uint64_t first_generation,
    std::span<const std::int32_t> prompt_tokens,
    std::string_view trace_id, std::string_view request_id,
    TargetK0ExecutionProgress* progress) {
  if (!executor_)
    throw std::logic_error("K0 startup executor was not published");
  const auto execution = executor_->execute_prefill(
      first_generation, prompt_tokens, trace_id, request_id, progress);
  return {execution,
          detokenize_target_k0_token(tokenizer_, execution.token)};
}

}  // namespace rocket::qwen38::decode

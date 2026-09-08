// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string_view>

namespace rocket::qwen38::output {

inline constexpr int kHidden = 2'560;
inline constexpr int kHyperConnections = 4;
inline constexpr int kHyperHidden = kHidden * kHyperConnections;
inline constexpr int kVocab = 248'320;
inline constexpr int kTpSize = 2;
inline constexpr int kLocalVocab = kVocab / kTpSize;
inline constexpr float kRmsEpsilon = 1.0e-6F;
inline constexpr int kBuckets[] = {1, 2, 4, 8, 16};

// Exact rank-target slab descriptors from plan
// a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4.
struct SlabDescriptor {
  std::string_view name;
  std::uint64_t offset_bytes;
  std::uint64_t length_bytes;
};
inline constexpr SlabDescriptor kLmHead{"lm_head.weight", 0, 635'699'200};
inline constexpr SlabDescriptor kEmbedding{
    "model.language_model.embed_tokens.weight", 635'699'200, 635'699'200};
inline constexpr SlabDescriptor kFinalNorm{
    "model.language_model.hyper_connection_mixer.hc_norm.weight",
    1'271'398'400, 20'480};
inline constexpr SlabDescriptor kFinalDown{
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    1'271'418'880, 6'553'600};
inline constexpr SlabDescriptor kFinalUp{
    "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    1'277'972'480, 6'553'600};

struct Winner {
  float value;
  std::int32_t token;
};
static_assert(sizeof(Winner) == 8);

enum class Operation : std::uint8_t {
  kEmbedding,
  kFinalGroupedNorm,
  kLmHead,
  kLocalArgmax,
  kGlobalGreedy,
};
enum class Outcome : std::uint8_t { kOk, kContractError, kCudaError };

struct OtelDimensions {
  std::string_view operation;
  int rank;
  int m_bucket;
  std::string_view mode;
  std::string_view outcome;
};

// Cardinality: 5 operations x 3 rank values (0,1,invalid) x 6 M values
// (five buckets plus invalid) x 1 mode x 3 outcomes = 270 series maximum.
[[nodiscard]] constexpr bool allowed_m(int m) noexcept {
  for (const int bucket : kBuckets)
    if (m == bucket) return true;
  return false;
}
[[nodiscard]] constexpr bool allowed_rank(int rank) noexcept {
  return rank == 0 || rank == 1;
}
[[nodiscard]] constexpr std::string_view operation_name(Operation operation) noexcept {
  switch (operation) {
    case Operation::kEmbedding: return "embedding";
    case Operation::kFinalGroupedNorm: return "final_grouped_norm";
    case Operation::kLmHead: return "lm_head";
    case Operation::kLocalArgmax: return "local_argmax";
    case Operation::kGlobalGreedy: return "global_greedy";
  }
  return "global_greedy";
}
[[nodiscard]] constexpr std::string_view outcome_name(Outcome outcome) noexcept {
  switch (outcome) {
    case Outcome::kOk: return "ok";
    case Outcome::kContractError: return "contract_error";
    case Outcome::kCudaError: return "cuda_error";
  }
  return "contract_error";
}
[[nodiscard]] constexpr OtelDimensions otel_dimensions(
    Operation operation, int rank, int m, Outcome outcome) noexcept {
  return {operation_name(operation), allowed_rank(rank) ? rank : -1,
          allowed_m(m) ? m : 0, "greedy", outcome_name(outcome)};
}

// The pinned production path is greedy. Temperature/top-p is rejected until
// the engine owns a counter-based RNG and distributed probability contract.
[[nodiscard]] constexpr bool sampling_supported(float temperature,
                                                float top_p) noexcept {
  return temperature == 0.0F && top_p == 1.0F;
}

// All buffers are borrowed device memory valid through stream completion.
// invalid_token is a caller-owned device int initialized to zero before launch;
// an out-of-range token sets it to one and emits a zero vector without OOB I/O.
[[nodiscard]] cudaError_t embedding_lookup_rank(
    const std::int32_t* token_ids, const __nv_bfloat16* rank_weight,
    __nv_bfloat16* rank_output, std::int32_t* invalid_token, int m, int rank,
    cudaStream_t stream = nullptr) noexcept;

// Qwen final HC normalization applies Gemma's (1 + weight) affine separately
// to four contiguous 2560-wide streams. It does not include the final mixer's
// two skinny GEMMs or gated stream reduction.
[[nodiscard]] cudaError_t final_grouped_rms_norm(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int m, cudaStream_t stream = nullptr) noexcept;

// Produces FP32 rank-local logits [m,124160]. M=1 uses the fixed GB10 vocab
// GEMV path adapted from myllmbox/b12x's BLOCK_K=1024, eight-warp design;
// larger graph buckets use cuBLAS GEMM so the head weights are reused across
// rows. The handle must be initialized and exclusively owned by this stream.
[[nodiscard]] cublasStatus_t lm_head(
    cublasHandle_t handle, const __nv_bfloat16* hidden,
    const __nv_bfloat16* rank_weight, float* rank_logits, int m, int rank,
    cudaStream_t stream = nullptr) noexcept;

// Produces one finite local winner per row or token=-1 when no finite logit
// exists. Equal values choose the lower global token id.
[[nodiscard]] cudaError_t local_argmax(const float* rank_logits,
                                       Winner* winners, int m, int rank,
                                       cudaStream_t stream = nullptr) noexcept;

// rank_winners is [m,2], ordered rank0 then rank1 within each row. An invalid
// pair writes token=-1; valid equal values choose the lower global token id.
[[nodiscard]] cudaError_t global_greedy(const Winner* rank_winners,
                                        std::int32_t* tokens, int m,
                                        float temperature, float top_p,
                                        cudaStream_t stream = nullptr) noexcept;

}  // namespace rocket::qwen38::output

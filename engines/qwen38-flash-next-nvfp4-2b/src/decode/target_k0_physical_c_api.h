// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

extern "C" {

struct Qwen38TargetK0Oracle35Result {
  std::int32_t token;
  std::int32_t rows;
  std::uint64_t final_generation;
  std::uint64_t lifecycle_outcomes[4];
  std::uint64_t moe_components[5];
  std::uint64_t stage_counters[4];
  std::uint64_t state_outcomes[4];
  std::uint64_t nccl_stages[9];
  std::uint64_t nccl_outcomes[7];
  std::uint64_t duration_samples;
  std::uint64_t total_bytes;
};

enum Qwen38TargetK0Oracle35Status : int {
  QWEN38_TARGET_K0_SUCCESS = 0,
  QWEN38_TARGET_K0_VALIDATION = 10,
  QWEN38_TARGET_K0_LAYER_PAIR_REDUCE_BOOTSTRAP = 20,
  QWEN38_TARGET_K0_EMBEDDING_PAIR_REDUCE_BOOTSTRAP = 21,
  QWEN38_TARGET_K0_NCCL_BOOTSTRAP = 22,
  QWEN38_TARGET_K0_TOKEN_IO_CONSTRUCTION = 30,
  QWEN38_TARGET_K0_PHYSICAL_LAYER_CONSTRUCTION = 31,
  QWEN38_TARGET_K0_COMPARATOR_STARTUP_CONSTRUCTION = 32,
  QWEN38_TARGET_K0_SOURCE_WAITS = 40,
  QWEN38_TARGET_K0_PROMPT_EXECUTION = 41,
  QWEN38_TARGET_K0_TERMINAL_FENCE = 42,
  QWEN38_TARGET_K0_ORACLE_COMPARISON = 43,
  QWEN38_TARGET_K0_CLEANUP = 50,
  QWEN38_TARGET_K0_QUARANTINE = 51,
  QWEN38_TARGET_K0_UNKNOWN = 255,
};

// Synchronous two-rank oracle35 entry point. Both processes must enter with
// matching ports and per-session material. The accepted-loader lease and all
// strings are borrowed until return. Session/authentication arrays are exactly
// 32 bytes and are never retained or emitted. Status is a closed value:
// Values are from Qwen38TargetK0Oracle35Status only.
int qwen38_target_k0_oracle35_run(
    int device, int rank, void* accepted_loader_lease_handle,
    const char* descriptor_directory, const char* qsa_sidecar_payload,
    const char* tokenizer_root, const char* oracle_capture,
    const char* bootstrap_host, int layer_port, int embedding_port,
    int nccl_port, std::uint32_t timeout_ms,
    const std::uint8_t layer_session_sha256[32],
    const std::uint8_t embedding_session_sha256[32],
    const std::uint8_t nccl_session_sha256[32],
    const std::uint8_t nccl_authentication_key[32],
    Qwen38TargetK0Oracle35Result* result) noexcept;

}

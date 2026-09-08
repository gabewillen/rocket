// SPDX-License-Identifier: Apache-2.0
#include "output/native_token_io_preflight.h"

#include <cstdint>
#include <exception>
#include <string>

#include "output/native_token_io.h"

namespace {
thread_local std::string last_error;
}

extern "C" int qwen38_token_io_owner_preflight(
    int device, int rank, void* accepted_loader_lease_handle,
    void* embedding_transport, void* winner_exchange, void* telemetry,
    const char* tokenizer_root, const char* oracle_capture,
    Qwen38TokenIoPreflightEvidence* evidence) noexcept {
  last_error.clear();
  if (!accepted_loader_lease_handle || !embedding_transport ||
      !winner_exchange || !telemetry || !tokenizer_root || !oracle_capture ||
      !evidence) {
    last_error = "token I/O preflight argument contract changed";
    return 1;
  }
  *evidence = {};
  try {
    namespace output = rocket::qwen38::output;
    auto roots = output::authenticate_token_io_artifact_roots(tokenizer_root,
                                                               oracle_capture);
    auto owner = output::NativeTokenIoOwner::create(
        device, rank, accepted_loader_lease_handle,
        *static_cast<rocket::qwen38::pair_reduce::Transport*>(
            embedding_transport),
        *static_cast<rocket::qwen38::mtp::WinnerExchangePort*>(winner_exchange),
        *static_cast<rocket::qwen38::pair_reduce::OtelStageSink*>(telemetry),
        std::move(roots));
    const auto arena = owner->arena();
    if (!owner->authenticated() || owner->rank() != rank ||
        !arena.final_hidden || !arena.local_logits || !arena.local_winner ||
        !arena.rank_winners || !arena.global_token)
      throw std::runtime_error("token I/O preflight construction incomplete");
    evidence->fixed_device_bytes =
        sizeof(std::int32_t) * 3 +
        sizeof(__nv_bfloat16) * (output::kHidden + output::kHyperConnections +
                                 output::kHyperHidden + output::kHidden) +
        sizeof(float) * (output::kHidden + output::kHidden +
                         output::kLocalVocab) +
        sizeof(output::Winner) * (1 + output::kTpSize);
    evidence->final_hidden_address =
        reinterpret_cast<std::uintptr_t>(arena.final_hidden);
    evidence->local_logits_address =
        reinterpret_cast<std::uintptr_t>(arena.local_logits);
    evidence->local_winner_address =
        reinterpret_cast<std::uintptr_t>(arena.local_winner);
    return 0;
  } catch (const std::exception& error) {
    last_error = std::string(error.what()).substr(0, 384);
  } catch (...) {
    last_error = "unknown token I/O preflight failure";
  }
  return 1;
}

extern "C" const char* qwen38_token_io_owner_preflight_last_error() noexcept {
  return last_error.c_str();
}

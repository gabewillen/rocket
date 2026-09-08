// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

extern "C" {

struct Qwen38TokenIoPreflightEvidence {
  std::uint64_t fixed_device_bytes;
  std::uint64_t final_hidden_address;
  std::uint64_t local_logits_address;
  std::uint64_t local_winner_address;
};

// Startup-only construction harness. The three opaque interface pointers are
// borrowed C++ pair_reduce::Transport, mtp::WinnerExchangePort, and
// pair_reduce::OtelStageSink objects. The call constructs and destroys the
// production owner without calling wait_source, embed_row, or finish_prefill.
__attribute__((visibility("default"))) int qwen38_token_io_owner_preflight(
    int device, int rank, void* accepted_loader_lease_handle,
    void* embedding_transport, void* winner_exchange, void* telemetry,
    const char* tokenizer_root, const char* oracle_capture,
    Qwen38TokenIoPreflightEvidence* evidence) noexcept;

__attribute__((visibility("default"))) const char*
qwen38_token_io_owner_preflight_last_error() noexcept;

}

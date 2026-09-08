// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_physical_startup.h"

#include <array>
#include <stdexcept>

namespace rocket::qwen38::decode {
namespace {
bool nonzero(const std::array<std::uint8_t, 32>& digest) noexcept {
  for (const auto byte : digest)
    if (byte != 0) return true;
  return false;
}
}  // namespace

void validate_target_k0_physical_startup_config(
    const TargetK0PhysicalStartupConfig& config) {
  if (config.device < 0 || (config.rank != 0 && config.rank != 1) ||
      !config.accepted_loader_lease_handle ||
      config.descriptor_directory.empty() ||
      config.qsa_sidecar_payload.empty() ||
      config.token_io_roots.tokenizer.empty() ||
      config.token_io_roots.oracle_capture.empty())
    throw std::invalid_argument("K0 physical startup identity changed");
  const auto& layer = config.layer_reductions;
  if (layer.rank != config.rank || layer.peer_rank != 1 - config.rank ||
      layer.bootstrap_host.empty() || layer.bootstrap_port <= 0 ||
      layer.bootstrap_port > 65'535 ||
      !pair_reduce::valid_operation_timeout_ms(layer.timeout_ms) ||
      !nonzero(layer.session_sha256))
    throw std::invalid_argument("K0 layer transport identity changed");
  const auto& embedding = config.embedding_reduction;
  if (embedding.rank != config.rank ||
      embedding.bootstrap_host != layer.bootstrap_host ||
      embedding.bootstrap_port <= 0 || embedding.bootstrap_port > 65'535 ||
      embedding.bootstrap_port == layer.bootstrap_port ||
      embedding.operation_timeout_ms != layer.timeout_ms ||
      embedding.session_sha256 == layer.session_sha256 ||
      !nonzero(embedding.session_sha256) || embedding.devices.size() != 2 ||
      embedding.devices[0] != "rocep1s0f1" ||
      embedding.devices[1] != "roceP2p1s0f1" || embedding.gid_index != 3 ||
      embedding.rail_split_bytes != 65'536)
    throw std::invalid_argument("K0 embedding transport identity changed");
}

}  // namespace rocket::qwen38::decode

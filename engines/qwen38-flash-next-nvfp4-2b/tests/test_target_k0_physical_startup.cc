// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_physical_startup.h"

#include <array>
#include <stdexcept>

using namespace rocket::qwen38;

namespace {
decode::TargetK0PhysicalStartupConfig valid() {
  decode::TargetK0PhysicalStartupConfig value;
  value.device = 0;
  value.rank = 0;
  value.accepted_loader_lease_handle = reinterpret_cast<void*>(0x1);
  value.descriptor_directory = "/descriptors";
  value.qsa_sidecar_payload = "/sidecar";
  value.token_io_roots.tokenizer = "/tokenizer";
  value.token_io_roots.oracle_capture = "/oracle";
  value.layer_reductions.rank = 0;
  value.layer_reductions.peer_rank = 1;
  value.layer_reductions.bootstrap_host = "192.0.2.1";
  value.layer_reductions.bootstrap_port = 18838;
  value.layer_reductions.timeout_ms = 120000;
  value.layer_reductions.session_sha256.fill(7);
  value.embedding_reduction.rank = 0;
  value.embedding_reduction.bootstrap_host = "192.0.2.1";
  value.embedding_reduction.bootstrap_port = 18839;
  value.embedding_reduction.operation_timeout_ms = 120000;
  value.embedding_reduction.session_sha256.fill(8);
  return value;
}

template <class Mutate> void rejects(Mutate mutate) {
  auto value = valid();
  mutate(value);
  try {
    decode::validate_target_k0_physical_startup_config(value);
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("invalid physical startup config accepted");
}
}  // namespace

int main() {
  decode::validate_target_k0_physical_startup_config(valid());
  rejects([](auto& v) { v.accepted_loader_lease_handle = nullptr; });
  rejects([](auto& v) { v.rank = 2; });
  rejects([](auto& v) { v.embedding_reduction.rank = 1; });
  rejects([](auto& v) { v.embedding_reduction.bootstrap_port = 18838; });
  rejects([](auto& v) { v.embedding_reduction.session_sha256.fill(0); });
  rejects([](auto& v) {
    v.embedding_reduction.session_sha256 =
        v.layer_reductions.session_sha256;
  });
  rejects([](auto& v) { v.embedding_reduction.devices[0] = "wrong"; });
  rejects([](auto& v) { v.embedding_reduction.gid_index = 2; });
  rejects([](auto& v) { v.embedding_reduction.rail_split_bytes = 4096; });
  rejects([](auto& v) { v.descriptor_directory.clear(); });
  rejects([](auto& v) { v.token_io_roots.oracle_capture.clear(); });
}

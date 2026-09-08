// SPDX-License-Identifier: Apache-2.0
#include "pair_reduce/rdma.h"

#include <cstdio>
#include <stdexcept>

namespace pr = rocket::qwen38::pair_reduce;

namespace {
void check_rejected(const pr::RdmaConfig& local,
                    const pr::RdmaPeerBootstrapIdentity& peer) {
  try {
    pr::validate_peer_bootstrap_identity(local, peer);
  } catch (const pr::RdmaError&) {
    return;
  }
  throw std::runtime_error("bootstrap identity mutation was accepted");
}
}  // namespace

int main() {
  try {
    pr::RdmaConfig local;
    local.rank = 0;
    local.operation_timeout_ms = 120000;
    local.session_sha256.fill(0x5a);
    pr::RdmaPeerBootstrapIdentity peer{
        pr::kRdmaBootstrapSchema, 1, 2, 2, 65536,
        local.operation_timeout_ms, local.session_sha256};
    pr::validate_peer_bootstrap_identity(local, peer);
    auto changed = peer;
    changed.session_sha256[31] ^= 1;
    check_rejected(local, changed);
    changed = peer; changed.rank = 0; check_rejected(local, changed);
    changed = peer; changed.schema += 1; check_rejected(local, changed);
    changed = peer; changed.operation_timeout_ms -= 1; check_rejected(local, changed);
    changed = peer; changed.rails = 1; check_rejected(local, changed);
    changed = peer; changed.page_bytes = 4096; check_rejected(local, changed);
    std::puts("qwen38 RDMA bootstrap: peer frame mutations rejected");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

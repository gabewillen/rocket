// SPDX-License-Identifier: Apache-2.0
#include "output/native_token_io.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace output = rocket::qwen38::output;
namespace decode = rocket::qwen38::decode;
namespace pair_reduce = rocket::qwen38::pair_reduce;
namespace mtp = rocket::qwen38::mtp;

namespace {
class Sink final : public pair_reduce::OtelStageSink {
 public:
  void emit_span_and_log(const pair_reduce::SpanRecord&) noexcept override {}
  void record_duration(const pair_reduce::MetricPoint&) noexcept override {}
};
class Transport final : public pair_reduce::Transport {
 public:
  int rank() const noexcept override { return 0; }
  int world_size() const noexcept override { return 2; }
  int register_region(void*, std::size_t) override { return 0; }
  void unregister_region(int) noexcept override {}
  std::uint64_t next_sequence() override { return 1; }
  void post_unsignaled_write(int, std::size_t, std::size_t,
                             std::size_t) override {}
  void signal_sequence(std::uint64_t) override {}
  void wait_peer(std::uint64_t) override {}
  void flush_signaled() override {}
  void acknowledge_consumed(std::uint64_t) override {}
  void wait_peer_consumed(std::uint64_t) override {}
};
class Exchange final : public mtp::WinnerExchangePort {
 public:
  void enqueue(const output::Winner*, output::Winner*, int, int,
               cudaStream_t) override {}
  void validate_after_fence() override {}
};
}  // namespace

int main(int argc, char** argv) {
  static_assert(std::is_base_of_v<decode::TargetK0TokenIoPort,
                                  output::NativeTokenIoOwner>);
  static_assert(!std::is_copy_constructible_v<output::NativeTokenIoOwner>);
  if (argc == 1) {
    bool rejected = false;
    try {
      (void)output::authenticate_token_io_artifact_roots(
          "/root/that-does-not-exist", "/oracle/that-does-not-exist");
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) return 1;
    std::puts("token I/O missing-root rejection: PASS");
    return 0;
  }
  if (argc != 3) return 2;
  try {
    const auto roots = output::authenticate_token_io_artifact_roots(argv[1],
                                                                     argv[2]);
    if (roots.tokenizer_identity_sha256.size() != 64 ||
        roots.oracle_manifest_sha256 != output::kTokenIoOracleManifestSha256 ||
        roots.tokenizer != std::filesystem::path(argv[1]) ||
        roots.oracle_capture != std::filesystem::path(argv[2]))
      return 3;
    Sink sink;
    Transport transport;
    Exchange exchange;
    bool rejected = false;
    try {
      (void)output::NativeTokenIoOwner::create(
          0, 0, nullptr, transport, exchange, sink, roots);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) return 5;
    auto forged = roots;
    forged.tokenizer_identity_sha256.assign(64, '0');
    rejected = false;
    try {
      (void)output::NativeTokenIoOwner::create(
          0, 0, nullptr, transport, exchange, sink, forged);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) return 6;
    std::puts("token I/O tokenizer/oracle roots: PASS");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 4;
  }
}

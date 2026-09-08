// SPDX-License-Identifier: Apache-2.0
#include "mtp/nccl_communicator_owner.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mtp = rocket::qwen38::mtp;

namespace {
int parse_int(std::string_view value, const char* name) {
  int result = 0;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    throw std::invalid_argument(std::string("invalid ") + name);
  return result;
}

std::array<std::uint8_t, 32> read_hex_fd(int fd, const char* name) {
  struct stat status {};
  if (fd < 0 || ::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_uid != ::geteuid() || (status.st_mode & 077) != 0)
    throw std::invalid_argument(std::string(name) +
                                " fd must be an owner-only regular file");
  std::array<char, 65> text{};
  const ssize_t count = ::pread(fd, text.data(), text.size(), 0);
  if (count != 64 && !(count == 65 && text[64] == '\n'))
    throw std::invalid_argument(std::string(name) +
                                " must contain exactly 64 hex digits");
  std::array<std::uint8_t, 32> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    unsigned int byte = 0;
    const auto parsed = std::from_chars(text.data() + index * 2,
                                        text.data() + index * 2 + 2, byte, 16);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + index * 2 + 2)
      throw std::invalid_argument(std::string(name) + " contains non-hex data");
    result[index] = static_cast<std::uint8_t>(byte);
  }
  return result;
}

class Sink final : public mtp::NcclBootstrapOtelSink {
 public:
  void emit_span_and_log(
      const mtp::NcclBootstrapTelemetryRecord& record) noexcept override {
    std::fprintf(stderr,
                 "{\"signal\":\"span_log\",\"stage\":\"%.*s\","
                 "\"outcome\":\"%.*s\",\"rank\":%d,\"world_size\":%d,"
                 "\"duration_ns\":%llu}\n",
                 static_cast<int>(mtp::stage_name(record.stage).size()),
                 mtp::stage_name(record.stage).data(),
                 static_cast<int>(mtp::outcome_name(record.outcome).size()),
                 mtp::outcome_name(record.outcome).data(), record.rank,
                 record.world_size,
                 static_cast<unsigned long long>(record.duration_ns));
  }
  void record_duration(
      const mtp::NcclBootstrapTelemetryRecord&) noexcept override {}
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 11)
      throw std::invalid_argument(
          "usage: qwen38-nccl-bootstrap-no-kernel RANK HOST NCCL_PORT "
          "PAIR_REDUCE_PORT TIMEOUT_MS SESSION_FD AUTH_KEY_FD NCCL_LIBRARY "
          "CUDA_RUNTIME_LIBRARY DEVICE");
    mtp::NcclCommunicatorConfig config;
    config.rank = parse_int(argv[1], "rank");
    config.peer_rank = 1 - config.rank;
    config.bootstrap_host = argv[2];
    config.bootstrap_port = parse_int(argv[3], "NCCL port");
    config.pair_reduce_bootstrap_port = parse_int(argv[4], "PairReduce port");
    config.timeout_ms =
        static_cast<std::uint32_t>(parse_int(argv[5], "timeout"));
    config.session_sha256 = read_hex_fd(parse_int(argv[6], "session fd"),
                                        "session identity");
    config.authentication_key = read_hex_fd(
        parse_int(argv[7], "authentication key fd"), "authentication key");
    config.nccl_library = argv[8];
    config.cuda_runtime_library = argv[9];
    config.device = parse_int(argv[10], "device");
    Sink sink;
    mtp::NcclCommunicatorOwner owner(config, sink);
    std::printf(
        "{\"result\":\"qwen38-nccl-bootstrap-no-kernel\",\"rank\":%d,"
        "\"world_size\":%d,\"nccl_version\":%d}\n",
        owner.rank(), owner.world_size(), mtp::kPinnedNcclRawVersion);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

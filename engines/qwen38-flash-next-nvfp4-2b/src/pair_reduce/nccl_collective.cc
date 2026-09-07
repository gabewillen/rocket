#include "pair_reduce/nccl_collective.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>

namespace rocket::qwen38::pair_reduce {
namespace {

static_assert(NCCL_VERSION_CODE == 23007,
              "PairReduce requires the measured NCCL 2.30.7 headers");

[[noreturn]] void fail(const std::string& reason) {
  throw PairReduceTransportError("qwen38 PairReduce NCCL: " + reason);
}

void check(ncclResult_t result, const char* operation) {
  if (result != ncclSuccess)
    fail(std::string(operation) + ": " + ncclGetErrorString(result));
}

void write_all(int socket, const void* source, std::size_t bytes) {
  const auto* cursor = static_cast<const std::byte*>(source);
  while (bytes) {
    const ssize_t count = ::write(socket, cursor, bytes);
    if (count < 0) {
      if (errno == EINTR) continue;
      fail(std::string("bootstrap write: ") + std::strerror(errno));
    }
    cursor += count;
    bytes -= static_cast<std::size_t>(count);
  }
}

void read_all(int socket, void* destination, std::size_t bytes) {
  auto* cursor = static_cast<std::byte*>(destination);
  while (bytes) {
    const ssize_t count = ::read(socket, cursor, bytes);
    if (count == 0) fail("bootstrap peer closed");
    if (count < 0) {
      if (errno == EINTR) continue;
      fail(std::string("bootstrap read: ") + std::strerror(errno));
    }
    cursor += count;
    bytes -= static_cast<std::size_t>(count);
  }
}

int bootstrap_socket(const NcclConfig& config) {
  int socket = ::socket(AF_INET, SOCK_STREAM, 0);
  if (socket < 0) fail("bootstrap socket failed");
  timeval timeout{60, 0};
  ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  ::setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  int one = 1;
  ::setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(config.bootstrap_port));
  if (config.rank == 0) {
    address.sin_addr.s_addr = INADDR_ANY;
    ::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (::bind(socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ||
        ::listen(socket, 1)) {
      ::close(socket);
      fail("bootstrap listen failed");
    }
    const int peer = ::accept(socket, nullptr, nullptr);
    ::close(socket);
    if (peer < 0) fail("bootstrap accept failed");
    return peer;
  }
  if (::inet_pton(AF_INET, config.bootstrap_host.c_str(), &address.sin_addr) != 1) {
    ::close(socket);
    fail("invalid bootstrap IPv4 address");
  }
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(60);
  while (::connect(socket, reinterpret_cast<sockaddr*>(&address),
                   sizeof(address))) {
    ::close(socket);
    if (std::chrono::steady_clock::now() >= deadline)
      fail("bootstrap connect timed out");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    socket = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socket < 0) fail("bootstrap socket failed");
    ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    ::setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    ::setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  }
  return socket;
}

}  // namespace

NcclCollective::NcclCollective(const NcclConfig& config) : rank_(config.rank) {
  if ((rank_ != 0 && rank_ != 1) || config.bootstrap_port < 1 ||
      config.bootstrap_port > 65'535 || config.operation_timeout_ms < 100 ||
      config.operation_timeout_ms > 120'000)
    fail("rank, port, or timeout contract rejected");
  int runtime_version = 0;
  check(ncclGetVersion(&runtime_version), "query runtime version");
  if (runtime_version != NCCL_VERSION_CODE)
    fail("header/runtime ABI mismatch");

  ncclUniqueId id{};
  if (rank_ == 0) check(ncclGetUniqueId(&id), "create unique id");
  const int socket = bootstrap_socket(config);
  try {
    if (rank_ == 0)
      write_all(socket, &id, sizeof(id));
    else
      read_all(socket, &id, sizeof(id));
    ::close(socket);
  } catch (...) {
    ::close(socket);
    throw;
  }
  check(ncclCommInitRank(&communicator_, kWorldSize, id, rank_),
        "initialize communicator");
}

NcclCollective::~NcclCollective() {
  if (communicator_ && !aborted_) ncclCommDestroy(communicator_);
}

void NcclCollective::enqueue_sum(const __nv_bfloat16* input,
                                 __nv_bfloat16* output,
                                 std::size_t elements, cudaStream_t stream) {
  if (aborted_ || !communicator_) fail("faulted communicator cannot enqueue");
  check(ncclAllReduce(input, output, elements, ncclBfloat16, ncclSum,
                      communicator_, stream),
        "enqueue all-reduce");
}

bool NcclCollective::healthy() noexcept {
  if (aborted_ || !communicator_) return false;
  ncclResult_t asynchronous = ncclSuccess;
  return ncclCommGetAsyncError(communicator_, &asynchronous) == ncclSuccess &&
         asynchronous == ncclSuccess;
}

void NcclCollective::abort() noexcept {
  if (aborted_ || !communicator_) return;
  aborted_ = true;
  ncclComm_t communicator = communicator_;
  communicator_ = nullptr;
  // NCCL abort may wait for graph-owned work to release. The transaction
  // owner must regain control within its timeout so state publication can be
  // suppressed. The detached cleanup owns only the opaque communicator.
  std::thread([communicator] { ncclCommAbort(communicator); }).detach();
}

}  // namespace rocket::qwen38::pair_reduce

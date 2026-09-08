// SPDX-License-Identifier: Apache-2.0
#include "mtp/nccl_communicator_owner.h"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

struct evp_md_st;
extern "C" {
const evp_md_st* EVP_sha256();
unsigned char* HMAC(const evp_md_st*, const void*, int, const unsigned char*,
                    std::size_t, unsigned char*, unsigned int*);
int CRYPTO_memcmp(const void*, const void*, std::size_t);
void OPENSSL_cleanse(void*, std::size_t);
}

namespace rocket::qwen38::mtp {
namespace {

using Clock = std::chrono::steady_clock;
using NcclResult = int;
using CudaResult = int;
constexpr NcclResult kNcclSuccess = 0;
constexpr CudaResult kCudaSuccess = 0;
constexpr std::uint32_t kBootstrapSchema = 1;
constexpr std::uint32_t kHelloPhase = 1;
constexpr std::uint32_t kReadyPhase = 2;
constexpr std::size_t kFrameBytes = 224;
constexpr std::size_t kUniqueIdOffset = 32;
constexpr std::size_t kSessionOffset = 160;
constexpr std::size_t kTagOffset = 192;
constexpr std::array<std::uint8_t, 8> kMagic{'Q', '3', '8', 'N', 'C', 'C', 'L', '1'};

struct NcclUniqueId {
  char internal[kNcclUniqueIdBytes];
};
static_assert(sizeof(NcclUniqueId) == kNcclUniqueIdBytes);

[[noreturn]] void contract_fail(const char* reason) {
  throw NcclBootstrapContractError(
      std::string("qwen38 NCCL bootstrap contract: ") + reason);
}

[[noreturn]] void transport_fail(const char* reason) {
  throw NcclBootstrapTransportError(
      std::string("qwen38 NCCL bootstrap transport: ") + reason);
}

[[noreturn]] void transport_errno(const char* reason) {
  const int observed_errno = errno;
  throw NcclBootstrapTransportError(
      std::string("qwen38 NCCL bootstrap transport: ") + reason + ": " +
      std::strerror(observed_errno));
}

bool nonzero(const std::array<std::uint8_t, 32>& value) noexcept {
  return std::any_of(value.begin(), value.end(),
                     [](std::uint8_t byte) { return byte != 0; });
}

bool nonzero_unique_id(const NcclUniqueId& value) noexcept {
  return std::any_of(std::begin(value.internal), std::end(value.internal),
                     [](char byte) { return byte != 0; });
}

std::uint64_t elapsed_ns(Clock::time_point start) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

void put_u32(std::array<std::uint8_t, kFrameBytes>& frame,
             std::size_t offset, std::uint32_t value) noexcept {
  const std::uint32_t network = htonl(value);
  std::memcpy(frame.data() + offset, &network, sizeof(network));
}

std::uint32_t get_u32(const std::array<std::uint8_t, kFrameBytes>& frame,
                      std::size_t offset) noexcept {
  std::uint32_t network = 0;
  std::memcpy(&network, frame.data() + offset, sizeof(network));
  return ntohl(network);
}

std::array<std::uint8_t, 32> authenticate(
    const std::array<std::uint8_t, kFrameBytes>& frame,
    const std::array<std::uint8_t, 32>& key) {
  std::array<std::uint8_t, 32> tag{};
  unsigned int tag_bytes = 0;
  if (HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()), frame.data(),
           kTagOffset, tag.data(), &tag_bytes) == nullptr ||
      tag_bytes != tag.size())
    throw NcclBootstrapAuthenticationError(
        "qwen38 NCCL bootstrap authentication primitive failed");
  return tag;
}

std::array<std::uint8_t, kFrameBytes> make_frame(
    std::uint32_t phase, int rank, const NcclUniqueId& unique_id,
    const std::array<std::uint8_t, 32>& session,
    const std::array<std::uint8_t, 32>& key) {
  std::array<std::uint8_t, kFrameBytes> frame{};
  std::copy(kMagic.begin(), kMagic.end(), frame.begin());
  put_u32(frame, 8, kBootstrapSchema);
  put_u32(frame, 12, phase);
  put_u32(frame, 16, static_cast<std::uint32_t>(rank));
  put_u32(frame, 20, kNcclWorldSize);
  put_u32(frame, 24, kNcclUniqueIdBytes);
  put_u32(frame, 28, kPinnedNcclRawVersion);
  std::memcpy(frame.data() + kUniqueIdOffset, unique_id.internal,
              kNcclUniqueIdBytes);
  std::copy(session.begin(), session.end(), frame.begin() + kSessionOffset);
  const auto tag = authenticate(frame, key);
  std::copy(tag.begin(), tag.end(), frame.begin() + kTagOffset);
  return frame;
}

NcclUniqueId validate_frame(
    const std::array<std::uint8_t, kFrameBytes>& frame,
    std::uint32_t expected_phase, int expected_rank,
    const std::array<std::uint8_t, 32>& expected_session,
    const std::array<std::uint8_t, 32>& key) {
  const auto expected_tag = authenticate(frame, key);
  if (CRYPTO_memcmp(expected_tag.data(), frame.data() + kTagOffset,
                    expected_tag.size()) != 0)
    throw NcclBootstrapAuthenticationError(
        "qwen38 NCCL bootstrap peer authentication failed");
  if (!std::equal(kMagic.begin(), kMagic.end(), frame.begin()) ||
      get_u32(frame, 8) != kBootstrapSchema ||
      get_u32(frame, 12) != expected_phase ||
      get_u32(frame, 16) != static_cast<std::uint32_t>(expected_rank) ||
      get_u32(frame, 20) != kNcclWorldSize ||
      get_u32(frame, 24) != kNcclUniqueIdBytes ||
      get_u32(frame, 28) != kPinnedNcclRawVersion ||
      CRYPTO_memcmp(expected_session.data(), frame.data() + kSessionOffset,
                    expected_session.size()) != 0)
    throw NcclBootstrapAuthenticationError(
        "qwen38 NCCL bootstrap authenticated peer contract changed");
  NcclUniqueId unique_id{};
  std::memcpy(unique_id.internal, frame.data() + kUniqueIdOffset,
              kNcclUniqueIdBytes);
  return unique_id;
}

class UniqueFd final {
 public:
  UniqueFd() = default;
  explicit UniqueFd(int fd) noexcept : fd_(fd) {}
  ~UniqueFd() {
    if (fd_ >= 0) ::close(fd_);
  }
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      if (fd_ >= 0) ::close(fd_);
      fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
  }
  int get() const noexcept { return fd_; }

 private:
  int fd_ = -1;
};

int remaining_ms(Clock::time_point deadline) {
  const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - Clock::now());
  if (remaining.count() <= 0) return 0;
  return static_cast<int>(std::min<std::int64_t>(
      remaining.count(), std::numeric_limits<int>::max()));
}

void wait_socket(int socket, short events, Clock::time_point deadline,
                 const char* operation) {
  while (true) {
    pollfd descriptor{socket, events, 0};
    const int result = ::poll(&descriptor, 1, remaining_ms(deadline));
    if (result == 0) transport_fail(operation);
    if (result < 0) {
      if (errno == EINTR) continue;
      transport_errno(operation);
    }
    if ((descriptor.revents & events) != 0) return;
    transport_fail(operation);
  }
}

void set_nonblocking(int socket) {
  const int flags = ::fcntl(socket, F_GETFL, 0);
  if (flags < 0 || ::fcntl(socket, F_SETFL, flags | O_NONBLOCK) != 0)
    transport_errno("configure nonblocking socket");
}

void configure_connected_socket(int socket) {
  int one = 1;
  if (::setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0)
    transport_errno("configure TCP_NODELAY");
  set_nonblocking(socket);
}

UniqueFd listen_bounded(int port, std::uint32_t timeout_ms) {
  UniqueFd listener(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (listener.get() < 0) transport_errno("create listener");
  int one = 1;
  if (::setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &one,
                   sizeof(one)) != 0)
    transport_errno("configure listener");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  address.sin_addr.s_addr = INADDR_ANY;
  if (::bind(listener.get(), reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0)
    transport_errno("bind listener");
  if (::listen(listener.get(), 1) != 0) transport_errno("listen");
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  wait_socket(listener.get(), POLLIN, deadline, "accept timed out");
  UniqueFd connected(
      ::accept4(listener.get(), nullptr, nullptr, SOCK_CLOEXEC));
  if (connected.get() < 0) transport_errno("accept peer");
  configure_connected_socket(connected.get());
  return connected;
}

UniqueFd connect_bounded(const std::string& host, int port,
                         std::uint32_t timeout_ms) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1)
    contract_fail("bootstrap host must be a numeric IPv4 address");
  UniqueFd socket(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (socket.get() < 0) transport_errno("create client socket");
  set_nonblocking(socket.get());
  const int result = ::connect(socket.get(), reinterpret_cast<sockaddr*>(&address),
                               sizeof(address));
  if (result != 0 && errno != EINPROGRESS) transport_errno("connect peer");
  if (result != 0) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    wait_socket(socket.get(), POLLOUT, deadline, "connect timed out");
    int socket_error = 0;
    socklen_t bytes = sizeof(socket_error);
    if (::getsockopt(socket.get(), SOL_SOCKET, SO_ERROR, &socket_error, &bytes) !=
        0)
      transport_errno("query connect result");
    if (socket_error != 0) {
      errno = socket_error;
      transport_errno("connect peer");
    }
  }
  int one = 1;
  if (::setsockopt(socket.get(), IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) !=
      0)
    transport_errno("configure TCP_NODELAY");
  return socket;
}

void write_all(int socket, const void* source, std::size_t bytes,
               std::uint32_t timeout_ms) {
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  const auto* cursor = static_cast<const std::byte*>(source);
  while (bytes != 0) {
    wait_socket(socket, POLLOUT, deadline, "write timed out");
    const ssize_t written = ::send(socket, cursor, bytes, MSG_NOSIGNAL);
    if (written < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      transport_errno("write peer frame");
    }
    if (written == 0) transport_fail("peer stopped accepting bootstrap data");
    cursor += written;
    bytes -= static_cast<std::size_t>(written);
  }
}

void read_all(int socket, void* destination, std::size_t bytes,
              std::uint32_t timeout_ms) {
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  auto* cursor = static_cast<std::byte*>(destination);
  while (bytes != 0) {
    wait_socket(socket, POLLIN, deadline, "read timed out");
    const ssize_t received = ::recv(socket, cursor, bytes, 0);
    if (received < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      transport_errno("read peer frame");
    }
    if (received == 0) transport_fail("peer closed during bootstrap");
    cursor += received;
    bytes -= static_cast<std::size_t>(received);
  }
}

template <class T>
T load_symbol(void* library, const char* name) {
  ::dlerror();
  void* address = ::dlsym(library, name);
  if (address == nullptr || ::dlerror() != nullptr)
    throw NcclBootstrapLibraryError(
        std::string("qwen38 NCCL bootstrap required symbol absent: ") + name);
  return reinterpret_cast<T>(address);
}

NcclBootstrapOutcome classify(const std::exception& error) noexcept {
  if (dynamic_cast<const NcclBootstrapContractError*>(&error) != nullptr)
    return NcclBootstrapOutcome::kContractError;
  if (dynamic_cast<const NcclBootstrapLibraryError*>(&error) != nullptr)
    return NcclBootstrapOutcome::kLibraryError;
  if (dynamic_cast<const NcclBootstrapTransportError*>(&error) != nullptr)
    return NcclBootstrapOutcome::kTransportError;
  if (dynamic_cast<const NcclBootstrapAuthenticationError*>(&error) != nullptr)
    return NcclBootstrapOutcome::kAuthenticationError;
  if (dynamic_cast<const NcclBootstrapCudaError*>(&error) != nullptr)
    return NcclBootstrapOutcome::kCudaError;
  return NcclBootstrapOutcome::kNcclError;
}

}  // namespace

struct NcclCommunicatorOwner::Impl {
  using GetVersion = NcclResult (*)(int*);
  using GetUniqueId = NcclResult (*)(NcclUniqueId*);
  using CommInitRank = NcclResult (*)(void**, int, NcclUniqueId, int);
  using CommInt = NcclResult (*)(const void*, int*);
  using AsyncError = NcclResult (*)(void*, NcclResult*);
  using CommCleanup = NcclResult (*)(void*);
  using ErrorString = const char* (*)(NcclResult);
  using SetDevice = CudaResult (*)(int);
  using CudaErrorString = const char* (*)(CudaResult);

  Impl(const NcclCommunicatorConfig& value, NcclBootstrapOtelSink& sink)
      : config(value), telemetry(sink) {}

  ~Impl() {
    if (communicator != nullptr) destroy_communicator();
    OPENSSL_cleanse(config.authentication_key.data(),
                    config.authentication_key.size());
    if (cuda_library != nullptr) ::dlclose(cuda_library);
    if (nccl_library != nullptr) ::dlclose(nccl_library);
  }

  void emit(NcclBootstrapStage stage, NcclBootstrapOutcome outcome,
            Clock::time_point start) const noexcept {
    const NcclBootstrapTelemetryRecord record{
        stage, outcome, (config.rank == 0 || config.rank == 1) ? config.rank : -1,
        kNcclWorldSize, elapsed_ns(start)};
    telemetry.emit_span_and_log(record);
    telemetry.record_duration(record);
  }

  [[noreturn]] void fail_nccl(const char* operation, NcclResult result) const {
    const char* detail = error_string != nullptr ? error_string(result) : nullptr;
    throw NcclBootstrapNcclError(
        std::string("qwen38 NCCL bootstrap ") + operation + ": " +
        (detail != nullptr ? detail : "unknown NCCL failure"));
  }

  void abort_communicator() noexcept {
    if (communicator == nullptr) return;
    const auto start = Clock::now();
    const NcclResult result = comm_abort(communicator);
    communicator = nullptr;
    emit(NcclBootstrapStage::kTeardown,
         result == kNcclSuccess ? NcclBootstrapOutcome::kOk
                                : NcclBootstrapOutcome::kNcclError,
         start);
  }

  void destroy_communicator() noexcept {
    if (communicator == nullptr) return;
    const auto start = Clock::now();
    const NcclResult result = comm_destroy(communicator);
    communicator = nullptr;
    emit(NcclBootstrapStage::kTeardown,
         result == kNcclSuccess ? NcclBootstrapOutcome::kOk
                                : NcclBootstrapOutcome::kNcclError,
         start);
  }

  void clear_authentication_key() noexcept {
    OPENSSL_cleanse(config.authentication_key.data(),
                    config.authentication_key.size());
  }

  NcclCommunicatorConfig config;
  NcclBootstrapOtelSink& telemetry;
  void* nccl_library = nullptr;
  void* cuda_library = nullptr;
  void* communicator = nullptr;
  GetVersion get_version = nullptr;
  GetUniqueId get_unique_id = nullptr;
  CommInitRank comm_init_rank = nullptr;
  CommInt comm_count = nullptr;
  CommInt comm_user_rank = nullptr;
  AsyncError comm_async_error = nullptr;
  CommCleanup comm_abort = nullptr;
  CommCleanup comm_destroy = nullptr;
  ErrorString error_string = nullptr;
  SetDevice set_device = nullptr;
  CudaErrorString cuda_error_string = nullptr;
};

void NcclCommunicatorOwner::validate_config(
    const NcclCommunicatorConfig& config) {
  if ((config.rank != 0 && config.rank != 1) ||
      config.peer_rank != 1 - config.rank)
    contract_fail("topology must be exactly ranks zero and one");
  if (config.device != 0) contract_fail("device must be process-local CUDA zero");
  if (config.bootstrap_host.empty())
    contract_fail("bootstrap host is required");
  if (config.bootstrap_port <= 0 || config.bootstrap_port > 65'535 ||
      config.pair_reduce_bootstrap_port <= 0 ||
      config.pair_reduce_bootstrap_port > 65'535)
    contract_fail("bootstrap ports must be within 1..65535");
  if (config.bootstrap_port == config.pair_reduce_bootstrap_port)
    contract_fail("NCCL and PairReduce bootstrap ports must be distinct");
  if (config.timeout_ms < 100 || config.timeout_ms > 120'000)
    contract_fail("timeout must be within 100..120000 milliseconds");
  if (!nonzero(config.session_sha256))
    contract_fail("session identity is required");
  if (!nonzero(config.authentication_key))
    contract_fail("ephemeral authentication key is required");
  if (config.nccl_library.empty() || config.cuda_runtime_library.empty())
    contract_fail("runtime library paths are required");
}

NcclCommunicatorOwner::NcclCommunicatorOwner(
    const NcclCommunicatorConfig& config, NcclBootstrapOtelSink& telemetry)
    : impl_(std::make_unique<Impl>(config, telemetry)) {
  NcclBootstrapStage stage = NcclBootstrapStage::kValidate;
  auto start = Clock::now();
  try {
    validate_config(config);
    impl_->emit(stage, NcclBootstrapOutcome::kOk, start);

    stage = NcclBootstrapStage::kLoad;
    start = Clock::now();
    impl_->nccl_library =
        ::dlopen(config.nccl_library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (impl_->nccl_library == nullptr)
      throw NcclBootstrapLibraryError(
          "qwen38 NCCL bootstrap pinned library unavailable");
    impl_->cuda_library =
        ::dlopen(config.cuda_runtime_library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (impl_->cuda_library == nullptr)
      throw NcclBootstrapLibraryError(
          "qwen38 NCCL bootstrap CUDA runtime unavailable");
    impl_->get_version =
        load_symbol<Impl::GetVersion>(impl_->nccl_library, "ncclGetVersion");
    impl_->get_unique_id = load_symbol<Impl::GetUniqueId>(
        impl_->nccl_library, "ncclGetUniqueId");
    impl_->comm_init_rank = load_symbol<Impl::CommInitRank>(
        impl_->nccl_library, "ncclCommInitRank");
    impl_->comm_count =
        load_symbol<Impl::CommInt>(impl_->nccl_library, "ncclCommCount");
    impl_->comm_user_rank = load_symbol<Impl::CommInt>(
        impl_->nccl_library, "ncclCommUserRank");
    impl_->comm_async_error = load_symbol<Impl::AsyncError>(
        impl_->nccl_library, "ncclCommGetAsyncError");
    impl_->comm_abort = load_symbol<Impl::CommCleanup>(
        impl_->nccl_library, "ncclCommAbort");
    impl_->comm_destroy = load_symbol<Impl::CommCleanup>(
        impl_->nccl_library, "ncclCommDestroy");
    impl_->error_string = load_symbol<Impl::ErrorString>(
        impl_->nccl_library, "ncclGetErrorString");
    impl_->set_device = load_symbol<Impl::SetDevice>(
        impl_->cuda_library, "cudaSetDevice");
    impl_->cuda_error_string = load_symbol<Impl::CudaErrorString>(
        impl_->cuda_library, "cudaGetErrorString");
    int version = 0;
    NcclResult result = impl_->get_version(&version);
    if (result != kNcclSuccess) impl_->fail_nccl("version query", result);
    if (version != kPinnedNcclRawVersion)
      throw NcclBootstrapLibraryError(
          "qwen38 NCCL bootstrap runtime version differs from pinned 23007");
    impl_->emit(stage, NcclBootstrapOutcome::kOk, start);

    NcclUniqueId unique_id{};
    if (config.rank == 0) {
      result = impl_->get_unique_id(&unique_id);
      if (result != kNcclSuccess) impl_->fail_nccl("unique ID", result);
      if (!nonzero_unique_id(unique_id))
        throw NcclBootstrapNcclError(
            "qwen38 NCCL bootstrap unique ID was empty");
    }

    stage = NcclBootstrapStage::kConnect;
    start = Clock::now();
    UniqueFd socket = config.rank == 0
                          ? listen_bounded(config.bootstrap_port,
                                           config.timeout_ms)
                          : connect_bounded(config.bootstrap_host,
                                            config.bootstrap_port,
                                            config.timeout_ms);
    impl_->emit(stage, NcclBootstrapOutcome::kOk, start);

    stage = NcclBootstrapStage::kAuthenticate;
    start = Clock::now();
    const auto local_hello =
        make_frame(kHelloPhase, config.rank, unique_id, config.session_sha256,
                   config.authentication_key);
    std::array<std::uint8_t, kFrameBytes> peer_hello{};
    write_all(socket.get(), local_hello.data(), local_hello.size(),
              config.timeout_ms);
    read_all(socket.get(), peer_hello.data(), peer_hello.size(),
             config.timeout_ms);
    const NcclUniqueId peer_id = validate_frame(
        peer_hello, kHelloPhase, config.peer_rank, config.session_sha256,
        config.authentication_key);
    if (config.rank == 0) {
      if (nonzero_unique_id(peer_id))
        throw NcclBootstrapAuthenticationError(
            "qwen38 NCCL bootstrap non-owner supplied a unique ID");
    } else {
      if (!nonzero_unique_id(peer_id))
        throw NcclBootstrapAuthenticationError(
            "qwen38 NCCL bootstrap rank zero supplied an empty unique ID");
      unique_id = peer_id;
    }
    impl_->emit(stage, NcclBootstrapOutcome::kOk, start);

    stage = NcclBootstrapStage::kBindDevice;
    start = Clock::now();
    const CudaResult cuda_result = impl_->set_device(config.device);
    if (cuda_result != kCudaSuccess) {
      const char* detail = impl_->cuda_error_string(cuda_result);
      throw NcclBootstrapCudaError(
          std::string("qwen38 NCCL bootstrap bind CUDA device zero: ") +
          (detail != nullptr ? detail : "unknown CUDA failure"));
    }
    impl_->emit(stage, NcclBootstrapOutcome::kOk, start);

    stage = NcclBootstrapStage::kInitialize;
    start = Clock::now();
    result = impl_->comm_init_rank(&impl_->communicator, kNcclWorldSize,
                                   unique_id, config.rank);
    if (result != kNcclSuccess) impl_->fail_nccl("communicator init", result);
    if (impl_->communicator == nullptr)
      throw NcclBootstrapNcclError(
          "qwen38 NCCL bootstrap communicator init returned null");
    impl_->emit(stage, NcclBootstrapOutcome::kOk, start);

    stage = NcclBootstrapStage::kVerify;
    start = Clock::now();
    int observed_count = 0;
    int observed_rank = -1;
    NcclResult async = kNcclSuccess;
    result = impl_->comm_count(impl_->communicator, &observed_count);
    if (result != kNcclSuccess) impl_->fail_nccl("communicator count", result);
    result = impl_->comm_user_rank(impl_->communicator, &observed_rank);
    if (result != kNcclSuccess) impl_->fail_nccl("communicator user rank", result);
    result = impl_->comm_async_error(impl_->communicator, &async);
    if (result != kNcclSuccess) impl_->fail_nccl("async state query", result);
    if (observed_count != kNcclWorldSize || observed_rank != config.rank)
      throw NcclBootstrapContractError(
          "qwen38 NCCL bootstrap communicator topology changed");
    if (async != kNcclSuccess) impl_->fail_nccl("async state", async);
    impl_->emit(stage, NcclBootstrapOutcome::kOk, start);

    stage = NcclBootstrapStage::kReady;
    start = Clock::now();
    const auto local_ready =
        make_frame(kReadyPhase, config.rank, unique_id, config.session_sha256,
                   config.authentication_key);
    std::array<std::uint8_t, kFrameBytes> peer_ready{};
    write_all(socket.get(), local_ready.data(), local_ready.size(),
              config.timeout_ms);
    read_all(socket.get(), peer_ready.data(), peer_ready.size(),
             config.timeout_ms);
    const NcclUniqueId ready_id = validate_frame(
        peer_ready, kReadyPhase, config.peer_rank, config.session_sha256,
        config.authentication_key);
    if (std::memcmp(ready_id.internal, unique_id.internal,
                    kNcclUniqueIdBytes) != 0)
      throw NcclBootstrapAuthenticationError(
          "qwen38 NCCL bootstrap peer initialized a different communicator");
    impl_->emit(stage, NcclBootstrapOutcome::kOk, start);
    impl_->clear_authentication_key();
  } catch (const std::exception& error) {
    impl_->emit(stage, classify(error), start);
    impl_->abort_communicator();
    impl_->clear_authentication_key();
    throw;
  } catch (...) {
    impl_->emit(stage, NcclBootstrapOutcome::kNcclError, start);
    impl_->abort_communicator();
    impl_->clear_authentication_key();
    throw;
  }
}

NcclCommunicatorOwner::~NcclCommunicatorOwner() = default;

void* NcclCommunicatorOwner::communicator() const noexcept {
  return impl_->communicator;
}

int NcclCommunicatorOwner::rank() const noexcept { return impl_->config.rank; }

std::string_view NcclCommunicatorOwner::nccl_library_path() const noexcept {
  return impl_->config.nccl_library;
}

void NcclCommunicatorOwner::validate_async() {
  const auto start = Clock::now();
  if (impl_->communicator == nullptr)
    throw NcclBootstrapContractError(
        "qwen38 NCCL bootstrap communicator is not active");
  NcclResult async = kNcclSuccess;
  const NcclResult result =
      impl_->comm_async_error(impl_->communicator, &async);
  if (result == kNcclSuccess && async == kNcclSuccess) {
    impl_->emit(NcclBootstrapStage::kVerify, NcclBootstrapOutcome::kOk, start);
    return;
  }
  try {
    if (result != kNcclSuccess) impl_->fail_nccl("async state query", result);
    impl_->fail_nccl("async state", async);
  } catch (const std::exception& error) {
    impl_->emit(NcclBootstrapStage::kVerify, classify(error), start);
    impl_->abort_communicator();
    throw;
  }
}

void NcclCommunicatorOwner::abort() noexcept { impl_->abort_communicator(); }

}  // namespace rocket::qwen38::mtp

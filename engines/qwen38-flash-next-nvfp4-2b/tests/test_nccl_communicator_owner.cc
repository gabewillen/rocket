// SPDX-License-Identifier: Apache-2.0
#include "mtp/nccl_communicator_owner.h"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mtp = rocket::qwen38::mtp;

namespace {
using Owner = mtp::NcclCommunicatorOwner;

void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

class Sink final : public mtp::NcclBootstrapOtelSink {
 public:
  void emit_span_and_log(
      const mtp::NcclBootstrapTelemetryRecord& record) noexcept override {
    std::lock_guard lock(mutex);
    spans.push_back(record);
  }
  void record_duration(
      const mtp::NcclBootstrapTelemetryRecord& record) noexcept override {
    std::lock_guard lock(mutex);
    metrics.push_back(record);
  }
  std::mutex mutex;
  std::vector<mtp::NcclBootstrapTelemetryRecord> spans;
  std::vector<mtp::NcclBootstrapTelemetryRecord> metrics;
};

int reserve_port() {
  const int socket = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (socket < 0) throw std::runtime_error("test socket failed");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) !=
      0) {
    ::close(socket);
    throw std::runtime_error("test bind failed");
  }
  socklen_t bytes = sizeof(address);
  if (::getsockname(socket, reinterpret_cast<sockaddr*>(&address), &bytes) != 0) {
    ::close(socket);
    throw std::runtime_error("test getsockname failed");
  }
  ::close(socket);
  return ntohs(address.sin_port);
}

mtp::NcclCommunicatorConfig config(int rank, int port,
                                   const std::string& library) {
  mtp::NcclCommunicatorConfig result;
  result.rank = rank;
  result.peer_rank = 1 - rank;
  result.device = 0;
  result.bootstrap_host = "127.0.0.1";
  result.bootstrap_port = port;
  result.pair_reduce_bootstrap_port = port == 65'535 ? port - 1 : port + 1;
  result.timeout_ms = 500;
  result.session_sha256.fill(0x35);
  result.authentication_key.fill(0xa7);
  result.nccl_library = library;
  result.cuda_runtime_library = library;
  return result;
}

struct PairResult {
  std::array<std::unique_ptr<Owner>, 2> owners;
  std::array<std::exception_ptr, 2> errors;
  std::array<std::unique_ptr<Sink>, 2> sinks;
};

PairResult run_pair(mtp::NcclCommunicatorConfig rank0,
                    mtp::NcclCommunicatorConfig rank1) {
  PairResult result;
  result.sinks[0] = std::make_unique<Sink>();
  result.sinks[1] = std::make_unique<Sink>();
  std::thread first([&] {
    try {
      result.owners[0] = std::make_unique<Owner>(rank0, *result.sinks[0]);
    } catch (...) {
      result.errors[0] = std::current_exception();
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  std::thread second([&] {
    try {
      result.owners[1] = std::make_unique<Owner>(rank1, *result.sinks[1]);
    } catch (...) {
      result.errors[1] = std::current_exception();
    }
  });
  first.join();
  second.join();
  return result;
}

template <class Error>
bool is_error(const std::exception_ptr& value) {
  if (!value) return false;
  try {
    std::rethrow_exception(value);
  } catch (const Error&) {
    return true;
  } catch (...) {
    return false;
  }
}

template <class Function>
Function function(void* library, const char* name) {
  void* address = ::dlsym(library, name);
  if (address == nullptr) throw std::runtime_error("fake control symbol absent");
  return reinterpret_cast<Function>(address);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    check(argc == 2, "fake library path required");
    const std::string library = argv[1];
    void* controls = ::dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    check(controls != nullptr, "fake library unavailable");
    const auto reset = function<void (*)()>(controls, "fake_nccl_reset");
    const auto set_count =
        function<void (*)(int)>(controls, "fake_nccl_set_count");
    const auto set_init =
        function<void (*)(int)>(controls, "fake_nccl_set_init_result");
    const auto set_rank_delta =
        function<void (*)(int)>(controls, "fake_nccl_set_rank_delta");
    const auto set_async =
        function<void (*)(int)>(controls, "fake_nccl_set_async");
    const auto set_cuda =
        function<void (*)(int)>(controls, "fake_cuda_set_result");
    const auto abort_calls =
        function<int (*)()>(controls, "fake_nccl_abort_calls");
    const auto destroy_calls =
        function<int (*)()>(controls, "fake_nccl_destroy_calls");

    reset();
    auto invalid = config(0, reserve_port(), library);
    invalid.bootstrap_port = invalid.pair_reduce_bootstrap_port;
    Sink invalid_sink;
    bool rejected = false;
    try {
      Owner owner(invalid, invalid_sink);
    } catch (const mtp::NcclBootstrapContractError&) {
      rejected = true;
    }
    check(rejected && invalid_sink.spans.size() == 1,
          "same-port contract was accepted or unobserved");

    reset();
    auto missing = config(0, reserve_port(), "/missing/libnccl.so.2");
    Sink missing_sink;
    rejected = false;
    try {
      Owner owner(missing, missing_sink);
    } catch (const mtp::NcclBootstrapLibraryError&) {
      rejected = true;
    }
    check(rejected && missing_sink.spans.size() == 2,
          "missing dynamic library was accepted or unobserved");

    reset();
    const int auth_port = reserve_port();
    auto auth0 = config(0, auth_port, library);
    auto auth1 = config(1, auth_port, library);
    auth1.authentication_key[31] ^= 1;
    auto auth = run_pair(auth0, auth1);
    check(is_error<mtp::NcclBootstrapAuthenticationError>(auth.errors[0]) &&
              is_error<mtp::NcclBootstrapAuthenticationError>(auth.errors[1]) &&
              abort_calls() == 0,
          "authentication-key mutation was accepted");

    reset();
    const int session_port = reserve_port();
    auto session0 = config(0, session_port, library);
    auto session1 = config(1, session_port, library);
    session1.session_sha256[0] ^= 1;
    auto session = run_pair(session0, session1);
    check(is_error<mtp::NcclBootstrapAuthenticationError>(session.errors[0]) &&
              is_error<mtp::NcclBootstrapAuthenticationError>(session.errors[1]),
          "session mutation was accepted");

    reset();
    const int success_port = reserve_port();
    auto success = run_pair(config(0, success_port, library),
                            config(1, success_port, library));
    check(!success.errors[0] && !success.errors[1] && success.owners[0] &&
              success.owners[1] && success.owners[0]->rank() == 0 &&
              success.owners[1]->rank() == 1,
          "valid two-rank construction failed");
    success.owners[0].reset();
    success.owners[1].reset();
    check(destroy_calls() == 2 && abort_calls() == 0,
          "normal teardown did not destroy exactly twice");

    reset();
    set_init(6);
    const int init_port = reserve_port();
    auto init = run_pair(config(0, init_port, library),
                         config(1, init_port, library));
    check(is_error<mtp::NcclBootstrapNcclError>(init.errors[0]) &&
              is_error<mtp::NcclBootstrapNcclError>(init.errors[1]) &&
              abort_calls() == 0 && destroy_calls() == 0,
          "communicator init fault was accepted or mis-cleaned");

    reset();
    set_count(3);
    const int count_port = reserve_port();
    auto count = run_pair(config(0, count_port, library),
                          config(1, count_port, library));
    check(is_error<mtp::NcclBootstrapContractError>(count.errors[0]) &&
              is_error<mtp::NcclBootstrapContractError>(count.errors[1]) &&
              abort_calls() == 2 && destroy_calls() == 0,
          "communicator count drift was accepted or not aborted");

    reset();
    set_rank_delta(1);
    const int rank_port = reserve_port();
    auto rank = run_pair(config(0, rank_port, library),
                         config(1, rank_port, library));
    check(is_error<mtp::NcclBootstrapContractError>(rank.errors[0]) &&
              is_error<mtp::NcclBootstrapContractError>(rank.errors[1]) &&
              abort_calls() == 2,
          "communicator rank drift was accepted or not aborted");

    reset();
    set_async(7);
    const int async_port = reserve_port();
    auto async = run_pair(config(0, async_port, library),
                          config(1, async_port, library));
    check(is_error<mtp::NcclBootstrapNcclError>(async.errors[0]) &&
              is_error<mtp::NcclBootstrapNcclError>(async.errors[1]) &&
              abort_calls() == 2,
          "async communicator fault was accepted or not aborted");

    reset();
    set_cuda(8);
    const int cuda_port = reserve_port();
    auto cuda = run_pair(config(0, cuda_port, library),
                         config(1, cuda_port, library));
    check(is_error<mtp::NcclBootstrapCudaError>(cuda.errors[0]) &&
              is_error<mtp::NcclBootstrapCudaError>(cuda.errors[1]) &&
              abort_calls() == 0,
          "CUDA bind fault was accepted");

    reset();
    Sink timeout_sink;
    auto timeout = config(0, reserve_port(), library);
    timeout.timeout_ms = 100;
    const auto timeout_start = std::chrono::steady_clock::now();
    rejected = false;
    try {
      Owner owner(timeout, timeout_sink);
    } catch (const mtp::NcclBootstrapTransportError&) {
      rejected = true;
    }
    const auto timeout_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - timeout_start)
                                .count();
    check(rejected && timeout_ms >= 80 && timeout_ms < 1'000,
          "listener timeout was not bounded");

    reset();
    const int abort_port = reserve_port();
    auto aborted = run_pair(config(0, abort_port, library),
                            config(1, abort_port, library));
    check(!aborted.errors[0] && !aborted.errors[1],
          "abort setup failed");
    aborted.owners[0]->abort();
    aborted.owners[1]->abort();
    aborted.owners[0].reset();
    aborted.owners[1].reset();
    check(abort_calls() == 2 && destroy_calls() == 0,
          "explicit abort was not exactly once");

    ::dlclose(controls);
    std::puts("qwen38 NCCL bootstrap owner: CPU success and fault contracts passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

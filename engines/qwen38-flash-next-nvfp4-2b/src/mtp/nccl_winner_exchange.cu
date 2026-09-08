// SPDX-License-Identifier: Apache-2.0
#include "mtp/nccl_winner_exchange.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace rocket::qwen38::mtp {
namespace {
using Result = int;
constexpr Result kSuccess = 0;
constexpr int kUint64 = 5;
template <class T> T symbol(void* library, const char* name) {
  void* value = dlsym(library, name);
  if (!value) throw std::runtime_error(std::string("NCCL symbol absent: ") + name);
  return reinterpret_cast<T>(value);
}
__global__ void transpose_rank_major(const output::Winner* gathered,
                                     output::Winner* ordered, int m) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m) {
    ordered[row * 2] = gathered[row];
    ordered[row * 2 + 1] = gathered[m + row];
  }
}
}  // namespace
struct NcclWinnerExchange::Impl {
  using AllGather = Result (*)(const void*, void*, std::size_t, int, void*, cudaStream_t);
  using ErrorString = const char* (*)(Result);
  std::unique_ptr<NcclCommunicatorOwner> owner;
  void* library = nullptr;
  output::Winner* gathered = nullptr;
  AllGather all_gather = nullptr;
  ErrorString error_string = nullptr;
  ~Impl() { cudaFree(gathered); if (library) dlclose(library); }
  [[noreturn]] void fail(const char* operation, Result result) const {
    const char* reason = error_string ? error_string(result) : "unknown";
    throw std::runtime_error(std::string("MTP NCCL winner ") + operation +
                             ": " + (reason ? reason : "unknown"));
  }
};
NcclWinnerExchange::NcclWinnerExchange(
    std::unique_ptr<NcclCommunicatorOwner> communicator_owner)
    : impl_(std::make_unique<Impl>()) {
  if (!communicator_owner || !communicator_owner->communicator()) {
    throw std::invalid_argument("MTP NCCL winner owner changed");
  }
  try {
    impl_->owner = std::move(communicator_owner);
    const std::string path(impl_->owner->nccl_library_path());
    impl_->library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!impl_->library) throw std::runtime_error("NCCL library unavailable");
    impl_->all_gather = symbol<Impl::AllGather>(impl_->library, "ncclAllGather");
    impl_->error_string = symbol<Impl::ErrorString>(impl_->library, "ncclGetErrorString");
    const auto status = cudaMalloc(&impl_->gathered,
                                   32 * sizeof(output::Winner));
    if (status != cudaSuccess)
      throw std::runtime_error("MTP NCCL winner scratch allocation failed");
  } catch (...) { impl_.reset(); throw; }
}
NcclWinnerExchange::~NcclWinnerExchange() = default;
void NcclWinnerExchange::enqueue(const output::Winner* local,
                                 output::Winner* ordered, int m, int rank,
                                 cudaStream_t stream) {
  if (!local || !ordered || !output::allowed_m(m) ||
      rank != impl_->owner->rank() || !stream)
    throw std::invalid_argument("MTP NCCL winner enqueue changed");
  const Result result = impl_->all_gather(local, impl_->gathered,
                                          static_cast<std::size_t>(m), kUint64,
                                          impl_->owner->communicator(), stream);
  if (result != kSuccess) impl_->fail("all-gather", result);
  transpose_rank_major<<<1, 32, 0, stream>>>(impl_->gathered, ordered, m);
  if (cudaPeekAtLastError() != cudaSuccess)
    throw std::runtime_error("MTP NCCL winner transpose launch failed");
}
void NcclWinnerExchange::validate_after_fence() {
  impl_->owner->validate_async();
}

std::unique_ptr<WinnerExchangePort> make_nccl_winner_exchange(
    const NcclCommunicatorConfig& config, NcclBootstrapOtelSink& telemetry) {
  return std::make_unique<NcclWinnerExchange>(
      std::make_unique<NcclCommunicatorOwner>(config, telemetry));
}
}  // namespace rocket::qwen38::mtp

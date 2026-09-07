// SPDX-License-Identifier: Apache-2.0
#include "mtp/nccl_winner_exchange.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
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
  using AsyncError = Result (*)(void*, Result*);
  using CommInt = Result (*)(const void*, int*);
  using ErrorString = const char* (*)(Result);
  void* library = nullptr; void* communicator = nullptr; int rank = -1;
  output::Winner* gathered = nullptr;
  AllGather all_gather = nullptr; AsyncError async_error = nullptr;
  ErrorString error_string = nullptr;
  ~Impl() { cudaFree(gathered); if (library) dlclose(library); }
  [[noreturn]] void fail(const char* operation, Result result) const {
    const char* reason = error_string ? error_string(result) : "unknown";
    throw std::runtime_error(std::string("MTP NCCL winner ") + operation +
                             ": " + (reason ? reason : "unknown"));
  }
};
NcclWinnerExchange::NcclWinnerExchange(void* communicator, int rank,
                                       std::string_view library)
    : impl_(new Impl) {
  if (!communicator || (rank != 0 && rank != 1) || library.empty()) {
    delete impl_; impl_ = nullptr;
    throw std::invalid_argument("MTP NCCL winner binding changed");
  }
  try {
    const std::string path(library);
    impl_->library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!impl_->library) throw std::runtime_error("NCCL library unavailable");
    impl_->communicator = communicator; impl_->rank = rank;
    impl_->all_gather = symbol<Impl::AllGather>(impl_->library, "ncclAllGather");
    impl_->async_error = symbol<Impl::AsyncError>(impl_->library, "ncclCommGetAsyncError");
    impl_->error_string = symbol<Impl::ErrorString>(impl_->library, "ncclGetErrorString");
    const auto count = symbol<Impl::CommInt>(impl_->library, "ncclCommCount");
    const auto user_rank = symbol<Impl::CommInt>(impl_->library, "ncclCommUserRank");
    int observed_count = 0, observed_rank = -1;
    Result result = count(communicator, &observed_count);
    if (result != kSuccess) impl_->fail("count", result);
    result = user_rank(communicator, &observed_rank);
    if (result != kSuccess) impl_->fail("rank", result);
    if (observed_count != 2 || observed_rank != rank)
      throw std::invalid_argument("MTP NCCL communicator topology changed");
    const auto status = cudaMalloc(&impl_->gathered,
                                   32 * sizeof(output::Winner));
    if (status != cudaSuccess)
      throw std::runtime_error("MTP NCCL winner scratch allocation failed");
  } catch (...) { delete impl_; impl_ = nullptr; throw; }
}
NcclWinnerExchange::~NcclWinnerExchange() { delete impl_; }
void NcclWinnerExchange::enqueue(const output::Winner* local,
                                 output::Winner* ordered, int m, int rank,
                                 cudaStream_t stream) {
  if (!local || !ordered || !output::allowed_m(m) || rank != impl_->rank || !stream)
    throw std::invalid_argument("MTP NCCL winner enqueue changed");
  const Result result = impl_->all_gather(local, impl_->gathered,
                                          static_cast<std::size_t>(m), kUint64,
                                          impl_->communicator, stream);
  if (result != kSuccess) impl_->fail("all-gather", result);
  transpose_rank_major<<<1, 32, 0, stream>>>(impl_->gathered, ordered, m);
  if (cudaPeekAtLastError() != cudaSuccess)
    throw std::runtime_error("MTP NCCL winner transpose launch failed");
}
void NcclWinnerExchange::validate_after_fence() {
  Result async = kSuccess;
  const Result result = impl_->async_error(impl_->communicator, &async);
  if (result != kSuccess) impl_->fail("async query", result);
  if (async != kSuccess) impl_->fail("async completion", async);
}
}  // namespace rocket::qwen38::mtp

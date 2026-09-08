// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {
struct UniqueId { char internal[128]; };
struct Comm { int count; int rank; };
std::atomic<int> get_id_result{0};
std::atomic<int> init_result{0};
std::atomic<int> count_value{2};
std::atomic<int> rank_delta{0};
std::atomic<int> async_value{0};
std::atomic<int> cuda_result{0};
std::atomic<int> abort_calls{0};
std::atomic<int> destroy_calls{0};
}  // namespace

extern "C" {
void fake_nccl_reset() {
  get_id_result = 0;
  init_result = 0;
  count_value = 2;
  rank_delta = 0;
  async_value = 0;
  cuda_result = 0;
  abort_calls = 0;
  destroy_calls = 0;
}
void fake_nccl_set_get_id_result(int value) { get_id_result = value; }
void fake_nccl_set_init_result(int value) { init_result = value; }
void fake_nccl_set_count(int value) { count_value = value; }
void fake_nccl_set_rank_delta(int value) { rank_delta = value; }
void fake_nccl_set_async(int value) { async_value = value; }
void fake_cuda_set_result(int value) { cuda_result = value; }
int fake_nccl_abort_calls() { return abort_calls; }
int fake_nccl_destroy_calls() { return destroy_calls; }

int ncclGetVersion(int* version) {
  *version = 23'007;
  return 0;
}
int ncclGetUniqueId(UniqueId* id) {
  if (get_id_result != 0) return get_id_result;
  for (std::size_t index = 0; index < sizeof(id->internal); ++index)
    id->internal[index] = static_cast<char>(index + 1);
  return 0;
}
int ncclCommInitRank(void** output, int count, UniqueId id, int rank) {
  if (init_result != 0) return init_result;
  if (id.internal[0] == 0) return 91;
  auto* comm = static_cast<Comm*>(std::malloc(sizeof(Comm)));
  if (comm == nullptr) return 92;
  comm->count = count;
  comm->rank = rank;
  *output = comm;
  return 0;
}
int ncclCommCount(const void* value, int* count) {
  if (value == nullptr) return 93;
  *count = count_value;
  return 0;
}
int ncclCommUserRank(const void* value, int* rank) {
  if (value == nullptr) return 94;
  *rank = static_cast<const Comm*>(value)->rank + rank_delta;
  return 0;
}
int ncclCommGetAsyncError(void* value, int* result) {
  if (value == nullptr) return 95;
  *result = async_value;
  return 0;
}
int ncclCommAbort(void* value) {
  ++abort_calls;
  std::free(value);
  return 0;
}
int ncclCommDestroy(void* value) {
  ++destroy_calls;
  std::free(value);
  return 0;
}
const char* ncclGetErrorString(int) { return "fake NCCL fault"; }
int ncclAllGather(const void*, void*, std::size_t, int, void*, void*) {
  return 0;
}
int cudaSetDevice(int device) { return device == 0 ? cuda_result.load() : 96; }
const char* cudaGetErrorString(int) { return "fake CUDA fault"; }
}

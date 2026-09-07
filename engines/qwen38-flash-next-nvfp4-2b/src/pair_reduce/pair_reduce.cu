#include "pair_reduce/pair_reduce.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace rocket::qwen38::pair_reduce {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kMagic = 0x3152494150383351ull;  // Q38PAIR1
constexpr std::uint32_t kSchema = 1;
constexpr std::uint32_t kBf16 = 1;

[[noreturn]] void contract_fail(const std::string& reason) {
  throw PairReduceContractError("qwen38 PairReduce contract: " + reason);
}

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw PairReduceCudaError(std::string("qwen38 PairReduce CUDA ") + operation + ": " +
                              cudaGetErrorString(status));
  }
}

bool allowed_m(int m) noexcept {
  for (const int allowed : kAllowedM)
    if (m == allowed) return true;
  return false;
}

int metric_m_bucket(int m) noexcept { return allowed_m(m) ? m : 0; }

std::uint64_t elapsed_ns(Clock::time_point start) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

void validate_header(const WireHeader& header, int peer_rank, int m,
                     std::size_t payload_bytes, std::uint64_t sequence) {
  if (header.magic != kMagic || header.schema != kSchema)
    contract_fail("peer wire schema drift");
  if (header.rank != static_cast<std::uint32_t>(peer_rank) ||
      header.world_size != kWorldSize)
    contract_fail("peer topology drift");
  if (header.page_bytes != kPageBytes)
    contract_fail("peer page-size drift");
  if (header.hidden != kHidden || header.m != static_cast<std::uint32_t>(m) ||
      header.dtype != kBf16 || header.payload_bytes != payload_bytes)
    contract_fail("peer message drift");
  if (header.sequence != sequence)
    contract_fail("peer sequence drift");
}

__global__ void accumulate_rank_order(const __nv_bfloat16* rank0,
                                      const __nv_bfloat16* rank1,
                                      float* output, std::size_t elements) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < elements) {
    const float first = __bfloat162float(rank0[index]);
    const float second = __bfloat162float(rank1[index]);
    output[index] = first + second;
  }
}

}  // namespace

PairReduce::PairReduce(Transport& transport, OtelStageSink& telemetry)
    : transport_(transport), telemetry_(telemetry) {
  if (transport_.world_size() != kWorldSize ||
      (transport_.rank() != 0 && transport_.rank() != 1))
    contract_fail("topology must be exactly ranks 0 and 1");
  if (::sysconf(_SC_PAGESIZE) != static_cast<long>(kPageBytes))
    contract_fail("host page size must be 65536 bytes");

  int device = -1;
  cudaDeviceProp properties{};
  cuda_check(cudaGetDevice(&device), "get device");
  cuda_check(cudaGetDeviceProperties(&properties, device), "get device properties");
  if (properties.major != 12 || properties.minor != 1)
    contract_fail("device must be GB10 sm_121");

  cuda_check(cudaHostAlloc(&region_, region_bytes(), cudaHostAllocMapped),
             "allocate anonymous pinned region");
  if (reinterpret_cast<std::uintptr_t>(region_) % kPageBytes != 0) {
    cudaFreeHost(region_);
    region_ = nullptr;
    contract_fail("pinned region is not 65536-byte aligned");
  }
  std::memset(region_, 0, region_bytes());
  try {
    cuda_check(cudaHostGetDevicePointer(&device_region_, region_, 0), "map pinned region");
    region_handle_ = transport_.register_region(region_, region_bytes());
  } catch (...) {
    cudaFreeHost(region_);
    region_ = nullptr;
    device_region_ = nullptr;
    throw;
  }
}

PairReduce::~PairReduce() {
  if (region_ != nullptr) {
    transport_.unregister_region(region_handle_);
    cudaFreeHost(region_);
  }
}

void PairReduce::reduce(const __nv_bfloat16* input, float* output, int m,
                        std::string_view trace_id, std::string_view request_id,
                        cudaStream_t stream) {
  const auto start = Clock::now();
  Outcome outcome = Outcome::kOk;
  std::size_t payload_bytes = 0;
  auto emit = [&]() noexcept {
    const std::uint64_t duration = elapsed_ns(start);
    telemetry_.emit_span_and_log({"rocket.qwen38.pair_reduce", trace_id, request_id,
                                  transport_.rank(), metric_m_bucket(m), kDtype, outcome, duration,
                                  payload_bytes});
    telemetry_.record_duration(
        {transport_.rank(), metric_m_bucket(m), kDtype, outcome, duration});
  };

  try {
    if (input == nullptr || output == nullptr) contract_fail("input and output are required");
    if (!allowed_m(m)) contract_fail("M must be one of 1,2,4,8,16");
    const std::size_t elements = static_cast<std::size_t>(m) * kHidden;
    payload_bytes = elements * sizeof(__nv_bfloat16);
    if (sizeof(WireHeader) + payload_bytes > slot_bytes())
      contract_fail("message exceeds its two-page slot");

    const std::uint64_t sequence = transport_.next_sequence();
    if (sequence == 0 || sequence <= last_sequence_)
      contract_fail("transport sequence is not strictly monotonic");
    last_sequence_ = sequence;

    auto* local = static_cast<std::byte*>(region_);
    const WireHeader header{kMagic, sequence, kSchema,
                            static_cast<std::uint32_t>(transport_.rank()),
                            static_cast<std::uint32_t>(kWorldSize),
                            static_cast<std::uint32_t>(kPageBytes),
                            static_cast<std::uint32_t>(kHidden),
                            static_cast<std::uint32_t>(m), kBf16,
                            static_cast<std::uint32_t>(payload_bytes), {0, 0}};
    std::memcpy(local, &header, sizeof(header));
    cuda_check(cudaMemcpyAsync(local + sizeof(WireHeader), input, payload_bytes,
                               cudaMemcpyDeviceToHost, stream),
               "stage local BF16 partial");
    cuda_check(cudaStreamSynchronize(stream), "publish local BF16 partial");

    const std::size_t wire_bytes = sizeof(WireHeader) + payload_bytes;
    transport_.post_unsignaled_write(region_handle_, 0, peer_offset(), wire_bytes);
    transport_.signal_sequence(sequence);
    transport_.wait_peer(sequence);
    transport_.flush_signaled();

    const auto* peer_header = reinterpret_cast<const WireHeader*>(local + peer_offset());
    validate_header(*peer_header, 1 - transport_.rank(), m, payload_bytes, sequence);

    const auto* mapped = static_cast<const std::byte*>(device_region_);
    const auto* local_values = reinterpret_cast<const __nv_bfloat16*>(
        mapped + sizeof(WireHeader));
    const auto* peer_values = reinterpret_cast<const __nv_bfloat16*>(
        mapped + peer_offset() + sizeof(WireHeader));
    const __nv_bfloat16* rank0 = transport_.rank() == 0 ? local_values : peer_values;
    const __nv_bfloat16* rank1 = transport_.rank() == 0 ? peer_values : local_values;
    accumulate_rank_order<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
        rank0, rank1, output, elements);
    cuda_check(cudaGetLastError(), "launch deterministic accumulation");
    cuda_check(cudaStreamSynchronize(stream), "complete deterministic accumulation");
  } catch (const PairReduceContractError&) {
    outcome = Outcome::kContractError;
    emit();
    throw;
  } catch (const PairReduceCudaError&) {
    outcome = Outcome::kCudaError;
    emit();
    throw;
  } catch (const std::exception& error) {
    outcome = Outcome::kTransportError;
    emit();
    throw PairReduceTransportError(std::string("qwen38 PairReduce transport: ") + error.what());
  } catch (...) {
    outcome = Outcome::kTransportError;
    emit();
    throw PairReduceTransportError("qwen38 PairReduce transport: unknown failure");
  }
  emit();
}

}  // namespace rocket::qwen38::pair_reduce

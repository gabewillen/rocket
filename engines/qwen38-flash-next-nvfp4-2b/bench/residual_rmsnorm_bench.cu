// SPDX-License-Identifier: Apache-2.0
#include "norm/residual_rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace qn = rocket::qwen38::norm;

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) fail(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    cuda_check(cudaMalloc(&data_, count * sizeof(T)), "allocate device buffer");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const noexcept { return data_; }
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }

 private:
  T* data_{};
  std::size_t count_{};
};

__global__ void launch_floor_kernel() {}

template <class Launch>
float measure(Launch launch, int warmup, int iterations) {
  cudaEvent_t begin{}, end{};
  cuda_check(cudaEventCreate(&begin), "create begin event");
  cuda_check(cudaEventCreate(&end), "create end event");
  for (int i = 0; i < warmup; ++i) cuda_check(launch(), "warmup launch");
  cuda_check(cudaDeviceSynchronize(), "warmup synchronize");
  cuda_check(cudaEventRecord(begin), "record begin");
  for (int i = 0; i < iterations; ++i) cuda_check(launch(), "measured launch");
  cuda_check(cudaEventRecord(end), "record end");
  cuda_check(cudaEventSynchronize(end), "synchronize end");
  float milliseconds = 0.0F;
  cuda_check(cudaEventElapsedTime(&milliseconds, begin, end), "elapsed time");
  cudaEventDestroy(end); cudaEventDestroy(begin);
  return milliseconds * 1.0e6F / iterations;
}

std::uint64_t hash_output(const __nv_bfloat16* device, std::size_t count) {
  std::vector<__nv_bfloat16> host(count);
  cuda_check(cudaMemcpy(host.data(), device, count * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost), "copy hash output");
  std::uint64_t hash = 1469598103934665603ULL;
  for (const auto value : host) {
    const std::uint16_t bits = std::bit_cast<std::uint16_t>(value);
    hash ^= bits & 0xffU; hash *= 1099511628211ULL;
    hash ^= bits >> 8; hash *= 1099511628211ULL;
  }
  return hash;
}

void run(int hidden, int iterations, int sm_count) {
  constexpr int m = 16;
  const std::size_t count = static_cast<std::size_t>(m) * hidden;
  const std::size_t bytes = count * sizeof(__nv_bfloat16);
  std::vector<__nv_bfloat16> host(count), weights(hidden);
  for (std::size_t i = 0; i < count; ++i)
    host[i] = __float2bfloat16(static_cast<float>(static_cast<int>(i % 251) - 125) / 64.0F);
  for (int i = 0; i < hidden; ++i)
    weights[i] = __float2bfloat16(0.75F + static_cast<float>(i % 31) / 64.0F);
  DeviceBuffer<__nv_bfloat16> input(count), residual(count), weight(hidden);
  DeviceBuffer<__nv_bfloat16> add(count), norm(count), residual_out(count);
  cuda_check(cudaMemcpy(input.get(), host.data(), bytes, cudaMemcpyHostToDevice), "copy input");
  cuda_check(cudaMemcpy(residual.get(), host.data(), bytes, cudaMemcpyHostToDevice), "copy residual");
  cuda_check(cudaMemcpy(weight.get(), weights.data(), hidden * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice), "copy weight");

  const float add_ns = measure([&] {
    return qn::residual_add(input.get(), residual.get(), add.get(), m, hidden);
  }, 100, iterations);
  const float rms_ns = measure([&] {
    return qn::rms_norm(input.get(), weight.get(), norm.get(), m, hidden);
  }, 100, iterations);
  const float separate_ns = measure([&] {
    cudaError_t status = qn::residual_add(input.get(), residual.get(), add.get(), m, hidden);
    return status == cudaSuccess
               ? qn::rms_norm(add.get(), weight.get(), norm.get(), m, hidden)
               : status;
  }, 100, iterations);
  const float fused_ns = measure([&] {
    return qn::fused_add_rms_norm(input.get(), residual.get(), weight.get(),
                                  residual_out.get(), norm.get(), m, hidden);
  }, 100, iterations);
  const float launch_floor_ns = measure([&] {
    launch_floor_kernel<<<m, 256>>>();
    return cudaPeekAtLastError();
  }, 100, iterations);

  constexpr std::size_t copy_control_bytes = 64ULL << 20;
  DeviceBuffer<std::byte> copy_source(copy_control_bytes);
  DeviceBuffer<std::byte> copy_destination(copy_control_bytes);
  const float copy_ns = measure([&] {
    return cudaMemcpyAsync(copy_destination.get(), copy_source.get(),
                           copy_control_bytes, cudaMemcpyDeviceToDevice);
  }, 10, 200);

  const double copy_bytes = 2.0 * copy_control_bytes;
  const double copy_gbps = copy_bytes / copy_ns;
  const double fused_bytes = 8.0 * count + 2.0 * hidden;
  const double fused_gbps = fused_bytes / fused_ns;
  const double roofline_ns = fused_bytes / copy_gbps;
  std::printf(
      "{\"result\":\"qwen38_residual_rmsnorm\",\"device\":\"NVIDIA GB10\","
      "\"sm\":121,\"m\":16,\"hidden\":%d,\"dtype\":\"bf16_fp32\","
      "\"bottleneck\":\"c16_launch_and_cta_occupancy\",\"iterations\":%d,"
      "\"sm_count\":%d,\"launch_floor_ns\":%.3f,"
      "\"residual_add_ns\":%.3f,\"rms_norm_ns\":%.3f,"
      "\"separate_add_rms_ns\":%.3f,\"fused_add_rms_ns\":%.3f,"
      "\"fusion_speedup\":%.6f,\"empirical_copy_gbps\":%.3f,"
      "\"fused_effective_gbps\":%.3f,\"memory_roofline_ns\":%.3f,"
      "\"roofline_fraction\":%.6f,\"output_hash\":%llu}\n",
      hidden, iterations, sm_count, launch_floor_ns,
      add_ns, rms_ns, separate_ns, fused_ns,
      separate_ns / fused_ns, copy_gbps, fused_gbps, roofline_ns,
      roofline_ns / fused_ns,
      static_cast<unsigned long long>(hash_output(norm.get(), count)));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int device = 0;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device), "get CUDA device");
    cuda_check(cudaGetDeviceProperties(&properties, device), "get CUDA properties");
    const int iterations = argc == 2 ? std::stoi(argv[1]) : 10'000;
    if (iterations <= 0 || iterations > 1'000'000) fail("invalid iteration count");
    run(qn::kHiddenTp, iterations, properties.multiProcessorCount);
    run(qn::kHiddenFull, iterations, properties.multiProcessorCount);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "qwen38 residual/RMSNorm bench failed: %s\n", error.what());
    return 1;
  }
}

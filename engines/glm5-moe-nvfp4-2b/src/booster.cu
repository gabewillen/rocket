#include "booster.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <sys/utsname.h>

namespace rocket {
namespace {

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::ostringstream msg;
    msg << what << ": " << cudaGetErrorString(status);
    throw std::runtime_error(msg.str());
  }
}

int device_attr(cudaDeviceAttr attr, int device) {
  int value = 0;
  check(cudaDeviceGetAttribute(&value, attr, device), "cudaDeviceGetAttribute");
  return value;
}

// MemAvailable, not MemTotal. MemTotal never sizes an allocation here
// (blog/rocket.qmd#memory).
std::size_t read_mem_available_bytes() {
  std::ifstream meminfo("/proc/meminfo");
  std::string key;
  std::size_t value_kb = 0;
  std::string unit;
  while (meminfo >> key >> value_kb >> unit) {
    if (key == "MemAvailable:") return value_kb * 1024ull;
    meminfo.ignore(1024, '\n');
  }
  throw std::runtime_error("/proc/meminfo has no MemAvailable");
}

__global__ void ignition_kernel(std::uint32_t* out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *out = 0x52434B54u;  // 'RCKT'
  }
}

}  // namespace

int compiled_arch() { return ROCKET_CUDA_ARCH; }

BoosterFacts probe_booster(int device_ordinal) {
  BoosterFacts facts;

  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, device_ordinal), "cudaGetDeviceProperties");
  check(cudaSetDevice(device_ordinal), "cudaSetDevice");

  facts.device_name = prop.name;
  facts.compute_major = prop.major;
  facts.compute_minor = prop.minor;
  facts.sm_count = prop.multiProcessorCount;
  facts.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  facts.max_shared_bytes_per_sm = prop.sharedMemPerMultiprocessor;
  facts.async_engine_count = prop.asyncEngineCount;
  facts.concurrent_kernels = prop.concurrentKernels;
  facts.unified_addressing = prop.unifiedAddressing != 0;

  facts.max_blocks_per_sm =
      device_attr(cudaDevAttrMaxBlocksPerMultiprocessor, device_ordinal);
  facts.max_shared_bytes_per_block_optin =
      device_attr(cudaDevAttrMaxSharedMemoryPerBlockOptin, device_ordinal);

  // On unified memory this counts clean page cache as used, so it is a floor
  // and not a capacity (blog/posts/hardware/2026-09-06-cuda-free-counts-page-cache/).
  check(cudaMemGetInfo(&facts.device_free_bytes, &facts.device_total_bytes),
        "cudaMemGetInfo");

  long page = sysconf(_SC_PAGESIZE);
  if (page <= 0) throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
  facts.host_page_bytes = static_cast<std::size_t>(page);
  facts.host_mem_available_bytes = read_mem_available_bytes();

  utsname uts{};
  if (uname(&uts) != 0) throw std::runtime_error("uname failed");
  facts.kernel_release = uts.release;
  facts.hostname = uts.nodename;

  return facts;
}

std::uint32_t ignite(int device_ordinal) {
  check(cudaSetDevice(device_ordinal), "cudaSetDevice");

  std::uint32_t* device_out = nullptr;
  check(cudaMalloc(&device_out, sizeof(std::uint32_t)), "cudaMalloc");
  check(cudaMemset(device_out, 0, sizeof(std::uint32_t)), "cudaMemset");

  ignition_kernel<<<1, 32>>>(device_out);
  check(cudaGetLastError(), "ignition_kernel launch");
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::uint32_t host_out = 0;
  check(cudaMemcpy(&host_out, device_out, sizeof(host_out), cudaMemcpyDeviceToHost),
        "cudaMemcpy");
  check(cudaFree(device_out), "cudaFree");
  return host_out;
}

}  // namespace rocket

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  const std::size_t gib = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 90;
  const std::size_t bytes = gib << 30;
  void* p = nullptr;
  auto t = std::chrono::steady_clock::now();
  check(cudaMalloc(&p, bytes), "cudaMalloc");
  const double legacy = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  check(cudaFree(p), "cudaFree");
  cudaStream_t stream;
  check(cudaStreamCreate(&stream), "stream");
  t = std::chrono::steady_clock::now();
  check(cudaMallocAsync(&p, bytes, stream), "cudaMallocAsync");
  const double async_call = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t).count();
  check(cudaStreamSynchronize(stream), "alloc sync");
  const double async = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  check(cudaFreeAsync(p, stream), "cudaFreeAsync");
  check(cudaStreamSynchronize(stream), "free sync");
  cudaStreamDestroy(stream);
  void* host = nullptr;
  t = std::chrono::steady_clock::now();
  const cudaError_t host_result = cudaHostAlloc(&host, bytes, cudaHostAllocMapped);
  const double mapped = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  if (host_result == cudaSuccess) check(cudaFreeHost(host), "cudaFreeHost");
  std::printf("bytes=%zu cudaMalloc=%.6f_s cudaMallocAsync_call=%.6f_s "
              "cudaMallocAsync_sync=%.6f_s cudaHostAllocMapped=%.6f_s status=%s\n",
              bytes, legacy, async_call, async, mapped,
              cudaGetErrorString(host_result));
}

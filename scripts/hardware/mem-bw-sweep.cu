// Is 238 GB/s a plateau? Sweep grid size and per-thread ILP on the read path.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("err %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

template <int U>
__global__ void kread(const float4* __restrict__ a, size_t n4, float* __restrict__ out) {
  size_t base = blockIdx.x * (size_t)blockDim.x * U + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x * U;
  float acc = 0.f;
  for (size_t i = base; i + (size_t)(U - 1) * blockDim.x < n4; i += stride) {
    float4 v[U];
#pragma unroll
    for (int u = 0; u < U; u++) v[u] = a[i + (size_t)u * blockDim.x];
#pragma unroll
    for (int u = 0; u < U; u++) acc += v[u].x + v[u].y + v[u].z + v[u].w;
  }
  if (acc == 1.2345e-30f) out[0] = acc;
}

__global__ void kfill(float4* a, size_t n4, float v) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += s) a[i] = make_float4(v, v, v, v);
}

int main() {
  size_t bytes = (size_t)8 << 30;
  size_t n4 = bytes / sizeof(float4);
  cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, 0));
  printf("# SMs=%d maxThreadsPerSM=%d\n", p.multiProcessorCount, p.maxThreadsPerMultiProcessor);

  float4* a; float* out;
  CHECK(cudaMalloc(&a, bytes)); CHECK(cudaMalloc(&out, sizeof(float)));
  kfill<<<1024, 256>>>(a, n4, 1.f); CHECK(cudaDeviceSynchronize());

  cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

  auto bench = [&](int u, int bpsm, int threads) {
    int blocks = p.multiProcessorCount * bpsm;
    std::vector<double> ms;
    for (int r = 0; r < 8; r++) {
      CHECK(cudaEventRecord(e0));
      switch (u) {
        case 1: kread<1><<<blocks, threads>>>(a, n4, out); break;
        case 2: kread<2><<<blocks, threads>>>(a, n4, out); break;
        case 4: kread<4><<<blocks, threads>>>(a, n4, out); break;
        case 8: kread<8><<<blocks, threads>>>(a, n4, out); break;
      }
      CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
      float t; CHECK(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t);
    }
    CHECK(cudaGetLastError());
    std::sort(ms.begin(), ms.end());
    printf("unroll=%d blocks/SM=%-3d threads=%-4d  %.1f GB/s\n",
           u, bpsm, threads, bytes / (ms.front() * 1e6));
  };

  for (int u : {1, 2, 4, 8})
    for (int bpsm : {8, 16, 32, 64})
      bench(u, bpsm, 256);
  printf("--- thread sweep, unroll=4 ---\n");
  for (int th : {128, 256, 512, 1024}) bench(4, 32, th);
  return 0;
}

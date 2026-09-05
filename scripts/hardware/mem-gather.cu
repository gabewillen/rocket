// Is read bandwidth sensitive to access pattern, and therefore to TLB reach?
// Sequential vs block-gather (the MoE expert pattern) vs page-stride (worst case).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("err %s %d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

__global__ void kseq(const float4* __restrict__ a, size_t n4, float* out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, s = (size_t)gridDim.x * blockDim.x;
  float acc = 0;
  for (; i < n4; i += s) { float4 v = a[i]; acc += v.x + v.y + v.z + v.w; }
  if (acc == 1.2345e-30f) out[0] = acc;
}

// Each block streams one randomly chosen contiguous chunk. This is how a MoE
// layer reads top-k experts: few, large, contiguous, but scattered in memory.
__global__ void kblock(const float4* __restrict__ a, const size_t* __restrict__ off,
                       size_t chunk4, int nchunk, float* out) {
  float acc = 0;
  for (int c = blockIdx.x; c < nchunk; c += gridDim.x) {
    const float4* base = a + off[c];
    for (size_t i = threadIdx.x; i < chunk4; i += blockDim.x) {
      float4 v = base[i]; acc += v.x + v.y + v.z + v.w;
    }
  }
  if (acc == 1.2345e-30f) out[0] = acc;
}

// One float4 per page-sized step: defeats prefetch, maximises TLB pressure.
__global__ void kpagestride(const float4* __restrict__ a, size_t n4, size_t step4, float* out) {
  size_t npage = n4 / step4;
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, s = (size_t)gridDim.x * blockDim.x;
  float acc = 0;
  for (; i < npage; i += s) { float4 v = a[i * step4]; acc += v.x + v.y + v.z + v.w; }
  if (acc == 1.2345e-30f) out[0] = acc;
}

__global__ void kfill(float4* a, size_t n4, float v) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, s = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += s) a[i] = make_float4(v, v, v, v);
}

int main(int argc, char** argv) {
  size_t gib = (argc > 1) ? atoll(argv[1]) : 8;
  size_t bytes = gib << 30, n4 = bytes / sizeof(float4);
  cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, 0));
  int blocks = p.multiProcessorCount * 32, th = 256;

  float4* a; float* out;
  CHECK(cudaMalloc(&a, bytes)); CHECK(cudaMalloc(&out, 4));
  kfill<<<blocks, th>>>(a, n4, 1.f); CHECK(cudaDeviceSynchronize());

  cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
  auto time_it = [&](auto&& launch) {
    std::vector<double> ms;
    for (int r = 0; r < 8; r++) {
      CHECK(cudaEventRecord(e0)); launch();
      CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
      float t; CHECK(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t);
    }
    CHECK(cudaGetLastError());
    std::sort(ms.begin(), ms.end());
    return ms.front();
  };

  printf("sequential            %.1f GB/s\n", bytes / (time_it([&]{ kseq<<<blocks,th>>>(a,n4,out); }) * 1e6));

  for (size_t chunk_kib : {64, 256, 1024, 4096, 16384}) {
    size_t chunk4 = (chunk_kib << 10) / sizeof(float4);
    int nchunk = (int)(n4 / chunk4);
    std::vector<size_t> h(nchunk);
    for (int i = 0; i < nchunk; i++) h[i] = (size_t)i * chunk4;
    // shuffle so chunk order is unrelated to address order
    for (int i = nchunk - 1; i > 0; i--) std::swap(h[i], h[rand() % (i + 1)]);
    size_t* d; CHECK(cudaMalloc(&d, nchunk * sizeof(size_t)));
    CHECK(cudaMemcpy(d, h.data(), nchunk * sizeof(size_t), cudaMemcpyHostToDevice));
    double t = time_it([&]{ kblock<<<blocks,th>>>(a,d,chunk4,nchunk,out); });
    printf("block-gather %5zu KiB  %.1f GB/s  (%d chunks)\n", chunk_kib, bytes / (t * 1e6), nchunk);
    CHECK(cudaFree(d));
  }

  for (size_t pg_kib : {4, 64}) {
    size_t step4 = (pg_kib << 10) / sizeof(float4);
    size_t touched = (n4 / step4) * sizeof(float4);
    double t = time_it([&]{ kpagestride<<<blocks,th>>>(a,n4,step4,out); });
    printf("page-stride %3zu KiB   %.2f GB/s useful (%.1f M accesses)\n",
           pg_kib, touched / (t * 1e6), (n4 / step4) / 1e6);
  }
  return 0;
}

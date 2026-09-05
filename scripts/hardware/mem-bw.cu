// GB10 achieved memory bandwidth: read, copy, triad.
// Decode is weight-read bound, so the read kernel is the number that matters.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// Pure read: vectorized float4 loads, reduced so the compiler cannot elide them.
__global__ void kread(const float4* __restrict__ a, size_t n4, float* __restrict__ out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n4; i += stride) {
    float4 v = a[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1.2345e-30f) out[0] = acc;  // never true; keeps loads live
}

__global__ void kcopy(const float4* __restrict__ a, float4* __restrict__ b, size_t n4) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) b[i] = a[i];
}

__global__ void ktriad(const float4* __restrict__ a, const float4* __restrict__ b,
                       float4* __restrict__ c, size_t n4, float s) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float4 x = a[i], y = b[i];
    c[i] = make_float4(x.x + s * y.x, x.y + s * y.y, x.z + s * y.z, x.w + s * y.w);
  }
}

__global__ void kfill(float4* a, size_t n4, float v) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) a[i] = make_float4(v, v, v, v);
}

int main(int argc, char** argv) {
  // bytes per array
  size_t gib = (argc > 1) ? atoll(argv[1]) : 8;
  int reps = (argc > 2) ? atoi(argv[2]) : 12;
  size_t bytes = gib << 30;
  size_t n4 = bytes / sizeof(float4);

  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));
  int blocks = p.multiProcessorCount * 32;
  int threads = 256;

  float4 *a, *b, *c;
  float* out;
  CHECK(cudaMalloc(&a, bytes));
  CHECK(cudaMalloc(&b, bytes));
  CHECK(cudaMalloc(&c, bytes));
  CHECK(cudaMalloc(&out, sizeof(float)));
  kfill<<<blocks, threads>>>(a, n4, 1.0f);
  kfill<<<blocks, threads>>>(b, n4, 2.0f);
  kfill<<<blocks, threads>>>(c, n4, 0.0f);
  CHECK(cudaDeviceSynchronize());

  cudaEvent_t e0, e1;
  CHECK(cudaEventCreate(&e0));
  CHECK(cudaEventCreate(&e1));

  auto run = [&](const char* name, size_t moved, auto&& launch) {
    std::vector<double> ms;
    for (int r = 0; r < reps; r++) {
      CHECK(cudaEventRecord(e0));
      launch();
      CHECK(cudaEventRecord(e1));
      CHECK(cudaEventSynchronize(e1));
      float t;
      CHECK(cudaEventElapsedTime(&t, e0, e1));
      ms.push_back(t);
    }
    CHECK(cudaGetLastError());
    std::sort(ms.begin(), ms.end());
    double best = ms.front(), med = ms[ms.size() / 2], worst = ms.back();
    // GB/s, decimal, to match the 273 GB/s datasheet convention
    printf("{\"kernel\":\"%s\",\"gib_per_array\":%zu,\"bytes_moved\":%zu,"
           "\"best_ms\":%.3f,\"median_ms\":%.3f,\"worst_ms\":%.3f,"
           "\"best_gbs\":%.1f,\"median_gbs\":%.1f}\n",
           name, gib, moved, best, med, worst,
           moved / (best * 1e6), moved / (med * 1e6));
  };

  run("read",  bytes,     [&] { kread<<<blocks, threads>>>(a, n4, out); });
  run("copy",  2 * bytes, [&] { kcopy<<<blocks, threads>>>(a, b, n4); });
  run("triad", 3 * bytes, [&] { ktriad<<<blocks, threads>>>(a, b, c, n4, 3.0f); });

  CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(c)); CHECK(cudaFree(out));
  return 0;
}

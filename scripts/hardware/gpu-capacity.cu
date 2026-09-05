// What does CUDA actually expose on a unified-memory GB10, against MemTotal?
// Allocates in chunks until failure, so the answer is achieved, not reported.
// On unified memory cudaMalloc does not fail before the host OOMs, so the
// probe stops itself when host MemAvailable drops below a floor (arg 2, GiB).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

static double gib(size_t b) { return (double)b / (1024.0 * 1024.0 * 1024.0); }

// Host MemAvailable in bytes, or 0 on parse failure (treated as exhausted).
static size_t mem_available(void) {
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  size_t kb = 0;
  while (fgets(line, sizeof line, f)) {
    if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
  }
  fclose(f);
  return kb << 10;
}

int main(int argc, char** argv) {
  size_t chunk_mib = (argc > 1) ? atoll(argv[1]) : 512;
  size_t chunk = chunk_mib << 20;
  size_t floor_gib = (argc > 2) ? atoll(argv[2]) : 12;
  size_t floor = floor_gib << 30;

  size_t f0 = 0, t0 = 0;
  cudaError_t e = cudaMemGetInfo(&f0, &t0);
  printf("cudaMemGetInfo: %s free=%.2f GiB total=%.2f GiB\n",
         e == cudaSuccess ? "ok" : cudaGetErrorString(e), gib(f0), gib(t0));
  printf("host MemAvailable at start: %.2f GiB, floor %.2f GiB\n",
         gib(mem_available()), gib(floor));

  std::vector<void*> blocks;
  size_t got = 0;
  bool hit_floor = false;
  for (;;) {
    if (mem_available() < floor + chunk) { hit_floor = true; break; }
    void* p = nullptr;
    if (cudaMalloc(&p, chunk) != cudaSuccess) { cudaGetLastError(); break; }
    blocks.push_back(p);
    got += chunk;
  }
  printf("cudaMalloc achieved: %.2f GiB in %zu x %zu MiB chunks%s\n",
         gib(got), blocks.size(), chunk_mib,
         hit_floor ? " (stopped at host floor)" : " (allocator refused)");

  // Touch every block from the device so the pages are really backed.
  size_t touched = 0;
  for (void* p : blocks) {
    if (mem_available() < floor) { hit_floor = true; break; }
    if (cudaMemset(p, 1, chunk) != cudaSuccess) { cudaGetLastError(); break; }
    touched += chunk;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) cudaGetLastError();
  printf("device-touched:      %.2f GiB\n", gib(touched));

  size_t f1 = 0, t1 = 0;
  cudaMemGetInfo(&f1, &t1);
  printf("after alloc:         free=%.2f GiB total=%.2f GiB\n", gib(f1), gib(t1));
  printf("host MemAvailable at end:   %.2f GiB\n", gib(mem_available()));

  for (void* p : blocks) cudaFree(p);
  return 0;
}

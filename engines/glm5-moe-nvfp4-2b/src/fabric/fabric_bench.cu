// What the RoCE transport in fabric.h actually delivers at the message sizes
// the expert-parallel decode step uses, measured before anything is built on
// top of it.
//
// Three modes, all on cudaHostAlloc'd memory (the decode step's staging
// buffers are the same allocation, so this measures the real configuration
// and not a malloc'd stand-in):
//   pingpong   symmetric Fabric::exchange, one in flight. Reports the time a
//              decode step would actually block per MoE layer.
//   duplex     both ranks stream `depth` writes per doorbell, both directions
//              at once. This is the shape of the row exchange under load.
//   oneway     rank 0 writes, rank 1 only acknowledges. Directly comparable
//              to ib_write_bw, which is also unidirectional.
//
// It also reports device read bandwidth over the registered host region,
// because the whole no-extra-copy argument for staging in cudaHostAlloc
// memory rests on the GPU reaching that memory at normal speed on GB10.
//
// Run through scripts/fabric/fabric-microbench.sh, which starts both ranks.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fabric/fabric.h"

namespace {

using Clock = std::chrono::steady_clock;
double us_since(Clock::time_point t) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}

const char* arg_value(int argc, char** argv, const char* key, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return fallback;
}

__global__ void read_region(const float4* __restrict__ a, size_t n4, float* __restrict__ out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n4; i += stride) {
    const float4 v = a[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1.2345e-30f) out[0] = acc;
}

// Median is reported for pingpong because a single RoCE exchange occasionally
// picks up a scheduler or interrupt delay an order of magnitude above the
// rest, and a mean over a few thousand iterations hides neither.
double median(std::vector<double>& v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  rocket::fabric::Config cfg;
  cfg.rank = std::atoi(arg_value(argc, argv, "--rank", "0"));
  cfg.bootstrap_host = arg_value(argc, argv, "--host", "192.168.100.10");
  cfg.bootstrap_port = std::atoi(arg_value(argc, argv, "--port", "18777"));
  const int rails = std::atoi(arg_value(argc, argv, "--rails", "2"));
  if (rails == 1) cfg.devices = {"rocep1s0f1"};
  cfg.rail_split_min_bytes =
      static_cast<std::size_t>(std::atoll(arg_value(argc, argv, "--split-min", "65536")));
  const int depth = std::atoi(arg_value(argc, argv, "--depth", "32"));

  const std::size_t kMax = 4u << 20;
  const std::size_t region_bytes = kMax * static_cast<std::size_t>(depth + 1);

  void* buf = nullptr;
  if (cudaHostAlloc(&buf, region_bytes, cudaHostAllocDefault) != cudaSuccess) {
    std::fprintf(stderr, "cudaHostAlloc %zu bytes failed\n", region_bytes);
    return 1;
  }
  std::memset(buf, cfg.rank + 1, region_bytes);

  rocket::fabric::Fabric fab(cfg);
  const int reg = fab.register_region(buf, region_bytes);
  fab.barrier();

  if (cfg.rank == 0) {
    std::printf("rocket fabric microbench: rails=%d split-min=%zu depth=%d region=%.0f MiB\n",
                fab.rails(), cfg.rail_split_min_bytes, depth, region_bytes / 1048576.0);
    std::printf("%10s %14s %14s %14s %14s\n", "bytes", "pingpong us", "duplex GB/s",
                "oneway GB/s", "duplex us");
  }

  const std::size_t sizes[] = {2048,   8192,    32768,   131072,
                               524288, 1048576, 2097152, 4194304};

  // Iteration counts are derived from the size alone, so both ranks run the
  // same number of doorbell rounds. A time-bounded loop would let one rank
  // finish a size first and leave the other blocked on a sequence number that
  // is never posted.
  auto clampi = [](long long v, long long lo, long long hi) {
    return static_cast<long long>(v < lo ? lo : (v > hi ? hi : v));
  };

  for (const std::size_t n : sizes) {
    const long long ping_iters =
        clampi(static_cast<long long>((32u << 20) / n), 300, 4000);
    const long long batches =
        clampi(static_cast<long long>((1ull << 30) / (n * static_cast<std::size_t>(depth))), 4, 4096);

    // --- pingpong: one symmetric exchange in flight -------------------------
    for (int i = 0; i < 50; ++i) fab.exchange(reg, 0, kMax, n);
    std::vector<double> lat;
    lat.reserve(static_cast<std::size_t>(ping_iters));
    for (long long i = 0; i < ping_iters; ++i) {
      const auto t0 = Clock::now();
      fab.exchange(reg, 0, kMax, n);
      lat.push_back(us_since(t0));
    }
    const double ping_us = median(lat);

    // --- duplex and oneway: `depth` writes per doorbell ----------------------
    double streamed[2] = {0.0, 0.0};
    std::vector<double> batch_us;
    for (int mode = 0; mode < 2; ++mode) {  // 0 duplex, 1 oneway (rank 0 sends)
      const std::size_t local = (mode == 0 || cfg.rank == 0) ? n : 0;
      for (int b = 0; b < 2; ++b) {  // warmup
        for (int d = 0; d < depth; ++d)
          fab.post_write(reg, 0, kMax * static_cast<std::size_t>(d + 1), local);
        const std::uint64_t s = fab.next_seq();
        fab.signal(s);
        fab.wait_peer(s);
        fab.flush();
      }
      fab.barrier();
      const auto t0 = Clock::now();
      for (long long b = 0; b < batches; ++b) {
        const auto tb = Clock::now();
        for (int d = 0; d < depth; ++d)
          fab.post_write(reg, 0, kMax * static_cast<std::size_t>(d + 1), local);
        const std::uint64_t s = fab.next_seq();
        fab.signal(s);
        fab.wait_peer(s);
        fab.flush();
        if (mode == 0) batch_us.push_back(us_since(tb));
      }
      const double elapsed_us = us_since(t0);
      const double bytes = static_cast<double>(batches) * depth * static_cast<double>(n);
      // duplex counts both directions, oneway counts rank 0's direction only
      const double total = (mode == 0) ? 2.0 * bytes : bytes;
      streamed[mode] = total / (elapsed_us * 1000.0);  // bytes/us -> GB/s
      fab.barrier();
    }
    const double dup_us = median(batch_us);

    if (cfg.rank == 0)
      std::printf("%10zu %14.2f %14.2f %14.2f %14.1f\n", n, ping_us, streamed[0], streamed[1],
                  dup_us);
    std::fflush(stdout);
  }

  // --- device bandwidth over the registered host region ---------------------
  fab.barrier();
  if (cfg.rank == 0) {
    const std::size_t n4 = region_bytes / sizeof(float4);
    float* sink = nullptr;
    cudaMalloc(&sink, sizeof(float));
    for (int i = 0; i < 3; ++i) {
      read_region<<<1024, 256>>>(static_cast<const float4*>(buf), n4, sink);
      cudaDeviceSynchronize();
    }
    double best = 0.0;
    for (int i = 0; i < 8; ++i) {
      const auto t0 = Clock::now();
      read_region<<<1024, 256>>>(static_cast<const float4*>(buf), n4, sink);
      cudaDeviceSynchronize();
      const double gbps = static_cast<double>(region_bytes) / (us_since(t0) * 1000.0);
      best = std::max(best, gbps);
    }
    std::printf("\ndevice read over the registered cudaHostAlloc region: %.1f GB/s (%.0f MiB)\n",
                best, region_bytes / 1048576.0);
    cudaFree(sink);
  }

  fab.barrier();
  const rocket::fabric::Stats& s = fab.stats();
  std::printf("rank %d: %llu exchanges, %.2f GiB out, post %.1f ms, wait %.1f ms\n", cfg.rank,
              static_cast<unsigned long long>(s.exchanges), s.bytes_out / 1073741824.0, s.post_ms,
              s.wait_ms);
  cudaFreeHost(buf);
  return 0;
}

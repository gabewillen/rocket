// The engine probes the booster it is mounted on. A booster is one GB10
// spark node and can mount one or more engines (blog/rocket.qmd#boosters).
//
// Everything here is read once at startup and compared against the compiled
// plan. Nothing in the serving path calls it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rocket {

struct MountFacts {
  std::string device_name;
  int compute_major = 0;
  int compute_minor = 0;
  int sm_count = 0;
  int max_threads_per_sm = 0;
  int max_blocks_per_sm = 0;
  int max_shared_bytes_per_sm = 0;
  int max_shared_bytes_per_block_optin = 0;
  int async_engine_count = 0;
  int concurrent_kernels = 0;
  bool unified_addressing = false;
  std::size_t device_free_bytes = 0;
  std::size_t device_total_bytes = 0;

  // Host side. Memory is unified, so these bound the engine as hard as the
  // device numbers do (blog/posts/hardware/2026-09-06-cuda-free-counts-page-cache/).
  std::size_t host_page_bytes = 0;
  std::size_t host_mem_available_bytes = 0;
  std::string kernel_release;
  std::string hostname;
};

// Throws std::runtime_error on any CUDA failure.
MountFacts probe_mount(int device_ordinal = 0);

// Runs one trivial kernel on the device and returns the value it wrote.
// Proves the compiled cubin loads and launches on this booster.
std::uint32_t ignite(int device_ordinal = 0);

// The architecture this translation unit was compiled for, as sm_<n>.
int compiled_arch();

}  // namespace rocket

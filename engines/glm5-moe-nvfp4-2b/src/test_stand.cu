// Build sanity target. Reports what this booster is, then launches one kernel.
//
// Refusals here mirror the ones the engine owes at startup
// (blog/rocket.qmd#launch): wrong architecture or wrong page size is a
// refusal, not a fallback.
#include "engine.h"

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

int expected_arch(const rocket::MountFacts& facts) {
  return facts.compute_major * 10 + facts.compute_minor;
}

}  // namespace

int main() {
  try {
    const rocket::MountFacts facts = rocket::probe_mount();

    std::printf("booster            %s (%s)\n",
                facts.hostname.c_str(), facts.kernel_release.c_str());
    std::printf("device             %s, cc %d.%d, %d SMs\n",
                facts.device_name.c_str(), facts.compute_major,
                facts.compute_minor, facts.sm_count);
    std::printf("compiled for       sm_%d\n", rocket::compiled_arch());
    std::printf("resident threads   %d (%d SMs x %d)\n",
                facts.sm_count * facts.max_threads_per_sm,
                facts.sm_count, facts.max_threads_per_sm);
    std::printf("blocks per SM      %d\n", facts.max_blocks_per_sm);
    std::printf("shared per SM      %d B, opt-in per block %d B\n",
                facts.max_shared_bytes_per_sm,
                facts.max_shared_bytes_per_block_optin);
    std::printf("concurrency        concurrent kernels %s, copy engines %d, unified addressing %s\n",
                facts.concurrent_kernels ? "yes" : "no",
                facts.async_engine_count,
                facts.unified_addressing ? "yes" : "no");
    std::printf("cudaMemGetInfo     free %.2f GiB of %.2f GiB (counts clean page cache as used)\n",
                facts.device_free_bytes / kGiB, facts.device_total_bytes / kGiB);
    std::printf("host page          %zu B, MemAvailable %.2f GiB\n",
                facts.host_page_bytes, facts.host_mem_available_bytes / kGiB);

    int failures = 0;

    if (rocket::compiled_arch() != expected_arch(facts)) {
      std::fprintf(stderr, "refuse: compiled sm_%d, device is sm_%d\n",
                   rocket::compiled_arch(), expected_arch(facts));
      ++failures;
    }
    if (facts.host_page_bytes != ROCKET_PAGE_BYTES) {
      std::fprintf(stderr, "refuse: plan is aligned to %d B pages, host uses %zu B\n",
                   ROCKET_PAGE_BYTES, facts.host_page_bytes);
      ++failures;
    }

    const std::uint32_t token = rocket::ignite();
    if (token != 0x52434B54u) {
      std::fprintf(stderr, "refuse: ignition kernel wrote 0x%08x\n", token);
      ++failures;
    } else {
      std::printf("ignition           kernel launched, wrote 0x%08x\n", token);
    }

    if (failures != 0) {
      std::fprintf(stderr, "test stand FAILED (%d)\n", failures);
      return EXIT_FAILURE;
    }
    std::printf("test stand        ok\n");
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "test stand FAILED: %s\n", error.what());
    return EXIT_FAILURE;
  }
}

#include "fabric/expert_parallel.h"

#include <chrono>
#include <stdexcept>
#include <string>

namespace rocket::fabric {
namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("rocket::fabric::expert_parallel: " + what);
}

void cuda_check(cudaError_t e, const std::string& what) {
  if (e != cudaSuccess) fail(what + ": " + cudaGetErrorString(e));
}

}  // namespace

ExpertParallel::ExpertParallel(const Config& cfg, int n_routed_experts, int max_rows, int width)
    : width_(width) {
  if (n_routed_experts % 2 != 0) fail("routed experts must split evenly across the pair");
  count_ = n_routed_experts / 2;
  first_ = cfg.rank * count_;

  fab_ = std::make_unique<Fabric>(cfg);

  // The staging region is cudaHostAlloc'd rather than cudaMalloc'd because
  // ibv_reg_mr needs pinned host pages, and on GB10 the LPDDR5X is unified, so
  // this is the same physical memory a device allocation would have used.
  // src/fabric/fabric_bench.cu measures the device reaching it at 201 GB/s
  // against 238 GB/s for cudaMalloc; the block staged here is a few hundred
  // KiB per layer, so that gap costs microseconds.
  stage_bytes_ = static_cast<std::size_t>(max_rows) * width * sizeof(std::uint16_t);
  cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&stage_), stage_bytes_, cudaHostAllocDefault),
             "staging cudaHostAlloc");
  if (fab_->register_region(stage_, stage_bytes_) != 0) fail("staging region is not handle 0");
  fab_->barrier();
}

ExpertParallel::~ExpertParallel() {
  // The Fabric goes first: its MRs point into stage_.
  fab_.reset();
  if (stage_ != nullptr) cudaFreeHost(stage_);
}

void ExpertParallel::exchange_rows(void* rows_dev, int own_lo, int own_n, int total_rows,
                                   cudaStream_t s, double* ms_sink) {
  const auto t0 = Clock::now();
  const std::size_t rb = static_cast<std::size_t>(width_) * sizeof(std::uint16_t);
  const std::size_t own_off = static_cast<std::size_t>(own_lo) * rb;
  const std::size_t own_bytes = static_cast<std::size_t>(own_n) * rb;
  const int own_hi = own_lo + own_n;
  if (static_cast<std::size_t>(total_rows) * rb > stage_bytes_)
    fail("row block of " + std::to_string(total_rows) + " rows exceeds the staging region");

  auto* rows = static_cast<std::uint8_t*>(rows_dev);
  if (own_bytes > 0)
    cuda_check(cudaMemcpyAsync(stage_ + own_off, rows + own_off, own_bytes, cudaMemcpyDefault, s),
               "own rows to staging");
  // The NIC reads this memory next, so the GPU's writes have to be complete
  // and not merely enqueued.
  cuda_check(cudaStreamSynchronize(s), "sync before RDMA write");

  fab_->exchange(0, own_off, own_off, own_bytes);

  // The peer wrote into exactly the rows this rank did not compute: the split
  // is by expert id and the row order is by expert id, so with two ranks the
  // complement of [own_lo, own_hi) is at most one prefix and one suffix.
  if (own_lo > 0)
    cuda_check(cudaMemcpyAsync(rows, stage_, own_off, cudaMemcpyDefault, s), "peer prefix rows");
  if (own_hi < total_rows)
    cuda_check(cudaMemcpyAsync(rows + static_cast<std::size_t>(own_hi) * rb,
                               stage_ + static_cast<std::size_t>(own_hi) * rb,
                               static_cast<std::size_t>(total_rows - own_hi) * rb,
                               cudaMemcpyDefault, s),
               "peer suffix rows");

  if (ms_sink != nullptr)
    *ms_sink += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace rocket::fabric

#include "fabric/expert_parallel.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

ExpertParallel::ExpertParallel(const Config& cfg, const std::vector<int>& owner, int max_rows,
                               int width)
    : owner_(owner), width_(width) {
  if (owner_.empty()) fail("no expert ownership vector");
  for (const int r : owner_)
    if (r != 0 && r != 1) fail("expert ownership names a rank other than 0 or 1");
  count_ = 0;
  for (const int r : owner_)
    if (r == cfg.rank) ++count_;
  if (count_ == 0) fail("this rank was given no routed experts to compute");

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

std::vector<int> ExpertParallel::owned_experts() const {
  std::vector<int> out;
  out.reserve(static_cast<std::size_t>(count_));
  for (std::size_t e = 0; e < owner_.size(); ++e)
    if (owner_[e] == rank()) out.push_back(static_cast<int>(e));
  return out;
}

bool ExpertParallel::contiguous() const {
  const std::vector<int> mine = owned_experts();
  for (std::size_t i = 1; i < mine.size(); ++i)
    if (mine[i] != mine[i - 1] + 1) return false;
  return true;
}

void ExpertParallel::exchange_begin(int off_rows, int n_rows) {
  if (in_flight_) fail("exchange_begin called twice without an exchange_end");
  const std::size_t rb = static_cast<std::size_t>(width_) * sizeof(std::uint16_t);
  const std::size_t off = static_cast<std::size_t>(off_rows) * rb;
  const std::size_t bytes = static_cast<std::size_t>(n_rows) * rb;
  if (off + bytes > stage_bytes_)
    fail("row block [" + std::to_string(off_rows) + ", " + std::to_string(off_rows + n_rows) +
         ") exceeds the staging region");
  const auto t0 = Clock::now();
  // One doorbell sequence number per exchange, taken from the fabric's own
  // counter rather than invented here, so barrier() and this share one stream
  // of values (fabric.h::next_seq).
  seq_ = fab_->next_seq();
  static const bool trace = std::getenv("ROCKET_FABRIC_TRACE") != nullptr;
  if (trace)
    std::fprintf(stderr, "[ep] rank=%d begin seq=%llu rows=%d off_rows=%d\n", rank(),
                 static_cast<unsigned long long>(seq_), n_rows, off_rows);
  fab_->post_write(0, off, off, bytes);
  fab_->signal(seq_);
  post_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  in_flight_ = true;
}

void ExpertParallel::exchange_end(double* ms_sink) {
  if (!in_flight_) fail("exchange_end without a matching exchange_begin");
  const auto t0 = Clock::now();
  static const bool trace = std::getenv("ROCKET_FABRIC_TRACE") != nullptr;
  if (trace) {
    std::fprintf(stderr, "[ep] rank=%d end   wait seq=%llu\n", rank(),
                 static_cast<unsigned long long>(seq_));
    std::fflush(stderr);
  }
  fab_->wait_peer(seq_);
  fab_->flush();
  in_flight_ = false;
  if (ms_sink != nullptr)
    *ms_sink += post_ms_ + std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  post_ms_ = 0.0;
}

void ExpertParallel::exchange_draft_candidates(const std::vector<int>& local_ids,
                                               const std::vector<float>& local_vals,
                                               std::vector<int>& peer_ids,
                                               std::vector<float>& peer_vals) {
  if (local_ids.size() != local_vals.size()) fail("draft candidate vector sizes differ");
  const std::size_t ids_bytes = local_ids.size() * sizeof(int);
  const std::size_t vals_bytes = local_vals.size() * sizeof(float);
  const std::size_t bytes = ids_bytes + vals_bytes;
  if (2 * bytes > stage_bytes_) fail("draft candidates exceed the staging region");
  auto* local = stage_ + static_cast<std::size_t>(rank()) * bytes;
  auto* peer = stage_ + static_cast<std::size_t>(1 - rank()) * bytes;
  std::memcpy(local, local_ids.data(), ids_bytes);
  std::memcpy(local + ids_bytes, local_vals.data(), vals_bytes);
  const std::uint64_t seq = fab_->next_seq();
  const std::size_t off = static_cast<std::size_t>(rank()) * bytes;
  fab_->post_write(0, off, off, bytes);
  fab_->signal(seq); fab_->flush(); fab_->wait_peer(seq); fab_->flush();
  peer_ids.resize(local_ids.size()); peer_vals.resize(local_vals.size());
  std::memcpy(peer_ids.data(), peer, ids_bytes);
  std::memcpy(peer_vals.data(), peer + ids_bytes, vals_bytes);
}

void ExpertParallel::sync_draft_tokens(std::vector<int>& tokens) {
  const std::size_t bytes = tokens.size() * sizeof(int);
  if (bytes > stage_bytes_) fail("draft tokens exceed the staging region");
  auto* control = reinterpret_cast<int*>(stage_);
  const std::uint64_t seq = fab_->next_seq();
  if (rank() == 0) {
    std::copy(tokens.begin(), tokens.end(), control);
    fab_->post_write(0, 0, 0, bytes);
    fab_->signal(seq); fab_->flush();
  } else {
    fab_->wait_peer(seq);
    std::copy(control, control + tokens.size(), tokens.begin());
    fab_->signal(seq); fab_->flush();
  }
  if (rank() == 0) fab_->wait_peer(seq);
}

void ExpertParallel::sync_prefix_boundary(int& tokens) {
  auto* control = reinterpret_cast<int*>(stage_);
  control[rank()] = tokens;
  const std::uint64_t seq = fab_->next_seq();
  const std::size_t off = static_cast<std::size_t>(rank()) * sizeof(int);
  fab_->post_write(0, off, off, sizeof(int));
  fab_->signal(seq); fab_->flush(); fab_->wait_peer(seq); fab_->flush();
  const std::uint64_t decision_seq = fab_->next_seq();
  if (rank() == 0) {
    control[0] = std::min(control[0], control[1]);
    fab_->post_write(0, 0, 0, sizeof(int));
    fab_->signal(decision_seq); fab_->flush(); fab_->wait_peer(decision_seq);
    tokens = control[0];
  } else {
    fab_->wait_peer(decision_seq);
    tokens = control[0];
    fab_->signal(decision_seq); fab_->flush();
  }
}

void ExpertParallel::sync_round_decision(std::vector<int>& accepted, std::vector<int>& last) {
  if (accepted.size() != last.size()) fail("round decision vector sizes differ");
  const std::size_t n = accepted.size();
  const std::size_t bytes = 2 * n * sizeof(int);
  if (bytes > stage_bytes_) fail("round decision exceeds the staging region");
  auto* control = reinterpret_cast<int*>(stage_);
  const std::uint64_t seq = fab_->next_seq();
  if (rank() == 0) {
    std::copy(accepted.begin(), accepted.end(), control);
    std::copy(last.begin(), last.end(), control + n);
    fab_->post_write(0, 0, 0, bytes);
    fab_->signal(seq);
    fab_->flush();
  } else {
    fab_->wait_peer(seq);
    std::copy(control, control + n, accepted.begin());
    std::copy(control + n, control + 2 * n, last.begin());
    // Acknowledge consumption so rank 0 cannot reuse the staging region for
    // the next MoE exchange while rank 1 is still reading the decision.
    fab_->signal(seq);
    fab_->flush();
  }
  if (rank() == 0) fab_->wait_peer(seq);
}

}  // namespace rocket::fabric

// The two-booster routed-expert split.
//
// The 182 GiB NVFP4 checkpoint does not fit one booster's ~119.5 GiB
// (blog/posts/hardware/2026-09-05-spark-cluster-baseline/), so the pair is the
// engine's only arrangement. The 16.62 GiB BF16 half is replicated; the 288
// routed experts split 144/144 by id, rank 0 taking the low half.
//
// What crosses the fabric, and what does not:
//
//   nothing forward   Both ranks run the whole non-expert stack for all M
//                     streams on the same inputs with the same weights, so
//                     `normed_` is already bit-identical on both. There is no
//                     activation to send.
//   nothing meta      Both ranks run the same router on that same input, so
//                     both already know the complete (stream, expert) row set
//                     and the expert-ascending order run_moe_grouped puts it
//                     in. Neither rank has to be told how many rows the other
//                     computed or where they go.
//   the rows          Each rank runs the grouped GEMM for its own experts and
//                     writes the resulting [rows, hidden] BF16 block. Because
//                     the split is by expert id and the order is by expert id,
//                     rank 0's rows are exactly the prefix of that block and
//                     rank 1's exactly the suffix. One RDMA write each way,
//                     into the same offsets the sender used.
//
// Exchanging rows rather than partial sums is what makes token parity a
// property instead of a hope. After the exchange both ranks hold the identical
// full row set and run the identical `moe_scatter_add`, which accumulates in
// FP32 over rows in ascending order and rounds to BF16 once
// (kernels.cu::moe_scatter_add_kernel), so the accumulator is bit-identical to
// the single-booster one. Summing FP32 partials instead would be half the
// bytes and not bit-identical, because (a+b)+(c+d) is not ((a+b)+c)+d.
//
// Cost of that choice, at hidden=4096: 8 KiB per row per direction, against
// 16 KiB per stream per direction for a partial-sum allreduce. It is more
// bytes above top_k=4 and fewer below, and at the measured rates neither is
// close to the MoE compute it overlaps.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "fabric/fabric.h"

namespace rocket::fabric {

class ExpertParallel {
 public:
  // `owner` is [n_routed_experts] with owner[e] the rank that computes expert
  // e; both ranks must pass the identical vector, which they do because both
  // derive it from the same histogram (src/fabric/expert_balance.h) or from
  // the same id split. `max_rows` bounds max_batch * num_experts_per_tok;
  // `width` is hidden_size in elements. Throws std::runtime_error if the
  // fabric or the allocation fails.
  ExpertParallel(const Config& cfg, const std::vector<int>& owner, int max_rows, int width);
  ~ExpertParallel();
  ExpertParallel(const ExpertParallel&) = delete;
  ExpertParallel& operator=(const ExpertParallel&) = delete;

  int rank() const { return fab_->rank(); }
  int expert_count() const { return count_; }
  bool owns(int expert_id) const {
    return expert_id >= 0 && expert_id < static_cast<int>(owner_.size()) &&
           owner_[static_cast<std::size_t>(expert_id)] == rank();
  }
  int owner_of(int expert_id) const { return owner_[static_cast<std::size_t>(expert_id)]; }
  std::vector<int> owned_experts() const;
  // True when this rank's experts are one contiguous ascending id range, which
  // is the arrangement the id split produces and the balanced partition does
  // not. Reported so a run says which split it measured.
  bool contiguous() const;

  // The registered staging region, [max_rows, width] BF16. The engine gathers
  // this rank's computed expert-output rows into its own slice of it and reads
  // the peer's rows out of the peer's slice; the fabric only moves bytes and
  // never interprets rows (model.cu::run_moe_grouped owns the layout).
  void* stage() { return stage_; }

  // Sends [off_rows, off_rows + n_rows) of the staging region to the same
  // offset in the peer's, and returns without waiting. The caller must have
  // completed the device writes into that slice first: the NIC reads this
  // memory next, so a cudaStreamSynchronize has to have happened, not just an
  // enqueue.
  void exchange_begin(int off_rows, int n_rows);
  // Waits for the peer's matching write to land. Adds the exposed cost of the
  // split to *ms_sink when that is not null: the time posting this rank's own
  // write plus the time blocked on the peer's doorbell, and nothing else. The
  // local sync before exchange_begin is this rank's own GEMM finishing, so it
  // is compute and is deliberately not counted here.
  void exchange_end(double* ms_sink);

  void barrier() { fab_->barrier(); }
  const Stats& stats() const { return fab_->stats(); }
  void reset_stats() { fab_->reset_stats(); }
  Fabric& fabric() { return *fab_; }

 private:
  std::unique_ptr<Fabric> fab_;
  std::uint8_t* stage_ = nullptr;  // cudaHostAlloc, registered with both rails
  std::size_t stage_bytes_ = 0;
  std::vector<int> owner_;
  int count_ = 0, width_ = 0;
  std::uint64_t seq_ = 0;
  double post_ms_ = 0.0;
  bool in_flight_ = false;
};

}  // namespace rocket::fabric

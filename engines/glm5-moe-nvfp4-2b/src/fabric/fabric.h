// Point-to-point RoCE transport between the two boosters of one rocket.
//
// The pair is symmetric and fixed at two ranks: rank 0 is the head, rank 1 the
// peer (blog/rocket.qmd#boosters). There is no collective library here and no
// process group. The only thing the decode step needs across the fabric is a
// symmetric exchange of routed-expert output rows, once per MoE layer, whose
// size both ranks already know without asking (src/fabric/README below), so a
// two-rank RC transport with a doorbell is the whole requirement.
//
// Transport shape, per rail:
//   payload   one RDMA_WRITE, unsignaled, into the peer's registered region
//   doorbell  one 8-byte inline RDMA_WRITE of a sequence number, signaled
//
// RC guarantees writes on one queue pair land in order, so a doorbell value
// arriving means that rail's payload is already in the peer's memory. The
// receiver polls its own doorbell words; nothing is posted to a receive queue
// and no completion is processed on the receive side. The doorbell is written
// on every rail whether or not that rail carried payload, so the receiver
// waits on a fixed number of words and never has to be told how the sender
// split the message.
//
// Both rails are used because either one alone is bounded by a PCIe Gen5 x4
// host link at 111.86 Gb/s and the pair reaches 23.1 GB/s
// (blog/posts/hardware/2026-09-06-measured-bandwidth/). Messages below
// Config::rail_split_min_bytes go on rail 0 alone, because splitting a small
// message adds a second doorbell round to save nothing.
//
// Registered memory is whatever the caller hands over. On GB10 the LPDDR5X is
// unified, so a cudaHostAlloc region is readable by kernels at full device
// bandwidth and registers with ibv_reg_mr as ordinary pinned host pages; no
// GPUDirect peer memory path is needed to keep the NIC and the GPU on the same
// bytes. src/fabric/fabric_bench.cu measures that claim.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocket::fabric {

struct Config {
  int rank = 0;  // 0 listens, 1 connects
  // rank 0's rail-1 address. Both ranks name the same host here.
  std::string bootstrap_host = "192.168.100.10";
  int bootstrap_port = 18777;
  std::vector<std::string> devices{"rocep1s0f1", "roceP2p1s0f1"};
  int gid_index = 3;  // RoCE v2, IPv4 (scripts/hardware/fabric-bw.sh)
  std::size_t rail_split_min_bytes = 65536;
};

// Cumulative fabric cost, read by the engine's per-step telemetry
// (model.h::StageMs::fabric). wait_ms is time blocked on the peer's doorbell
// and therefore includes the peer's compute skew, so it is an upper bound on
// transport cost, not transport cost alone; post_ms is this rank's own time
// inside ibv_post_send.
struct Stats {
  std::uint64_t exchanges = 0;
  std::uint64_t bytes_out = 0;
  std::uint64_t bytes_in = 0;
  double post_ms = 0.0;
  double wait_ms = 0.0;
};

class Fabric {
 public:
  // Connects both rails and both ranks. Throws std::runtime_error on any
  // verbs or bootstrap failure; a half-open fabric is never returned.
  explicit Fabric(const Config& cfg);
  ~Fabric();
  Fabric(const Fabric&) = delete;
  Fabric& operator=(const Fabric&) = delete;

  int rank() const { return rank_; }
  int peer_rank() const { return 1 - rank_; }

  // Registers a region both ranks allocate at the same size. Both ranks must
  // register the same regions in the same order; the returned handle is the
  // registration index and is the same on both. Blocks until the peer has
  // registered its matching region.
  int register_region(void* addr, std::size_t bytes);

  // Writes [src_off, src_off + bytes) of this rank's region into the peer's
  // region at dst_off, and returns once the peer's own matching write into
  // this rank's memory has landed. The two ranks may pass different `bytes`.
  void exchange(int handle, std::size_t src_off, std::size_t dst_off, std::size_t bytes);

  // Doorbell-only rendezvous. Same cost as an exchange of zero payload.
  void barrier();

  // --- pipelined form, used by the microbench and by nothing in the engine ---
  void post_write(int handle, std::size_t src_off, std::size_t dst_off, std::size_t bytes);
  // Doorbell values are one monotonic sequence per rank, shared by exchange(),
  // barrier(), and the pipelined form. A caller that invents its own sequence
  // numbers can hand wait_peer() a value the doorbell already passed, which
  // returns without the peer having written anything; take the next value from
  // here instead.
  std::uint64_t next_seq();
  void signal(std::uint64_t seq);      // doorbell on every rail
  void flush();                        // wait for this rank's own writes
  void wait_peer(std::uint64_t seq);   // wait for the peer's doorbell

  int rails() const;
  const Stats& stats() const;
  void reset_stats();

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  int rank_ = 0;
};

}  // namespace rocket::fabric

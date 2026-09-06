#include "fabric/fabric.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <infiniband/verbs.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace rocket::fabric {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("rocket::fabric: " + what);
}
[[noreturn]] void fail_errno(const std::string& what) {
  fail(what + ": " + std::strerror(errno));
}

// Doorbell words are one per rail and are the only memory the peer polls, so
// they are spread a cache line apart: two rails landing in one line would put
// the NIC's writes for rail 0 and rail 1 in the same coherence unit.
constexpr std::size_t kDoorStride = 64;
constexpr int kWaitTimeoutSeconds = 120;

// Exchanged over the bootstrap socket. Fixed-width and byte-copied; both ends
// are the same aarch64 build, so no conversion is done and none is implied.
struct RailInfo {
  std::uint32_t qpn = 0;
  std::uint32_t psn = 0;
  std::uint32_t door_rkey = 0;
  std::uint32_t pad = 0;
  std::uint64_t door_addr = 0;
  std::uint8_t gid[16] = {};
};

struct RegionInfo {
  std::uint64_t addr = 0;
  std::uint64_t bytes = 0;
  std::uint32_t rkey[8] = {};
};

void write_all(int fd, const void* buf, std::size_t n) {
  const auto* p = static_cast<const std::uint8_t*>(buf);
  while (n > 0) {
    const ssize_t w = ::write(fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      fail_errno("bootstrap write");
    }
    p += w;
    n -= static_cast<std::size_t>(w);
  }
}

void read_all(int fd, void* buf, std::size_t n) {
  auto* p = static_cast<std::uint8_t*>(buf);
  while (n > 0) {
    const ssize_t r = ::read(fd, p, n);
    if (r == 0) fail("bootstrap peer closed the connection");
    if (r < 0) {
      if (errno == EINTR) continue;
      fail_errno("bootstrap read");
    }
    p += r;
    n -= static_cast<std::size_t>(r);
  }
}

int bootstrap_listen(const std::string& host, int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) fail_errno("bootstrap socket");
  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  a.sin_addr.s_addr = INADDR_ANY;
  (void)host;
  if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) fail_errno("bootstrap bind");
  if (::listen(fd, 1) != 0) fail_errno("bootstrap listen");
  const int c = ::accept(fd, nullptr, nullptr);
  ::close(fd);
  if (c < 0) fail_errno("bootstrap accept");
  ::setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  return c;
}

// rank 1 may start before rank 0 has bound; ssh launch ordering is not a
// synchronization primitive, so retry for a bounded time instead.
int bootstrap_connect(const std::string& host, int port) {
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) fail("bad bootstrap host " + host);
  const auto t0 = Clock::now();
  for (;;) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) fail_errno("bootstrap socket");
    if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0) {
      int one = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      return fd;
    }
    ::close(fd);
    if (ms_since(t0) > 60000.0) fail("bootstrap connect to " + host + " timed out");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

ibv_context* open_device(const std::string& name) {
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  if (list == nullptr) fail_errno("ibv_get_device_list");
  ibv_context* ctx = nullptr;
  for (int i = 0; i < n; ++i) {
    if (name == ibv_get_device_name(list[i])) {
      ctx = ibv_open_device(list[i]);
      break;
    }
  }
  ibv_free_device_list(list);
  if (ctx == nullptr) fail("no RDMA device named " + name);
  return ctx;
}

}  // namespace

struct Fabric::Impl {
  struct Rail {
    std::string name;
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* cq = nullptr;
    ibv_qp* qp = nullptr;
    ibv_mr* door_mr = nullptr;
    std::uint32_t peer_qpn = 0;
    std::uint32_t peer_door_rkey = 0;
    std::uint64_t peer_door_addr = 0;
    ibv_mtu mtu = IBV_MTU_1024;
    std::uint64_t seq_scratch = 0;  // inline source for the doorbell write
    int outstanding = 0;            // signaled sends not yet reaped
  };

  struct Region {
    void* addr = nullptr;
    std::size_t bytes = 0;
    std::vector<ibv_mr*> mr;  // one per rail, one PD each
    std::uint64_t peer_addr = 0;
    std::uint32_t peer_rkey[8] = {};
  };

  Config cfg;
  int sock = -1;
  std::vector<Rail> rails;
  std::vector<Region> regions;
  std::uint8_t* door = nullptr;  // rails * kDoorStride, written by the peer
  std::uint64_t seq = 0;
  Stats stats;

  volatile std::uint64_t* door_word(int r) {
    return reinterpret_cast<volatile std::uint64_t*>(door + static_cast<std::size_t>(r) * kDoorStride);
  }
};

Fabric::Fabric(const Config& cfg) : p_(new Impl), rank_(cfg.rank) {
  p_->cfg = cfg;
  if (cfg.rank != 0 && cfg.rank != 1) fail("rank must be 0 or 1");
  if (cfg.devices.empty()) fail("no rails configured");
  if (cfg.devices.size() > 8) fail("at most 8 rails (RegionInfo::rkey)");

  const int nr = static_cast<int>(cfg.devices.size());
  p_->rails.resize(static_cast<std::size_t>(nr));

  const std::size_t door_bytes = static_cast<std::size_t>(nr) * kDoorStride;
  p_->door = static_cast<std::uint8_t*>(std::aligned_alloc(kDoorStride, door_bytes));
  if (p_->door == nullptr) fail("doorbell allocation");
  std::memset(p_->door, 0, door_bytes);

  std::vector<RailInfo> mine(static_cast<std::size_t>(nr)), theirs(static_cast<std::size_t>(nr));

  for (int r = 0; r < nr; ++r) {
    Impl::Rail& rail = p_->rails[static_cast<std::size_t>(r)];
    rail.name = cfg.devices[static_cast<std::size_t>(r)];
    rail.ctx = open_device(rail.name);
    rail.pd = ibv_alloc_pd(rail.ctx);
    if (rail.pd == nullptr) fail("ibv_alloc_pd on " + rail.name);
    rail.cq = ibv_create_cq(rail.ctx, 512, nullptr, nullptr, 0);
    if (rail.cq == nullptr) fail("ibv_create_cq on " + rail.name);
    rail.door_mr = ibv_reg_mr(rail.pd, p_->door, door_bytes,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (rail.door_mr == nullptr) fail("ibv_reg_mr doorbell on " + rail.name);

    ibv_qp_init_attr qa{};
    qa.send_cq = rail.cq;
    qa.recv_cq = rail.cq;
    qa.qp_type = IBV_QPT_RC;
    qa.sq_sig_all = 0;
    qa.cap.max_send_wr = 256;
    qa.cap.max_recv_wr = 1;
    qa.cap.max_send_sge = 1;
    qa.cap.max_recv_sge = 1;
    qa.cap.max_inline_data = 64;
    rail.qp = ibv_create_qp(rail.pd, &qa);
    if (rail.qp == nullptr) fail("ibv_create_qp on " + rail.name);

    ibv_port_attr pa{};
    if (ibv_query_port(rail.ctx, 1, &pa) != 0) fail("ibv_query_port on " + rail.name);
    if (pa.state != IBV_PORT_ACTIVE) fail(rail.name + " port 1 is not ACTIVE");
    rail.mtu = pa.active_mtu;

    ibv_gid gid{};
    if (ibv_query_gid(rail.ctx, 1, cfg.gid_index, &gid) != 0)
      fail("ibv_query_gid " + std::to_string(cfg.gid_index) + " on " + rail.name);

    RailInfo& info = mine[static_cast<std::size_t>(r)];
    info.qpn = rail.qp->qp_num;
    info.psn = static_cast<std::uint32_t>(0x1000 + r * 0x100 + cfg.rank);
    info.door_rkey = rail.door_mr->rkey;
    info.door_addr = reinterpret_cast<std::uint64_t>(p_->door);
    std::memcpy(info.gid, gid.raw, 16);
  }

  p_->sock = (cfg.rank == 0) ? bootstrap_listen(cfg.bootstrap_host, cfg.bootstrap_port)
                             : bootstrap_connect(cfg.bootstrap_host, cfg.bootstrap_port);

  // Symmetric: both ranks write first. The socket buffer holds a few hundred
  // bytes, so neither side can block the other here.
  write_all(p_->sock, mine.data(), mine.size() * sizeof(RailInfo));
  read_all(p_->sock, theirs.data(), theirs.size() * sizeof(RailInfo));

  for (int r = 0; r < nr; ++r) {
    Impl::Rail& rail = p_->rails[static_cast<std::size_t>(r)];
    const RailInfo& t = theirs[static_cast<std::size_t>(r)];
    rail.peer_qpn = t.qpn;
    rail.peer_door_rkey = t.door_rkey;
    rail.peer_door_addr = t.door_addr;

    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(rail.qp, &a,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0)
      fail("qp -> INIT on " + rail.name);

    ibv_qp_attr rtr{};
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = rail.mtu;
    rtr.dest_qp_num = t.qpn;
    rtr.rq_psn = t.psn;
    rtr.max_dest_rd_atomic = 1;
    rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = 1;  // RoCE v2 is always GRH
    rtr.ah_attr.port_num = 1;
    rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0;
    rtr.ah_attr.grh.hop_limit = 64;
    rtr.ah_attr.grh.sgid_index = static_cast<std::uint8_t>(cfg.gid_index);
    rtr.ah_attr.grh.traffic_class = 0;
    rtr.ah_attr.grh.flow_label = 0;
    std::memcpy(rtr.ah_attr.grh.dgid.raw, t.gid, 16);
    if (ibv_modify_qp(rail.qp, &rtr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0)
      fail("qp -> RTR on " + rail.name);

    ibv_qp_attr rts{};
    rts.qp_state = IBV_QPS_RTS;
    rts.sq_psn = mine[static_cast<std::size_t>(r)].psn;
    rts.timeout = 14;
    rts.retry_cnt = 7;
    rts.rnr_retry = 7;
    rts.max_rd_atomic = 1;
    if (ibv_modify_qp(rail.qp, &rts,
                      IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC) != 0)
      fail("qp -> RTS on " + rail.name);
  }
}

Fabric::~Fabric() {
  if (!p_) return;
  for (auto& reg : p_->regions)
    for (ibv_mr* mr : reg.mr)
      if (mr != nullptr) ibv_dereg_mr(mr);
  for (auto& rail : p_->rails) {
    if (rail.qp != nullptr) ibv_destroy_qp(rail.qp);
    if (rail.door_mr != nullptr) ibv_dereg_mr(rail.door_mr);
    if (rail.cq != nullptr) ibv_destroy_cq(rail.cq);
    if (rail.pd != nullptr) ibv_dealloc_pd(rail.pd);
    if (rail.ctx != nullptr) ibv_close_device(rail.ctx);
  }
  if (p_->door != nullptr) std::free(p_->door);
  if (p_->sock >= 0) ::close(p_->sock);
}

int Fabric::rails() const { return static_cast<int>(p_->rails.size()); }
const Stats& Fabric::stats() const { return p_->stats; }
void Fabric::reset_stats() { p_->stats = Stats{}; }

int Fabric::register_region(void* addr, std::size_t bytes) {
  Impl::Region reg;
  reg.addr = addr;
  reg.bytes = bytes;
  RegionInfo mine{};
  mine.addr = reinterpret_cast<std::uint64_t>(addr);
  mine.bytes = bytes;
  for (std::size_t r = 0; r < p_->rails.size(); ++r) {
    ibv_mr* mr = ibv_reg_mr(p_->rails[r].pd, addr, bytes,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (mr == nullptr)
      fail("ibv_reg_mr of " + std::to_string(bytes) + " bytes on " + p_->rails[r].name);
    reg.mr.push_back(mr);
    mine.rkey[r] = mr->rkey;
  }

  RegionInfo theirs{};
  write_all(p_->sock, &mine, sizeof(mine));
  read_all(p_->sock, &theirs, sizeof(theirs));
  if (theirs.bytes != bytes)
    fail("peer registered " + std::to_string(theirs.bytes) + " bytes where this rank registered " +
         std::to_string(bytes) + "; both ranks must register the same regions in the same order");
  reg.peer_addr = theirs.addr;
  for (std::size_t r = 0; r < p_->rails.size(); ++r) reg.peer_rkey[r] = theirs.rkey[r];

  p_->regions.push_back(reg);
  return static_cast<int>(p_->regions.size()) - 1;
}

void Fabric::post_write(int handle, std::size_t src_off, std::size_t dst_off, std::size_t bytes) {
  if (bytes == 0) return;
  if (handle < 0 || handle >= static_cast<int>(p_->regions.size())) fail("bad region handle");
  Impl::Region& reg = p_->regions[static_cast<std::size_t>(handle)];
  if (src_off + bytes > reg.bytes || dst_off + bytes > reg.bytes) fail("write outside region");

  const int nr = static_cast<int>(p_->rails.size());
  // One rail below the split threshold: a second rail would add a second
  // doorbell round to a message already dominated by fixed cost.
  int use = (bytes >= p_->cfg.rail_split_min_bytes) ? nr : 1;
  const auto t0 = Clock::now();

  std::size_t done = 0;
  for (int r = 0; r < use; ++r) {
    const std::size_t remaining = bytes - done;
    std::size_t chunk = (r == use - 1) ? remaining : ((bytes / use) & ~std::size_t{63});
    if (chunk > remaining) chunk = remaining;
    if (chunk == 0) continue;

    Impl::Rail& rail = p_->rails[static_cast<std::size_t>(r)];
    ibv_sge sge{};
    sge.addr = reinterpret_cast<std::uint64_t>(reg.addr) + src_off + done;
    sge.length = static_cast<std::uint32_t>(chunk);
    sge.lkey = reg.mr[static_cast<std::size_t>(r)]->lkey;

    ibv_send_wr wr{};
    wr.wr_id = 0;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = 0;  // unsignaled: the doorbell's completion covers it
    wr.wr.rdma.remote_addr = reg.peer_addr + dst_off + done;
    wr.wr.rdma.rkey = reg.peer_rkey[static_cast<std::size_t>(r)];
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(rail.qp, &wr, &bad) != 0) fail_errno("ibv_post_send payload on " + rail.name);
    done += chunk;
  }
  p_->stats.bytes_out += bytes;
  p_->stats.post_ms += ms_since(t0);
}

std::uint64_t Fabric::next_seq() { return ++p_->seq; }

void Fabric::signal(std::uint64_t seq) {
  const auto t0 = Clock::now();
  for (auto& rail : p_->rails) {
    rail.seq_scratch = seq;
    ibv_sge sge{};
    sge.addr = reinterpret_cast<std::uint64_t>(&rail.seq_scratch);
    sge.length = sizeof(std::uint64_t);
    sge.lkey = 0;  // unused for inline

    ibv_send_wr wr{};
    wr.wr_id = seq;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    wr.wr.rdma.remote_addr = rail.peer_door_addr +
                             static_cast<std::uint64_t>(&rail - p_->rails.data()) * kDoorStride;
    wr.wr.rdma.rkey = rail.peer_door_rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(rail.qp, &wr, &bad) != 0) fail_errno("ibv_post_send doorbell on " + rail.name);
    ++rail.outstanding;
  }
  p_->stats.post_ms += ms_since(t0);
}

void Fabric::flush() {
  for (auto& rail : p_->rails) {
    const auto t0 = Clock::now();
    while (rail.outstanding > 0) {
      ibv_wc wc[16];
      const int n = ibv_poll_cq(rail.cq, 16, wc);
      if (n < 0) fail("ibv_poll_cq on " + rail.name);
      for (int i = 0; i < n; ++i) {
        if (wc[i].status != IBV_WC_SUCCESS)
          fail("completion on " + rail.name + ": " + ibv_wc_status_str(wc[i].status));
      }
      rail.outstanding -= n;
      if (n == 0 && ms_since(t0) > kWaitTimeoutSeconds * 1000.0)
        fail("send completion on " + rail.name + " timed out");
    }
  }
}

void Fabric::wait_peer(std::uint64_t seq) {
  const auto t0 = Clock::now();
  for (int r = 0; r < static_cast<int>(p_->rails.size()); ++r) {
    volatile std::uint64_t* w = p_->door_word(r);
    unsigned spins = 0;
    while (*w < seq) {
      if ((++spins & 0x3ff) == 0) {
        if (ms_since(t0) > kWaitTimeoutSeconds * 1000.0)
          fail("peer doorbell " + std::to_string(seq) + " on rail " + std::to_string(r) +
               " timed out (peer stalled or died)");
      }
#if defined(__aarch64__)
      asm volatile("yield" ::: "memory");
#endif
    }
  }
  // The doorbell landing means every payload byte on that rail landed with it
  // (RC ordering), but this thread may still read stale lines without a
  // barrier against its own subsequent loads.
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  p_->stats.wait_ms += ms_since(t0);
}

void Fabric::exchange(int handle, std::size_t src_off, std::size_t dst_off, std::size_t bytes) {
  const std::uint64_t seq = next_seq();
  post_write(handle, src_off, dst_off, bytes);
  signal(seq);
  wait_peer(seq);
  flush();
  ++p_->stats.exchanges;
}

void Fabric::barrier() {
  const std::uint64_t seq = next_seq();
  signal(seq);
  wait_peer(seq);
  flush();
}

}  // namespace rocket::fabric

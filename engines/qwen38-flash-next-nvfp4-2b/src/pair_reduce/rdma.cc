#include "pair_reduce/rdma.h"

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace rocket::qwen38::pair_reduce {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int kRails = 2;
constexpr std::size_t kDoorStride = 64;
constexpr std::size_t kReadyOffset = 0;
constexpr std::size_t kConsumedOffset = sizeof(std::uint64_t);
constexpr std::array<std::string_view, kRails> kDevices{"rocep1s0f1", "roceP2p1s0f1"};

[[noreturn]] void fail(const std::string& reason) {
  throw RdmaError("qwen38 PairReduce RDMA: " + reason);
}
[[noreturn]] void fail_errno(const std::string& reason) {
  fail(reason + ": " + std::strerror(errno));
}
double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void write_all(int socket, const void* source, std::size_t bytes) {
  const auto* cursor = static_cast<const std::byte*>(source);
  while (bytes != 0) {
    const ssize_t written = ::write(socket, cursor, bytes);
    if (written < 0) {
      if (errno == EINTR) continue;
      fail_errno("bootstrap write");
    }
    cursor += written;
    bytes -= static_cast<std::size_t>(written);
  }
}

void read_all(int socket, void* destination, std::size_t bytes) {
  auto* cursor = static_cast<std::byte*>(destination);
  while (bytes != 0) {
    const ssize_t received = ::read(socket, cursor, bytes);
    if (received == 0) fail("bootstrap peer closed");
    if (received < 0) {
      if (errno == EINTR) continue;
      fail_errno("bootstrap read");
    }
    cursor += received;
    bytes -= static_cast<std::size_t>(received);
  }
}

int listen_once(int port) {
  const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listener < 0) fail_errno("bootstrap socket");
  int one = 1;
  ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  address.sin_addr.s_addr = INADDR_ANY;
  if (::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    ::close(listener);
    fail_errno("bootstrap bind");
  }
  if (::listen(listener, 1) != 0) {
    ::close(listener);
    fail_errno("bootstrap listen");
  }
  const int connected = ::accept(listener, nullptr, nullptr);
  ::close(listener);
  if (connected < 0) fail_errno("bootstrap accept");
  ::setsockopt(connected, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  return connected;
}

int connect_bounded(const std::string& host, int port) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1)
    fail("invalid bootstrap IPv4 address");
  const auto start = Clock::now();
  while (elapsed_ms(start) <= 60'000.0) {
    const int socket = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socket < 0) fail_errno("bootstrap socket");
    if (::connect(socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
      int one = 1;
      ::setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      return socket;
    }
    ::close(socket);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  fail("bootstrap connect timed out");
}

ibv_context* open_device(const std::string& name) {
  int count = 0;
  ibv_device** devices = ibv_get_device_list(&count);
  if (devices == nullptr) fail_errno("ibv_get_device_list");
  ibv_context* result = nullptr;
  for (int index = 0; index < count; ++index) {
    if (name == ibv_get_device_name(devices[index])) {
      result = ibv_open_device(devices[index]);
      break;
    }
  }
  ibv_free_device_list(devices);
  if (result == nullptr) fail("required RDMA device is unavailable: " + name);
  return result;
}

struct RailWire {
  std::uint32_t schema;
  std::uint32_t rank;
  std::uint32_t world_size;
  std::uint32_t rails;
  std::uint32_t qpn;
  std::uint32_t psn;
  std::uint32_t door_rkey;
  std::uint32_t page_bytes;
  std::uint32_t operation_timeout_ms;
  std::uint32_t reserved;
  std::uint64_t door_address;
  std::uint8_t gid[16];
  std::uint8_t session_sha256[32];
};
static_assert(sizeof(RailWire) == 96);

struct RegionWire {
  std::uint64_t address;
  std::uint64_t bytes;
  std::uint32_t rkey[kRails];
};

}  // namespace

void validate_peer_bootstrap_identity(
    const RdmaConfig& local, const RdmaPeerBootstrapIdentity& peer) {
  if (peer.schema != kRdmaBootstrapSchema ||
      peer.rank != static_cast<std::uint32_t>(1 - local.rank) ||
      peer.world_size != 2 || peer.rails != kRails ||
      peer.page_bytes != 65'536 ||
      peer.operation_timeout_ms != local.operation_timeout_ms)
    fail("peer bootstrap topology or page contract drift");
  if (peer.session_sha256 != local.session_sha256)
    fail("peer bootstrap session identity drift");
}

struct RdmaTransport::Impl {
  struct Rail {
    std::string name;
    ibv_context* context = nullptr;
    ibv_pd* protection_domain = nullptr;
    ibv_cq* completion_queue = nullptr;
    ibv_qp* queue_pair = nullptr;
    ibv_mr* door_mr = nullptr;
    std::uint64_t peer_door_address = 0;
    std::uint32_t peer_door_rkey = 0;
    std::uint64_t sequence_source = 0;
    int outstanding = 0;
    ibv_mtu mtu = IBV_MTU_1024;
  };
  struct Region {
    void* address = nullptr;
    std::size_t bytes = 0;
    std::array<ibv_mr*, kRails> memory_regions{};
    std::uint64_t peer_address = 0;
    std::array<std::uint32_t, kRails> peer_rkeys{};
    bool active = false;
  };

  explicit Impl(const RdmaConfig& value) : config(value) {}
  ~Impl() {
    for (auto& region : regions)
      for (ibv_mr*& memory_region : region.memory_regions)
        if (memory_region != nullptr) ibv_dereg_mr(memory_region);
    for (auto& rail : rails) {
      if (rail.queue_pair != nullptr) ibv_destroy_qp(rail.queue_pair);
      if (rail.door_mr != nullptr) ibv_dereg_mr(rail.door_mr);
      if (rail.completion_queue != nullptr) ibv_destroy_cq(rail.completion_queue);
      if (rail.protection_domain != nullptr) ibv_dealloc_pd(rail.protection_domain);
      if (rail.context != nullptr) ibv_close_device(rail.context);
    }
    if (door != nullptr) std::free(door);
    if (socket >= 0) ::close(socket);
  }

  volatile std::uint64_t* door_word(int rail, std::size_t offset) noexcept {
    return reinterpret_cast<volatile std::uint64_t*>(
        door + static_cast<std::size_t>(rail) * kDoorStride + offset);
  }

  RdmaConfig config;
  std::array<Rail, kRails> rails;
  std::vector<Region> regions;
  std::byte* door = nullptr;
  int socket = -1;
  std::uint64_t sequence = 0;
};

RdmaTransport::RdmaTransport(const RdmaConfig& config) : impl_(new Impl(config)) {
  if (config.rank != 0 && config.rank != 1) fail("rank must be 0 or 1");
  if (config.devices.size() != kRails) fail("topology requires exactly two RDMA rails");
  for (int index = 0; index < kRails; ++index)
    if (config.devices[static_cast<std::size_t>(index)] !=
        kDevices[static_cast<std::size_t>(index)])
      fail("RDMA rail identity or order drift");
  if (config.gid_index != 3) fail("RoCE GID index must be 3");
  if (config.bootstrap_port <= 0 || config.bootstrap_port > 65'535)
    fail("bootstrap port is outside 1..65535");
  if (config.rail_split_bytes != 65'536)
    fail("rail split must equal one 65536-byte host page");
  if (!valid_operation_timeout_ms(config.operation_timeout_ms))
    fail("operation timeout must be within 100..120000 milliseconds");

  impl_->door = static_cast<std::byte*>(std::aligned_alloc(kDoorStride, kRails * kDoorStride));
  if (impl_->door == nullptr) fail("doorbell allocation failed");
  std::memset(impl_->door, 0, kRails * kDoorStride);

  std::array<RailWire, kRails> local{}, peer{};
  for (int index = 0; index < kRails; ++index) {
    auto& rail = impl_->rails[static_cast<std::size_t>(index)];
    rail.name = config.devices[static_cast<std::size_t>(index)];
    rail.context = open_device(rail.name);
    rail.protection_domain = ibv_alloc_pd(rail.context);
    if (rail.protection_domain == nullptr) fail("ibv_alloc_pd failed on " + rail.name);
    rail.completion_queue = ibv_create_cq(rail.context, 256, nullptr, nullptr, 0);
    if (rail.completion_queue == nullptr) fail("ibv_create_cq failed on " + rail.name);
    rail.door_mr = ibv_reg_mr(rail.protection_domain, impl_->door, kRails * kDoorStride,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (rail.door_mr == nullptr) fail("doorbell registration failed on " + rail.name);

    ibv_qp_init_attr init{};
    init.send_cq = rail.completion_queue;
    init.recv_cq = rail.completion_queue;
    init.qp_type = IBV_QPT_RC;
    init.sq_sig_all = 0;
    init.cap.max_send_wr = 256;
    init.cap.max_recv_wr = 1;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    init.cap.max_inline_data = 64;
    rail.queue_pair = ibv_create_qp(rail.protection_domain, &init);
    if (rail.queue_pair == nullptr) fail("ibv_create_qp failed on " + rail.name);

    ibv_port_attr port{};
    if (ibv_query_port(rail.context, 1, &port) != 0 || port.state != IBV_PORT_ACTIVE)
      fail("RDMA port 1 is not active on " + rail.name);
    rail.mtu = port.active_mtu;
    ibv_gid gid{};
    if (ibv_query_gid(rail.context, 1, config.gid_index, &gid) != 0)
      fail("RoCE GID query failed on " + rail.name);

    auto& wire = local[static_cast<std::size_t>(index)];
    wire.schema = kRdmaBootstrapSchema;
    wire.rank = static_cast<std::uint32_t>(config.rank);
    wire.world_size = 2;
    wire.rails = kRails;
    wire.qpn = rail.queue_pair->qp_num;
    wire.psn = static_cast<std::uint32_t>(0x3800 + 0x100 * index + config.rank);
    wire.door_rkey = rail.door_mr->rkey;
    wire.page_bytes = 65'536;
    wire.operation_timeout_ms = config.operation_timeout_ms;
    wire.door_address = reinterpret_cast<std::uint64_t>(impl_->door);
    std::memcpy(wire.gid, gid.raw, sizeof(wire.gid));
    std::memcpy(wire.session_sha256, config.session_sha256.data(),
                config.session_sha256.size());
  }

  impl_->socket = config.rank == 0 ? listen_once(config.bootstrap_port)
                                   : connect_bounded(config.bootstrap_host,
                                                     config.bootstrap_port);
  write_all(impl_->socket, local.data(), sizeof(local));
  read_all(impl_->socket, peer.data(), sizeof(peer));

  for (int index = 0; index < kRails; ++index) {
    auto& rail = impl_->rails[static_cast<std::size_t>(index)];
    const auto& remote = peer[static_cast<std::size_t>(index)];
    RdmaPeerBootstrapIdentity identity{
        remote.schema, remote.rank, remote.world_size, remote.rails,
        remote.page_bytes, remote.operation_timeout_ms, {}};
    std::memcpy(identity.session_sha256.data(), remote.session_sha256,
                identity.session_sha256.size());
    validate_peer_bootstrap_identity(config, identity);
    rail.peer_door_address = remote.door_address;
    rail.peer_door_rkey = remote.door_rkey;

    ibv_qp_attr initial{};
    initial.qp_state = IBV_QPS_INIT;
    initial.pkey_index = 0;
    initial.port_num = 1;
    initial.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(rail.queue_pair, &initial,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0)
      fail("QP INIT failed on " + rail.name);

    ibv_qp_attr ready{};
    ready.qp_state = IBV_QPS_RTR;
    ready.path_mtu = rail.mtu;
    ready.dest_qp_num = remote.qpn;
    ready.rq_psn = remote.psn;
    ready.max_dest_rd_atomic = 1;
    ready.min_rnr_timer = 12;
    ready.ah_attr.is_global = 1;
    ready.ah_attr.port_num = 1;
    ready.ah_attr.grh.hop_limit = 64;
    ready.ah_attr.grh.sgid_index = static_cast<std::uint8_t>(config.gid_index);
    std::memcpy(ready.ah_attr.grh.dgid.raw, remote.gid, sizeof(remote.gid));
    if (ibv_modify_qp(rail.queue_pair, &ready,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0)
      fail("QP RTR failed on " + rail.name);

    ibv_qp_attr send{};
    send.qp_state = IBV_QPS_RTS;
    send.sq_psn = local[static_cast<std::size_t>(index)].psn;
    send.timeout = 14;
    send.retry_cnt = 7;
    send.rnr_retry = 7;
    send.max_rd_atomic = 1;
    if (ibv_modify_qp(rail.queue_pair, &send,
                      IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                          IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                          IBV_QP_MAX_QP_RD_ATOMIC) != 0)
      fail("QP RTS failed on " + rail.name);
  }
}

RdmaTransport::~RdmaTransport() = default;
int RdmaTransport::rank() const noexcept { return impl_->config.rank; }

int RdmaTransport::register_region(void* address, std::size_t bytes) {
  if (address == nullptr || bytes == 0 || bytes % 65'536 != 0 ||
      reinterpret_cast<std::uintptr_t>(address) % 65'536 != 0)
    fail("registered region must be nonempty and 65536-byte aligned");
  Impl::Region region;
  region.address = address;
  region.bytes = bytes;
  RegionWire local{reinterpret_cast<std::uint64_t>(address), bytes, {0, 0}};
  for (int index = 0; index < kRails; ++index) {
    ibv_mr* mr = ibv_reg_mr(impl_->rails[static_cast<std::size_t>(index)].protection_domain,
                            address, bytes,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (mr == nullptr) {
      for (ibv_mr*& registered : region.memory_regions) {
        if (registered != nullptr) {
          ibv_dereg_mr(registered);
          registered = nullptr;
        }
      }
      fail("region registration failed");
    }
    region.memory_regions[static_cast<std::size_t>(index)] = mr;
    local.rkey[index] = mr->rkey;
  }
  RegionWire peer{};
  write_all(impl_->socket, &local, sizeof(local));
  read_all(impl_->socket, &peer, sizeof(peer));
  if (peer.bytes != bytes) {
    for (ibv_mr*& registered : region.memory_regions) {
      if (registered != nullptr) {
        ibv_dereg_mr(registered);
        registered = nullptr;
      }
    }
    fail("peer region-size drift");
  }
  region.peer_address = peer.address;
  for (int index = 0; index < kRails; ++index)
    region.peer_rkeys[static_cast<std::size_t>(index)] = peer.rkey[index];
  region.active = true;
  impl_->regions.push_back(region);
  return static_cast<int>(impl_->regions.size() - 1);
}

void RdmaTransport::unregister_region(int handle) noexcept {
  if (handle < 0 || handle >= static_cast<int>(impl_->regions.size())) return;
  auto& region = impl_->regions[static_cast<std::size_t>(handle)];
  if (!region.active) return;
  for (ibv_mr*& mr : region.memory_regions) {
    if (mr != nullptr) {
      ibv_dereg_mr(mr);
      mr = nullptr;
    }
  }
  region.active = false;
}

std::uint64_t RdmaTransport::next_sequence() { return ++impl_->sequence; }

void RdmaTransport::post_unsignaled_write(int handle, std::size_t source_offset,
                                          std::size_t peer_offset, std::size_t bytes) {
  if (handle < 0 || handle >= static_cast<int>(impl_->regions.size()))
    fail("invalid region handle");
  auto& region = impl_->regions[static_cast<std::size_t>(handle)];
  if (!region.active || source_offset > region.bytes || peer_offset > region.bytes ||
      bytes > region.bytes - source_offset || bytes > region.bytes - peer_offset)
    fail("write is outside registered region");

  const int rails = bytes >= impl_->config.rail_split_bytes ? kRails : 1;
  std::size_t completed = 0;
  for (int index = 0; index < rails; ++index) {
    const std::size_t remaining = bytes - completed;
    std::size_t chunk = index == rails - 1 ? remaining : (bytes / rails) & ~std::size_t{63};
    if (chunk == 0) continue;
    auto& rail = impl_->rails[static_cast<std::size_t>(index)];
    ibv_sge scatter{};
    scatter.addr = reinterpret_cast<std::uint64_t>(region.address) + source_offset + completed;
    scatter.length = static_cast<std::uint32_t>(chunk);
    scatter.lkey = region.memory_regions[static_cast<std::size_t>(index)]->lkey;
    ibv_send_wr write{};
    write.sg_list = &scatter;
    write.num_sge = 1;
    write.opcode = IBV_WR_RDMA_WRITE;
    write.send_flags = 0;
    write.wr.rdma.remote_addr = region.peer_address + peer_offset + completed;
    write.wr.rdma.rkey = region.peer_rkeys[static_cast<std::size_t>(index)];
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(rail.queue_pair, &write, &bad) != 0)
      fail("unsignaled payload post failed on " + rail.name);
    completed += chunk;
  }
}

void RdmaTransport::signal_sequence(std::uint64_t sequence) {
  if (sequence == 0 || sequence != impl_->sequence) fail("sequence signal drift");
  for (int index = 0; index < kRails; ++index) {
    auto& rail = impl_->rails[static_cast<std::size_t>(index)];
    rail.sequence_source = sequence;
    ibv_sge scatter{};
    scatter.addr = reinterpret_cast<std::uint64_t>(&rail.sequence_source);
    scatter.length = sizeof(sequence);
    ibv_send_wr write{};
    write.wr_id = sequence;
    write.sg_list = &scatter;
    write.num_sge = 1;
    write.opcode = IBV_WR_RDMA_WRITE;
    write.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    write.wr.rdma.remote_addr = rail.peer_door_address +
                                static_cast<std::size_t>(index) * kDoorStride + kReadyOffset;
    write.wr.rdma.rkey = rail.peer_door_rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(rail.queue_pair, &write, &bad) != 0)
      fail("signaled doorbell post failed on " + rail.name);
    ++rail.outstanding;
  }
}

void RdmaTransport::wait_peer(std::uint64_t sequence) {
  const auto start = Clock::now();
  for (int index = 0; index < kRails; ++index) {
    volatile std::uint64_t* word = impl_->door_word(index, kReadyOffset);
    std::uint64_t spins = 0;
    while (*word < sequence) {
      if ((++spins & 0x3ffu) == 0 &&
          elapsed_ms(start) > impl_->config.operation_timeout_ms)
        fail("peer doorbell timed out");
#if defined(__aarch64__)
      asm volatile("yield" ::: "memory");
#endif
    }
  }
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
}

void RdmaTransport::acknowledge_consumed(std::uint64_t sequence) {
  if (sequence == 0 || sequence != impl_->sequence) fail("consumed sequence drift");
  for (int index = 0; index < kRails; ++index) {
    auto& rail = impl_->rails[static_cast<std::size_t>(index)];
    rail.sequence_source = sequence;
    ibv_sge scatter{};
    scatter.addr = reinterpret_cast<std::uint64_t>(&rail.sequence_source);
    scatter.length = sizeof(sequence);
    ibv_send_wr write{};
    write.wr_id = sequence;
    write.sg_list = &scatter;
    write.num_sge = 1;
    write.opcode = IBV_WR_RDMA_WRITE;
    write.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    write.wr.rdma.remote_addr = rail.peer_door_address +
                                static_cast<std::size_t>(index) * kDoorStride +
                                kConsumedOffset;
    write.wr.rdma.rkey = rail.peer_door_rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(rail.queue_pair, &write, &bad) != 0)
      fail("consumed doorbell post failed on " + rail.name);
    ++rail.outstanding;
  }
}

void RdmaTransport::wait_peer_consumed(std::uint64_t sequence) {
  const auto start = Clock::now();
  for (int index = 0; index < kRails; ++index) {
    volatile std::uint64_t* word = impl_->door_word(index, kConsumedOffset);
    std::uint64_t spins = 0;
    while (*word < sequence) {
      if ((++spins & 0x3ffu) == 0 &&
          elapsed_ms(start) > impl_->config.operation_timeout_ms)
        fail("peer consumption acknowledgment timed out");
#if defined(__aarch64__)
      asm volatile("yield" ::: "memory");
#endif
    }
  }
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
}

void RdmaTransport::flush_signaled() {
  for (auto& rail : impl_->rails) {
    const auto start = Clock::now();
    while (rail.outstanding > 0) {
      ibv_wc completions[8];
      const int count = ibv_poll_cq(rail.completion_queue, 8, completions);
      if (count < 0) fail("completion poll failed on " + rail.name);
      for (int index = 0; index < count; ++index)
        if (completions[index].status != IBV_WC_SUCCESS)
          fail("send completion failed on " + rail.name + ": " +
               ibv_wc_status_str(completions[index].status));
      rail.outstanding -= count;
      if (count == 0 && elapsed_ms(start) > impl_->config.operation_timeout_ms)
        fail("send completion timed out on " + rail.name);
    }
  }
}

}  // namespace rocket::qwen38::pair_reduce

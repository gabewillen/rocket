// SPDX-License-Identifier: Apache-2.0
// Live-only proof for the authenticated rank-0/layer-0 fixed GDN graph.
#include "linear_attention/gdn_cutlass.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
constexpr std::uint64_t kSlabBytes = 63'212'748'800ULL;
constexpr int kRows = 16, kHidden = 2'560, kConvRows = 6;
constexpr int kConvWidth = 5'120, kHeads = 24, kDim = 128, kOut = 2'560;
constexpr float kLocalRoofGbps = 238.0F;
struct Extent { std::uint64_t offset, bytes; };
constexpr Extent kALog{1'297'735'680, 48};
constexpr Extent kConv{1'297'735'936, 40'960};
constexpr Extent kDt{1'297'776'896, 48};
constexpr Extent kAWeight{1'297'777'152, 30'720};
constexpr Extent kAScale{1'297'807'872, 20'480};
constexpr Extent kAGlobal{1'297'828'352, 4};
constexpr Extent kBWeight{1'297'828'864, 30'720};
constexpr Extent kBScale{1'297'859'584, 20'480};
constexpr Extent kBGlobal{1'297'880'064, 4};
constexpr Extent kQWeight{1'297'880'576, 6'553'600};
constexpr Extent kQScale{1'304'434'176, 819'200};
constexpr Extent kQGlobal{1'305'253'376, 4};
constexpr Extent kZWeight{1'305'253'888, 3'932'160};
constexpr Extent kZScale{1'309'186'048, 491'520};
constexpr Extent kZGlobal{1'309'677'568, 4};
constexpr Extent kNorm{1'309'678'080, 256};
constexpr Extent kOWeight{1'309'678'336, 3'932'160};
constexpr Extent kOScale{1'313'610'496, 491'520};
constexpr Extent kOGlobal{1'314'102'016, 4};

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

struct DeviceBlob {
  void* pointer = nullptr;
  explicit DeviceBlob(std::size_t bytes) { check(cudaMalloc(&pointer, bytes), "cudaMalloc"); }
  ~DeviceBlob() { cudaFree(pointer); }
  DeviceBlob(const DeviceBlob&) = delete;
  DeviceBlob& operator=(const DeviceBlob&) = delete;
};

std::vector<std::uint8_t> read_exact(int fd, Extent extent) {
  std::vector<std::uint8_t> result(extent.bytes);
  std::size_t done = 0;
  while (done != result.size()) {
    const ssize_t count = pread(fd, result.data() + done, result.size() - done,
                                static_cast<off_t>(extent.offset + done));
    if (count <= 0) throw std::runtime_error("short authenticated slab read");
    done += static_cast<std::size_t>(count);
  }
  return result;
}

void load_blob(int fd, Extent extent, DeviceBlob& device) {
  const auto host = read_exact(fd, extent);
  check(cudaMemcpy(device.pointer, host.data(), host.size(),
                   cudaMemcpyHostToDevice), "copy slab extent");
}

float load_scalar(int fd, Extent extent) {
  const auto bytes = read_exact(fd, extent);
  float value;
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}

__global__ void initialize(__nv_bfloat16* input, std::int32_t* indices) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kRows * kHidden) {
    input[index] = __float2bfloat16(static_cast<float>((index % 29) - 14) / 32.0F);
  }
  if (index < kRows) indices[index] = index + 1;
}

std::uint64_t hash(const void* data, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  std::uint64_t value = 14'695'981'039'346'656'037ULL;
  for (std::size_t i = 0; i < bytes; ++i) value = (value ^ p[i]) * 1'099'511'628'211ULL;
  return value;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc != 3) throw std::invalid_argument("usage: qwen38-gdn-graph-smoke SLAB DEVICE");
  const int device = std::stoi(argv[2]);
  check(cudaSetDevice(device), "cudaSetDevice");
  const int fd = open(argv[1], O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error(std::string("open slab: ") + std::strerror(errno));
  struct stat info {};
  if (fstat(fd, &info) || static_cast<std::uint64_t>(info.st_size) != kSlabBytes) {
    close(fd); throw std::runtime_error("rank0-target slab byte identity changed");
  }

  DeviceBlob qw(kQWeight.bytes), qs(kQScale.bytes), zw(kZWeight.bytes), zs(kZScale.bytes);
  DeviceBlob bw(kBWeight.bytes), bs(kBScale.bytes), aw(kAWeight.bytes), as(kAScale.bytes);
  DeviceBlob ow(kOWeight.bytes), os(kOScale.bytes), conv(kConv.bytes), alog(kALog.bytes);
  DeviceBlob dt(kDt.bytes), norm(kNorm.bytes);
  load_blob(fd, kQWeight, qw); load_blob(fd, kQScale, qs);
  load_blob(fd, kZWeight, zw); load_blob(fd, kZScale, zs);
  load_blob(fd, kBWeight, bw); load_blob(fd, kBScale, bs);
  load_blob(fd, kAWeight, aw); load_blob(fd, kAScale, as);
  load_blob(fd, kOWeight, ow); load_blob(fd, kOScale, os);
  load_blob(fd, kConv, conv); load_blob(fd, kALog, alog);
  load_blob(fd, kDt, dt); load_blob(fd, kNorm, norm);
  const float qg = load_scalar(fd, kQGlobal), zg = load_scalar(fd, kZGlobal);
  const float bg = load_scalar(fd, kBGlobal), ag = load_scalar(fd, kAGlobal);
  const float og = load_scalar(fd, kOGlobal);
  close(fd);

  void* graph = nullptr;
  if (qwen38_gdn_graph_create(
          device, static_cast<std::uint8_t*>(qw.pointer), static_cast<std::uint8_t*>(qs.pointer), qg,
          static_cast<std::uint8_t*>(zw.pointer), static_cast<std::uint8_t*>(zs.pointer), zg,
          static_cast<std::uint8_t*>(bw.pointer), static_cast<std::uint8_t*>(bs.pointer), bg,
          static_cast<std::uint8_t*>(aw.pointer), static_cast<std::uint8_t*>(as.pointer), ag,
          static_cast<std::uint8_t*>(ow.pointer), static_cast<std::uint8_t*>(os.pointer), og,
          static_cast<__nv_bfloat16*>(conv.pointer), static_cast<__nv_bfloat16*>(alog.pointer),
          static_cast<__nv_bfloat16*>(dt.pointer), static_cast<__nv_bfloat16*>(norm.pointer), &graph)) {
    throw std::runtime_error(qwen38_gdn_graph_last_error());
  }
  DeviceBlob input(kRows * kHidden * 2), conv_state(17ULL * kConvRows * kConvWidth * 2);
  DeviceBlob recurrent(17ULL * kHeads * kDim * kDim * 4), indices(kRows * 4);
  initialize<<<(kRows * kHidden + 255) / 256, 256>>>(
      static_cast<__nv_bfloat16*>(input.pointer), static_cast<std::int32_t*>(indices.pointer));
  check(cudaMemset(conv_state.pointer, 0, 17ULL * kConvRows * kConvWidth * 2), "clear conv state");
  check(cudaMemset(recurrent.pointer, 0, 17ULL * kHeads * kDim * kDim * 4), "clear recurrent state");
  cudaStream_t stream; check(cudaStreamCreate(&stream), "create stream");

  void* output = nullptr; std::size_t elements = 0;
  if (qwen38_gdn_graph_output(graph, &output, &elements)) throw std::runtime_error(qwen38_gdn_graph_last_error());
  std::vector<std::uint8_t> host(elements * 2);
  constexpr std::size_t conv_bytes = 17ULL * kConvRows * kConvWidth * 2;
  constexpr std::size_t recurrent_bytes = 17ULL * kHeads * kDim * kDim * 4;
  std::vector<std::uint8_t> first_output(host.size());
  std::vector<std::uint8_t> first_conv(conv_bytes), first_recurrent(recurrent_bytes);
  std::vector<std::uint8_t> second_conv(conv_bytes), second_recurrent(recurrent_bytes);
  for (const int m : std::array<int, 5>{1, 2, 4, 8, 16}) {
    cudaGraph_t captured; cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    if (qwen38_gdn_graph_launch(graph, static_cast<__nv_bfloat16*>(input.pointer),
                               static_cast<__nv_bfloat16*>(conv_state.pointer),
                               static_cast<float*>(recurrent.pointer),
                               static_cast<std::int32_t*>(indices.pointer), m, stream)) {
      throw std::runtime_error(qwen38_gdn_graph_last_error());
    }
    check(cudaStreamEndCapture(stream, &captured), "end capture");
    check(cudaGraphInstantiate(&executable, captured, 0), "instantiate graph");
    check(cudaMemsetAsync(conv_state.pointer, 0, 17ULL * kConvRows * kConvWidth * 2, stream), "reset conv");
    check(cudaMemsetAsync(recurrent.pointer, 0, 17ULL * kHeads * kDim * kDim * 4, stream), "reset recurrent");
    check(cudaGraphLaunch(executable, stream), "graph replay 1"); check(cudaStreamSynchronize(stream), "sync replay 1");
    check(cudaMemcpy(host.data(), output, host.size(), cudaMemcpyDeviceToHost), "copy output 1");
    first_output = host;
    check(cudaMemcpy(first_conv.data(), conv_state.pointer, conv_bytes,
                     cudaMemcpyDeviceToHost), "copy conv state 1");
    check(cudaMemcpy(first_recurrent.data(), recurrent.pointer, recurrent_bytes,
                     cudaMemcpyDeviceToHost), "copy recurrent state 1");
    const auto first = hash(host.data(), static_cast<std::size_t>(m) * kOut * 2);
    check(cudaMemsetAsync(conv_state.pointer, 0, 17ULL * kConvRows * kConvWidth * 2, stream), "reset conv 2");
    check(cudaMemsetAsync(recurrent.pointer, 0, 17ULL * kHeads * kDim * kDim * 4, stream), "reset recurrent 2");
    check(cudaGraphLaunch(executable, stream), "graph replay 2"); check(cudaStreamSynchronize(stream), "sync replay 2");
    check(cudaMemcpy(host.data(), output, host.size(), cudaMemcpyDeviceToHost), "copy output 2");
    check(cudaMemcpy(second_conv.data(), conv_state.pointer, conv_bytes,
                     cudaMemcpyDeviceToHost), "copy conv state 2");
    check(cudaMemcpy(second_recurrent.data(), recurrent.pointer, recurrent_bytes,
                     cudaMemcpyDeviceToHost), "copy recurrent state 2");
    const auto second = hash(host.data(), static_cast<std::size_t>(m) * kOut * 2);
    const auto conv_hash = hash(first_conv.data(), conv_bytes);
    const auto recurrent_hash = hash(first_recurrent.data(), recurrent_bytes);
    if (first != second || first_conv != second_conv ||
        first_recurrent != second_recurrent) {
      std::size_t differing_output = 0;
      for (std::size_t i = 0; i < static_cast<std::size_t>(m) * kOut * 2; ++i) {
        differing_output += first_output[i] != host[i];
      }
      throw std::runtime_error(
          "GDN graph replay differs: m=" + std::to_string(m) +
          " output_bytes=" + std::to_string(differing_output) +
          " conv=" + (first_conv == second_conv ? "same" : "different") +
          " recurrent=" +
          (first_recurrent == second_recurrent ? "same" : "different"));
    }
    std::array<float, 5> samples{};
    cudaEvent_t begin, end; check(cudaEventCreate(&begin), "create begin event");
    check(cudaEventCreate(&end), "create end event");
    for (auto& sample : samples) {
      check(cudaMemsetAsync(conv_state.pointer, 0, 17ULL * kConvRows * kConvWidth * 2, stream), "profile reset conv");
      check(cudaMemsetAsync(recurrent.pointer, 0, 17ULL * kHeads * kDim * kDim * 4, stream), "profile reset recurrent");
      check(cudaEventRecord(begin, stream), "record begin");
      check(cudaGraphLaunch(executable, stream), "profile graph replay");
      check(cudaEventRecord(end, stream), "record end");
      check(cudaEventSynchronize(end), "sync profile replay");
      check(cudaEventElapsedTime(&sample, begin, end), "elapsed profile replay");
    }
    cudaEventDestroy(end); cudaEventDestroy(begin);
    const float mean = std::accumulate(samples.begin(), samples.end(), 0.0F) /
                       static_cast<float>(samples.size());
    const auto [minimum, maximum] =
        std::minmax_element(samples.begin(), samples.end());
    constexpr std::uint64_t weights =
        static_cast<std::uint64_t>(8'240) * 2'560 * 9 / 16 +
        static_cast<std::uint64_t>(2'560) * 3'072 * 9 / 16 + 41'312;
    constexpr std::uint64_t state_per_row =
        2ULL * kHeads * kDim * kDim * 4 + 6ULL * kConvWidth * 2;
    const std::uint64_t traffic = weights + static_cast<std::uint64_t>(m) *
        (state_per_row + 2ULL * kHidden + 2ULL * kOut);
    const float gbps = static_cast<float>(traffic) / (mean * 1.0e6F);
    std::cout << "m=" << m << " output_fnv64=" << first
              << " conv_fnv64=" << conv_hash
              << " recurrent_fnv64=" << recurrent_hash
              << " replay=bit-exact mean_ms=" << mean
              << " min_ms=" << *minimum << " max_ms=" << *maximum
              << " spread_percent=" << ((*maximum - *minimum) / mean * 100.0F)
              << " traffic_bytes="
              << traffic << " effective_GBps=" << gbps
              << " local_roof_fraction=" << gbps / kLocalRoofGbps << '\n';
    cudaGraphExecDestroy(executable); cudaGraphDestroy(captured);
  }
  // The fixed output GEMM reads 16 rows for every bucket. A c1 launch after
  // c16 must clear the inactive recurrent rows before that read.
  if (qwen38_gdn_graph_launch(graph, static_cast<__nv_bfloat16*>(input.pointer),
                             static_cast<__nv_bfloat16*>(conv_state.pointer),
                             static_cast<float*>(recurrent.pointer),
                             static_cast<std::int32_t*>(indices.pointer), 1,
                             stream)) {
    throw std::runtime_error(qwen38_gdn_graph_last_error());
  }
  check(cudaStreamSynchronize(stream), "sync c16-to-c1 isolation launch");
  check(cudaMemcpy(host.data(), output, host.size(), cudaMemcpyDeviceToHost),
        "copy c16-to-c1 output");
  for (std::size_t byte = static_cast<std::size_t>(kOut) * 2;
       byte < host.size(); ++byte) {
    if (host[byte] != 0) {
      throw std::runtime_error("c16-to-c1 inactive output retained data");
    }
  }
  std::cout << "c16_to_c1=inactive-output-zero\n";
  check(cudaMemsetAsync(indices.pointer, 0, kRows * 4, stream), "set null slots");
  check(cudaMemsetAsync(conv_state.pointer, 0x35, conv_bytes, stream), "seed null conv");
  check(cudaMemsetAsync(recurrent.pointer, 0x5a, recurrent_bytes, stream), "seed null recurrent");
  std::vector<std::uint8_t> conv_before(conv_bytes), recurrent_before(recurrent_bytes);
  check(cudaMemcpyAsync(conv_before.data(), conv_state.pointer, conv_bytes,
                        cudaMemcpyDeviceToHost, stream), "copy null conv before");
  check(cudaMemcpyAsync(recurrent_before.data(), recurrent.pointer, recurrent_bytes,
                        cudaMemcpyDeviceToHost, stream), "copy null recurrent before");
  if (qwen38_gdn_graph_launch(graph, static_cast<__nv_bfloat16*>(input.pointer),
                             static_cast<__nv_bfloat16*>(conv_state.pointer),
                             static_cast<float*>(recurrent.pointer),
                             static_cast<std::int32_t*>(indices.pointer), 16, stream)) {
    throw std::runtime_error(qwen38_gdn_graph_last_error());
  }
  check(cudaStreamSynchronize(stream), "sync null-slot launch");
  std::vector<std::uint8_t> conv_after(conv_bytes), recurrent_after(recurrent_bytes);
  check(cudaMemcpy(conv_after.data(), conv_state.pointer, conv_bytes,
                   cudaMemcpyDeviceToHost), "copy null conv after");
  check(cudaMemcpy(recurrent_after.data(), recurrent.pointer, recurrent_bytes,
                   cudaMemcpyDeviceToHost), "copy null recurrent after");
  check(cudaMemcpy(host.data(), output, host.size(), cudaMemcpyDeviceToHost),
        "copy null output");
  if (conv_before != conv_after || recurrent_before != recurrent_after) {
    throw std::runtime_error("slot 0 mutated recurrent state");
  }
  for (const auto byte : host) {
    if (byte != 0) throw std::runtime_error("slot 0 produced nonzero output");
  }
  std::cout << "slot0=null state=unchanged output=zero\n";
  cudaStreamDestroy(stream);
  if (qwen38_gdn_graph_destroy(graph)) throw std::runtime_error(qwen38_gdn_graph_last_error());
  return 0;
} catch (const std::exception& error) {
  std::cerr << "gdn_graph_smoke: " << error.what() << '\n';
  return 1;
}

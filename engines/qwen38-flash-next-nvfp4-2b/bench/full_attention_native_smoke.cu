// SPDX-License-Identifier: Apache-2.0
// Layer-3 K0 live proof. Authenticate the manifest/chunks with
// qwen38_slab.full_attention_graph before invoking this binary.
#include "attention/full_attention_native.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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

namespace attention = rocket::qwen38::attention;
namespace {
constexpr std::uint64_t kSlabBytes = 63'212'748'800ULL;
struct Extent { std::uint64_t offset, bytes; };
constexpr Extent kIndex{45'042'313'216ULL, 1'638'400};
constexpr Extent kIndexKNorm{45'043'951'616ULL, 256};
constexpr Extent kIndexQNorm{45'043'951'872ULL, 256};
constexpr Extent kKNorm{45'043'952'128ULL, 512};
constexpr Extent kKWeight{45'043'952'640ULL, 327'680};
constexpr Extent kKScale{45'044'280'320ULL, 40'960};
constexpr Extent kKGlobal{45'044'321'280ULL, 4};
constexpr Extent kOWeight{45'044'321'792ULL, 3'932'160};
constexpr Extent kOScale{45'048'253'952ULL, 491'520};
constexpr Extent kOGlobal{45'048'745'472ULL, 4};
constexpr Extent kQNorm{45'048'745'984ULL, 512};
constexpr Extent kQWeight{45'048'746'496ULL, 7'864'320};
constexpr Extent kQScale{45'056'610'816ULL, 983'040};
constexpr Extent kQGlobal{45'057'593'856ULL, 4};
constexpr Extent kVWeight{45'057'594'368ULL, 327'680};
constexpr Extent kVScale{45'057'922'048ULL, 40'960};
constexpr Extent kVGlobal{45'057'963'008ULL, 4};

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}
struct Blob {
  void* p = nullptr; std::size_t bytes;
  explicit Blob(std::size_t n) : bytes(n) { check(cudaMalloc(&p, n), "cudaMalloc"); }
  ~Blob() { cudaFree(p); }
};
std::vector<std::uint8_t> read_extent(int fd, Extent extent) {
  std::vector<std::uint8_t> result(extent.bytes);
  std::size_t done = 0;
  while (done < result.size()) {
    const ssize_t count = pread(fd, result.data() + done, result.size() - done,
                                extent.offset + done);
    if (count <= 0) throw std::runtime_error("short authenticated slab extent");
    done += count;
  }
  return result;
}
void load(int fd, Extent extent, Blob& blob) {
  const auto host = read_extent(fd, extent);
  check(cudaMemcpy(blob.p, host.data(), host.size(), cudaMemcpyHostToDevice),
        "load extent");
}
float scalar(int fd, Extent extent) {
  const auto bytes = read_extent(fd, extent); float result;
  std::memcpy(&result, bytes.data(), sizeof(result)); return result;
}
int open_slab(const char* path) {
  const int fd = open(path, O_RDONLY | O_CLOEXEC);
  struct stat info {};
  if (fd < 0 || fstat(fd, &info) ||
      static_cast<std::uint64_t>(info.st_size) != kSlabBytes)
    throw std::runtime_error("rank target slab byte identity changed");
  return fd;
}
__global__ void initialize(__nv_bfloat16* input, std::int64_t* rope,
                           std::int64_t* logical, std::int32_t* lengths,
                           std::int32_t* requests, int rows,
                           int verify_width) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < rows * 2560)
    input[index] = __float2bfloat16(static_cast<float>((index % 31) - 15) / 32.0F);
  if (index < rows) {
    const int request = index / verify_width;
    const std::int64_t position = attention::kQsaStateContext - verify_width +
                                  index % verify_width;
    logical[index] = position;
    requests[index] = request;
    rope[index] = position;
    rope[128 + index] = position;
    rope[256 + index] = position;
  }
  if (index < 16) lengths[index] = attention::kQsaStateContext;
}
std::uint64_t hash(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t value = 14695981039346656037ULL;
  for (auto byte : bytes) value = (value ^ byte) * 1099511628211ULL;
  return value;
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 4 && argc != 5)
    throw std::invalid_argument("usage: qwen38-full-attention-native-smoke LOCAL_SLAB PEER_SLAB DEVICE [VERIFY_WIDTH]");
  const int device = std::stoi(argv[3]); check(cudaSetDevice(device), "device");
  const int verify_width = argc == 5 ? std::stoi(argv[4]) : 1;
  if (verify_width < 1 || verify_width > 8)
    throw std::invalid_argument("VERIFY_WIDTH must be 1..8");
  const int rows = 16 * verify_width;
  const int local = open_slab(argv[1]), peer = open_slab(argv[2]);
  Blob qw(kQWeight.bytes), qs(kQScale.bytes), kw(kKWeight.bytes), ks(kKScale.bytes),
      vw(kVWeight.bytes), vs(kVScale.bytes), ow(kOWeight.bytes), os(kOScale.bytes),
      qn(kQNorm.bytes), kn(kKNorm.bytes), iqn(kIndexQNorm.bytes), ikn(kIndexKNorm.bytes),
      index0(kIndex.bytes), index1(kIndex.bytes);
  load(local,kQWeight,qw); load(local,kQScale,qs); load(local,kKWeight,kw);
  load(local,kKScale,ks); load(local,kVWeight,vw); load(local,kVScale,vs);
  load(local,kOWeight,ow); load(local,kOScale,os); load(local,kQNorm,qn);
  load(local,kKNorm,kn); load(local,kIndexQNorm,iqn); load(local,kIndexKNorm,ikn);
  // Canonical reconstruction is always rank0 rows followed by rank1 rows.
  const bool local_is_rank0 = std::string(argv[1]).find("rank0-target") != std::string::npos;
  load(local_is_rank0 ? local : peer, kIndex, index0);
  load(local_is_rank0 ? peer : local, kIndex, index1);
  const float qg=scalar(local,kQGlobal), kg=scalar(local,kKGlobal),
              vg=scalar(local,kVGlobal), og=scalar(local,kOGlobal);
  close(local); close(peer);

  constexpr std::size_t main_bytes=2147483648ULL, raw_bytes=35840,
                        compressed_bytes=268435456;
  Blob main_state(main_bytes), raw_state(raw_bytes), compressed_state(compressed_bytes),
      input(128ULL*2560*2), rope(3ULL*128*8), logical(128*8), lengths(16*4), requests(128*4);
  check(cudaMemset(main_state.p,0,main_state.bytes),"clear main");
  check(cudaMemset(raw_state.p,0,raw_state.bytes),"clear raw");
  check(cudaMemset(compressed_state.p,0,compressed_state.bytes),"clear compressed");
  initialize<<<(rows*2560+255)/256,256>>>(
      static_cast<__nv_bfloat16*>(input.p),static_cast<std::int64_t*>(rope.p),
      static_cast<std::int64_t*>(logical.p),static_cast<std::int32_t*>(lengths.p),
      static_cast<std::int32_t*>(requests.p),rows,verify_width);
  check(cudaDeviceSynchronize(),"initialize");
  std::uint64_t generation=1;
  attention::FullAttentionNativeConfig config{
      device, static_cast<std::uint8_t*>(qw.p),static_cast<std::uint8_t*>(qs.p),
      static_cast<std::uint8_t*>(kw.p),static_cast<std::uint8_t*>(ks.p),
      static_cast<std::uint8_t*>(vw.p),static_cast<std::uint8_t*>(vs.p),
      static_cast<std::uint8_t*>(ow.p),static_cast<std::uint8_t*>(os.p),
      qg,kg,vg,og,static_cast<__nv_bfloat16*>(qn.p),static_cast<__nv_bfloat16*>(kn.p),
      static_cast<__nv_bfloat16*>(index0.p),static_cast<__nv_bfloat16*>(index1.p),
      static_cast<__nv_bfloat16*>(iqn.p),static_cast<__nv_bfloat16*>(ikn.p),
      {static_cast<std::uint8_t*>(main_state.p),main_bytes,
       static_cast<std::uint8_t*>(raw_state.p),raw_bytes,
       static_cast<std::uint8_t*>(compressed_state.p),compressed_bytes},
      static_cast<std::int64_t*>(rope.p),static_cast<std::int64_t*>(logical.p),
      static_cast<std::int32_t*>(lengths.p),static_cast<std::int32_t*>(requests.p),
      &generation};
  attention::FullAttentionNativeProgram program(config);
  auto callbacks=program.callbacks(); cudaStream_t stream; check(cudaStreamCreate(&stream),"stream");
  auto control_config = config;
  control_config.use_scalar_attention_control = true;
  attention::FullAttentionNativeProgram control(control_config);
  auto control_callbacks = control.callbacks();
  if(control_callbacks.stage(control_callbacks.context,
                             static_cast<__nv_bfloat16*>(input.p),
                             {16,verify_width,rows},stream))
    throw std::runtime_error(control.last_error());
  check(cudaStreamSynchronize(stream),"control stage fence");
  std::vector<__nv_bfloat16> control_output(
      static_cast<std::size_t>(rows)*2560);
  check(cudaMemcpy(control_output.data(),control.projected_output(),
                   control_output.size()*sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToHost),"control output");
  if(control_callbacks.reset(control_callbacks.context,stream))
    throw std::runtime_error(control.last_error());
  std::array<double,5> ms{}; std::uint64_t expected_hash=0;
  double maximum_error=0.0, mean_error=0.0;
  attention::FullAttentionNativeProfile profile{};
  for (double& sample:ms) {
    const auto start=std::chrono::steady_clock::now();
    if(callbacks.stage(callbacks.context,static_cast<__nv_bfloat16*>(input.p),
                       {16,verify_width,rows},stream))
      throw std::runtime_error(program.last_error());
    check(cudaStreamSynchronize(stream),"stage fence");
    profile=program.profile();
    sample=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::vector<std::uint8_t> output(
        static_cast<std::size_t>(rows)*2560*2);
    check(cudaMemcpy(output.data(),program.projected_output(),output.size(),cudaMemcpyDeviceToHost),"output");
    const auto observed=hash(output);
    if(expected_hash && expected_hash!=observed) throw std::runtime_error("reset replay output changed");
    expected_hash=observed;
    if (&sample == &ms[0]) {
      const auto* selected_output =
          reinterpret_cast<const __nv_bfloat16*>(output.data());
      for (std::size_t index=0; index<control_output.size(); ++index) {
        const double error=std::abs(
            static_cast<double>(__bfloat162float(selected_output[index]))-
            static_cast<double>(__bfloat162float(control_output[index])));
        maximum_error=std::max(maximum_error,error);
        mean_error+=error/control_output.size();
      }
    }
    if(callbacks.reset(callbacks.context,stream)) throw std::runtime_error(program.last_error());
  }
  if(callbacks.stage(callbacks.context,static_cast<__nv_bfloat16*>(input.p),
                     {16,verify_width,rows},stream))
    throw std::runtime_error(program.last_error());
  std::array<std::int32_t,16> accepted{};
  accepted.fill(verify_width);
  if(callbacks.accept(callbacks.context,accepted.data(),16,2,stream))
    throw std::runtime_error(program.last_error());
  cudaGraph_t graph=nullptr;
  cudaGraphExec_t graph_exec=nullptr;
  check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),
        "begin native graph capture");
  if(callbacks.stage(callbacks.context,static_cast<__nv_bfloat16*>(input.p),
                     {16,verify_width,rows},stream))
    throw std::runtime_error(program.last_error());
  check(cudaStreamEndCapture(stream,&graph),"end native graph capture");
  if(callbacks.reset(callbacks.context,stream))
    throw std::runtime_error(program.last_error());
  check(cudaStreamSynchronize(stream),"capture reset fence");
  check(cudaGraphInstantiate(&graph_exec,graph,0),"instantiate native graph");
  std::uint64_t graph_hash=0;
  for(int replay=0; replay<2; ++replay) {
    check(cudaGraphLaunch(graph_exec,stream),"launch native graph");
    check(cudaStreamSynchronize(stream),"native graph fence");
    std::vector<std::uint8_t> replay_output(
        static_cast<std::size_t>(rows)*2560*2);
    check(cudaMemcpy(replay_output.data(),program.projected_output(),
                     replay_output.size(),cudaMemcpyDeviceToHost),
          "graph output");
    const auto observed=hash(replay_output);
    if(graph_hash && graph_hash!=observed)
      throw std::runtime_error("CUDA graph replay output changed");
    graph_hash=observed;
  }
  std::vector<std::uint8_t> first_row(512);
  const auto* published = static_cast<std::uint8_t*>(main_state.p) +
      static_cast<std::size_t>(attention::kQsaStateContext - 1) * 512;
  check(cudaMemcpy(first_row.data(),published,first_row.size(),cudaMemcpyDeviceToHost),"published row");
  const double mean=std::accumulate(ms.begin(),ms.end(),0.0)/ms.size();
  std::cout << "rows=" << rows << " verify_width=" << verify_width
            << " mean_ms=" << mean << " min_ms="
            << *std::min_element(ms.begin(),ms.end()) << " max_ms="
            << *std::max_element(ms.begin(),ms.end()) << " output_hash="
            << expected_hash << " state_hash=" << hash(first_row)
            << " generation=" << generation
            << " graph_hash=" << graph_hash
            << " qkv_ms=" << profile.qkv_ms
            << " preprocess_ms=" << profile.preprocess_ms
            << " state_format_ms=" << profile.state_format_ms
            << " state_fork_ms=" << profile.state_fork_ms
            << " score_ms=" << profile.score_ms
            << " select_ms=" << profile.select_ms
            << " attention_ms=" << profile.attention_ms
            << " gate_output_ms=" << profile.gate_output_ms << "\n";
  std::cout << "scalar_control_max_abs=" << maximum_error
            << " scalar_control_mean_abs=" << mean_error << "\n";
  check(cudaGraphExecDestroy(graph_exec),"destroy native graph exec");
  check(cudaGraphDestroy(graph),"destroy native graph");
  check(cudaStreamDestroy(stream),"destroy stream"); return 0;
} catch(const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << "\n"; return 1;
}

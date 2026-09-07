#include "projection/cutlass_qkv.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
constexpr int kRows = 1;
constexpr int kBlockTopk = 512;
constexpr int kOutputWidth = 2051;

bool cuda_ok(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}
}  // namespace

int main() {
  std::vector<std::int32_t> blocks(kBlockTopk);
  for (int index = 0; index < kBlockTopk; ++index) blocks[index] = index;
  const std::int64_t position = 10;
  const std::int32_t sequence_length = 11;
  const std::int32_t request = 0;
  std::vector<std::int32_t> expected(kOutputWidth, -1), observed(kOutputWidth);
  for (int token = 0; token < 11; ++token) expected[token] = token;

  std::int32_t *d_blocks = nullptr, *d_sequence = nullptr, *d_request = nullptr,
               *d_output = nullptr;
  std::int64_t* d_position = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  bool ok = cuda_ok(cudaMalloc(&d_blocks, blocks.size() * sizeof(std::int32_t)), "alloc blocks") &&
            cuda_ok(cudaMalloc(&d_position, sizeof(position)), "alloc position") &&
            cuda_ok(cudaMalloc(&d_sequence, sizeof(sequence_length)), "alloc sequence") &&
            cuda_ok(cudaMalloc(&d_request, sizeof(request)), "alloc request") &&
            cuda_ok(cudaMalloc(&d_output, observed.size() * sizeof(std::int32_t)), "alloc output") &&
            cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream") &&
            cuda_ok(cudaMemcpy(d_blocks, blocks.data(), blocks.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice), "copy blocks") &&
            cuda_ok(cudaMemcpy(d_position, &position, sizeof(position), cudaMemcpyHostToDevice),
                    "copy position") &&
            cuda_ok(cudaMemcpy(d_sequence, &sequence_length, sizeof(sequence_length),
                               cudaMemcpyHostToDevice), "copy sequence") &&
            cuda_ok(cudaMemcpy(d_request, &request, sizeof(request), cudaMemcpyHostToDevice),
                    "copy request");
  if (ok) {
    ok = cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture") &&
         qwen38_qsa_expand_topk(d_blocks, d_position, d_sequence, d_request,
                                d_output, kRows, stream) == 0 &&
         cuda_ok(cudaStreamEndCapture(stream, &graph), "end capture") &&
         cuda_ok(cudaGraphInstantiate(&executable, graph, 0), "instantiate graph") &&
         cuda_ok(cudaGraphLaunch(executable, stream), "launch graph") &&
         cuda_ok(cudaStreamSynchronize(stream), "synchronize graph") &&
         cuda_ok(cudaMemcpy(observed.data(), d_output,
                            observed.size() * sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost), "copy output") &&
         observed == expected;
  }
  std::printf("qsa_expand rows=1 block_topk=512 output_width=2051 result=%s\n",
              ok ? "match" : "failure");
  if (executable) cudaGraphExecDestroy(executable);
  if (graph) cudaGraphDestroy(graph);
  if (stream) cudaStreamDestroy(stream);
  cudaFree(d_output); cudaFree(d_request); cudaFree(d_sequence);
  cudaFree(d_position); cudaFree(d_blocks);
  return ok ? 0 : 1;
}

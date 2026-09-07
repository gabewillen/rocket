#include "projection/cutlass_qkv.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
constexpr int kRows = 1;
constexpr int kBlockTopk = 512;
constexpr int kOutputWidth = 2051;
constexpr int kFullRows = 16;
constexpr int kColumns = 65536;

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
  if (!ok) return 1;
  stream = nullptr;
  if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "recreate QSA stream")) return 1;

  void* plan = nullptr;
  std::vector<std::int64_t> positions(kFullRows, 262143);
  std::vector<std::int32_t> lengths(kFullRows, 262144), requests(kFullRows);
  for (int row = 0; row < kFullRows; ++row) requests[row] = row;
  requests.back() = -1;
  std::int64_t* d_positions = nullptr;
  std::int32_t *d_lengths = nullptr, *d_requests = nullptr;
  cudaGraph_t index_graph = nullptr;
  cudaGraphExec_t index_exec = nullptr;
  std::vector<std::int32_t> first(kFullRows * kOutputWidth), second(first.size());
  ok = qwen38_qsa_indexer_create(0, &plan) == 0 &&
       cuda_ok(cudaMalloc(&d_positions, positions.size() * 8), "alloc QSA positions") &&
       cuda_ok(cudaMalloc(&d_lengths, lengths.size() * 4), "alloc QSA lengths") &&
       cuda_ok(cudaMalloc(&d_requests, requests.size() * 4), "alloc QSA requests") &&
       cuda_ok(cudaMemcpy(d_positions, positions.data(), positions.size() * 8,
                          cudaMemcpyHostToDevice), "copy QSA positions") &&
       cuda_ok(cudaMemcpy(d_lengths, lengths.data(), lengths.size() * 4,
                          cudaMemcpyHostToDevice), "copy QSA lengths") &&
       cuda_ok(cudaMemcpy(d_requests, requests.data(), requests.size() * 4,
                          cudaMemcpyHostToDevice), "copy QSA requests") &&
       cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "begin QSA capture") &&
       qwen38_qsa_indexer_launch(plan, d_positions, d_lengths, d_requests, stream) == 0 &&
       cuda_ok(cudaStreamEndCapture(stream, &index_graph), "end QSA capture") &&
       cuda_ok(cudaGraphInstantiate(&index_exec, index_graph, 0), "instantiate QSA graph");
  void* d_index_output = nullptr;
  std::size_t index_elements = 0;
  if (ok) ok = qwen38_qsa_indexer_output(plan, &d_index_output, &index_elements) == 0 &&
               index_elements == first.size();
  for (auto* destination : {&first, &second}) {
    if (!ok) break;
    ok = cuda_ok(cudaGraphLaunch(index_exec, stream), "launch QSA graph") &&
         cuda_ok(cudaStreamSynchronize(stream), "sync QSA graph") &&
         cuda_ok(cudaMemcpy(destination->data(), d_index_output,
                            destination->size() * 4, cudaMemcpyDeviceToHost),
                 "copy QSA output");
  }
  ok = ok && first == second;
  for (int rank = 0; ok && rank < kBlockTopk; ++rank) {
    const int block = kColumns - 1 - rank;
    for (int offset = 0; offset < 4; ++offset)
      ok = first[rank * 4 + offset] == block * 4 + offset;
  }
  for (int column = 0; ok && column < kOutputWidth; ++column)
    ok = first[(kFullRows - 1) * kOutputWidth + column] == -1;

  auto elapsed = [&](bool score) {
    cudaEvent_t start = nullptr, end = nullptr;
    cudaEventCreate(&start); cudaEventCreate(&end);
    for (int warm = 0; warm < 3; ++warm)
      (score ? qwen38_qsa_indexer_score(plan, d_positions, d_lengths, d_requests, stream)
             : qwen38_qsa_indexer_select_expand(plan, d_positions, d_lengths, d_requests, stream));
    cudaStreamSynchronize(stream); cudaEventRecord(start, stream);
    constexpr int iterations = 20;
    for (int iteration = 0; iteration < iterations; ++iteration)
      (score ? qwen38_qsa_indexer_score(plan, d_positions, d_lengths, d_requests, stream)
             : qwen38_qsa_indexer_select_expand(plan, d_positions, d_lengths, d_requests, stream));
    cudaEventRecord(end, stream); cudaEventSynchronize(end);
    float ms = 0; cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(end); cudaEventDestroy(start); return ms / iterations;
  };
  const float score_ms = elapsed(true), select_ms = elapsed(false);
  const double physical_gb =
      (15.0 * kColumns * (128.0 * 2.0 + 4.0)) / 1.0e9;
  const double bandwidth = physical_gb / (score_ms / 1000.0);
  std::printf("qsa_indexer rows=16 columns=65536 score_ms=%.6f "
              "score_gbps=%.3f local_roof_pct=%.3f select_expand_ms=%.6f "
              "repeat=%s result=%s\n", score_ms, bandwidth,
              bandwidth / 238.0 * 100.0, select_ms,
              first == second ? "bit-exact" : "mismatch", ok ? "match" : "failure");
  if (index_exec) cudaGraphExecDestroy(index_exec);
  if (index_graph) cudaGraphDestroy(index_graph);
  qwen38_qsa_indexer_destroy(plan);
  cudaFree(d_requests); cudaFree(d_lengths); cudaFree(d_positions);
  cudaStreamDestroy(stream);
  return ok ? 0 : 1;
}

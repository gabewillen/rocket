// SPDX-License-Identifier: Apache-2.0
#include "output/token_output.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace qo = rocket::qwen38::output;

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

void cuda_ok(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "FAIL: %s: %s\n", message, cudaGetErrorString(status));
    std::exit(1);
  }
}

void cublas_ok(cublasStatus_t status, const char* message) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "FAIL: %s: cublas %d\n", message, static_cast<int>(status));
    std::exit(1);
  }
}

void test_contract() {
  require(qo::kVocab == 248320 && qo::kLocalVocab == 124160 && qo::kHidden == 2560,
          "fixed model geometry");
  require(qo::kLmHead.offset_bytes == 0 && qo::kLmHead.length_bytes == 635699200,
          "lm-head slab descriptor");
  require(qo::kEmbedding.offset_bytes == 635699200 &&
              qo::kFinalNorm.offset_bytes == 1271398400,
          "embedding/final-norm slab descriptors");
  for (const int m : qo::kBuckets) require(qo::allowed_m(m), "M bucket accepted");
  require(!qo::allowed_m(3) && !qo::allowed_rank(2), "invalid shape/rank rejected");
  require(qo::sampling_supported(0.0F, 1.0F), "greedy accepted");
  require(!qo::sampling_supported(0.8F, 1.0F) &&
              !qo::sampling_supported(0.0F, 0.9F),
          "random sampling rejected");
  const auto dims = qo::otel_dimensions(qo::Operation::kLmHead, 1, 16,
                                         qo::Outcome::kCudaError);
  require(dims.operation == "lm_head" && dims.rank == 1 && dims.m_bucket == 16 &&
              dims.mode == "greedy" && dims.outcome == "cuda_error",
          "bounded OTEL dimensions");
  require(qo::embedding_lookup_rank(nullptr, nullptr, nullptr, nullptr, 3, 2) ==
              cudaErrorInvalidValue,
          "embedding invalid contract");
  require(qo::final_grouped_rms_norm(nullptr, nullptr, nullptr, 3) ==
              cudaErrorInvalidValue,
          "norm invalid contract");
  require(qo::lm_head(nullptr, nullptr, nullptr, nullptr, 3, 2) ==
              CUBLAS_STATUS_INVALID_VALUE,
          "head invalid contract");
  require(qo::global_greedy(nullptr, nullptr, 3, 0.8F, 0.9F) ==
              cudaErrorInvalidValue,
          "sampler invalid contract");
}

void test_final_norm() {
  constexpr int m = 2;
  const std::size_t elements = static_cast<std::size_t>(m) * qo::kHyperHidden;
  std::vector<__nv_bfloat16> input(elements), weight(qo::kHyperHidden), output(elements);
  for (std::size_t i = 0; i < elements; ++i)
    input[i] = __float2bfloat16(static_cast<float>(static_cast<int>(i % 29) - 14) / 8.0F);
  for (int i = 0; i < qo::kHyperHidden; ++i)
    weight[i] = __float2bfloat16(static_cast<float>(static_cast<int>(i % 7) - 3) / 64.0F);

  __nv_bfloat16 *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
  cuda_ok(cudaMalloc(&d_input, elements * sizeof(*d_input)), "norm input alloc");
  cuda_ok(cudaMalloc(&d_weight, weight.size() * sizeof(*d_weight)), "norm weight alloc");
  cuda_ok(cudaMalloc(&d_output, elements * sizeof(*d_output)), "norm output alloc");
  cuda_ok(cudaMemcpy(d_input, input.data(), elements * sizeof(*d_input), cudaMemcpyHostToDevice),
          "norm input upload");
  cuda_ok(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(*d_weight),
                     cudaMemcpyHostToDevice), "norm weight upload");
  cuda_ok(qo::final_grouped_rms_norm(d_input, d_weight, d_output, m), "norm launch");
  cuda_ok(cudaMemcpy(output.data(), d_output, elements * sizeof(*d_output),
                     cudaMemcpyDeviceToHost), "norm output download");

  float max_abs = 0.0F;
  for (int row = 0; row < m; ++row) {
    for (int group = 0; group < qo::kHyperConnections; ++group) {
      const std::size_t base = static_cast<std::size_t>(row) * qo::kHyperHidden +
                               group * qo::kHidden;
      float square_sum = 0.0F;
      for (int column = 0; column < qo::kHidden; ++column) {
        const float value = __bfloat162float(input[base + column]);
        square_sum += value * value;
      }
      const float inverse = 1.0F / std::sqrt(square_sum / qo::kHidden + qo::kRmsEpsilon);
      for (int column = 0; column < qo::kHidden; ++column) {
        const float expected = __bfloat162float(input[base + column]) * inverse *
                               (1.0F + __bfloat162float(weight[group * qo::kHidden + column]));
        max_abs = std::max(max_abs,
                           std::abs(__bfloat162float(output[base + column]) - expected));
      }
    }
  }
  require(max_abs <= 0.008F, "grouped norm BF16 vs FP32 error");
  cuda_ok(cudaFree(d_output), "norm output free");
  cuda_ok(cudaFree(d_weight), "norm weight free");
  cuda_ok(cudaFree(d_input), "norm input free");
}

void test_embedding() {
  constexpr int m = 4;
  __nv_bfloat16* d_weight = nullptr;
  __nv_bfloat16* d_output = nullptr;
  std::int32_t *d_tokens = nullptr, *d_invalid = nullptr;
  cuda_ok(cudaMalloc(&d_weight, qo::kEmbedding.length_bytes), "embedding weight alloc");
  cuda_ok(cudaMemset(d_weight, 0, qo::kEmbedding.length_bytes), "embedding weight clear");
  std::vector<__nv_bfloat16> row(qo::kHidden, __float2bfloat16(1.25F));
  cuda_ok(cudaMemcpy(d_weight + 7LL * qo::kHidden, row.data(),
                     row.size() * sizeof(row[0]), cudaMemcpyHostToDevice), "embedding row upload");
  std::fill(row.begin(), row.end(), __float2bfloat16(1.5F));
  cuda_ok(cudaMemcpy(d_weight + 3LL * qo::kHidden, row.data(),
                     row.size() * sizeof(row[0]), cudaMemcpyHostToDevice),
          "rank1 embedding row upload");
  cuda_ok(cudaMalloc(&d_tokens, m * sizeof(*d_tokens)), "token alloc");
  cuda_ok(cudaMalloc(&d_output, m * qo::kHidden * sizeof(*d_output)), "embedding output alloc");
  cuda_ok(cudaMalloc(&d_invalid, sizeof(*d_invalid)), "invalid flag alloc");
  const std::int32_t tokens[m] = {7, qo::kLocalVocab + 3, -1, qo::kVocab};
  cuda_ok(cudaMemcpy(d_tokens, tokens, sizeof(tokens), cudaMemcpyHostToDevice), "token upload");
  cuda_ok(cudaMemset(d_invalid, 0, sizeof(*d_invalid)), "invalid flag clear");
  cuda_ok(qo::embedding_lookup_rank(d_tokens, d_weight, d_output, d_invalid, m, 0),
          "embedding launch");
  std::vector<__nv_bfloat16> output(static_cast<std::size_t>(m) * qo::kHidden);
  std::int32_t invalid = 0;
  cuda_ok(cudaMemcpy(output.data(), d_output, output.size() * sizeof(output[0]),
                     cudaMemcpyDeviceToHost), "embedding output download");
  cuda_ok(cudaMemcpy(&invalid, d_invalid, sizeof(invalid), cudaMemcpyDeviceToHost),
          "invalid flag download");
  require(invalid == 1, "invalid token reported");
  require(__bfloat162float(output[0]) == 1.25F &&
              __bfloat162float(output[qo::kHidden]) == 0.0F &&
              __bfloat162float(output[2 * qo::kHidden]) == 0.0F,
          "rank-owned embedding or zero");
  cuda_ok(cudaMemset(d_invalid, 0, sizeof(*d_invalid)), "rank1 invalid flag clear");
  cuda_ok(qo::embedding_lookup_rank(d_tokens, d_weight, d_output, d_invalid, m, 1),
          "rank1 embedding launch");
  cuda_ok(cudaMemcpy(output.data(), d_output, output.size() * sizeof(output[0]),
                     cudaMemcpyDeviceToHost), "rank1 embedding output download");
  require(__bfloat162float(output[0]) == 0.0F &&
              __bfloat162float(output[qo::kHidden]) == 1.5F,
          "rank1 embedding ownership");
  cuda_ok(cudaFree(d_invalid), "invalid flag free");
  cuda_ok(cudaFree(d_output), "embedding output free");
  cuda_ok(cudaFree(d_tokens), "token free");
  cuda_ok(cudaFree(d_weight), "embedding weight free");
}

void test_head_greedy_graph() {
  constexpr int m = 16;
  __nv_bfloat16 *d_hidden = nullptr, *d_weight = nullptr;
  float* d_logits = nullptr;
  qo::Winner *d_local = nullptr, *d_pairs = nullptr;
  std::int32_t* d_tokens = nullptr;
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  cuda_ok(cudaStreamCreate(&stream), "stream create");
  cublas_ok(cublasCreate(&handle), "cublas create");
  cuda_ok(cudaMalloc(&d_hidden, static_cast<std::size_t>(m) * qo::kHidden * sizeof(*d_hidden)),
          "hidden alloc");
  cuda_ok(cudaMalloc(&d_weight, qo::kLmHead.length_bytes), "head weight alloc");
  cuda_ok(cudaMalloc(&d_logits,
                     static_cast<std::size_t>(m) * qo::kLocalVocab * sizeof(*d_logits)),
          "logits alloc");
  cuda_ok(cudaMalloc(&d_local, m * sizeof(*d_local)), "local winners alloc");
  cuda_ok(cudaMalloc(&d_pairs, m * 2 * sizeof(*d_pairs)), "pair winners alloc");
  cuda_ok(cudaMalloc(&d_tokens, m * sizeof(*d_tokens)), "sample tokens alloc");

  std::vector<__nv_bfloat16> hidden(static_cast<std::size_t>(m) * qo::kHidden,
                                    __float2bfloat16(1.0F));
  std::vector<__nv_bfloat16> winning_row(qo::kHidden, __float2bfloat16(0.5F));
  cuda_ok(cudaMemcpyAsync(d_hidden, hidden.data(), hidden.size() * sizeof(hidden[0]),
                          cudaMemcpyHostToDevice, stream), "hidden upload");
  cuda_ok(cudaMemsetAsync(d_weight, 0, qo::kLmHead.length_bytes, stream), "head weight clear");
  cuda_ok(cudaMemcpyAsync(d_weight + 11LL * qo::kHidden, winning_row.data(),
                          winning_row.size() * sizeof(winning_row[0]),
                          cudaMemcpyHostToDevice, stream), "winner row upload");
  cublas_ok(qo::lm_head(handle, d_hidden, d_weight, d_logits, m, 0, stream), "head warmup");
  cuda_ok(qo::local_argmax(d_logits, d_local, m, 0, stream), "argmax warmup");
  cuda_ok(cudaStreamSynchronize(stream), "warmup sync");

  std::vector<qo::Winner> first(m);
  for (const int bucket : qo::kBuckets) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture begin");
    cublas_ok(qo::lm_head(handle, d_hidden, d_weight, d_logits, bucket, 0, stream),
              "captured head");
    cuda_ok(qo::local_argmax(d_logits, d_local, bucket, 0, stream), "captured argmax");
    cuda_ok(cudaStreamEndCapture(stream, &graph), "capture end");
    cuda_ok(cudaGraphInstantiate(&executable, graph, 0), "graph instantiate");

    std::vector<qo::Winner> bucket_first(bucket), bucket_second(bucket);
    cuda_ok(cudaGraphLaunch(executable, stream), "graph replay one");
    cuda_ok(cudaMemcpyAsync(bucket_first.data(), d_local,
                            bucket_first.size() * sizeof(bucket_first[0]),
                            cudaMemcpyDeviceToHost, stream), "winner one download");
    cuda_ok(cudaStreamSynchronize(stream), "replay one sync");
    cuda_ok(cudaGraphLaunch(executable, stream), "graph replay two");
    cuda_ok(cudaMemcpyAsync(bucket_second.data(), d_local,
                            bucket_second.size() * sizeof(bucket_second[0]),
                            cudaMemcpyDeviceToHost, stream), "winner two download");
    cuda_ok(cudaStreamSynchronize(stream), "replay two sync");
    for (int row = 0; row < bucket; ++row) {
      require(bucket_first[row].token == 11 && bucket_first[row].value == 1280.0F,
              "head FP32 reference and local argmax");
      require(bucket_first[row].token == bucket_second[row].token &&
                  bucket_first[row].value == bucket_second[row].value,
              "bit-exact graph replay");
    }
    if (bucket == m) first = bucket_first;
    cuda_ok(cudaGraphExecDestroy(executable), "graph exec destroy");
    cuda_ok(cudaGraphDestroy(graph), "graph destroy");
  }

  std::vector<qo::Winner> pairs(static_cast<std::size_t>(m) * 2);
  for (int row = 0; row < m; ++row) {
    pairs[row * 2] = first[row];
    pairs[row * 2 + 1] = {first[row].value, qo::kLocalVocab + 3};
  }
  cuda_ok(cudaMemcpyAsync(d_pairs, pairs.data(), pairs.size() * sizeof(pairs[0]),
                          cudaMemcpyHostToDevice, stream), "pair upload");
  cuda_ok(qo::global_greedy(d_pairs, d_tokens, m, 0.0F, 1.0F, stream),
          "global greedy launch");
  std::vector<std::int32_t> tokens(m);
  cuda_ok(cudaMemcpyAsync(tokens.data(), d_tokens, tokens.size() * sizeof(tokens[0]),
                          cudaMemcpyDeviceToHost, stream), "sample download");
  cuda_ok(cudaStreamSynchronize(stream), "sample sync");
  for (const int token : tokens) require(token == 11, "global tie chooses lower token");
  pairs[1].token = -1;
  cuda_ok(cudaMemcpyAsync(d_pairs, pairs.data(), pairs.size() * sizeof(pairs[0]),
                          cudaMemcpyHostToDevice, stream), "invalid pair upload");
  cuda_ok(qo::global_greedy(d_pairs, d_tokens, m, 0.0F, 1.0F, stream),
          "invalid pair selection launch");
  cuda_ok(cudaMemcpyAsync(tokens.data(), d_tokens, tokens.size() * sizeof(tokens[0]),
                          cudaMemcpyDeviceToHost, stream), "invalid sample download");
  cuda_ok(cudaStreamSynchronize(stream), "invalid sample sync");
  require(tokens[0] == -1, "invalid rank winner fails closed");
  require(qo::global_greedy(d_pairs, d_tokens, m, 0.7F, 0.9F, stream) ==
              cudaErrorInvalidValue,
          "temperature/top-p fails closed");

  cuda_ok(cudaFree(d_tokens), "sample tokens free");
  cuda_ok(cudaFree(d_pairs), "pairs free");
  cuda_ok(cudaFree(d_local), "local winners free");
  cuda_ok(cudaFree(d_logits), "logits free");
  cuda_ok(cudaFree(d_weight), "head weight free");
  cuda_ok(cudaFree(d_hidden), "hidden free");
  cublas_ok(cublasDestroy(handle), "cublas destroy");
  cuda_ok(cudaStreamDestroy(stream), "stream destroy");
}

}  // namespace

int main() {
  test_contract();
  test_final_norm();
  test_embedding();
  test_head_greedy_graph();
  std::puts("token output contract: PASS");
  return 0;
}

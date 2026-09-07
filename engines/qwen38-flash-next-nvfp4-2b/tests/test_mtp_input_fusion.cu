#include "mtp/input_fusion.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace mtp = rocket::qwen38::mtp;

namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}
template <typename T>
T* allocate(std::size_t count) {
  T* result = nullptr;
  cuda_check(cudaMalloc(&result, count * sizeof(T)), "allocate");
  return result;
}
}  // namespace

int main() {
  try {
    constexpr int rows = 16;
    constexpr std::size_t projection_elements =
        static_cast<std::size_t>(mtp::kFusionLocalHidden) *
        mtp::kFusionHidden;
    auto* embedding_norm = allocate<__nv_bfloat16>(mtp::kFusionHidden);
    auto* hidden_norm = allocate<__nv_bfloat16>(mtp::kFusionHyperHidden);
    auto* embedding_projection =
        allocate<__nv_bfloat16>(projection_elements);
    auto* hidden_projection = allocate<__nv_bfloat16>(projection_elements);
    auto* embedding = allocate<__nv_bfloat16>(rows * mtp::kFusionHidden);
    auto* hidden = allocate<__nv_bfloat16>(rows * mtp::kFusionHyperHidden);
    auto* embedding_partial =
        allocate<__nv_bfloat16>(rows * mtp::kFusionHidden);
    auto* hidden_partial =
        allocate<__nv_bfloat16>(rows * mtp::kFusionHyperHidden);
    auto* reduced_embedding = allocate<float>(rows * mtp::kFusionHidden);
    auto* reduced_hidden = allocate<float>(rows * mtp::kFusionHyperHidden);
    auto* fused = allocate<__nv_bfloat16>(rows * mtp::kFusionHyperHidden);
    cuda_check(cudaMemset(embedding_norm, 0, mtp::kFusionHidden * 2),
               "clear embedding norm");
    cuda_check(cudaMemset(hidden_norm, 0, mtp::kFusionHyperHidden * 2),
               "clear hidden norm");
    cuda_check(cudaMemset(embedding_projection, 0, projection_elements * 2),
               "clear embedding projection");
    cuda_check(cudaMemset(hidden_projection, 0, projection_elements * 2),
               "clear hidden projection");
    cuda_check(cudaMemset(embedding, 1, rows * mtp::kFusionHidden * 2),
               "initialize embedding");
    cuda_check(cudaMemset(hidden, 1, rows * mtp::kFusionHyperHidden * 2),
               "initialize hidden");

    std::vector<float> host_embedding(rows * mtp::kFusionHidden);
    std::vector<float> host_hidden(rows * mtp::kFusionHyperHidden);
    for (std::size_t i = 0; i < host_embedding.size(); ++i)
      host_embedding[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
    for (std::size_t i = 0; i < host_hidden.size(); ++i)
      host_hidden[i] = static_cast<float>(static_cast<int>(i % 11) - 5);
    cuda_check(cudaMemcpy(reduced_embedding, host_embedding.data(),
                          host_embedding.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy reduced embedding");
    cuda_check(cudaMemcpy(reduced_hidden, host_hidden.data(),
                          host_hidden.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy reduced hidden");

    mtp::InputFusionPlan plan(
        0, 1, {embedding_norm, hidden_norm, embedding_projection,
               hidden_projection});
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create stream");
    plan.local_project(embedding, hidden, embedding_partial, hidden_partial,
                       rows, stream);
    plan.finish(reduced_embedding, reduced_hidden, fused, rows, stream);
    cuda_check(cudaStreamSynchronize(stream), "warm input fusion");

    for (const int m : {1, 16}) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t executable = nullptr;
      cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                 "begin input fusion capture");
      plan.local_project(embedding, hidden, embedding_partial, hidden_partial,
                         m, stream);
      cuda_check(cudaStreamEndCapture(stream, &graph),
                 "end input fusion capture");
      cuda_check(cudaGraphInstantiate(&executable, graph, 0),
                 "instantiate input fusion graph");
      cuda_check(cudaGraphLaunch(executable, stream),
                 "replay input fusion graph");
      cuda_check(cudaStreamSynchronize(stream),
                 "complete input fusion graph");
      cudaGraphExecDestroy(executable);
      cudaGraphDestroy(graph);
    }

    std::vector<__nv_bfloat16> partial(rows * mtp::kFusionHyperHidden);
    std::vector<__nv_bfloat16> result(partial.size());
    cuda_check(cudaMemcpy(partial.data(), hidden_partial,
                          partial.size() * 2, cudaMemcpyDeviceToHost),
               "copy hidden partial");
    check(std::all_of(partial.begin(), partial.end(), [](__nv_bfloat16 value) {
            return __bfloat162float(value) == 0.0F;
          }),
          "zero projection produced nonzero PairReduce input");
    cuda_check(cudaMemcpy(result.data(), fused, result.size() * 2,
                          cudaMemcpyDeviceToHost),
               "copy fused output");
    for (int row = 0; row < rows; ++row)
      for (int stream_index = 0; stream_index < mtp::kFusionStreams;
           ++stream_index)
        for (int column = 0; column < mtp::kFusionHidden; ++column) {
          const std::size_t output_index =
              static_cast<std::size_t>(row) * mtp::kFusionHyperHidden +
              stream_index * mtp::kFusionHidden + column;
          const std::size_t embedding_index =
              static_cast<std::size_t>(row) * mtp::kFusionHidden + column;
          const float expected = host_hidden[output_index] +
                                 host_embedding[embedding_index];
          check(__bfloat162float(result[output_index]) ==
                    __bfloat162float(__float2bfloat16(expected)),
                "post-PairReduce fusion drift");
        }
    std::printf("qwen38_mtp_input_fusion rank=1 captured=1,16 boundary=pair_reduce result=match\n");
    cudaStreamDestroy(stream);
    cudaFree(fused);
    cudaFree(reduced_hidden);
    cudaFree(reduced_embedding);
    cudaFree(hidden_partial);
    cudaFree(embedding_partial);
    cudaFree(hidden);
    cudaFree(embedding);
    cudaFree(hidden_projection);
    cudaFree(embedding_projection);
    cudaFree(hidden_norm);
    cudaFree(embedding_norm);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

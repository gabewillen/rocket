#include "hyperconnection/hyperconnection.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace hc = rocket::qwen38::hyperconnection;

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
    const std::size_t hyper = static_cast<std::size_t>(rows) * hc::kHyperHidden;
    std::vector<__nv_bfloat16> hidden(hyper), zero_norm(hc::kHyperHidden),
        zero_down(static_cast<std::size_t>(hc::kLowRank) * hc::kHyperHidden),
        zero_injection(static_cast<std::size_t>(hc::kStreams) * hc::kHyperHidden),
        zero_up(static_cast<std::size_t>(hc::kHyperHidden) * hc::kLowRank);
    std::vector<float> reduced(static_cast<std::size_t>(rows) * hc::kHidden);
    for (std::size_t index = 0; index < hidden.size(); ++index)
      hidden[index] = __float2bfloat16((static_cast<int>(index % 29) - 14) / 32.0F);
    for (std::size_t index = 0; index < reduced.size(); ++index)
      reduced[index] = (static_cast<int>(index % 17) - 8) / 64.0F;

    auto* d_hidden = allocate<__nv_bfloat16>(hyper);
    auto* d_norm = allocate<__nv_bfloat16>(zero_norm.size());
    auto* d_down = allocate<__nv_bfloat16>(zero_down.size());
    auto* d_injection_weight = allocate<__nv_bfloat16>(zero_injection.size());
    auto* d_up = allocate<__nv_bfloat16>(zero_up.size());
    auto* d_reduced = allocate<float>(reduced.size());
    auto* d_block = allocate<__nv_bfloat16>(static_cast<std::size_t>(rows) * hc::kHidden);
    auto* d_injection = allocate<__nv_bfloat16>(static_cast<std::size_t>(rows) * hc::kStreams);
    auto* d_updated = allocate<__nv_bfloat16>(hyper);
    auto* d_next_block = allocate<__nv_bfloat16>(static_cast<std::size_t>(rows) * hc::kHidden);
    auto* d_next_injection = allocate<__nv_bfloat16>(static_cast<std::size_t>(rows) * hc::kStreams);
    cuda_check(cudaMemcpy(d_hidden, hidden.data(), hidden.size() * 2,
                          cudaMemcpyHostToDevice), "copy hidden");
    cuda_check(cudaMemset(d_norm, 0, zero_norm.size() * 2), "clear norm");
    cuda_check(cudaMemset(d_down, 0, zero_down.size() * 2), "clear down");
    cuda_check(cudaMemset(d_injection_weight, 0, zero_injection.size() * 2),
               "clear injection weight");
    cuda_check(cudaMemset(d_up, 0, zero_up.size() * 2), "clear up");
    cuda_check(cudaMemcpy(d_reduced, reduced.data(), reduced.size() * 4,
                          cudaMemcpyHostToDevice), "copy reduction");
    hc::Weights weights{d_norm, d_down, d_injection_weight, d_up};
    hc::Plan plan(0, weights, weights);
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    for (const int m : {1, 2, 4, 8, 16}) {
      plan.mix(d_hidden, d_block, d_injection, m, stream);
      plan.combine_and_mix(d_hidden, d_reduced, d_injection, d_updated,
                           d_next_block, d_next_injection, m, stream);
      cuda_check(cudaStreamSynchronize(stream), "complete HC bucket");
    }
    std::vector<__nv_bfloat16> injection(rows * hc::kStreams), next(injection.size());
    cuda_check(cudaMemcpy(injection.data(), d_injection, injection.size() * 2,
                          cudaMemcpyDeviceToHost), "copy injection");
    cuda_check(cudaMemcpy(next.data(), d_next_injection, next.size() * 2,
                          cudaMemcpyDeviceToHost), "copy next injection");
    check(std::all_of(injection.begin(), injection.end(), [](__nv_bfloat16 value) {
            return __bfloat162float(value) == 0.0F;
          }) && std::all_of(next.begin(), next.end(), [](__nv_bfloat16 value) {
            return __bfloat162float(value) == 0.0F;
          }), "zero-weight injection drift");
    std::vector<__nv_bfloat16> updated(hyper);
    cuda_check(cudaMemcpy(updated.data(), d_updated, updated.size() * 2,
                          cudaMemcpyDeviceToHost), "copy updated hidden");
    for (int row = 0; row < rows; ++row) {
      for (int stream_index = 0; stream_index < hc::kStreams; ++stream_index) {
        for (int column = 0; column < hc::kHidden; ++column) {
          const std::size_t hidden_index =
              static_cast<std::size_t>(row) * hc::kHyperHidden +
              stream_index * hc::kHidden + column;
          const std::size_t reduced_index =
              static_cast<std::size_t>(row) * hc::kHidden + column;
          const float rounded_block = __bfloat162float(
              __float2bfloat16(reduced[reduced_index]));
          const __nv_bfloat16 expected = __float2bfloat16(
              __bfloat162float(hidden[hidden_index]) + rounded_block);
          check(__bfloat162float(updated[hidden_index]) ==
                    __bfloat162float(expected),
                "BF16-rounded HyperConnection residual drift");
        }
      }
    }
    std::vector<__nv_bfloat16> first(static_cast<std::size_t>(rows) * hc::kHidden),
        second(first.size());
    cuda_check(cudaMemcpy(first.data(), d_block, first.size() * 2,
                          cudaMemcpyDeviceToHost), "copy first block");
    cuda_check(cudaMemcpy(second.data(), d_next_block, second.size() * 2,
                          cudaMemcpyDeviceToHost), "copy second block");
    for (const auto& values : {first, second})
      check(std::all_of(values.begin(), values.end(), [](__nv_bfloat16 value) {
              return std::isfinite(__bfloat162float(value));
            }), "HC produced nonfinite output");
    std::printf("qwen38_hc buckets=1,2,4,8,16 output_elements=%zu result=match\n",
                first.size() + second.size());
    cudaStreamDestroy(stream);
    cudaFree(d_next_injection); cudaFree(d_next_block); cudaFree(d_updated);
    cudaFree(d_injection); cudaFree(d_block); cudaFree(d_reduced); cudaFree(d_up);
    cudaFree(d_injection_weight); cudaFree(d_down); cudaFree(d_norm); cudaFree(d_hidden);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

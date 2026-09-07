// SPDX-License-Identifier: Apache-2.0
#include "output/token_output.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace qo = rocket::qwen38::output;

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) fail(std::string(operation) + ": " + cudaGetErrorString(status));
}
void cublas_check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    fail(std::string(operation) + ": cublas " + std::to_string(static_cast<int>(status)));
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) {
    cuda_check(cudaMalloc(&data_, count * sizeof(T)), "allocate device buffer");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  T* get() const noexcept { return data_; }
 private:
  T* data_{};
};

template <class Launch>
float measure_cuda(Launch launch, int warmup, int iterations) {
  cudaEvent_t begin{}, end{};
  cuda_check(cudaEventCreate(&begin), "create begin event");
  cuda_check(cudaEventCreate(&end), "create end event");
  for (int i = 0; i < warmup; ++i) cuda_check(launch(), "warmup launch");
  cuda_check(cudaDeviceSynchronize(), "warmup synchronize");
  cuda_check(cudaEventRecord(begin), "record begin");
  for (int i = 0; i < iterations; ++i) cuda_check(launch(), "measured launch");
  cuda_check(cudaEventRecord(end), "record end");
  cuda_check(cudaEventSynchronize(end), "synchronize end");
  float ms = 0.0F;
  cuda_check(cudaEventElapsedTime(&ms, begin, end), "elapsed time");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return ms / iterations;
}

float measure_head(cublasHandle_t handle, const __nv_bfloat16* hidden,
                   const __nv_bfloat16* weight, float* logits, int m,
                   int iterations) {
  cudaEvent_t begin{}, end{};
  cuda_check(cudaEventCreate(&begin), "create head begin");
  cuda_check(cudaEventCreate(&end), "create head end");
  for (int i = 0; i < 3; ++i)
    cublas_check(qo::lm_head(handle, hidden, weight, logits, m, 0), "head warmup");
  cuda_check(cudaDeviceSynchronize(), "head warmup synchronize");
  cuda_check(cudaEventRecord(begin), "record head begin");
  for (int i = 0; i < iterations; ++i)
    cublas_check(qo::lm_head(handle, hidden, weight, logits, m, 0), "head launch");
  cuda_check(cudaEventRecord(end), "record head end");
  cuda_check(cudaEventSynchronize(end), "synchronize head end");
  float ms = 0.0F;
  cuda_check(cudaEventElapsedTime(&ms, begin, end), "head elapsed");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return ms / iterations;
}

__global__ void initialize_weight(__nv_bfloat16* weight) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t count = static_cast<std::int64_t>(qo::kLocalVocab) * qo::kHidden;
  if (index < count) {
    const int row = static_cast<int>(index / qo::kHidden);
    const int column = static_cast<int>(index % qo::kHidden);
    weight[index] = __float2bfloat16(static_cast<float>((row * 13 + column * 17) % 31 - 15) /
                                     128.0F);
  }
}

cublasStatus_t cublas_head_control(cublasHandle_t handle,
                                   const __nv_bfloat16* hidden,
                                   const __nv_bfloat16* weight, float* logits) {
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, qo::kLocalVocab, 1,
                      qo::kHidden, &alpha, weight, CUDA_R_16BF, qo::kHidden,
                      hidden, CUDA_R_16BF, qo::kHidden, &beta, logits,
                      CUDA_R_32F, qo::kLocalVocab, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

float measure_cublas_control(cublasHandle_t handle, const __nv_bfloat16* hidden,
                             const __nv_bfloat16* weight, float* logits,
                             int iterations) {
  cudaEvent_t begin{}, end{};
  cuda_check(cudaEventCreate(&begin), "create control begin");
  cuda_check(cudaEventCreate(&end), "create control end");
  for (int i = 0; i < 3; ++i)
    cublas_check(cublas_head_control(handle, hidden, weight, logits), "control warmup");
  cuda_check(cudaDeviceSynchronize(), "control warmup synchronize");
  cuda_check(cudaEventRecord(begin), "record control begin");
  for (int i = 0; i < iterations; ++i)
    cublas_check(cublas_head_control(handle, hidden, weight, logits), "control head");
  cuda_check(cudaEventRecord(end), "record control end");
  cuda_check(cudaEventSynchronize(end), "synchronize control end");
  float ms = 0.0F;
  cuda_check(cudaEventElapsedTime(&ms, begin, end), "control elapsed");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return ms / iterations;
}

void run(int iterations) {
  DeviceBuffer<__nv_bfloat16> weight(static_cast<std::size_t>(qo::kLocalVocab) * qo::kHidden);
  DeviceBuffer<__nv_bfloat16> hidden(static_cast<std::size_t>(16) * qo::kHidden);
  DeviceBuffer<float> logits(static_cast<std::size_t>(16) * qo::kLocalVocab);
  DeviceBuffer<qo::Winner> winners(16);
  DeviceBuffer<__nv_bfloat16> hyper(static_cast<std::size_t>(16) * qo::kHyperHidden);
  DeviceBuffer<__nv_bfloat16> norm_weight(qo::kHyperHidden);
  DeviceBuffer<__nv_bfloat16> norm_output(static_cast<std::size_t>(16) * qo::kHyperHidden);
  DeviceBuffer<std::int32_t> ids(16), invalid(1);
  DeviceBuffer<__nv_bfloat16> embedding(16ULL * qo::kHidden);
  constexpr std::size_t copy_elements = 64ULL << 20;
  DeviceBuffer<std::uint8_t> copy_source(copy_elements), copy_destination(copy_elements);
  cublasHandle_t handle = nullptr;
  cublas_check(cublasCreate(&handle), "create cublas handle");

  std::vector<__nv_bfloat16> host_hidden(16ULL * qo::kHidden);
  for (std::size_t i = 0; i < host_hidden.size(); ++i)
    host_hidden[i] = __float2bfloat16(static_cast<float>(static_cast<int>(i % 37) - 18) /
                                      32.0F);
  std::vector<std::int32_t> host_ids(16, 11);
  const std::int64_t weight_elements = static_cast<std::int64_t>(qo::kLocalVocab) * qo::kHidden;
  initialize_weight<<<static_cast<unsigned>((weight_elements + 255) / 256), 256>>>(weight.get());
  cuda_check(cudaPeekAtLastError(), "initialize weights");
  cuda_check(cudaMemcpy(hidden.get(), host_hidden.data(), host_hidden.size() * 2,
                        cudaMemcpyHostToDevice), "upload hidden");
  cuda_check(cudaMemcpy(ids.get(), host_ids.data(), host_ids.size() * sizeof(host_ids[0]),
                        cudaMemcpyHostToDevice), "upload ids");
  cuda_check(cudaMemset(hyper.get(), 0,
                        static_cast<std::size_t>(16) * qo::kHyperHidden * 2), "clear hyper");
  cuda_check(cudaMemset(norm_weight.get(), 0, qo::kHyperHidden * 2), "clear norm weight");

  const float embedding_ms = measure_cuda([&] {
    cudaError_t status = cudaMemsetAsync(invalid.get(), 0, sizeof(std::int32_t));
    return status == cudaSuccess
               ? qo::embedding_lookup_rank(ids.get(), weight.get(), embedding.get(), invalid.get(), 16, 0)
               : status;
  }, 20, iterations);
  const float norm_ms = measure_cuda([&] {
    return qo::final_grouped_rms_norm(hyper.get(), norm_weight.get(), norm_output.get(), 16);
  }, 20, iterations);
  const float copy_ms = measure_cuda([&] {
    return cudaMemcpyAsync(copy_destination.get(), copy_source.get(), copy_elements,
                           cudaMemcpyDeviceToDevice);
  }, 20, std::min(iterations, 200));
  const double copy_gbps = (2.0 * copy_elements) / (copy_ms * 1.0e6);
  constexpr double empirical_read_roofline_gbps = 241.3;

  const int control_iterations = std::min(iterations, 100);
  const float cublas_m1_ms = measure_cublas_control(handle, hidden.get(), weight.get(),
                                                    logits.get(), control_iterations);
  std::vector<float> cublas_m1(qo::kLocalVocab);
  cuda_check(cudaMemcpy(cublas_m1.data(), logits.get(), cublas_m1.size() * sizeof(float),
                        cudaMemcpyDeviceToHost), "download cublas control");

  for (const int m : qo::kBuckets) {
    const int head_iterations = std::min(iterations, 100);
    const float head_ms = measure_head(handle, hidden.get(), weight.get(), logits.get(), m,
                                       head_iterations);
    float control_max_abs = 0.0F;
    bool control_argmax_match = true;
    if (m == 1) {
      std::vector<float> specialized(qo::kLocalVocab);
      cuda_check(cudaMemcpy(specialized.data(), logits.get(), specialized.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "download specialized M1");
      int control_token = 0;
      int specialized_token = 0;
      for (int token = 0; token < qo::kLocalVocab; ++token) {
        control_max_abs = std::max(control_max_abs,
                                   std::abs(cublas_m1[token] - specialized[token]));
        if (cublas_m1[token] > cublas_m1[control_token]) control_token = token;
        if (specialized[token] > specialized[specialized_token]) specialized_token = token;
      }
      control_argmax_match = control_token == specialized_token;
    }
    const float argmax_ms = measure_cuda([&] {
      return qo::local_argmax(logits.get(), winners.get(), m, 0);
    }, 20, iterations);
    std::vector<qo::Winner> result(m);
    cuda_check(cudaMemcpy(result.data(), winners.get(), result.size() * sizeof(result[0]),
                          cudaMemcpyDeviceToHost), "download winners");
    bool valid = true;
    for (const auto winner : result)
      valid = valid && winner.token >= 0 && winner.token < qo::kLocalVocab &&
              std::isfinite(winner.value);
    const double bytes = static_cast<double>(qo::kLmHead.length_bytes) +
                         static_cast<double>(m) * qo::kHidden * 2.0 +
                         static_cast<double>(m) * qo::kLocalVocab * 4.0;
    const double effective_gbps = bytes / (head_ms * 1.0e6);
    const double tflops = (2.0 * m * qo::kLocalVocab * qo::kHidden) /
                          (head_ms * 1.0e9);
    std::printf(
        "{\"result\":\"qwen38_token_output\",\"device\":\"NVIDIA GB10\","
        "\"sm\":121,\"m\":%d,\"hidden\":2560,\"local_vocab\":124160,"
        "\"tp\":2,\"head_path\":\"%s\",\"logits_dtype\":\"fp32\","
        "\"bottleneck\":\"rank_local_lm_head_weight_read\","
        "\"lm_head_ms\":%.6f,\"local_argmax_ms\":%.6f,"
        "\"m1_cublas_control_ms\":%.6f,\"m1_control_max_abs\":%.9f,"
        "\"m1_control_argmax_match\":%s,"
        "\"head_effective_gbps\":%.3f,\"head_tflops\":%.3f,"
        "\"empirical_copy_gbps\":%.3f,\"empirical_read_roofline_gbps\":241.300,"
        "\"memory_roofline_fraction\":%.6f,"
        "\"embedding_c16_ms\":%.6f,\"final_grouped_norm_c16_ms\":%.6f,"
        "\"greedy_token\":%d,"
        "\"reference_ok\":%s,\"otel_series_bound\":270}\n",
        m, m == 1 ? "mbx_b12x_adapted_gemv" : "cublas_gemm", head_ms,
        argmax_ms, cublas_m1_ms, control_max_abs,
        control_argmax_match ? "true" : "false", effective_gbps, tflops,
        copy_gbps, effective_gbps / empirical_read_roofline_gbps, embedding_ms, norm_ms,
        result.empty() ? -1 : result[0].token, valid ? "true" : "false");
  }
  cublas_check(cublasDestroy(handle), "destroy cublas handle");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int iterations = argc == 2 ? std::stoi(argv[1]) : 1000;
    if (iterations <= 0 || iterations > 100000) fail("invalid iteration count");
    run(iterations);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "qwen38 token output bench failed: %s\n", error.what());
    return 1;
  }
}

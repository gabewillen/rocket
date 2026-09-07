// SPDX-License-Identifier: Apache-2.0
#include "norm/residual_rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qn = rocket::qwen38::norm;

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void check(bool value, const std::string& message) { if (!value) fail(message); }
void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) fail(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct Buffers {
  __nv_bfloat16 *input{}, *residual{}, *weight{}, *add{}, *norm{}, *fused_residual{}, *fused_norm{};
  Buffers() = default;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;
  Buffers(Buffers&& other) noexcept
      : input(std::exchange(other.input, nullptr)),
        residual(std::exchange(other.residual, nullptr)),
        weight(std::exchange(other.weight, nullptr)),
        add(std::exchange(other.add, nullptr)),
        norm(std::exchange(other.norm, nullptr)),
        fused_residual(std::exchange(other.fused_residual, nullptr)),
        fused_norm(std::exchange(other.fused_norm, nullptr)) {}
  ~Buffers() {
    cudaFree(fused_norm); cudaFree(fused_residual); cudaFree(norm); cudaFree(add);
    cudaFree(weight); cudaFree(residual); cudaFree(input);
  }
};

float observed_max_norm_error = 0.0F;
float observed_max_fused_error = 0.0F;

std::vector<__nv_bfloat16> inputs(int count, int salt) {
  std::vector<__nv_bfloat16> result(count);
  for (int i = 0; i < count; ++i) {
    const float value = static_cast<float>(((i * 37 + salt * 17) % 257) - 128) / 31.0F;
    result[i] = __float2bfloat16(value);
  }
  return result;
}

Buffers allocate(int m, int hidden, const std::vector<__nv_bfloat16>& input,
                 const std::vector<__nv_bfloat16>& residual,
                 const std::vector<__nv_bfloat16>& weight) {
  Buffers b;
  const std::size_t bytes = static_cast<std::size_t>(m) * hidden * sizeof(__nv_bfloat16);
  const std::size_t weight_bytes = static_cast<std::size_t>(hidden) * sizeof(__nv_bfloat16);
  cuda_check(cudaMalloc(&b.input, bytes), "allocate input");
  cuda_check(cudaMalloc(&b.residual, bytes), "allocate residual");
  cuda_check(cudaMalloc(&b.weight, weight_bytes), "allocate weight");
  cuda_check(cudaMalloc(&b.add, bytes), "allocate add");
  cuda_check(cudaMalloc(&b.norm, bytes), "allocate norm");
  cuda_check(cudaMalloc(&b.fused_residual, bytes), "allocate fused residual");
  cuda_check(cudaMalloc(&b.fused_norm, bytes), "allocate fused norm");
  cuda_check(cudaMemcpy(b.input, input.data(), bytes, cudaMemcpyHostToDevice), "copy input");
  cuda_check(cudaMemcpy(b.residual, residual.data(), bytes, cudaMemcpyHostToDevice), "copy residual");
  cuda_check(cudaMemcpy(b.weight, weight.data(), weight_bytes, cudaMemcpyHostToDevice), "copy weight");
  return b;
}

std::vector<float> reference(const std::vector<__nv_bfloat16>& input,
                             const std::vector<__nv_bfloat16>* residual,
                             const std::vector<__nv_bfloat16>& weight,
                             int m, int hidden) {
  std::vector<float> output(input.size());
  for (int row = 0; row < m; ++row) {
    double squares = 0.0;
    for (int column = 0; column < hidden; ++column) {
      const int index = row * hidden + column;
      float value = __bfloat162float(input[index]);
      if (residual != nullptr) value += __bfloat162float((*residual)[index]);
      squares += static_cast<double>(value) * value;
    }
    const float inverse = 1.0F / std::sqrt(static_cast<float>(squares / hidden) + qn::kEpsilon);
    for (int column = 0; column < hidden; ++column) {
      const int index = row * hidden + column;
      float value = __bfloat162float(input[index]);
      if (residual != nullptr) value += __bfloat162float((*residual)[index]);
      output[index] = value * inverse * __bfloat162float(weight[column]);
    }
  }
  return output;
}

void test_shapes_reference_and_replay() {
  for (const int hidden : {qn::kHiddenTp, qn::kHiddenFull}) {
    for (const int m : qn::kBuckets) {
      const int count = m * hidden;
      const auto input = inputs(count, 1);
      const auto residual = inputs(count, 2);
      auto weight = inputs(hidden, 3);
      for (int i = 0; i < hidden; ++i)
        weight[i] = __float2bfloat16(0.75F + std::fabs(__bfloat162float(weight[i])) / 16.0F);
      auto b = allocate(m, hidden, input, residual, weight);
      cuda_check(qn::residual_add(b.input, b.residual, b.add, m, hidden), "residual add");
      cuda_check(qn::rms_norm(b.input, b.weight, b.norm, m, hidden), "rms norm");
      cuda_check(qn::fused_add_rms_norm(b.input, b.residual, b.weight,
                                        b.fused_residual, b.fused_norm, m, hidden),
                 "fused norm");
      cuda_check(cudaDeviceSynchronize(), "synchronize kernels");
      std::vector<__nv_bfloat16> norm(count), fused(count), residual_out(count), first(count);
      cuda_check(cudaMemcpy(norm.data(), b.norm, count * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToHost), "copy norm");
      cuda_check(cudaMemcpy(fused.data(), b.fused_norm, count * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToHost), "copy fused norm");
      cuda_check(cudaMemcpy(residual_out.data(), b.fused_residual,
                            count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
                 "copy fused residual");
      const auto norm_ref = reference(input, nullptr, weight, m, hidden);
      const auto fused_ref = reference(input, &residual, weight, m, hidden);
      float max_norm = 0.0F, max_fused = 0.0F;
      for (int i = 0; i < count; ++i) {
        max_norm = std::max(max_norm, std::fabs(__bfloat162float(norm[i]) - norm_ref[i]));
        max_fused = std::max(max_fused, std::fabs(__bfloat162float(fused[i]) - fused_ref[i]));
        const auto expected_residual = __float2bfloat16(
            __bfloat162float(input[i]) + __bfloat162float(residual[i]));
        check(std::memcmp(&expected_residual, &residual_out[i], sizeof(expected_residual)) == 0,
              "fused residual BF16 mismatch");
      }
      check(max_norm <= 0.032F && max_fused <= 0.032F,
            "BF16 output exceeded FP32 reference bound");
      observed_max_norm_error = std::max(observed_max_norm_error, max_norm);
      observed_max_fused_error = std::max(observed_max_fused_error, max_fused);
      first = fused;
      cuda_check(qn::fused_add_rms_norm(b.input, b.residual, b.weight,
                                        b.fused_residual, b.fused_norm, m, hidden),
                 "fused replay");
      cuda_check(cudaMemcpy(fused.data(), b.fused_norm, count * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToHost), "copy replay");
      check(std::memcmp(first.data(), fused.data(), count * sizeof(__nv_bfloat16)) == 0,
            "fused replay is not bit-exact");
    }
  }
}

void test_graph_capture_and_contract() {
  constexpr int m = 16;
  constexpr int hidden = qn::kHiddenFull;
  const auto input = inputs(m * hidden, 4);
  const auto residual = inputs(m * hidden, 5);
  const auto weight = inputs(hidden, 6);
  auto b = allocate(m, hidden, input, residual, weight);
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
  cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
  cuda_check(qn::fused_add_rms_norm(b.input, b.residual, b.weight,
                                    b.fused_residual, b.fused_norm, m, hidden, stream),
             "capture fused norm");
  cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
  cuda_check(cudaGraphInstantiate(&executable, graph, 0), "instantiate graph");
  cuda_check(cudaGraphLaunch(executable, stream), "launch graph one");
  cuda_check(cudaStreamSynchronize(stream), "sync graph one");
  std::vector<__nv_bfloat16> first(m * hidden), second(m * hidden);
  cuda_check(cudaMemcpy(first.data(), b.fused_norm, first.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost), "copy graph one");
  cuda_check(cudaGraphLaunch(executable, stream), "launch graph two");
  cuda_check(cudaStreamSynchronize(stream), "sync graph two");
  cuda_check(cudaMemcpy(second.data(), b.fused_norm, second.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost), "copy graph two");
  check(std::memcmp(first.data(), second.data(), first.size() * sizeof(__nv_bfloat16)) == 0,
        "captured graph replay changed bits");
  cudaGraphExecDestroy(executable); cudaGraphDestroy(graph); cudaStreamDestroy(stream);

  check(qn::rms_norm(nullptr, b.weight, b.norm, 16, hidden) == cudaErrorInvalidValue,
        "null input accepted");
  check(qn::rms_norm(b.input, b.weight, b.norm, 3, hidden) == cudaErrorInvalidValue,
        "non-bucket M accepted");
  check(qn::rms_norm(b.input, b.weight, b.norm, 16, 4096) == cudaErrorInvalidValue,
        "unconfigured width accepted");
  const auto dimensions = qn::otel_dimensions(
      qn::Operation::kFusedAddRmsNorm, hidden, m, qn::Outcome::kOk);
  check(dimensions.operation == "fused_add_rms_norm" && dimensions.hidden == hidden &&
            dimensions.m_bucket == m && dimensions.dtype == "bf16_fp32" &&
            dimensions.outcome == "ok",
        "bounded OTEL dimensions changed");
  const auto invalid = qn::otel_dimensions(qn::Operation::kRmsNorm, 4096, 3,
                                            qn::Outcome::kContractError);
  check(invalid.hidden == 0 && invalid.m_bucket == 0,
        "invalid OTEL values were not bounded");
}

}  // namespace

int main() {
  try {
    test_shapes_reference_and_replay();
    test_graph_capture_and_contract();
    std::printf(
        "{\"result\":\"qwen38_residual_rmsnorm_contract\","
        "\"shape_count\":10,\"max_norm_abs_error\":%.9g,"
        "\"max_fused_abs_error\":%.9g,\"error_bound\":0.032,"
        "\"bit_exact_replay\":true,\"graph_capture\":true,"
        "\"otel_cardinality_bound\":162}\n",
        observed_max_norm_error, observed_max_fused_error);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "qwen38 residual/RMSNorm test failed: %s\n", error.what());
    return 1;
  }
}

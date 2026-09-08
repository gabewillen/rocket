// SPDX-License-Identifier: Apache-2.0
#include "linear_attention/gdn_prefill_owner.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cmath>
#include <vector>

namespace prefill = rocket::qwen38::linear_attention::prefill;

namespace {
void check(bool condition) {
  if (!condition) std::abort();
}

template <typename T>
T* allocate(std::size_t elements) {
  T* pointer = nullptr;
  check(cudaMalloc(&pointer, elements * sizeof(T)) == cudaSuccess);
  check(cudaMemset(pointer, 0, elements * sizeof(T)) == cudaSuccess);
  return pointer;
}

void publish(void* context, prefill::NativePrefillRecord record) noexcept {
  static_cast<std::vector<prefill::NativePrefillRecord>*>(context)->push_back(
      record);
}
}  // namespace

int main() {
  constexpr std::size_t kQk = 35ULL * 8 * 128;
  constexpr std::size_t kV = 35ULL * 24 * 128;
  constexpr std::size_t kGates = 35ULL * 24;
  constexpr std::size_t kState = 24ULL * 128 * 128;
  auto* q = allocate<__nv_bfloat16>(kQk);
  auto* k = allocate<__nv_bfloat16>(kQk);
  auto* v = allocate<__nv_bfloat16>(kV);
  auto* g = allocate<float>(kGates);
  auto* beta = allocate<float>(kGates);
  auto* initial = allocate<float>(kState);
  auto* output = allocate<__nv_bfloat16>(kV);
  auto* final_state = allocate<float>(kState);
  std::vector<__nv_bfloat16> q_host(kQk, __float2bfloat16(0.125F));
  std::vector<__nv_bfloat16> k_host(kQk, __float2bfloat16(0.0625F));
  std::vector<__nv_bfloat16> v_host(kV, __float2bfloat16(0.25F));
  std::vector<float> g_host(kGates, -0.125F);
  std::vector<float> beta_host(kGates, 0.5F);
  check(cudaMemcpy(q, q_host.data(), q_host.size() * sizeof(q_host.front()),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  check(cudaMemcpy(k, k_host.data(), k_host.size() * sizeof(k_host.front()),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  check(cudaMemcpy(v, v_host.data(), v_host.size() * sizeof(v_host.front()),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  check(cudaMemcpy(g, g_host.data(), g_host.size() * sizeof(g_host.front()),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  check(cudaMemcpy(beta, beta_host.data(),
                   beta_host.size() * sizeof(beta_host.front()),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream) == cudaSuccess);

  std::vector<prefill::NativePrefillRecord> records;
  bool rejected = false;
  try {
    const prefill::NativePrefillConfig invalid_config{
        .device = 0, .rank = 1, .layer = 0, .rows = 35,
        .publish = publish, .publish_context = &records};
    prefill::NativeGdnM35PrefillOwner invalid(invalid_config);
  } catch (...) {
    rejected = true;
  }
  check(rejected);
  check(records.back().phase == prefill::NativePrefillPhase::kValidation);
  check(records.back().status == prefill::NativePrefillStatus::kIdentity);

  const prefill::NativePrefillConfig config{
      .device = 0, .rank = 0, .layer = 0, .rows = 35,
      .publish = publish, .publish_context = &records};
  prefill::NativeGdnM35PrefillOwner owner(config);
  const prefill::NativePrefillTensors tensors{q, k, v, g, beta, initial,
                                               output, final_state};
  owner.launch(tensors, stream);
  check(cudaStreamSynchronize(stream) == cudaSuccess);
  check(records.back().phase == prefill::NativePrefillPhase::kLaunch);
  check(records.back().success);
  __nv_bfloat16 first_output{};
  float first_state = 0.0F;
  check(cudaMemcpy(&first_output, output, sizeof(first_output),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  check(cudaMemcpy(&first_state, final_state, sizeof(first_state),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  check(std::isfinite(__bfloat162float(first_output)) &&
        __bfloat162float(first_output) != 0.0F);
  check(std::isfinite(first_state) && first_state != 0.0F);

  const prefill::NativePrefillTensors in_place{q, k, v, g, beta, initial,
                                                output, initial};
  owner.launch(in_place, stream);
  check(cudaStreamSynchronize(stream) == cudaSuccess);

  rejected = false;
  const prefill::NativePrefillTensors overlapping{
      q, k, v, g, beta, initial,
      reinterpret_cast<__nv_bfloat16*>(final_state), final_state};
  try {
    owner.launch(overlapping, stream);
  } catch (...) {
    rejected = true;
  }
  check(rejected);
  check(records.back().status == prefill::NativePrefillStatus::kAliasing);

  cudaStreamDestroy(stream);
  cudaFree(final_state);
  cudaFree(output);
  cudaFree(initial);
  cudaFree(beta);
  cudaFree(g);
  cudaFree(v);
  cudaFree(k);
  cudaFree(q);
  return 0;
}

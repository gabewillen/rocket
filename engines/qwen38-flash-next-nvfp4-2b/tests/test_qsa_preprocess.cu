// SPDX-License-Identifier: Apache-2.0
#include "attention/qsa_preprocess.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kHidden = 2560, kHeads = 12, kDim = 256;
constexpr int kQkv = 6656, kIndex = 640;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct Blob {
  void* p = nullptr;
  std::size_t bytes;
  explicit Blob(std::size_t n) : bytes(n) { check(cudaMalloc(&p, n), "cudaMalloc"); }
  ~Blob() { cudaFree(p); }
};
void upload(Blob& blob, const void* host) {
  check(cudaMemcpy(blob.p, host, blob.bytes, cudaMemcpyHostToDevice), "upload");
}
float bf16(float value) { return __bfloat162float(__float2bfloat16(value)); }
float rope_expected(const std::vector<__nv_bfloat16>& input, int start,
                    int head_dim, int dimension,
                    const std::int64_t* positions) {
  float sum = 0.0F;
  for (int i = 0; i < head_dim; ++i) {
    const float value = __bfloat162float(input[start + i]);
    sum += value * value;
  }
  const float inverse = 1.0F / std::sqrt(sum / head_dim + 1.0e-6F);
  const int frequency = dimension % 32;
  const int axis = frequency % 3 == 1 && frequency < 33
                       ? 1
                       : (frequency % 3 == 2 && frequency < 30 ? 2 : 0);
  const float angle = static_cast<float>(positions[axis]) *
                      std::pow(1.0e7F, -2.0F * frequency / 64.0F);
  const float cosine = bf16(std::cos(angle));
  const float sine = bf16(std::sin(angle));
  const int pair = dimension < 32 ? dimension + 32 : dimension - 32;
  const float value = bf16(__bfloat162float(input[start + dimension]) * inverse);
  const float other = bf16(__bfloat162float(input[start + pair]) * inverse);
  return bf16(dimension < 32 ? value * cosine - other * sine
                             : value * cosine + other * sine);
}
}  // namespace

int main() try {
  std::vector<__nv_bfloat16> hidden(kHidden), qkv(kQkv), norm(kDim),
      index_norm(128), first(320ULL * kHidden), second(320ULL * kHidden);
  for (int i = 0; i < kHidden; ++i)
    hidden[i] = __float2bfloat16(static_cast<float>((i % 17) - 8) / 8.0F);
  for (int i = 0; i < kQkv; ++i)
    qkv[i] = __float2bfloat16(static_cast<float>((i % 23) - 11) / 16.0F);
  for (int column = 0; column < kIndex; ++column) {
    auto& matrix = column < 320 ? first : second;
    const int row = column < 320 ? column : column - 320;
    matrix[static_cast<std::size_t>(row) * kHidden + column % kHidden] =
        __float2bfloat16(1.0F);
  }
  Blob d_hidden(kHidden * 2), d_qkv(kQkv * 2), d_norm(kDim * 2),
      d_index_norm(128 * 2), d_first(first.size() * 2),
      d_second(second.size() * 2), d_positions(3 * sizeof(std::int64_t)),
      d_query(kHeads * kDim * 2), d_key(kDim * 2), d_value(kDim * 2),
      d_gate(kHeads * kDim * 2), d_iq(4 * 128 * 2), d_ik(128 * 2),
      d_scratch(kIndex * 2), d_active_raw(16ULL * 8 * 280),
      d_main_rows(512), d_raw_rows(280), d_compressed_rows(256),
      d_logical(sizeof(std::int64_t)), d_requests(sizeof(std::int32_t));
  upload(d_hidden, hidden.data()); upload(d_qkv, qkv.data());
  upload(d_norm, norm.data()); upload(d_index_norm, index_norm.data());
  upload(d_first, first.data()); upload(d_second, second.data());
  const std::int64_t positions[3] = {3, 5, 7}; upload(d_positions, positions);
  const std::int64_t logical = 3;
  const std::int32_t request = 0;
  upload(d_logical, &logical); upload(d_requests, &request);
  check(cudaMemset(d_active_raw.p, 0, d_active_raw.bytes), "clear raw state");
  cudaStream_t stream = nullptr; check(cudaStreamCreate(&stream), "stream");
  require(qwen38_qsa_preprocess(
              static_cast<__nv_bfloat16*>(d_hidden.p),
              static_cast<__nv_bfloat16*>(d_qkv.p),
              static_cast<__nv_bfloat16*>(d_norm.p),
              static_cast<__nv_bfloat16*>(d_norm.p),
              static_cast<__nv_bfloat16*>(d_first.p),
              static_cast<__nv_bfloat16*>(d_second.p),
              static_cast<__nv_bfloat16*>(d_index_norm.p),
              static_cast<__nv_bfloat16*>(d_index_norm.p),
              static_cast<std::int64_t*>(d_positions.p), 1,
              static_cast<__nv_bfloat16*>(d_query.p),
              static_cast<__nv_bfloat16*>(d_key.p),
              static_cast<__nv_bfloat16*>(d_value.p),
              static_cast<__nv_bfloat16*>(d_gate.p),
              static_cast<__nv_bfloat16*>(d_iq.p),
              static_cast<__nv_bfloat16*>(d_ik.p),
              static_cast<__nv_bfloat16*>(d_scratch.p), stream) == 0,
          qwen38_qsa_preprocess_last_error());
  check(cudaStreamSynchronize(stream), "preprocess fence");
  std::vector<__nv_bfloat16> query(kHeads * kDim), key(kDim), value(kDim),
      gate(kHeads * kDim), iq(4 * 128), ik(128);
  check(cudaMemcpy(query.data(), d_query.p, d_query.bytes, cudaMemcpyDeviceToHost), "query");
  check(cudaMemcpy(key.data(), d_key.p, d_key.bytes, cudaMemcpyDeviceToHost), "key");
  check(cudaMemcpy(value.data(), d_value.p, d_value.bytes, cudaMemcpyDeviceToHost), "value");
  check(cudaMemcpy(gate.data(), d_gate.p, d_gate.bytes, cudaMemcpyDeviceToHost), "gate");
  check(cudaMemcpy(iq.data(), d_iq.p, d_iq.bytes, cudaMemcpyDeviceToHost), "index query");
  check(cudaMemcpy(ik.data(), d_ik.p, d_ik.bytes, cudaMemcpyDeviceToHost), "index key");
  require(__bfloat162float(query[0]) ==
                  rope_expected(qkv, 0, kDim, 0, positions) &&
              __bfloat162float(query[1]) ==
                  rope_expected(qkv, 0, kDim, 1, positions) &&
              __bfloat162float(gate[0]) == __bfloat162float(qkv[kDim]) &&
              __bfloat162float(key[0]) == rope_expected(
                  qkv, kHeads * 2 * kDim, kDim, 0, positions) &&
              __bfloat162float(value[0]) ==
                  __bfloat162float(qkv[kHeads * 2 * kDim + kDim]),
          "QKV split, per-head norm, MRoPE, or gate drifted");
  require(__bfloat162float(iq[0]) ==
                  rope_expected(hidden, 0, 128, 0, positions) &&
              __bfloat162float(ik[0]) == __bfloat162float(hidden[512]),
          "two-half replicated index projection or index query norm drifted");
  require(qwen38_qsa_format_state_rows(
              static_cast<__nv_bfloat16*>(d_key.p),
              static_cast<__nv_bfloat16*>(d_value.p),
              static_cast<__nv_bfloat16*>(d_ik.p),
              static_cast<__nv_bfloat16*>(d_index_norm.p),
              static_cast<std::int64_t*>(d_positions.p),
              static_cast<std::int64_t*>(d_logical.p),
              static_cast<std::int32_t*>(d_requests.p), d_active_raw.p, 1,
              d_main_rows.p, d_raw_rows.p,
              static_cast<__nv_bfloat16*>(d_compressed_rows.p), stream) == 0,
          qwen38_qsa_preprocess_last_error());
  check(cudaStreamSynchronize(stream), "state row fence");
  std::vector<__nv_bfloat16> raw_row(140), compressed_row(128);
  check(cudaMemcpy(raw_row.data(), d_raw_rows.p, d_raw_rows.bytes,
                   cudaMemcpyDeviceToHost), "raw state row");
  check(cudaMemcpy(compressed_row.data(), d_compressed_rows.p,
                   d_compressed_rows.bytes, cudaMemcpyDeviceToHost),
        "compressed state row");
  require(__bfloat162float(raw_row[0]) == __bfloat162float(ik[0]) &&
              std::isfinite(__bfloat162float(compressed_row[0])),
          "raw or completed-group state formatting drifted");

  std::vector<__nv_bfloat16> attention(kHeads * kDim, __float2bfloat16(2.0F));
  upload(d_query, attention.data());
  require(qwen38_qsa_apply_output_gate(
              static_cast<__nv_bfloat16*>(d_query.p),
              static_cast<__nv_bfloat16*>(d_gate.p), 1, stream) == 0,
          qwen38_qsa_preprocess_last_error());
  check(cudaStreamSynchronize(stream), "gate fence");
  check(cudaMemcpy(attention.data(), d_query.p, d_query.bytes,
                   cudaMemcpyDeviceToHost), "gated output");
  const float expected = bf16(2.0F / (1.0F + std::exp(-__bfloat162float(gate[0]))));
  require(__bfloat162float(attention[0]) == expected,
          "sigmoid output gate drifted");
  check(cudaStreamDestroy(stream), "destroy stream");
  std::puts("qwen38 QSA native preprocess reference passed");
  return 0;
} catch (const std::exception& exception) {
  std::fprintf(stderr, "FAIL: %s\n", exception.what());
  return 1;
}

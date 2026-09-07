// SPDX-License-Identifier: Apache-2.0
#include "attention/qsa_prefill.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kHeads = 12;
constexpr int kDim = 256;
constexpr int kTopk = 2051;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

struct DeviceBuffer {
  void* pointer = nullptr;
  std::size_t bytes;
  explicit DeviceBuffer(std::size_t size) : bytes(size) {
    check(cudaMalloc(&pointer, bytes), "allocate benchmark buffer");
  }
  ~DeviceBuffer() { cudaFree(pointer); }
};

__global__ void initialize(__nv_bfloat16* query, __nv_bfloat16* key,
                           __nv_bfloat16* value, std::int32_t* indices,
                           int sequences, int query_tokens, int context_tokens,
                           bool disjoint) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t rows =
      static_cast<std::size_t>(sequences) * query_tokens;
  const std::size_t query_elements = rows * kHeads * kDim;
  const std::size_t state_elements =
      static_cast<std::size_t>(sequences) * context_tokens * kDim;
  const std::size_t index_elements = rows * kTopk;
  if (index < query_elements)
    query[index] = __float2bfloat16(
        static_cast<float>(static_cast<int>((index * 17 + 3) % 127) - 63) /
        64.0F);
  if (index < state_elements) {
    key[index] = __float2bfloat16(
        static_cast<float>(static_cast<int>((index * 13 + 5) % 113) - 56) /
        57.0F);
    value[index] = __float2bfloat16(
        static_cast<float>(static_cast<int>((index * 7 + 11) % 109) - 54) /
        55.0F);
  }
  if (index < index_elements) {
    const int row = index / kTopk;
    const int slot = index % kTopk;
    const int local_row = row % query_tokens;
    const int position = context_tokens - query_tokens + local_row;
    int first = max(0, position - (kTopk - 1));
    if (disjoint) {
      const int pair_row = local_row & 1;
      first = pair_row ? 4096 : 0;
    }
    const int logical = first + slot;
    indices[index] = logical <= position && logical < context_tokens
                         ? logical
                         : -1;
  }
}

std::uint64_t hash(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t value = 14695981039346656037ULL;
  for (std::uint8_t byte : bytes) value = (value ^ byte) * 1099511628211ULL;
  return value;
}

std::uint64_t device_hash(const DeviceBuffer& buffer) {
  std::vector<std::uint8_t> bytes(buffer.bytes);
  check(cudaMemcpy(bytes.data(), buffer.pointer, buffer.bytes,
                   cudaMemcpyDeviceToHost),
        "copy immutable state for hash");
  return hash(bytes);
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc < 3 || argc > 5)
    throw std::invalid_argument(
        "usage: qwen38-qsa-prefill-smoke SEQUENCES QUERY_TOKENS "
        "[overlap|disjoint] [OUTPUT_BF16]");
  const int sequences = std::stoi(argv[1]);
  const int query_tokens = std::stoi(argv[2]);
  const int context_tokens = query_tokens == 300 ? 8492 : 8192;
  const std::string pattern = argc >= 4 ? argv[3] : "overlap";
  if (pattern != "overlap" && pattern != "disjoint")
    throw std::invalid_argument("QSA prefill pattern must be overlap or disjoint");
  const bool disjoint = pattern == "disjoint";
  const std::string output_path = argc == 5 ? argv[4] : "";
  if (disjoint && query_tokens != 300)
    throw std::invalid_argument("disjoint control is defined for tool bursts");
  const std::size_t rows =
      static_cast<std::size_t>(sequences) * query_tokens;
  DeviceBuffer query(rows * kHeads * kDim * 2);
  DeviceBuffer key(static_cast<std::size_t>(sequences) * context_tokens * kDim *
                   2);
  DeviceBuffer value(key.bytes);
  DeviceBuffer indices(rows * kTopk * 4);
  DeviceBuffer output(rows * kHeads * kDim * 2);
  DeviceBuffer control(output.bytes);
  DeviceBuffer counters(sizeof(Qwen38QsaPrefillCounters));
  const std::size_t elements = std::max(
      {query.bytes / 2, key.bytes / 2, indices.bytes / 4});
  initialize<<<(elements + 255) / 256, 256>>>(
      static_cast<__nv_bfloat16*>(query.pointer),
      static_cast<__nv_bfloat16*>(key.pointer),
      static_cast<__nv_bfloat16*>(value.pointer),
      static_cast<std::int32_t*>(indices.pointer), sequences, query_tokens,
      context_tokens, disjoint);
  check(cudaDeviceSynchronize(), "initialize benchmark inputs");
  const std::uint64_t state_hash_before =
      device_hash(key) ^ (device_hash(value) * 0x9e3779b97f4a7c15ULL);
  if (qwen38_qsa_prefill_prepare(0))
    throw std::runtime_error(qwen38_qsa_prefill_last_error());
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream), "create benchmark stream");
  check(cudaMemsetAsync(counters.pointer, 0, counters.bytes, stream),
        "clear prefill counters");
  if (qwen38_qsa_prefill_union2(
          static_cast<__nv_bfloat16*>(query.pointer),
          static_cast<__nv_bfloat16*>(key.pointer),
          static_cast<__nv_bfloat16*>(value.pointer),
          static_cast<std::int32_t*>(indices.pointer), sequences, query_tokens,
          context_tokens, static_cast<__nv_bfloat16*>(output.pointer),
          static_cast<Qwen38QsaPrefillCounters*>(counters.pointer), stream))
    throw std::runtime_error(qwen38_qsa_prefill_last_error());
  check(cudaStreamSynchronize(stream), "warm QSA prefill");

  // The scalar oracle is intentionally limited to c1. It is a correctness
  // proof, not a performance baseline. It runs before all timing samples.
  double maximum_error = -1.0;
  double mean_error = -1.0;
  if (sequences == 1) {
    if (qwen38_qsa_prefill_control(
            static_cast<__nv_bfloat16*>(query.pointer),
            static_cast<__nv_bfloat16*>(key.pointer),
            static_cast<__nv_bfloat16*>(value.pointer),
            static_cast<std::int32_t*>(indices.pointer), sequences,
            query_tokens, context_tokens,
            static_cast<__nv_bfloat16*>(control.pointer), stream))
      throw std::runtime_error(qwen38_qsa_prefill_last_error());
    check(cudaStreamSynchronize(stream), "fence scalar control");
    std::vector<__nv_bfloat16> selected(output.bytes / 2), oracle(control.bytes / 2);
    check(cudaMemcpy(selected.data(), output.pointer, output.bytes,
                     cudaMemcpyDeviceToHost),
          "copy selected output");
    check(cudaMemcpy(oracle.data(), control.pointer, control.bytes,
                     cudaMemcpyDeviceToHost),
          "copy control output");
    maximum_error = 0.0;
    mean_error = 0.0;
    for (std::size_t index = 0; index < selected.size(); ++index) {
      const double difference = std::abs(
          static_cast<double>(__bfloat162float(selected[index])) -
          static_cast<double>(__bfloat162float(oracle[index])));
      maximum_error = std::max(maximum_error, difference);
      mean_error += difference / selected.size();
    }
  }

  std::array<float, 7> samples{};
  for (float& sample : samples) {
    cudaEvent_t begin = nullptr, end = nullptr;
    check(cudaEventCreate(&begin), "create begin event");
    check(cudaEventCreate(&end), "create end event");
    check(cudaEventRecord(begin, stream), "record begin event");
    if (qwen38_qsa_prefill_union2(
            static_cast<__nv_bfloat16*>(query.pointer),
            static_cast<__nv_bfloat16*>(key.pointer),
            static_cast<__nv_bfloat16*>(value.pointer),
            static_cast<std::int32_t*>(indices.pointer), sequences,
            query_tokens, context_tokens,
            static_cast<__nv_bfloat16*>(output.pointer),
            nullptr, stream))
      throw std::runtime_error(qwen38_qsa_prefill_last_error());
    check(cudaEventRecord(end, stream), "record end event");
    check(cudaEventSynchronize(end), "fence end event");
    check(cudaEventElapsedTime(&sample, begin, end), "read event time");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "begin prefill capture");
  if (qwen38_qsa_prefill_union2(
          static_cast<__nv_bfloat16*>(query.pointer),
          static_cast<__nv_bfloat16*>(key.pointer),
          static_cast<__nv_bfloat16*>(value.pointer),
          static_cast<std::int32_t*>(indices.pointer), sequences, query_tokens,
          context_tokens, static_cast<__nv_bfloat16*>(output.pointer),
          nullptr, stream))
    throw std::runtime_error(qwen38_qsa_prefill_last_error());
  check(cudaStreamEndCapture(stream, &graph), "end prefill capture");
  check(cudaGraphInstantiate(&executable, graph, 0), "instantiate prefill graph");
  std::uint64_t replay_hash = 0;
  for (int replay = 0; replay < 2; ++replay) {
    check(cudaGraphLaunch(executable, stream), "launch prefill graph");
    check(cudaStreamSynchronize(stream), "fence prefill graph");
    std::vector<std::uint8_t> bytes(output.bytes);
    check(cudaMemcpy(bytes.data(), output.pointer, output.bytes,
                     cudaMemcpyDeviceToHost),
          "copy replay output");
    const auto observed = hash(bytes);
    if (replay_hash && replay_hash != observed)
      throw std::runtime_error("QSA prefill graph replay changed output");
    replay_hash = observed;
  }
  Qwen38QsaPrefillCounters observed_counters{};
  check(cudaMemcpy(&observed_counters, counters.pointer,
                   sizeof(observed_counters),
                   cudaMemcpyDeviceToHost),
        "copy prefill counters");
  const std::uint64_t state_hash_after =
      device_hash(key) ^ (device_hash(value) * 0x9e3779b97f4a7c15ULL);
  if (state_hash_before != state_hash_after)
    throw std::runtime_error("QSA prefill mutated causal K/V state");
  if (!output_path.empty()) {
    std::vector<std::uint8_t> bytes(output.bytes);
    check(cudaMemcpy(bytes.data(), output.pointer, output.bytes,
                     cudaMemcpyDeviceToHost),
          "copy output artifact");
    std::ofstream artifact(output_path, std::ios::binary | std::ios::trunc);
    artifact.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!artifact)
      throw std::runtime_error("write QSA prefill output artifact");
  }
  auto ordered = samples;
  std::sort(ordered.begin(), ordered.end());
  const std::uint64_t union_per_launch = observed_counters.union_tokens;
  const std::uint64_t selected_per_launch =
      observed_counters.selected_tokens;
  const std::uint64_t index_loads_per_launch = observed_counters.index_loads;
  const std::uint64_t control_tokens = selected_per_launch;
  const std::uint64_t kv_bytes = union_per_launch * 2ULL * kDim * 2ULL;
  const std::uint64_t index_bytes = index_loads_per_launch * 4ULL;
  const std::uint64_t query_output_bytes = rows * kHeads * kDim * 4ULL;
  std::cout << "sequences=" << sequences << " query_tokens=" << query_tokens
            << " context_tokens=" << context_tokens
            << " pattern=" << (disjoint ? "disjoint" : "causal-overlap")
            << " median_ms=" << ordered[ordered.size() / 2]
            << " p95_ms=" << ordered.back()
            << " union_tokens=" << union_per_launch
            << " index_loads=" << index_loads_per_launch
            << " control_tokens=" << control_tokens
            << " kv_bytes=" << kv_bytes
            << " index_bytes=" << index_bytes
            << " query_output_bytes=" << query_output_bytes
            << " logical_hot_bytes="
            << kv_bytes + index_bytes + query_output_bytes
            << " control_kv_bytes=" << control_tokens * 2ULL * kDim * 2ULL
            << " max_abs=" << maximum_error << " mean_abs=" << mean_error
            << " graph_hash=" << replay_hash
            << " state_hash=" << state_hash_after
            << " state_unchanged=yes samples_ms=";
  for (std::size_t index = 0; index < samples.size(); ++index)
    std::cout << (index ? "," : "") << samples[index];
  std::cout << "\n";
  cudaGraphExecDestroy(executable);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  return 0;
} catch (const std::exception& exception) {
  std::cerr << "FAIL: " << exception.what() << "\n";
  return 1;
}

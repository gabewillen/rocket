// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_oracle_comparator.h"

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace decode = rocket::qwen38::decode;
namespace pr = rocket::qwen38::pair_reduce;

namespace {
void check(bool value) {
  if (!value) throw std::runtime_error("K0 oracle proof failed");
}
struct Sink final : pr::OtelStageSink {
  void emit_span_and_log(const pr::SpanRecord& record) noexcept override {
    ++spans;
    last = record.outcome;
  }
  void record_duration(const pr::MetricPoint&) noexcept override { ++metrics; }
  int spans = 0;
  int metrics = 0;
  pr::Outcome last = pr::Outcome::kContractError;
};
struct Cuda final : decode::TargetLayer3OracleCudaApi {
  cudaError_t host_alloc(void** pointer, std::size_t bytes) noexcept override {
    *pointer = std::malloc(bytes);
    return *pointer ? cudaSuccess : cudaErrorMemoryAllocation;
  }
  cudaError_t free_host(void* pointer) noexcept override {
    std::free(pointer);
    return cudaSuccess;
  }
  cudaError_t event_create(cudaEvent_t* event) noexcept override {
    *event = reinterpret_cast<cudaEvent_t>(0x1);
    return cudaSuccess;
  }
  cudaError_t event_destroy(cudaEvent_t) noexcept override { return cudaSuccess; }
  cudaError_t copy_d2h(void* destination, const void* source,
                       std::size_t bytes, cudaStream_t) noexcept override {
    std::memcpy(destination, source, bytes);
    return cudaSuccess;
  }
  cudaError_t event_record(cudaEvent_t, cudaStream_t) noexcept override {
    return cudaSuccess;
  }
  cudaError_t event_sync(cudaEvent_t) noexcept override { return cudaSuccess; }
};

std::vector<std::uint16_t> read_bf16(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  check(static_cast<bool>(input));
  const auto bytes = input.tellg();
  std::vector<std::uint16_t> result(
      static_cast<std::size_t>(bytes) / sizeof(std::uint16_t));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(result.data()), bytes);
  check(static_cast<bool>(input));
  return result;
}

float bf16_to_float(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    check(decode::accepted_target_k0_oracle(
              decode::kTargetK0OracleManifestSha256, 35) &&
          decode::accepted_target_k0_oracle(
              decode::kTargetK0ShortOracleManifestSha256, 87) &&
          !decode::accepted_target_k0_oracle(
              decode::kTargetK0OracleManifestSha256, 87));
    if (argc == 1) {
      std::puts("qwen38 K0 oracle: fixed manifest contracts passed");
      return 0;
    }
    check(argc == 2);
    const std::filesystem::path capture(argv[1]);
    Sink sink;
    Cuda cuda;
    decode::NativeTargetK0OracleComparator comparator(0, capture, sink, &cuda);
    check(comparator.authenticated() && comparator.rank() == 0 &&
          comparator.rows() == 35 &&
          comparator.expected_input_token(0) == 7734 &&
          comparator.expected_input_token(34) == 13);
    auto stream = reinterpret_cast<cudaStream_t>(0x1);
    decode::TargetK0LayerBoundaryEvidence boundary_evidence;
    std::vector<float> reduced(decode::kTargetK0Hidden, 0.0F);
    reduced[17] = 1.0F;
    comparator.observe(decode::TargetK0LayerBoundary::kAttentionReduction,
                       reduced.data(), reduced.size(),
                       decode::TargetK0DiagnosticDtype::kFloat32, stream,
                       boundary_evidence);
    const auto reduction_index = static_cast<std::size_t>(
        decode::TargetK0LayerBoundary::kAttentionReduction);
    check(boundary_evidence.hashes[reduction_index] != 0 &&
          boundary_evidence.elements[reduction_index] == reduced.size() &&
          boundary_evidence.zero_counts[reduction_index] == reduced.size() - 1 &&
          boundary_evidence.nonfinite_counts[reduction_index] == 0);
    bool duplicate_rejected = false;
    try {
      comparator.observe(decode::TargetK0LayerBoundary::kAttentionReduction,
                         reduced.data(), reduced.size(),
                         decode::TargetK0DiagnosticDtype::kFloat32, stream,
                         boundary_evidence);
    } catch (const std::invalid_argument&) {
      duplicate_rejected = true;
    }
    check(duplicate_rejected);
    auto embedding = read_bf16(capture / "embedding.bin");
    comparator.compare(decode::TargetK0Boundary::kEmbedding, 0, -1,
                       embedding.data(), decode::kTargetK0Hidden, stream);
    auto layer = read_bf16(capture / "layer-47.bin");
    comparator.compare(
        decode::TargetK0Boundary::kLayer, 34, 47,
        layer.data() + 34 * decode::kTargetK0HyperHidden,
        decode::kTargetK0HyperHidden, stream);
    auto final = read_bf16(capture / "final_norm.bin");
    comparator.compare(decode::TargetK0Boundary::kFinalNorm, 34, -1,
                       final.data() + 34 * decode::kTargetK0Hidden,
                       decode::kTargetK0Hidden, stream);
    auto logits = read_bf16(capture / "logits.bin");
    std::vector<float> rank0(decode::kTargetK0LocalVocab);
    for (std::size_t index = 0; index < rank0.size(); ++index)
      rank0[index] = bf16_to_float(logits[index]);
    comparator.compare(decode::TargetK0Boundary::kLocalLogits, 34, -1,
                       rank0.data(), rank0.size(), stream);
    comparator.compare_token(248'046);
    check(comparator.evidence().accepted && sink.last == pr::Outcome::kOk);

    Sink mismatch_sink;
    decode::NativeTargetK0OracleComparator mismatch(0, capture, mismatch_sink,
                                                     &cuda);
    embedding[0] = static_cast<std::uint16_t>(embedding[0] + 2);
    bool rejected = false;
    try {
      mismatch.compare(decode::TargetK0Boundary::kEmbedding, 0, -1,
                       embedding.data(), decode::kTargetK0Hidden, stream);
    } catch (const std::logic_error&) { rejected = true; }
    check(rejected && !mismatch.evidence().accepted &&
          mismatch.evidence().mismatch_count == 1 &&
          mismatch_sink.last == pr::Outcome::kContractError);
    std::puts("qwen38 K0 oracle: 51 artifacts and named boundaries passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

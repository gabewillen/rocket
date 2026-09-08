// SPDX-License-Identifier: Apache-2.0
#include "moe/route_compaction.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace moe = rocket::qwen38::moe;

namespace {

[[noreturn]] void fail(const std::string& message) {
  std::fprintf(stderr, "%s\n", message.c_str());
  std::exit(1);
}

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    fail(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void require(bool condition, const char* contract) {
  if (!condition) fail(contract);
}

template <typename T>
class DeviceArray {
 public:
  explicit DeviceArray(std::size_t count) : count_(count) {
    cuda_check(cudaMalloc(&data_, count_ * sizeof(T)), "cudaMalloc");
  }
  ~DeviceArray() { cudaFree(data_); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  [[nodiscard]] T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_;
};

struct DeviceFixture {
  DeviceArray<std::int32_t> input_ids{moe::kMaxRoutes};
  DeviceArray<float> input_weights{moe::kMaxRoutes};
  DeviceArray<std::uint64_t> source_generation{1};
  DeviceArray<std::uint64_t> requested_generation{1};
  DeviceArray<std::int32_t> active_ids{moe::kLocalExperts};
  DeviceArray<std::int32_t> local_to_active{moe::kLocalExperts};
  DeviceArray<std::int32_t> row_counts{moe::kLocalExperts};
  DeviceArray<std::int32_t> route_offsets{moe::kLocalExperts + 1};
  DeviceArray<std::int32_t> route_cursors{moe::kLocalExperts};
  DeviceArray<std::int32_t> owner_ids{moe::kMaxRoutes};
  DeviceArray<float> owner_weights{moe::kMaxRoutes};
  DeviceArray<std::int32_t> owner_rows{moe::kMaxRoutes};
  DeviceArray<std::uint8_t> owner_slots{moe::kMaxRoutes};
  DeviceArray<std::int32_t> route_indices{moe::kMaxRoutes};
  DeviceArray<moe::RouteCompactionDeviceSummary> summary{1};

  [[nodiscard]] moe::RouteCompactionInput input() const noexcept {
    return {
        .global_expert_ids = input_ids.get(),
        .routing_weights = input_weights.get(),
        .source_generation = source_generation.get(),
        .requested_generation = requested_generation.get(),
    };
  }

  [[nodiscard]] moe::RouteCompactionBuffers output() const noexcept {
    return {
        .active_global_expert_ids = active_ids.get(),
        .local_to_active = local_to_active.get(),
        .expert_row_counts = row_counts.get(),
        .expert_route_offsets = route_offsets.get(),
        .expert_route_cursors = route_cursors.get(),
        .owner_route_global_expert_ids = owner_ids.get(),
        .owner_route_weights = owner_weights.get(),
        .owner_route_rows = owner_rows.get(),
        .owner_route_slots = owner_slots.get(),
        .expert_route_indices = route_indices.get(),
        .summary = summary.get(),
    };
  }
};

struct HostInput {
  std::vector<std::int32_t> ids;
  std::vector<float> weights;
};

int target_union(int concurrency) {
  switch (concurrency) {
    case 1: return 19;
    case 2: return 30;
    case 4: return 44;
    case 8: return 51;
    case 16: return 256;
    default: fail("unsupported concurrency");
  }
}

HostInput make_input(int rows, int concurrency) {
  HostInput input;
  input.ids.reserve(static_cast<std::size_t>(rows * moe::kTopK));
  input.weights.reserve(static_cast<std::size_t>(rows * moe::kTopK));
  const int expert_union = target_union(concurrency);
  for (int row = 0; row < rows; ++row) {
    for (int slot = 0; slot < moe::kTopK; ++slot) {
      const int owner_local = (row * 5 + slot / 2) % expert_union;
      input.ids.push_back(slot % 2 == 0 ? owner_local : 256 + owner_local);
      input.weights.push_back(static_cast<float>(slot + 1) / 55.0F);
    }
  }
  return input;
}

void upload_input(const HostInput& input, std::uint64_t source_generation,
                  std::uint64_t requested_generation, DeviceFixture& device,
                  cudaStream_t stream) {
  cuda_check(cudaMemcpyAsync(device.input_ids.get(), input.ids.data(),
                             input.ids.size() * sizeof(std::int32_t),
                             cudaMemcpyHostToDevice, stream),
             "copy route ids");
  cuda_check(cudaMemcpyAsync(device.input_weights.get(), input.weights.data(),
                             input.weights.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream),
             "copy route weights");
  cuda_check(cudaMemcpyAsync(device.source_generation.get(), &source_generation,
                             sizeof(source_generation), cudaMemcpyHostToDevice,
                             stream),
             "copy source generation");
  cuda_check(cudaMemcpyAsync(device.requested_generation.get(),
                             &requested_generation,
                             sizeof(requested_generation),
                             cudaMemcpyHostToDevice, stream),
             "copy requested generation");
  cuda_check(cudaStreamSynchronize(stream), "finish input upload");
}

class CapturedCompaction {
 public:
  explicit CapturedCompaction(const moe::RouteCompactionLaunch& launch) {
    cuda_check(cudaStreamBeginCapture(launch.stream,
                                      cudaStreamCaptureModeThreadLocal),
               "begin compaction capture");
    require(moe::enqueue_route_compaction(launch) ==
                moe::RouteCompactionOutcome::kOk,
            "compaction enqueue failed during capture");
    cuda_check(cudaStreamEndCapture(launch.stream, &graph_),
               "end compaction capture");
    cuda_check(cudaGraphGetNodes(graph_, nullptr, &nodes_), "count graph nodes");
    require(nodes_ > 0, "captured graph is empty");
    cuda_check(cudaGraphInstantiate(&executable_, graph_, 0),
               "instantiate compaction graph");
  }
  ~CapturedCompaction() {
    if (executable_) cudaGraphExecDestroy(executable_);
    if (graph_) cudaGraphDestroy(graph_);
  }
  CapturedCompaction(const CapturedCompaction&) = delete;
  CapturedCompaction& operator=(const CapturedCompaction&) = delete;
  void launch(cudaStream_t stream) const {
    cuda_check(cudaGraphLaunch(executable_, stream), "launch compaction graph");
  }
  [[nodiscard]] std::size_t nodes() const noexcept { return nodes_; }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  std::size_t nodes_ = 0;
};

template <typename T>
std::vector<T> download(T* source, std::size_t count) {
  std::vector<T> output(count);
  cuda_check(cudaMemcpy(output.data(), source, count * sizeof(T),
                        cudaMemcpyDeviceToHost),
             "download result");
  return output;
}

moe::RouteCompactionDeviceSummary download_summary(const DeviceFixture& device) {
  return download(device.summary.get(), std::size_t{1}).front();
}

void verify_parity(const HostInput& input, moe::RouteCompactionShape shape,
                   std::uint64_t generation, const DeviceFixture& device) {
  const auto expected = moe::compact_owner_routes_reference({
      .shape = shape,
      .capacity = {},
      .global_expert_ids = input.ids,
      .routing_weights = input.weights,
      .source_generation = generation,
      .requested_generation = generation,
  });
  const auto actual = download_summary(device);
  require(actual.outcome == expected.summary.outcome &&
              actual.generation == expected.summary.generation &&
              actual.active_experts == expected.summary.active_experts &&
              actual.active_rows == expected.summary.active_rows &&
              actual.active_routes == expected.summary.active_routes &&
              actual.active_weight_bytes == expected.summary.active_weight_bytes,
          "device summary diverged from scalar oracle");

  const auto active_count = static_cast<std::size_t>(actual.active_experts);
  const auto route_count = static_cast<std::size_t>(actual.active_routes);
  require(download(device.active_ids.get(), active_count) ==
              expected.active_global_expert_ids,
          "active expert order diverged");
  require(download(device.row_counts.get(), active_count) ==
              expected.expert_row_counts,
          "expert row counts diverged");
  require(download(device.route_offsets.get(), active_count + 1) ==
              expected.expert_route_offsets,
          "expert offsets diverged");
  require(download(device.owner_ids.get(), route_count) ==
              expected.owner_route_global_expert_ids,
          "row-major owner ids diverged");
  require(download(device.owner_rows.get(), route_count) ==
              expected.owner_route_rows,
          "row-major owner rows diverged");
  require(download(device.owner_slots.get(), route_count) ==
              expected.owner_route_slots,
          "top-k slot order diverged");
  require(download(device.route_indices.get(), route_count) ==
              expected.expert_route_indices,
          "expert route permutation diverged");
  const auto actual_weights = download(device.owner_weights.get(), route_count);
  for (std::size_t route = 0; route < route_count; ++route) {
    require(std::bit_cast<std::uint32_t>(actual_weights[route]) ==
                std::bit_cast<std::uint32_t>(
                    expected.owner_route_weights[route]),
            "route weight bits diverged");
  }
  const auto cursors = download(device.route_cursors.get(), active_count);
  for (std::size_t expert = 0; expert < active_count; ++expert) {
    require(cursors[expert] == expected.expert_route_offsets[expert + 1],
            "expert cursor did not terminate at its offset");
  }
  const auto local_map = download(device.local_to_active.get(),
                                  std::size_t{moe::kLocalExperts});
  std::vector<std::int32_t> expected_map(moe::kLocalExperts, -1);
  for (std::size_t active = 0; active < active_count; ++active) {
    expected_map[static_cast<std::size_t>(
        expected.active_global_expert_ids[active])] =
        static_cast<std::int32_t>(active);
  }
  require(local_map == expected_map, "local-to-active map diverged");
}

struct Latency {
  float p50_us;
  float p95_us;
};

Latency measure(const CapturedCompaction& graph, cudaStream_t stream) {
  constexpr int kWarmups = 100;
  constexpr int kSamples = 500;
  for (int warmup = 0; warmup < kWarmups; ++warmup) graph.launch(stream);
  cuda_check(cudaStreamSynchronize(stream), "finish warmup");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  cuda_check(cudaEventCreate(&begin), "create begin event");
  cuda_check(cudaEventCreate(&end), "create end event");
  std::vector<float> samples;
  samples.reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    cuda_check(cudaEventRecord(begin, stream), "record begin");
    graph.launch(stream);
    cuda_check(cudaEventRecord(end, stream), "record end");
    cuda_check(cudaEventSynchronize(end), "wait timing sample");
    float elapsed_ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, begin, end),
               "read timing sample");
    samples.push_back(elapsed_ms * 1'000.0F);
  }
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  std::sort(samples.begin(), samples.end());
  return {
      .p50_us = samples[samples.size() / 2],
      .p95_us = samples[(samples.size() * 95) / 100],
  };
}

void require_failed_summary(const DeviceFixture& device,
                            moe::RouteCompactionOutcome outcome) {
  const auto summary = download_summary(device);
  require(summary.outcome == outcome && summary.generation == 0 &&
              summary.active_experts == 0 && summary.active_rows == 0 &&
              summary.active_routes == 0 && summary.active_weight_bytes == 0,
          "failure did not close the active prefix");
}

void require_empty_success(const DeviceFixture& device,
                           std::uint64_t generation) {
  const auto summary = download_summary(device);
  require(summary.outcome == moe::RouteCompactionOutcome::kOk &&
              summary.generation == generation &&
              summary.active_experts == 0 && summary.active_rows == 0 &&
              summary.active_routes == 0 && summary.active_weight_bytes == 0,
          "all-remote replay did not produce an empty active prefix");
}

void run_failure_matrix(DeviceFixture& device, cudaStream_t stream) {
  constexpr moe::RouteCompactionShape kShape{
      .rank = 0, .sequences = 1, .rows = 2};
  auto input = make_input(kShape.rows, kShape.sequences);
  upload_input(input, 10, 11, device, stream);
  const moe::RouteCompactionLaunch launch{
      .shape = kShape,
      .capacity = {},
      .input = device.input(),
      .output = device.output(),
      .stream = stream,
  };
  CapturedCompaction graph(launch);
  graph.launch(stream);
  cuda_check(cudaStreamSynchronize(stream), "finish stale replay");
  require_failed_summary(device, moe::RouteCompactionOutcome::kStaleGeneration);

  upload_input(input, 0, 0, device, stream);
  graph.launch(stream);
  cuda_check(cudaStreamSynchronize(stream), "finish zero-generation replay");
  require_failed_summary(device, moe::RouteCompactionOutcome::kStaleGeneration);

  input.ids[1] = input.ids[0];
  upload_input(input, 12, 12, device, stream);
  graph.launch(stream);
  cuda_check(cudaStreamSynchronize(stream), "finish duplicate replay");
  require_failed_summary(device, moe::RouteCompactionOutcome::kContractError);

  input = make_input(kShape.rows, kShape.sequences);
  for (std::size_t route = 0; route < input.ids.size(); ++route) {
    input.ids[route] = moe::kLocalExperts +
                       static_cast<std::int32_t>(route % moe::kLocalExperts);
  }
  upload_input(input, 13, 13, device, stream);
  graph.launch(stream);
  cuda_check(cudaStreamSynchronize(stream), "finish all-remote replay");
  require_empty_success(device, 13);

  input = make_input(kShape.rows, kShape.sequences);
  upload_input(input, 14, 14, device, stream);
  const moe::RouteCompactionLaunch expert_overflow{
      .shape = kShape,
      .capacity = {.experts = 1, .rows = moe::kMaxRows,
                   .routes = moe::kMaxRoutes},
      .input = device.input(),
      .output = device.output(),
      .stream = stream,
  };
  CapturedCompaction expert_overflow_graph(expert_overflow);
  expert_overflow_graph.launch(stream);
  cuda_check(cudaStreamSynchronize(stream), "finish expert overflow replay");
  require_failed_summary(device, moe::RouteCompactionOutcome::kOverflow);

  const moe::RouteCompactionLaunch route_overflow{
      .shape = kShape,
      .capacity = {.experts = moe::kLocalExperts, .rows = moe::kMaxRows,
                   .routes = 1},
      .input = device.input(),
      .output = device.output(),
      .stream = stream,
  };
  CapturedCompaction route_overflow_graph(route_overflow);
  route_overflow_graph.launch(stream);
  cuda_check(cudaStreamSynchronize(stream), "finish route overflow replay");
  require_failed_summary(device, moe::RouteCompactionOutcome::kOverflow);
}

void run_cell(int concurrency, int depth, DeviceFixture& device,
              cudaStream_t stream) {
  const int verify_width = depth + 1;
  const int rows = concurrency * verify_width;
  const moe::RouteCompactionShape shape{
      .rank = 0, .sequences = concurrency, .rows = rows};
  const auto input = make_input(rows, concurrency);
  constexpr std::uint64_t kFirstGeneration = 100;
  upload_input(input, kFirstGeneration, kFirstGeneration, device, stream);
  const moe::RouteCompactionLaunch launch{
      .shape = shape,
      .capacity = {},
      .input = device.input(),
      .output = device.output(),
      .stream = stream,
  };
  CapturedCompaction graph(launch);
  graph.launch(stream);
  cuda_check(cudaStreamSynchronize(stream), "finish first parity replay");
  verify_parity(input, shape, kFirstGeneration, device);

  constexpr std::uint64_t kNextGeneration = 101;
  cuda_check(cudaMemcpyAsync(device.source_generation.get(), &kNextGeneration,
                             sizeof(kNextGeneration), cudaMemcpyHostToDevice,
                             stream),
             "advance source generation");
  cuda_check(cudaMemcpyAsync(device.requested_generation.get(), &kNextGeneration,
                             sizeof(kNextGeneration), cudaMemcpyHostToDevice,
                             stream),
             "advance requested generation");
  graph.launch(stream);
  cuda_check(cudaStreamSynchronize(stream), "finish advanced parity replay");
  verify_parity(input, shape, kNextGeneration, device);

  const auto latency = measure(graph, stream);
  const auto summary = download_summary(device);
  std::printf(
      "{\"active_experts\":%d,\"active_rows\":%d,"
      "\"active_routes\":%d,\"active_weight_bytes\":%llu,"
      "\"concurrency\":%d,\"cuda_graph_nodes\":%zu,\"depth\":%d,"
      "\"iterations\":500,\"p50_us\":%.6f,\"p95_us\":%.6f,"
      "\"parity\":true,\"synthetic_input\":true,"
      "\"union_basis\":\"%s\",\"verify_width\":%d}\n",
      summary.active_experts, summary.active_rows, summary.active_routes,
      static_cast<unsigned long long>(summary.active_weight_bytes), concurrency,
      graph.nodes(), depth, latency.p50_us, latency.p95_us,
      concurrency == 16 ? "capacity_pattern" : "schema_v4_median_target",
      verify_width);
}

}  // namespace

int main() {
  int device_id = -1;
  cuda_check(cudaGetDevice(&device_id), "get device");
  cudaDeviceProp properties{};
  cuda_check(cudaGetDeviceProperties(&properties, device_id),
             "get device properties");
  require(properties.major == 12 && properties.minor == 1,
          "route compaction proof requires SM121");
  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreate(&stream), "create stream");
  {
    DeviceFixture fixture;
    run_failure_matrix(fixture, stream);
    for (const int concurrency : {1, 2, 4, 8, 16}) {
      for (const int depth : {1, 4, 7}) {
        run_cell(concurrency, depth, fixture, stream);
      }
    }
  }
  cuda_check(cudaStreamDestroy(stream), "destroy stream");
  std::printf(
      "{\"failure_matrix\":true,\"gpu\":\"%s\","
      "\"schema\":\"rocket.qwen38.route-compaction.physical.v1\"}\n",
      properties.name);
  return 0;
}

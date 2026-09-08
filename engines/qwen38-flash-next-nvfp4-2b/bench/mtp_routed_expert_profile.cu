// SPDX-License-Identifier: Apache-2.0
#include "moe/fp8_routed_experts.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe = rocket::qwen38::moe;

namespace {

void check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

template <typename T>
T* allocate(std::vector<void*>& allocations, std::size_t elements) {
  T* pointer = nullptr;
  check(cudaMalloc(&pointer, elements * sizeof(T)), "cudaMalloc");
  allocations.push_back(pointer);
  return pointer;
}

struct Events {
  cudaEvent_t start{};
  cudaEvent_t after_compaction{};
  cudaEvent_t after_prepare{};
  cudaEvent_t after_quantize{};
  cudaEvent_t after_gate_up{};
  cudaEvent_t after_silu{};
  cudaEvent_t after_down{};
  cudaEvent_t stop{};

  Events() {
    for (auto* event : {&start, &after_compaction, &after_prepare,
                        &after_quantize, &after_gate_up, &after_silu,
                        &after_down, &stop}) {
      check(cudaEventCreate(event), "cudaEventCreate");
    }
  }
  ~Events() {
    for (auto event : {start, after_compaction, after_prepare, after_quantize,
                       after_gate_up, after_silu, after_down, stop}) {
      if (event) cudaEventDestroy(event);
    }
  }
};

struct Stats {
  double mean;
  double stddev;
  float p50;
  float p95;
  float minimum;
  float maximum;
};

Stats summarize(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                      static_cast<double>(values.size());
  double squared = 0.0;
  for (float value : values) squared += (value - mean) * (value - mean);
  const auto percentile = [&](double p) {
    const auto index = static_cast<std::size_t>(
        std::ceil(p * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1)];
  };
  return {mean, std::sqrt(squared / static_cast<double>(values.size())),
          percentile(0.50), percentile(0.95), values.front(), values.back()};
}

struct Cohort {
  int concurrency;
  std::array<double, 2> k4_owner_routes;
  std::array<int, 2> median_union;
};

// Rank-local medians from accepted activation-telemetry v4 cohorts at K4.
// Source summary SHA256 values, ordered c1/c2/c4/c8:
// 0765452712ac36e87f89ab7d9a3d874c158f8b4bc69f6d845e6f308fcf73cbf4
// 1d55128c24fabe8d4e8e8206b750f6880aa119136f4a8792028272e233c04007
// ad14463360691a410eeedea0bb95b85df7028c9e64eed710f7e18c95956ad85b
// 5a073b4e7e66b59ccfff3479698575bc0cb04194e942d3f251392174327f4d4b
// Owner routes scale by verifier width. A K is omitted when that prefix cannot
// contain the accepted active-expert union. The c16/K7 row is a full-capacity
// control because its calibration attempt failed closed before publication.
constexpr std::array<Cohort, 4> kAcceptedCohorts{{
    {1, {25.0, 25.0}, {19, 19}},
    {2, {49.0, 51.0}, {29, 30}},
    {4, {99.0, 101.0}, {43, 44}},
    {8, {175.5, 182.5}, {48, 51}},
}};

struct Workload {
  int concurrency;
  int draft_depth;
  int rank;
  int rows;
  int owner_routes;
  int active_experts;
  bool capacity_control;
};

std::vector<Workload> workloads() {
  std::vector<Workload> result;
  for (const auto& cohort : kAcceptedCohorts) {
    for (int rank = 0; rank < 2; ++rank) {
      for (int depth : {1, 4, 7}) {
        const int rows = cohort.concurrency * (depth + 1);
        const int owner_routes = static_cast<int>(std::lround(
            cohort.k4_owner_routes[rank] * (depth + 1) / 5.0));
        if (owner_routes < cohort.median_union[rank]) continue;
        result.push_back({cohort.concurrency, depth, rank, rows, owner_routes,
                          cohort.median_union[rank], false});
      }
    }
  }
  for (int rank = 0; rank < 2; ++rank) {
    result.push_back({16, 7, rank, moe::kMaxRows, moe::kMaxRoutes,
                      moe::kLocalExperts, true});
  }
  return result;
}

struct DeviceArena {
  std::vector<void*> allocations;
  ~DeviceArena() {
    for (void* pointer : allocations) cudaFree(pointer);
  }
};

struct FixedBuffers {
  explicit FixedBuffers(DeviceArena& arena) {
    constexpr std::size_t weight_elements =
        moe::kHidden * moe::kLogicalIntermediate;
    constexpr std::size_t scale_elements =
        (moe::kHidden / moe::kFp8Block) *
        (moe::kLogicalIntermediate / moe::kFp8Block);
    gate = allocate<std::uint8_t>(arena.allocations,
                                  moe::kLocalExperts * weight_elements);
    up = allocate<std::uint8_t>(arena.allocations,
                                moe::kLocalExperts * weight_elements);
    down = allocate<std::uint8_t>(arena.allocations,
                                  moe::kLocalExperts * weight_elements);
    gate_scale = allocate<__nv_bfloat16>(arena.allocations,
                                         moe::kLocalExperts * scale_elements);
    up_scale = allocate<__nv_bfloat16>(arena.allocations,
                                       moe::kLocalExperts * scale_elements);
    down_scale = allocate<__nv_bfloat16>(arena.allocations,
                                         moe::kLocalExperts * scale_elements);
    check(cudaMemset(gate, 0, moe::kLocalExperts * weight_elements),
          "cudaMemset gate");
    check(cudaMemset(up, 0, moe::kLocalExperts * weight_elements),
          "cudaMemset up");
    check(cudaMemset(down, 0, moe::kLocalExperts * weight_elements),
          "cudaMemset down");
    check(cudaMemset(gate_scale, 0,
                     moe::kLocalExperts * scale_elements * sizeof(*gate_scale)),
          "cudaMemset gate scales");
    check(cudaMemset(up_scale, 0,
                     moe::kLocalExperts * scale_elements * sizeof(*up_scale)),
          "cudaMemset up scales");
    check(cudaMemset(down_scale, 0,
                     moe::kLocalExperts * scale_elements * sizeof(*down_scale)),
          "cudaMemset down scales");

    std::vector<const std::uint8_t*> gate_ptrs(moe::kLocalExperts);
    std::vector<const std::uint8_t*> up_ptrs(moe::kLocalExperts);
    std::vector<const std::uint8_t*> down_ptrs(moe::kLocalExperts);
    std::vector<const __nv_bfloat16*> gate_scale_ptrs(moe::kLocalExperts);
    std::vector<const __nv_bfloat16*> up_scale_ptrs(moe::kLocalExperts);
    std::vector<const __nv_bfloat16*> down_scale_ptrs(moe::kLocalExperts);
    for (int expert = 0; expert < moe::kLocalExperts; ++expert) {
      gate_ptrs[expert] = gate + expert * weight_elements;
      up_ptrs[expert] = up + expert * weight_elements;
      down_ptrs[expert] = down + expert * weight_elements;
      gate_scale_ptrs[expert] = gate_scale + expert * scale_elements;
      up_scale_ptrs[expert] = up_scale + expert * scale_elements;
      down_scale_ptrs[expert] = down_scale + expert * scale_elements;
    }
    d_gate_ptrs = allocate<const std::uint8_t*>(arena.allocations,
                                                moe::kLocalExperts);
    d_up_ptrs = allocate<const std::uint8_t*>(arena.allocations,
                                              moe::kLocalExperts);
    d_down_ptrs = allocate<const std::uint8_t*>(arena.allocations,
                                                moe::kLocalExperts);
    d_gate_scale_ptrs = allocate<const __nv_bfloat16*>(arena.allocations,
                                                       moe::kLocalExperts);
    d_up_scale_ptrs = allocate<const __nv_bfloat16*>(arena.allocations,
                                                     moe::kLocalExperts);
    d_down_scale_ptrs = allocate<const __nv_bfloat16*>(arena.allocations,
                                                       moe::kLocalExperts);
    check(cudaMemcpy(d_gate_ptrs, gate_ptrs.data(),
                     gate_ptrs.size() * sizeof(gate_ptrs[0]),
                     cudaMemcpyHostToDevice), "copy gate table");
    check(cudaMemcpy(d_up_ptrs, up_ptrs.data(),
                     up_ptrs.size() * sizeof(up_ptrs[0]),
                     cudaMemcpyHostToDevice), "copy up table");
    check(cudaMemcpy(d_down_ptrs, down_ptrs.data(),
                     down_ptrs.size() * sizeof(down_ptrs[0]),
                     cudaMemcpyHostToDevice), "copy down table");
    check(cudaMemcpy(d_gate_scale_ptrs, gate_scale_ptrs.data(),
                     gate_scale_ptrs.size() * sizeof(gate_scale_ptrs[0]),
                     cudaMemcpyHostToDevice), "copy gate scale table");
    check(cudaMemcpy(d_up_scale_ptrs, up_scale_ptrs.data(),
                     up_scale_ptrs.size() * sizeof(up_scale_ptrs[0]),
                     cudaMemcpyHostToDevice), "copy up scale table");
    check(cudaMemcpy(d_down_scale_ptrs, down_scale_ptrs.data(),
                     down_scale_ptrs.size() * sizeof(down_scale_ptrs[0]),
                     cudaMemcpyHostToDevice), "copy down scale table");

    route_ids = allocate<std::int32_t>(arena.allocations, moe::kMaxRoutes);
    route_weights = allocate<float>(arena.allocations, moe::kMaxRoutes);
    source_generation = allocate<std::uint64_t>(arena.allocations, 1);
    requested_generation = allocate<std::uint64_t>(arena.allocations, 1);
    routes = {
        .active_global_expert_ids = allocate<std::int32_t>(arena.allocations, moe::kLocalExperts),
        .local_to_active = allocate<std::int32_t>(arena.allocations, moe::kLocalExperts),
        .expert_row_counts = allocate<std::int32_t>(arena.allocations, moe::kLocalExperts),
        .expert_route_offsets = allocate<std::int32_t>(arena.allocations, moe::kLocalExperts + 1),
        .expert_route_cursors = allocate<std::int32_t>(arena.allocations, moe::kLocalExperts),
        .owner_route_global_expert_ids = allocate<std::int32_t>(arena.allocations, moe::kMaxRoutes),
        .owner_route_weights = allocate<float>(arena.allocations, moe::kMaxRoutes),
        .owner_route_rows = allocate<std::int32_t>(arena.allocations, moe::kMaxRoutes),
        .owner_route_slots = allocate<std::uint8_t>(arena.allocations, moe::kMaxRoutes),
        .expert_route_indices = allocate<std::int32_t>(arena.allocations, moe::kMaxRoutes),
        .summary = allocate<moe::RouteCompactionDeviceSummary>(arena.allocations, 1),
    };
    hidden = allocate<__nv_bfloat16>(arena.allocations,
                                     moe::kMaxRows * moe::kHidden);
    quantized_hidden = allocate<std::uint8_t>(arena.allocations,
                                              moe::kMaxRows * moe::kHidden);
    hidden_scale = allocate<float>(arena.allocations,
                                   moe::kMaxRows * moe::kHidden / moe::kFp8Block);
    gate_up_scratch = allocate<__nv_bfloat16>(
        arena.allocations, moe::kMaxRoutes * 2 * moe::kPhysicalIntermediate);
    activated = allocate<std::uint8_t>(
        arena.allocations, moe::kMaxRoutes * moe::kPhysicalIntermediate);
    activated_scale = allocate<float>(
        arena.allocations,
        moe::kMaxRoutes * moe::kLogicalIntermediate / moe::kFp8Block);
    output = allocate<float>(arena.allocations, moe::kMaxRows * moe::kHidden);
    expert_summary = allocate<moe::RoutedExpertDeviceSummary>(arena.allocations, 1);
    check(cudaMemset(hidden, 0, moe::kMaxRows * moe::kHidden * sizeof(*hidden)),
          "cudaMemset hidden");
  }

  std::uint8_t* gate{};
  std::uint8_t* up{};
  std::uint8_t* down{};
  __nv_bfloat16* gate_scale{};
  __nv_bfloat16* up_scale{};
  __nv_bfloat16* down_scale{};
  const std::uint8_t** d_gate_ptrs{};
  const std::uint8_t** d_up_ptrs{};
  const std::uint8_t** d_down_ptrs{};
  const __nv_bfloat16** d_gate_scale_ptrs{};
  const __nv_bfloat16** d_up_scale_ptrs{};
  const __nv_bfloat16** d_down_scale_ptrs{};
  std::int32_t* route_ids{};
  float* route_weights{};
  std::uint64_t* source_generation{};
  std::uint64_t* requested_generation{};
  moe::RouteCompactionBuffers routes{};
  __nv_bfloat16* hidden{};
  std::uint8_t* quantized_hidden{};
  float* hidden_scale{};
  __nv_bfloat16* gate_up_scratch{};
  std::uint8_t* activated{};
  float* activated_scale{};
  float* output{};
  moe::RoutedExpertDeviceSummary* expert_summary{};
};

void install_workload(const Workload& workload, FixedBuffers& buffers) {
  const int total_routes = workload.rows * moe::kTopK;
  std::vector<std::int32_t> ids(total_routes);
  std::vector<float> weights(total_routes, 0.1F);
  int owner_seen = 0;
  int remote_seen = 0;
  const int owner_base = workload.rank * moe::kLocalExperts;
  const int remote_base = (1 - workload.rank) * moe::kLocalExperts;
  for (int route = 0; route < total_routes; ++route) {
    const bool owner = ((route + 1) * workload.owner_routes / total_routes) !=
                       (route * workload.owner_routes / total_routes);
    if (owner) {
      ids[route] = owner_base + owner_seen++ % workload.active_experts;
    } else {
      ids[route] = remote_base + remote_seen++ % moe::kLocalExperts;
    }
  }
  check(cudaMemcpy(buffers.route_ids, ids.data(), ids.size() * sizeof(ids[0]),
                   cudaMemcpyHostToDevice), "copy route ids");
  check(cudaMemcpy(buffers.route_weights, weights.data(),
                   weights.size() * sizeof(weights[0]), cudaMemcpyHostToDevice),
        "copy route weights");
  const std::uint64_t generation =
      static_cast<std::uint64_t>(workload.concurrency * 100 +
                                 workload.draft_depth * 10 + workload.rank + 1);
  check(cudaMemcpy(buffers.source_generation, &generation, sizeof(generation),
                   cudaMemcpyHostToDevice), "copy source generation");
  check(cudaMemcpy(buffers.requested_generation, &generation,
                   sizeof(generation), cudaMemcpyHostToDevice),
        "copy requested generation");
}

cudaGraphExec_t capture(cudaStream_t stream, const Workload& workload,
                        FixedBuffers& buffers,
                        const moe::Fp8RoutedExperts& consumer,
                        Events* events, std::size_t* node_count) {
  const moe::RouteCompactionShape shape{workload.rank, workload.concurrency,
                                         workload.rows};
  const moe::RouteCompactionCapacity capacity{};
  const moe::RoutedExpertStageEvents stage_events{
      events ? events->after_prepare : nullptr,
      events ? events->after_quantize : nullptr,
      events ? events->after_gate_up : nullptr,
      events ? events->after_silu : nullptr,
      events ? events->after_down : nullptr,
  };
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture");
  if (moe::enqueue_route_compaction({
          shape, capacity,
          {buffers.route_ids, buffers.route_weights, buffers.source_generation,
           buffers.requested_generation},
          buffers.routes, stream}) != moe::RouteCompactionOutcome::kOk) {
    throw std::runtime_error("route compaction enqueue failed");
  }
  if (events) check(cudaEventRecord(events->after_compaction, stream),
                    "record compaction event");
  if (consumer.enqueue({
          shape, capacity, buffers.routes,
          {buffers.d_gate_ptrs, buffers.d_gate_scale_ptrs, buffers.d_up_ptrs,
           buffers.d_up_scale_ptrs, buffers.d_down_ptrs,
           buffers.d_down_scale_ptrs},
          {buffers.hidden, buffers.quantized_hidden, buffers.hidden_scale,
           buffers.gate_up_scratch, buffers.activated, buffers.activated_scale,
           buffers.output, buffers.expert_summary},
          stream, events ? &stage_events : nullptr}) !=
      moe::RoutedExpertOutcome::kOk) {
    throw std::runtime_error("routed expert enqueue failed");
  }
  cudaGraph_t graph = nullptr;
  check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
  check(cudaGraphGetNodes(graph, nullptr, node_count), "cudaGraphGetNodes");
  cudaGraphExec_t executable = nullptr;
  check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
        "cudaGraphInstantiate");
  check(cudaGraphDestroy(graph), "cudaGraphDestroy");
  return executable;
}

float elapsed(cudaEvent_t start, cudaEvent_t stop) {
  float milliseconds = 0.0F;
  check(cudaEventElapsedTime(&milliseconds, start, stop),
        "cudaEventElapsedTime");
  return milliseconds;
}

std::vector<float> run_total(cudaGraphExec_t graph, cudaStream_t stream,
                             Events& events, int warmup, int samples) {
  for (int i = 0; i < warmup; ++i)
    check(cudaGraphLaunch(graph, stream), "warmup graph launch");
  check(cudaStreamSynchronize(stream), "warmup synchronize");
  std::vector<float> result;
  result.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    check(cudaEventRecord(events.start, stream), "record start");
    check(cudaGraphLaunch(graph, stream), "sample graph launch");
    check(cudaEventRecord(events.stop, stream), "record stop");
    check(cudaEventSynchronize(events.stop), "sample synchronize");
    result.push_back(elapsed(events.start, events.stop));
  }
  return result;
}

void print_stats(const Workload& workload, const char* stage,
                 const Stats& stats, int warmup, int samples,
                 std::size_t graph_nodes) {
  const auto active_bytes = static_cast<std::uint64_t>(workload.active_experts) *
                            moe::kFp8BytesPerExpert;
  const auto expanded_bytes = static_cast<std::uint64_t>(workload.owner_routes) *
                              moe::kFp8BytesPerExpert;
  std::printf(
      "RESULT\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%d\t%d\t%zu\t%llu\t%llu\t%d\n",
      workload.rank, workload.concurrency, workload.draft_depth, workload.rows,
      workload.owner_routes, workload.active_experts, stage, stats.p50,
      stats.p95, stats.mean, stats.stddev,
      stats.mean == 0.0 ? 0.0 : stats.stddev / stats.mean,
      stats.minimum, warmup, samples, graph_nodes,
      static_cast<unsigned long long>(active_bytes),
      static_cast<unsigned long long>(expanded_bytes),
      workload.capacity_control ? 1 : 0);
}

void run_workload(const Workload& workload, FixedBuffers& buffers,
                  cudaStream_t stream, const moe::Fp8RoutedExperts& consumer,
                  int warmup, int samples) {
  install_workload(workload, buffers);
  Events events;
  std::size_t plain_nodes = 0;
  cudaGraphExec_t plain = capture(stream, workload, buffers, consumer, nullptr,
                                  &plain_nodes);
  const auto total = run_total(plain, stream, events, warmup, samples);

  const moe::RouteCompactionShape shape{workload.rank, workload.concurrency,
                                         workload.rows};
  const moe::RouteCompactionCapacity capacity{};
  const moe::RoutedExpertStageEvents stage_events{
      events.after_prepare, events.after_quantize, events.after_gate_up,
      events.after_silu, events.after_down};
  const auto enqueue_profiled = [&] {
    if (moe::enqueue_route_compaction({
            shape, capacity,
            {buffers.route_ids, buffers.route_weights,
             buffers.source_generation, buffers.requested_generation},
            buffers.routes, stream}) != moe::RouteCompactionOutcome::kOk)
      throw std::runtime_error("profile route compaction enqueue failed");
    check(cudaEventRecord(events.after_compaction, stream),
          "profile record compaction");
    if (consumer.enqueue({
            shape, capacity, buffers.routes,
            {buffers.d_gate_ptrs, buffers.d_gate_scale_ptrs,
             buffers.d_up_ptrs, buffers.d_up_scale_ptrs,
             buffers.d_down_ptrs, buffers.d_down_scale_ptrs},
            {buffers.hidden, buffers.quantized_hidden, buffers.hidden_scale,
             buffers.gate_up_scratch, buffers.activated,
             buffers.activated_scale, buffers.output, buffers.expert_summary},
            stream, &stage_events}) != moe::RoutedExpertOutcome::kOk)
      throw std::runtime_error("profile routed expert enqueue failed");
  };
  for (int i = 0; i < warmup; ++i) enqueue_profiled();
  check(cudaStreamSynchronize(stream), "profile warmup synchronize");
  std::array<std::vector<float>, 6> stages;
  for (auto& stage : stages) stage.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    check(cudaEventRecord(events.start, stream), "profile record start");
    enqueue_profiled();
    check(cudaEventRecord(events.stop, stream), "profile record stop");
    check(cudaEventSynchronize(events.stop), "profile synchronize");
    stages[0].push_back(elapsed(events.start, events.after_compaction));
    stages[1].push_back(elapsed(events.after_compaction, events.after_prepare));
    stages[2].push_back(elapsed(events.after_prepare, events.after_quantize));
    stages[3].push_back(elapsed(events.after_quantize, events.after_gate_up));
    stages[4].push_back(elapsed(events.after_gate_up, events.after_silu));
    stages[5].push_back(elapsed(events.after_silu, events.after_down));
  }

  moe::RouteCompactionDeviceSummary route_summary{};
  moe::RoutedExpertDeviceSummary expert_summary{};
  check(cudaMemcpy(&route_summary, buffers.routes.summary, sizeof(route_summary),
                   cudaMemcpyDeviceToHost), "copy route summary");
  check(cudaMemcpy(&expert_summary, buffers.expert_summary,
                   sizeof(expert_summary), cudaMemcpyDeviceToHost),
        "copy expert summary");
  const std::uint64_t generation =
      static_cast<std::uint64_t>(workload.concurrency * 100 +
                                 workload.draft_depth * 10 + workload.rank + 1);
  if (moe::validate_route_compaction_summary(
          {route_summary, generation, shape}) != moe::RouteCompactionOutcome::kOk ||
      moe::validate_routed_expert_summary({expert_summary, generation, shape}) !=
          moe::RoutedExpertOutcome::kOk ||
      route_summary.active_experts != workload.active_experts ||
      route_summary.active_routes != workload.owner_routes) {
    throw std::runtime_error("post-replay publication validation failed");
  }
  std::vector<float> output(workload.rows * moe::kHidden, 1.0F);
  check(cudaMemcpy(output.data(), buffers.output,
                   output.size() * sizeof(output[0]), cudaMemcpyDeviceToHost),
        "copy output");
  if (std::any_of(output.begin(), output.end(),
                  [](float value) { return value != 0.0F; })) {
    throw std::runtime_error("zero-weight parity failed");
  }

  print_stats(workload, "whole", summarize(total), warmup, samples, plain_nodes);
  constexpr std::array<const char*, 6> names{
      "route_compact", "prepare", "quantize_hidden", "gate_up",
      "silu_quantize", "down_reduce"};
  for (std::size_t i = 0; i < names.size(); ++i)
    print_stats(workload, names[i], summarize(stages[i]), warmup, samples,
                plain_nodes);
  check(cudaGraphExecDestroy(plain), "destroy plain graph");
}

}  // namespace

int main(int argc, char** argv) try {
  const int device = argc > 1 ? std::atoi(argv[1]) : 0;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 20;
  const int samples = argc > 3 ? std::atoi(argv[3]) : 100;
  if (device < 0 || warmup < 1 || samples < 2)
    throw std::invalid_argument("usage: profile [device] [warmup>=1] [samples>=2]");
  check(cudaSetDevice(device), "cudaSetDevice");
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  std::printf("META\tdevice=%d\tname=%s\tcc=%d.%d\twarmup=%d\tsamples=%d\tplain_kernel_launches=6\tcubin_launches=4\n",
              device, properties.name, properties.major, properties.minor,
              warmup, samples);
  std::printf("FIELDS\trank\tconcurrency\tdraft_depth\trows\towner_routes\tactive_experts\tstage\tp50_ms\tp95_ms\tmean_ms\tstddev_ms\tcv\tmin_ms\twarmup\tsamples\tgraph_nodes\tactive_weight_bytes\troute_expanded_weight_bytes\tcapacity_control\n");

  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream), "cudaStreamCreate");
  DeviceArena arena;
  FixedBuffers buffers(arena);
  moe::Fp8RoutedExperts rank0(device, 0);
  moe::Fp8RoutedExperts rank1(device, 1);
  for (const auto& workload : workloads()) {
    run_workload(workload, buffers, stream,
                 workload.rank == 0 ? rank0 : rank1, warmup, samples);
  }
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "ERROR\t%s\n", error.what());
  return 1;
}

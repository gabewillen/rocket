// SPDX-License-Identifier: Apache-2.0
#include "model/target_slab_owner.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <unistd.h>

namespace model = rocket::qwen38::model;
namespace {

constexpr std::size_t kSampleBytes = 64;
constexpr std::uint64_t kColdLoadRegressionGuardNs = 17'862'785'416ULL;

struct Sink final : model::TargetSlabTelemetrySink {
  void emit(const model::TargetSlabTelemetryRecord& record) noexcept override {
    if (count < records.size()) records[count] = record;
    else overflow = true;
    ++count;
  }
  std::array<model::TargetSlabTelemetryRecord, 4> records{};
  std::size_t count = 0;
  bool overflow = false;
};

void cuda_require(cudaError_t status, const char* message) {
  if (status != cudaSuccess) throw std::runtime_error(message);
}

bool sample_matches(const std::filesystem::path& payload,
                    const std::uint8_t* device, std::uint64_t offset) {
  const int fd = open(payload.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) throw std::runtime_error("sample source open failed");
  std::array<std::uint8_t, kSampleBytes> source{};
  const ssize_t count = pread(fd, source.data(), source.size(), offset);
  close(fd);
  if (count != static_cast<ssize_t>(source.size()))
    throw std::runtime_error("sample source read failed");
  std::array<std::uint8_t, kSampleBytes> observed{};
  cuda_require(cudaMemcpy(observed.data(), device + offset, observed.size(),
                          cudaMemcpyDeviceToHost),
               "sample device read failed");
  return observed == source;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  Sink telemetry;
  try {
    const std::string_view rank_text(argv[1]);
    const std::string_view device_text(argv[2]);
    if ((rank_text != "0" && rank_text != "1") || device_text != "0")
      throw std::invalid_argument("physical rank/device arguments changed");
    const int rank = rank_text == "0" ? 0 : 1;
    const int device = 0;
    const std::filesystem::path artifact(argv[3]);
    const auto metadata = model::authenticate_target_slab_metadata(artifact, rank);
    cuda_require(cudaSetDevice(device), "physical device selection failed");
    std::size_t free_before = 0, total = 0;
    cuda_require(cudaMemGetInfo(&free_before, &total),
                 "pre-load CUDA memory query failed");

    auto owner = model::TargetSlabDeviceOwner::load(
        device, rank, artifact, telemetry);
    const auto& view = owner->publication();
    std::size_t free_published = 0;
    cuda_require(cudaMemGetInfo(&free_published, &total),
                 "published CUDA memory query failed");
    if (!view.device_base || !view.ready_event ||
        cudaEventQuery(view.ready_event) != cudaSuccess ||
        view.bytes != model::kTargetSlabBytes ||
        view.device != device || view.rank != rank ||
        view.artifact_key != model::kTargetSlabArtifactKey ||
        view.slab_key != metadata.slab_key ||
        view.manifest_sha256 != model::kTargetSlabManifestSha256 ||
        view.layout_sha256 != metadata.layout_sha256 ||
        view.open_to_publish_ns == 0 ||
        view.chunks_authenticated != model::kTargetSlabChunks ||
        view.peak_host_pinned_bytes != model::kTargetSlabPeakPinnedBytes ||
        telemetry.overflow ||
        telemetry.count != 1 || !telemetry.records[0].success ||
        telemetry.records[0].phase != model::TargetSlabLoadPhase::kPublish)
      throw std::logic_error("target slab publication identity changed");

    const std::array<std::uint64_t, 3> offsets{
        0, model::kTargetSlabBytes / 2,
        model::kTargetSlabBytes - kSampleBytes};
    std::array<bool, 3> samples{};
    for (std::size_t index = 0; index < offsets.size(); ++index)
      samples[index] = sample_matches(metadata.payload, view.device_base,
                                      offsets[index]);
    if (!samples[0] || !samples[1] || !samples[2])
      throw std::logic_error("target slab device sample changed");

    const std::uint64_t elapsed_ns = view.open_to_publish_ns;
    if (elapsed_ns > kColdLoadRegressionGuardNs)
      throw std::logic_error("target slab cold-load regression guard exceeded");
    const std::size_t allocation_delta = free_before > free_published
                                             ? free_before - free_published : 0;
    owner.reset();
    std::size_t free_clean = 0;
    cuda_require(cudaMemGetInfo(&free_clean, &total),
                 "post-cleanup CUDA memory query failed");
    const std::size_t cleanup_delta = free_before > free_clean
                                          ? free_before - free_clean : 0;
    if (cleanup_delta != 0)
      throw std::logic_error("target slab CUDA allocation was not released");
    if (telemetry.overflow || telemetry.count != 1 ||
        !telemetry.records[0].success)
      throw std::logic_error("target slab cleanup telemetry changed");

    std::cout << "{\"schema\":\"rocket.qwen38.target-slab-load.v1\""
              << ",\"valid\":true,\"complete\":true"
              << ",\"rank\":" << rank << ",\"device\":0"
              << ",\"artifact_key\":\"" << model::kTargetSlabArtifactKey << "\""
              << ",\"slab_key\":\"" << metadata.slab_key << "\""
              << ",\"manifest_sha256\":\""
              << model::kTargetSlabManifestSha256 << "\""
              << ",\"layout_sha256\":\"" << metadata.layout_sha256 << "\""
              << ",\"bytes\":" << model::kTargetSlabBytes
              << ",\"chunks\":" << model::kTargetSlabChunks
              << ",\"chunks_authenticated\":" << model::kTargetSlabChunks
              << ",\"ring_depth\":" << model::kTargetSlabRingDepth
              << ",\"peak_host_pinned_bytes\":"
              << model::kTargetSlabPeakPinnedBytes
              << ",\"gpu_allocation_delta_bytes\":" << allocation_delta
              << ",\"gpu_cleanup_delta_bytes\":" << cleanup_delta
              << ",\"open_to_publish_ns\":" << elapsed_ns
              << ",\"cold_load_regression_guard_ns\":"
              << kColdLoadRegressionGuardNs
              << ",\"cold_load_regression_guard_passed\":true"
              << ",\"publication_fence_completed\":true"
              << ",\"bytes_per_second\":"
              << (static_cast<long double>(model::kTargetSlabBytes) * 1.0e9L /
                  static_cast<long double>(elapsed_ns))
              << ",\"sample_bytes\":" << kSampleBytes
              << ",\"sample_offsets\":[" << offsets[0] << ',' << offsets[1]
              << ',' << offsets[2] << "]"
              << ",\"samples_match\":[true,true,true]"
              << ",\"telemetry_records\":" << telemetry.count
              << ",\"telemetry_overflow\":false,\"cleanup\":true}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "{\"schema\":\"rocket.qwen38.target-slab-load.v1\""
              << ",\"valid\":false,\"complete\":false"
              << ",\"failure_class\":\"contract_or_runtime\""
              << ",\"telemetry_records\":" << telemetry.count
              << ",\"telemetry_overflow\":"
              << (telemetry.overflow ? "true" : "false")
              << ",\"reason\":\"" << error.what() << "\"}\n";
    return 1;
  }
}

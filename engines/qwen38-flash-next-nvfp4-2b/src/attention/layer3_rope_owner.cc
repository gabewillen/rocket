// SPDX-License-Identifier: Apache-2.0
#include "attention/layer3_rope_owner.h"

#include <cuda_runtime_api.h>

#include <array>
#include <bit>
#include <stdexcept>
#include <string>

namespace rocket::qwen38::attention {
namespace {

// Generated from pinned vLLM 8e685d198
// model_executor/layers/rotary_embedding/{base.py,mrope.py} and checked against
// models/qwen3_8_flash_next/nvidia/qsa.py: float32
// arange/pow/einsum/cos/sin followed by BF16 cast. The source checkpoint is
// fc694b54 and config SHA256 is declared above.
constexpr std::array<std::uint16_t,
                     kLayer3RopeRows * kLayer3RopeColumns> kHostBits = {{
#include "attention/layer3_rope_bits.inc"
}};
constexpr std::array<std::uint16_t, kLayer3RopeColumns> kPosition35Bits = {{
#include "attention/layer3_rope_position35_bits.inc"
}};
constexpr auto kC1HostBits = [] {
  std::array<std::uint16_t, kLayer3RopeC1Rows * kLayer3RopeColumns> result{};
  for (std::size_t index = 0; index < kHostBits.size(); ++index)
    result[index] = kHostBits[index];
  for (std::size_t index = 0; index < kPosition35Bits.size(); ++index)
    result[kHostBits.size() + index] = kPosition35Bits[index];
  return result;
}();
static_assert(sizeof(__nv_bfloat16) == sizeof(std::uint16_t));

void require_identity(const Layer3RopeIdentity& identity) {
  if (identity.checkpoint_revision != kLayer3RopeCheckpointRevision ||
      identity.config_sha256 != kLayer3RopeConfigSha256 ||
      identity.vllm_revision != kLayer3RopeVllmRevision ||
      (identity.rank != 0 && identity.rank != 1) ||
      identity.layer < 0 || identity.layer >= 48 || identity.layer % 4 != 3 ||
      identity.first_position != 0 ||
      identity.rows != kLayer3RopeRows ||
      identity.rotary_dim != kLayer3RopeColumns ||
      std::bit_cast<std::uint32_t>(identity.rope_theta) !=
          std::bit_cast<std::uint32_t>(kLayer3RopeTheta) ||
      identity.uses_mrope)
    throw std::invalid_argument(
        "layer3_rope valid=0 complete=0 phase=identity reason=drift");
}

void require_c1_identity(const Layer3RopeIdentity& identity) {
  if (identity.checkpoint_revision != kLayer3RopeCheckpointRevision ||
      identity.config_sha256 != kLayer3RopeConfigSha256 ||
      identity.vllm_revision != kLayer3RopeVllmRevision ||
      (identity.rank != 0 && identity.rank != 1) ||
      identity.layer < 0 || identity.layer >= 48 || identity.layer % 4 != 3 ||
      identity.first_position != 0 || identity.rows != kLayer3RopeC1Rows ||
      identity.rotary_dim != kLayer3RopeColumns ||
      std::bit_cast<std::uint32_t>(identity.rope_theta) !=
          std::bit_cast<std::uint32_t>(kLayer3RopeTheta) ||
      identity.uses_mrope)
    throw std::invalid_argument(
        "layer3_rope_c1 valid=0 complete=0 phase=identity reason=drift");
}

void check(cudaError_t status, const char* phase) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(
        "layer3_rope valid=0 complete=0 phase=") + phase +
        " reason=" + cudaGetErrorString(status));
}

}  // namespace

Layer3RopeIdentity layer3_rope_identity(int rank) {
  return target_qsa_rope_identity(rank, kLayer3RopeLayer);
}

Layer3RopeIdentity target_qsa_rope_identity(int rank, int layer) {
  if (rank != 0 && rank != 1)
    throw std::invalid_argument(
        "layer3_rope valid=0 complete=0 phase=identity reason=rank");
  if (layer < 0 || layer >= 48 || layer % 4 != 3)
    throw std::invalid_argument(
        "layer3_rope valid=0 complete=0 phase=identity reason=layer");
  return {kLayer3RopeCheckpointRevision, kLayer3RopeConfigSha256,
          kLayer3RopeVllmRevision, rank, layer, 0,
          kLayer3RopeRows, kLayer3RopeColumns, kLayer3RopeTheta, false};
}

Layer3RopeIdentity target_qsa_c1_rope_identity(int rank, int layer) {
  auto identity = target_qsa_rope_identity(rank, layer);
  identity.rows = kLayer3RopeC1Rows;
  return identity;
}

std::span<const std::uint16_t> layer3_rope_host_bits() noexcept {
  return kHostBits;
}

std::span<const std::uint16_t> layer3_rope_c1_host_bits() noexcept {
  return kC1HostBits;
}

Layer3RopeDeviceOwner::Layer3RopeDeviceOwner(
    int device, Layer3RopeIdentity identity)
    : device_(device), identity_(identity) {
  require_identity(identity);
  if (device < 0)
    throw std::invalid_argument(
        "layer3_rope valid=0 complete=0 phase=identity reason=device");
  check(cudaSetDevice(device), "device");
  try {
    check(cudaStreamCreateWithFlags(&initialization_stream_,
                                    cudaStreamNonBlocking),
          "stream_create");
    check(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming),
          "event_create");
    check(cudaMalloc(reinterpret_cast<void**>(&cos_sin_),
                     kHostBits.size() * sizeof(kHostBits[0])),
          "allocate");
    check(cudaMemcpyAsync(cos_sin_, kHostBits.data(),
                          kHostBits.size() * sizeof(kHostBits[0]),
                          cudaMemcpyHostToDevice, initialization_stream_),
          "fill");
    check(cudaEventRecord(ready_, initialization_stream_), "publish");
  } catch (...) {
    if (cos_sin_) cudaFree(cos_sin_);
    if (ready_) cudaEventDestroy(ready_);
    if (initialization_stream_) cudaStreamDestroy(initialization_stream_);
    cos_sin_ = nullptr;
    ready_ = nullptr;
    initialization_stream_ = nullptr;
    throw;
  }
}

Layer3RopeDeviceOwner::~Layer3RopeDeviceOwner() {
  if (device_ >= 0) cudaSetDevice(device_);
  if (cos_sin_) cudaFree(cos_sin_);
  if (ready_) cudaEventDestroy(ready_);
  if (initialization_stream_) cudaStreamDestroy(initialization_stream_);
}

Layer3RopeView Layer3RopeDeviceOwner::view() const noexcept {
  return {cos_sin_, ready_, kLayer3RopePayloadSha256, kLayer3RopeRows,
          kLayer3RopeColumns,
          kLayer3RopeColumns};
}

void Layer3RopeDeviceOwner::wait(cudaStream_t consumer_stream) const {
  if (!consumer_stream || !cos_sin_ || !ready_)
    throw std::invalid_argument(
        "layer3_rope valid=0 complete=0 phase=wait reason=binding");
  check(cudaStreamWaitEvent(consumer_stream, ready_, 0), "wait");
}

Layer3RopeC1DeviceOwner::Layer3RopeC1DeviceOwner(
    int device, Layer3RopeIdentity identity)
    : device_(device), identity_(identity) {
  require_c1_identity(identity);
  if (device < 0)
    throw std::invalid_argument(
        "layer3_rope_c1 valid=0 complete=0 phase=identity reason=device");
  check(cudaSetDevice(device), "device");
  try {
    check(cudaStreamCreateWithFlags(&initialization_stream_,
                                    cudaStreamNonBlocking),
          "stream_create");
    check(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming),
          "event_create");
    check(cudaMalloc(reinterpret_cast<void**>(&cos_sin_),
                     kC1HostBits.size() * sizeof(kC1HostBits[0])),
          "allocate");
    check(cudaMemcpyAsync(cos_sin_, kC1HostBits.data(),
                          kC1HostBits.size() * sizeof(kC1HostBits[0]),
                          cudaMemcpyHostToDevice, initialization_stream_),
          "fill");
    check(cudaEventRecord(ready_, initialization_stream_), "publish");
  } catch (...) {
    if (cos_sin_) cudaFree(cos_sin_);
    if (ready_) cudaEventDestroy(ready_);
    if (initialization_stream_) cudaStreamDestroy(initialization_stream_);
    cos_sin_ = nullptr;
    ready_ = nullptr;
    initialization_stream_ = nullptr;
    throw;
  }
}

Layer3RopeC1DeviceOwner::~Layer3RopeC1DeviceOwner() {
  if (device_ >= 0) cudaSetDevice(device_);
  if (cos_sin_) cudaFree(cos_sin_);
  if (ready_) cudaEventDestroy(ready_);
  if (initialization_stream_) cudaStreamDestroy(initialization_stream_);
}

Layer3RopeView Layer3RopeC1DeviceOwner::view() const noexcept {
  return {cos_sin_, ready_, kLayer3RopeC1PayloadSha256, kLayer3RopeC1Rows,
          kLayer3RopeColumns, kLayer3RopeColumns};
}

void Layer3RopeC1DeviceOwner::wait(cudaStream_t consumer_stream) const {
  if (!consumer_stream || !cos_sin_ || !ready_)
    throw std::invalid_argument(
        "layer3_rope_c1 valid=0 complete=0 phase=wait reason=binding");
  check(cudaStreamWaitEvent(consumer_stream, ready_, 0), "wait");
}

}  // namespace rocket::qwen38::attention

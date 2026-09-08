// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_physical_startup.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <utility>

namespace rocket::qwen38::decode {
namespace {

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) throw std::runtime_error(operation);
}

}  // namespace

std::unique_ptr<TargetK0PhysicalStartupOwner>
TargetK0PhysicalStartupOwner::create(
    TargetK0PhysicalStartupConfig config,
    std::unique_ptr<mtp::WinnerExchangePort> winner_exchange,
    std::shared_ptr<pair_reduce::OtelStageSink> lifecycle_telemetry,
    std::shared_ptr<moe::TargetFullMoeOtelSink> moe_telemetry,
    std::shared_ptr<moe::TargetMoeStageOtelSink> stage_telemetry,
    std::shared_ptr<attention::TargetK0OracleQsaStateOtelSink>
        state_telemetry) {
  validate_target_k0_physical_startup_config(config);
  if (!winner_exchange || !lifecycle_telemetry || !moe_telemetry ||
      !stage_telemetry || !state_telemetry)
    throw std::invalid_argument("K0 physical startup dependencies incomplete");

  auto result = std::unique_ptr<TargetK0PhysicalStartupOwner>(
      new TargetK0PhysicalStartupOwner);
  result->device_ = config.device;
  result->rank_ = config.rank;
  result->lifecycle_telemetry_ = std::move(lifecycle_telemetry);
  result->moe_telemetry_ = std::move(moe_telemetry);
  result->stage_telemetry_ = std::move(stage_telemetry);
  result->state_telemetry_ = std::move(state_telemetry);
  result->winner_exchange_ = std::move(winner_exchange);

  check(cudaSetDevice(result->device_), "select K0 startup device");
  check(cudaStreamCreateWithFlags(&result->stream_, cudaStreamNonBlocking),
        "create K0 startup stream");
  check(cudaMalloc(&result->hidden_a_,
                   kTargetK0HyperHidden * sizeof(__nv_bfloat16)),
        "allocate K0 hidden A");
  check(cudaMalloc(&result->hidden_b_,
                   kTargetK0HyperHidden * sizeof(__nv_bfloat16)),
        "allocate K0 hidden B");

  auto reductions = std::make_unique<TargetK0PairReduceOwner>(
      config.layer_reductions, *result->lifecycle_telemetry_);
  result->embedding_transport_ =
      std::make_unique<pair_reduce::RdmaTransport>(config.embedding_reduction);
  auto roots = output::authenticate_token_io_artifact_roots(
      config.token_io_roots.tokenizer,
      config.token_io_roots.oracle_capture);
  auto comparator = std::make_unique<NativeTargetK0OracleComparator>(
      config.rank, roots.oracle_capture, *result->lifecycle_telemetry_);
  result->prompt_tokens_.reserve(static_cast<std::size_t>(comparator->rows()));
  for (int row = 0; row < comparator->rows(); ++row)
    result->prompt_tokens_.push_back(comparator->expected_input_token(row));
  if (result->prompt_tokens_.size() != 35)
    throw std::invalid_argument("K0 physical startup is oracle35 only");

  auto token_io = output::NativeTokenIoOwner::create(
      config.device, config.rank, config.accepted_loader_lease_handle,
      *result->embedding_transport_, *result->winner_exchange_,
      *result->lifecycle_telemetry_, roots);
  auto plans = std::make_unique<TargetK0NativePlanInventory>(
      TargetK0NativePlanInventory::load(config.descriptor_directory));
  auto layers = TargetK0PhysicalLayerOwners::create(
      config.device, config.rank, config.accepted_loader_lease_handle,
      std::move(plans), config.qsa_sidecar_payload, reductions->schedule(),
      result->lifecycle_telemetry_, result->moe_telemetry_,
      result->stage_telemetry_, result->state_telemetry_);
  result->startup_ = TargetK0StartupOwner::create(
      config.rank, std::move(reductions), std::move(comparator),
      std::move(token_io), std::move(layers), *result->lifecycle_telemetry_,
      roots, {result->hidden_a_, result->hidden_b_}, result->stream_);
  result->authenticated_ = true;
  return result;
}

TargetK0PhysicalStartupOwner::~TargetK0PhysicalStartupOwner() {
  if (device_ >= 0) cudaSetDevice(device_);
  if (stream_) cudaStreamSynchronize(stream_);
  startup_.reset();
  winner_exchange_.reset();
  embedding_transport_.reset();
  cudaFree(hidden_b_);
  cudaFree(hidden_a_);
  if (stream_) cudaStreamDestroy(stream_);
}

TargetK0GeneratedToken TargetK0PhysicalStartupOwner::execute_oracle35(
    std::uint64_t first_generation, std::string_view trace_id,
    std::string_view request_id) {
  if (!authenticated_ || !startup_ || prompt_tokens_.size() != 35)
    throw std::logic_error("K0 physical startup was not published");
  return startup_->execute_prefill(first_generation, prompt_tokens_, trace_id,
                                   request_id);
}

}  // namespace rocket::qwen38::decode

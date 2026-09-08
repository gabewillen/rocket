// SPDX-License-Identifier: Apache-2.0
#include "decode/target_k0_physical_c_api.h"

#include "decode/target_k0_bounded_telemetry.h"
#include "decode/target_k0_physical_startup.h"
#include "mtp/nccl_winner_exchange.h"
#include "model/target_slab_owner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace {
namespace decode = rocket::qwen38::decode;
namespace mtp = rocket::qwen38::mtp;
namespace pair_reduce = rocket::qwen38::pair_reduce;
namespace model = rocket::qwen38::model;

std::array<std::atomic<void*>, 2> runtime_leases{};

std::array<std::uint8_t, 32> bytes(const std::uint8_t* source) {
  if (!source) throw std::invalid_argument("K0 session material absent");
  std::array<std::uint8_t, 32> result{};
  std::copy_n(source, result.size(), result.begin());
  return result;
}

void publish(const decode::TargetK0BoundedTelemetrySnapshot& source,
             Qwen38TargetK0Oracle35Result& target) noexcept {
  std::copy(source.lifecycle_outcomes.begin(), source.lifecycle_outcomes.end(),
            target.lifecycle_outcomes);
  std::copy(source.moe_components.begin(), source.moe_components.end(),
            target.moe_components);
  std::copy(source.stage_counters.begin(), source.stage_counters.end(),
            target.stage_counters);
  std::copy(source.state_outcomes.begin(), source.state_outcomes.end(),
            target.state_outcomes);
  std::copy(source.nccl_stages.begin(), source.nccl_stages.end(),
            target.nccl_stages);
  std::copy(source.nccl_outcomes.begin(), source.nccl_outcomes.end(),
            target.nccl_outcomes);
  target.duration_samples = source.duration_samples;
  target.total_bytes = source.total_bytes;
}

int status_for(decode::TargetK0PhysicalStartupStage stage) noexcept {
  using Stage = decode::TargetK0PhysicalStartupStage;
  switch (stage) {
    case Stage::kValidation:
      return QWEN38_TARGET_K0_VALIDATION;
    case Stage::kLayerPairReduceBootstrap:
      return QWEN38_TARGET_K0_LAYER_PAIR_REDUCE_BOOTSTRAP;
    case Stage::kEmbeddingPairReduceBootstrap:
      return QWEN38_TARGET_K0_EMBEDDING_PAIR_REDUCE_BOOTSTRAP;
    case Stage::kTokenIoConstruction:
      return QWEN38_TARGET_K0_TOKEN_IO_CONSTRUCTION;
    case Stage::kPhysicalLayerConstruction:
      return QWEN38_TARGET_K0_PHYSICAL_LAYER_CONSTRUCTION;
    case Stage::kComparatorStartupConstruction:
      return QWEN38_TARGET_K0_COMPARATOR_STARTUP_CONSTRUCTION;
  }
  return QWEN38_TARGET_K0_UNKNOWN;
}

}  // namespace

extern "C" int qwen38_target_k0_retain_accepted_loader(
    std::uintptr_t device_base, std::uintptr_t ready_event,
    std::size_t byte_count, int device, int rank, const char* slab_key,
    const char* layout_sha256, const char* receipt_sha256,
    std::uint64_t open_to_publish_ns, std::size_t chunks_authenticated,
    std::size_t peak_host_pinned_bytes, std::uint64_t receipt_started_ns,
    std::uint64_t receipt_completed_ns,
    const model::TargetSlabChunkReceipt* chunk_receipts,
    std::size_t chunk_receipt_count, void** lease_handle) noexcept {
  if (!lease_handle || (rank != 0 && rank != 1)) return 1;
  void* retained = nullptr;
  const int status = model::qwen38_target_slab_retain_accepted_loader(
      device_base, ready_event, byte_count, device, rank, slab_key,
      layout_sha256, receipt_sha256, open_to_publish_ns, chunks_authenticated,
      peak_host_pinned_bytes, receipt_started_ns, receipt_completed_ns,
      chunk_receipts, chunk_receipt_count, &retained);
  if (status != 0 || !retained) return status == 0 ? 1 : status;
  auto& slot = runtime_leases[static_cast<std::size_t>(rank)];
  void* expected = nullptr;
  if (!slot.compare_exchange_strong(expected, retained) &&
      expected != retained)
    return 2;
  *lease_handle = retained;
  return 0;
}

extern "C" int qwen38_target_k0_oracle35_run(
    int device, int rank, void* accepted_loader_lease_handle,
    const char* descriptor_directory, const char* qsa_sidecar_payload,
    const char* tokenizer_root, const char* oracle_capture,
    const char* bootstrap_host, int layer_port, int embedding_port,
    int nccl_port, std::uint32_t timeout_ms,
    const std::uint8_t layer_session_sha256[32],
    const std::uint8_t embedding_session_sha256[32],
    const std::uint8_t nccl_session_sha256[32],
    const std::uint8_t nccl_authentication_key[32],
    Qwen38TargetK0Oracle35Result* result) noexcept {
  if (!result) return QWEN38_TARGET_K0_VALIDATION;
  std::memset(result, 0, sizeof(*result));
  result->token = -1;
  result->physical_layer_index = -1;
  std::shared_ptr<decode::TargetK0BoundedTelemetry> telemetry;
  int failure_status = QWEN38_TARGET_K0_VALIDATION;
  decode::TargetK0PhysicalStartupStage startup_stage =
      decode::TargetK0PhysicalStartupStage::kValidation;
  decode::TargetK0PhysicalLayerConstructionProgress layer_progress;
  const auto resolved_status = [&]() noexcept {
    return failure_status < 0 ? status_for(startup_stage) : failure_status;
  };
  const auto publish_state = [&]() noexcept {
    result->physical_layer_substage =
        static_cast<std::int32_t>(layer_progress.stage);
    result->physical_layer_index = layer_progress.layer;
    result->gdn_owner_substage =
        static_cast<std::int32_t>(layer_progress.gdn_stage);
    result->moe_aot_cuda_failure =
        static_cast<std::int32_t>(layer_progress.moe_aot_cuda_failure);
    if (telemetry) publish(telemetry->snapshot(), *result);
  };
  try {
    telemetry = std::make_shared<decode::TargetK0BoundedTelemetry>();
    if (!accepted_loader_lease_handle || !descriptor_directory ||
        !qsa_sidecar_payload || !tokenizer_root || !oracle_capture ||
        !bootstrap_host || layer_port == embedding_port ||
        layer_port == nccl_port || embedding_port == nccl_port)
      throw std::invalid_argument("K0 C ABI dependencies incomplete");
    {
      if ((rank != 0 && rank != 1) ||
          runtime_leases[static_cast<std::size_t>(rank)].load() !=
              accepted_loader_lease_handle)
        throw std::invalid_argument("K0 runtime slab capability origin changed");
    }
    const auto layer_session = bytes(layer_session_sha256);
    const auto embedding_session = bytes(embedding_session_sha256);
    const auto nccl_session = bytes(nccl_session_sha256);
    if (layer_session == embedding_session || layer_session == nccl_session ||
        embedding_session == nccl_session)
      throw std::invalid_argument("K0 bootstrap sessions are not distinct");

    decode::TargetK0PhysicalStartupConfig config;
    config.device = device;
    config.rank = rank;
    config.accepted_loader_lease_handle = accepted_loader_lease_handle;
    config.descriptor_directory = descriptor_directory;
    config.qsa_sidecar_payload = qsa_sidecar_payload;
    config.token_io_roots = {tokenizer_root, oracle_capture, {}, {}};
    config.layer_reductions = {rank, 1 - rank, bootstrap_host, layer_port,
                               timeout_ms, layer_session};
    config.embedding_reduction.rank = rank;
    config.embedding_reduction.bootstrap_host = bootstrap_host;
    config.embedding_reduction.bootstrap_port = embedding_port;
    config.embedding_reduction.operation_timeout_ms = timeout_ms;
    config.embedding_reduction.session_sha256 = embedding_session;

    mtp::NcclCommunicatorConfig nccl;
    nccl.rank = rank;
    nccl.peer_rank = 1 - rank;
    nccl.device = device;
    nccl.bootstrap_host = bootstrap_host;
    nccl.bootstrap_port = nccl_port;
    nccl.pair_reduce_bootstrap_port = layer_port;
    nccl.timeout_ms = timeout_ms;
    nccl.session_sha256 = nccl_session;
    nccl.authentication_key = bytes(nccl_authentication_key);
    failure_status = QWEN38_TARGET_K0_NCCL_BOOTSTRAP;
    auto winner = mtp::make_nccl_winner_exchange(nccl, *telemetry);
    failure_status = -1;  // Startup owner reports its exact construction stage.
    auto owner = decode::TargetK0PhysicalStartupOwner::create(
        std::move(config), std::move(winner), telemetry, telemetry, telemetry,
        telemetry, &startup_stage, &layer_progress);
    failure_status = QWEN38_TARGET_K0_PROMPT_EXECUTION;
    const auto generated = owner->execute_oracle35(
        1, "k0-oracle35", "oracle-05ea3af");
    result->token = generated.execution.token;
    result->rows = generated.execution.rows;
    result->final_generation = generated.execution.final_generation;
    failure_status = QWEN38_TARGET_K0_CLEANUP;
    owner.reset();
    publish_state();
    return 0;
  } catch (const std::invalid_argument&) {
    publish_state();
    return resolved_status();
  } catch (const mtp::NcclBootstrapContractError&) {
    publish_state();
    return resolved_status();
  } catch (const mtp::NcclBootstrapTransportError&) {
    publish_state();
    return resolved_status();
  } catch (const pair_reduce::PairReduceTransportError&) {
    publish_state();
    return resolved_status();
  } catch (const mtp::NcclBootstrapAuthenticationError&) {
    publish_state();
    return resolved_status();
  } catch (const mtp::NcclBootstrapCudaError&) {
    publish_state();
    return resolved_status();
  } catch (const mtp::NcclBootstrapNcclError&) {
    publish_state();
    return resolved_status();
  } catch (const mtp::NcclBootstrapLibraryError&) {
    publish_state();
    return resolved_status();
  } catch (...) {
    publish_state();
    return resolved_status();
  }
}

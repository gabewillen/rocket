// SPDX-License-Identifier: Apache-2.0
#include "decode/target_qsa_k0_layer_owner.h"

#include "attention/native_qsa_graph.h"
#include "decode/target_full_layer.h"
#include "decode/target_layer3_generation.h"
#include "decode/target_layer_native_bindings.h"
#include "hyperconnection/native_adapter.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <utility>

namespace rocket::qwen38::decode {
namespace {

struct Cursor {
  std::uint8_t* base;
  std::size_t offset = 0;
  template <class T>
  T* take(std::size_t bytes) {
    offset = (offset + 255) & ~std::size_t{255};
    auto* result = reinterpret_cast<T*>(base + offset);
    offset += bytes;
    return result;
  }
};

constexpr std::size_t kStorageBytes = 812'544;

struct DeviceStorage {
  int device;
  void* pointer = nullptr;
  explicit DeviceStorage(int value) : device(value) {
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMalloc(&pointer, kStorageBytes) != cudaSuccess)
      throw std::runtime_error("target QSA K0 storage allocation failed");
  }
  ~DeviceStorage() {
    if (pointer && device >= 0 && cudaSetDevice(device) == cudaSuccess)
      cudaFree(pointer);
  }
};

}  // namespace

struct TargetQsaK0LayerOwner::Bundle {
  std::shared_ptr<const model::TargetSlabLease> slab;
  std::shared_ptr<pair_reduce::OtelStageSink> layer_telemetry;
  std::unique_ptr<DeviceStorage> storage;
  cudaEvent_t rope_ready = nullptr;
  cudaEvent_t state_ready = nullptr;
  std::unique_ptr<moe::TargetLayerMoeDeviceOwner> moe;
  attention::TargetQsaGraphArena arena{};
  TargetLayer3RowBuffers rows{};
  std::unique_ptr<hyperconnection::Plan> hyper_plan;
  std::unique_ptr<hyperconnection::NativeFullAttentionHyperConnection> hyper;
  std::unique_ptr<attention::NativeQsaFullAttentionGraph> qsa;
  std::unique_ptr<NativeTargetQsaGenerationOwner> generation;
  std::unique_ptr<TargetFullLayer> layer;
};

std::unique_ptr<TargetQsaK0LayerOwner> TargetQsaK0LayerOwner::create(
    int device, const TargetLayerNativePlan& plan,
    void* accepted_loader_lease_handle,
    const attention::QsaSidecarPublication& sidecar,
    const attention::Layer3RopeIdentity& rope_identity,
    const attention::Layer3RopeView& rope,
    const attention::TargetQsaStateView& state,
    cudaEvent_t state_ready,
    HiddenPartialReducer& attention_reducer,
    HiddenPartialReducer& moe_reducer,
    std::shared_ptr<pair_reduce::OtelStageSink> layer_telemetry,
    std::shared_ptr<moe::TargetFullMoeOtelSink> moe_telemetry,
    std::shared_ptr<moe::TargetMoeStageOtelSink> stage_telemetry,
    int max_rows) {
  return std::unique_ptr<TargetQsaK0LayerOwner>(new TargetQsaK0LayerOwner(
      device, plan, accepted_loader_lease_handle, sidecar, rope_identity, rope,
      state, state_ready, attention_reducer, moe_reducer,
      std::move(layer_telemetry),
      std::move(moe_telemetry), std::move(stage_telemetry), max_rows));
}

TargetQsaK0LayerOwner::TargetQsaK0LayerOwner(
    int device, const TargetLayerNativePlan& plan,
    void* accepted_loader_lease_handle,
    const attention::QsaSidecarPublication& sidecar,
    const attention::Layer3RopeIdentity& rope_identity,
    const attention::Layer3RopeView& rope,
    const attention::TargetQsaStateView& state,
    cudaEvent_t state_ready,
    HiddenPartialReducer& attention_reducer,
    HiddenPartialReducer& moe_reducer,
    std::shared_ptr<pair_reduce::OtelStageSink> layer_telemetry,
    std::shared_ptr<moe::TargetFullMoeOtelSink> moe_telemetry,
    std::shared_ptr<moe::TargetMoeStageOtelSink> stage_telemetry,
    int max_rows)
    : device_(device), rank_(plan.rank), layer_(plan.layer),
      attention_reducer_(&attention_reducer), moe_reducer_(&moe_reducer),
      bundle_(std::make_unique<Bundle>()) {
  validate_target_layer_native_plan_binding(plan);
  if (plan.attention_kind != TargetK0AttentionKind::kQsa ||
      attention_reducer.rank() != plan.rank ||
      moe_reducer.rank() != plan.rank || !layer_telemetry ||
      !moe_telemetry || !stage_telemetry)
    throw std::invalid_argument("target QSA K0 owner identity changed");
  bundle_->slab = model::TargetSlabStartupFactory::lease_from_handle(
      accepted_loader_lease_handle);
  bundle_->layer_telemetry = std::move(layer_telemetry);
  if (!bundle_->slab || device != bundle_->slab->publication().device)
    throw std::invalid_argument("target QSA K0 slab lease changed");
  const auto expected_sidecar =
      attention::target_qsa_sidecar_identity(plan.rank, plan.layer);
  const auto expected_rope =
      attention::target_qsa_rope_identity(plan.rank, plan.layer);
  if (!sidecar.device_base || sidecar.bytes != attention::kQsaSidecarBytes ||
      sidecar.device != device ||
      sidecar.identity.artifact_key != expected_sidecar.artifact_key ||
      sidecar.identity.payload_sha256 != expected_sidecar.payload_sha256 ||
      sidecar.identity.layer3_sha256 != expected_sidecar.layer3_sha256 ||
      sidecar.identity.rank != plan.rank || sidecar.identity.layer != plan.layer ||
      rope_identity.checkpoint_revision != expected_rope.checkpoint_revision ||
      rope_identity.config_sha256 != expected_rope.config_sha256 ||
      rope_identity.vllm_revision != expected_rope.vllm_revision ||
      rope_identity.rank != plan.rank || rope_identity.layer != plan.layer ||
      !rope.cos_sin || !rope.ready || rope.rows != 35 || rope.columns != 64 ||
      rope.row_stride != 64 || !state_ready ||
      rope.payload_sha256 != attention::kLayer3RopePayloadSha256)
    throw std::invalid_argument("target QSA K0 shared assets changed");
  bundle_->rope_ready = rope.ready;
  bundle_->state_ready = state_ready;
  bundle_->moe = moe::TargetLayerMoeDeviceOwner::create(
      device, plan, accepted_loader_lease_handle, std::move(moe_telemetry),
      std::move(stage_telemetry));
  const auto weights = bind_target_qsa_layer_native_weights(
      plan, bundle_->slab->publication(), sidecar, rope_identity, rope);
  bundle_->storage = std::make_unique<DeviceStorage>(device);
  Cursor c{static_cast<std::uint8_t*>(bundle_->storage->pointer)};
  bundle_->arena = {
      c.take<std::uint8_t>(1'280), c.take<std::uint8_t>(20'480),
      c.take<__nv_bfloat16>(13'312), c.take<__nv_bfloat16>(1'280),
      c.take<__nv_bfloat16>(6'144), c.take<__nv_bfloat16>(6'144),
      c.take<__nv_bfloat16>(1'024), c.take<float>(262'144),
      c.take<std::int32_t>(4), c.take<std::int32_t>(2'048),
      c.take<std::int32_t>(8'204), c.take<float>(393'216),
      c.take<float>(1'536), c.take<__nv_bfloat16>(6'144),
      c.take<__nv_bfloat16>(6'144), c.take<std::uint8_t>(1'536),
      c.take<std::uint8_t>(24'576), c.take<__nv_bfloat16>(5'120)};
  bundle_->rows = {
      c.take<__nv_bfloat16>(5'120), c.take<__nv_bfloat16>(8),
      c.take<float>(10'240), c.take<__nv_bfloat16>(20'480),
      c.take<__nv_bfloat16>(5'120), c.take<__nv_bfloat16>(8),
      c.take<float>(10'240)};
  if (c.offset > kStorageBytes)
    throw std::logic_error("target QSA K0 storage layout changed");
  bundle_->hyper_plan = std::make_unique<hyperconnection::Plan>(
      device, weights.attention_hyperconnection, weights.mlp_hyperconnection);
  bundle_->hyper =
      std::make_unique<hyperconnection::NativeFullAttentionHyperConnection>(
          *bundle_->hyper_plan);
  bundle_->qsa = std::make_unique<attention::NativeQsaFullAttentionGraph>(
      device, plan.rank, plan.layer, plan.indexer_sidecar_key,
      weights.projection, weights.preprocess, bundle_->arena);
  bundle_->generation = std::make_unique<NativeTargetQsaGenerationOwner>(
      plan.rank, plan.layer, state, bundle_->moe->requested_generation(),
      max_rows);
  bundle_->layer = std::make_unique<TargetFullLayer>(
      *bundle_->qsa, bundle_->moe->graph(), attention_reducer, moe_reducer,
      *bundle_->hyper, *bundle_->layer_telemetry);
  authenticated_ = bundle_->moe->authenticated() &&
                   bundle_->generation->authenticated();
  if (!authenticated_)
    throw std::invalid_argument("target QSA K0 construction was unauthenticated");
}

TargetQsaK0LayerOwner::~TargetQsaK0LayerOwner() {
  if (!bundle_) return;
  bundle_->layer.reset();
  bundle_->generation.reset();
  bundle_->qsa.reset();
  bundle_->hyper.reset();
  bundle_->hyper_plan.reset();
  bundle_->moe.reset();
  bundle_->storage.reset();
}

void TargetQsaK0LayerOwner::wait_source(cudaStream_t stream) {
  if (!authenticated_ || !stream) throw std::logic_error("target QSA K0 source wait changed");
  if (cudaStreamWaitEvent(stream, bundle_->slab->publication().ready_event, 0) !=
      cudaSuccess)
    throw std::runtime_error("target QSA slab wait failed");
  if (cudaStreamWaitEvent(stream, bundle_->rope_ready, 0) != cudaSuccess)
    throw std::runtime_error("target QSA shared RoPE wait failed");
  if (cudaStreamWaitEvent(stream, bundle_->state_ready, 0) != cudaSuccess)
    throw std::runtime_error("target QSA oracle state wait failed");
  bundle_->moe->wait_source(stream);
}

void TargetQsaK0LayerOwner::execute_row(
    std::uint64_t generation, const __nv_bfloat16* replicated_pre_layer,
    __nv_bfloat16* replicated_post_layer, cudaStream_t stream,
    TargetK0ExecutionProgress* progress) {
  target_k0_enter_layer(progress,
                        TargetK0LayerExecutionStage::kStatePreparation);
  const auto& state = bundle_->generation->view(row_, generation);
  bundle_->generation->enqueue_prepare(row_, generation, stream);
  bundle_->layer->execute(
      generation, state, replicated_pre_layer, bundle_->rows.attention_input,
      bundle_->rows.attention_injection, bundle_->rows.reduced_attention,
      bundle_->rows.post_attention_hidden, bundle_->rows.moe_input,
      bundle_->rows.moe_injection, bundle_->rows.reduced_moe,
      replicated_post_layer, "k0-target", "prefill", stream, progress);
  ++row_;
}

}  // namespace rocket::qwen38::decode

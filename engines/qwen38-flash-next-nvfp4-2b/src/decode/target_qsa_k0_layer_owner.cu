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
  std::unique_ptr<attention::QsaSidecarDeviceOwner> sidecar;
  std::unique_ptr<attention::Layer3RopeDeviceOwner> rope;
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
    const std::filesystem::path& sidecar_payload,
    const attention::TargetQsaStateView& state,
    HiddenPartialReducer& attention_reducer,
    HiddenPartialReducer& moe_reducer,
    std::shared_ptr<pair_reduce::OtelStageSink> layer_telemetry,
    std::shared_ptr<moe::TargetFullMoeOtelSink> moe_telemetry,
    std::shared_ptr<moe::TargetMoeStageOtelSink> stage_telemetry,
    int max_rows) {
  return std::unique_ptr<TargetQsaK0LayerOwner>(new TargetQsaK0LayerOwner(
      device, plan, accepted_loader_lease_handle, sidecar_payload, state,
      attention_reducer, moe_reducer, std::move(layer_telemetry),
      std::move(moe_telemetry), std::move(stage_telemetry), max_rows));
}

TargetQsaK0LayerOwner::TargetQsaK0LayerOwner(
    int device, const TargetLayerNativePlan& plan,
    void* accepted_loader_lease_handle,
    const std::filesystem::path& sidecar_payload,
    const attention::TargetQsaStateView& state,
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
  bundle_->sidecar = std::make_unique<attention::QsaSidecarDeviceOwner>(
      device, sidecar_payload,
      attention::target_qsa_sidecar_identity(plan.rank, plan.layer));
  bundle_->rope = std::make_unique<attention::Layer3RopeDeviceOwner>(
      device, attention::target_qsa_rope_identity(plan.rank, plan.layer));
  bundle_->moe = moe::TargetLayerMoeDeviceOwner::create(
      device, plan, accepted_loader_lease_handle, std::move(moe_telemetry),
      std::move(stage_telemetry));
  const auto weights = bind_target_qsa_layer_native_weights(
      plan, bundle_->slab->publication(), bundle_->sidecar->publication(),
      bundle_->rope->identity(), bundle_->rope->view());
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
  bundle_->rope->wait(stream);
  bundle_->moe->wait_source(stream);
}

void TargetQsaK0LayerOwner::execute_row(
    std::uint64_t generation, const __nv_bfloat16* replicated_pre_layer,
    __nv_bfloat16* replicated_post_layer, cudaStream_t stream) {
  const auto& state = bundle_->generation->view(row_, generation);
  bundle_->generation->enqueue_prepare(row_, generation, stream);
  bundle_->layer->execute(
      generation, state, replicated_pre_layer, bundle_->rows.attention_input,
      bundle_->rows.attention_injection, bundle_->rows.reduced_attention,
      bundle_->rows.post_attention_hidden, bundle_->rows.moe_input,
      bundle_->rows.moe_injection, bundle_->rows.reduced_moe,
      replicated_post_layer, "k0-target", "prefill", stream);
  ++row_;
}

}  // namespace rocket::qwen38::decode

// SPDX-License-Identifier: Apache-2.0
#include "moe/target_layer_moe_owner.h"

#include <cuda_runtime_api.h>

#include <array>
#include <stdexcept>
#include <utility>

namespace rocket::qwen38::moe {
namespace {

struct Cursor {
  std::uint8_t* base;
  std::size_t offset = 0;
  template <class T>
  T* take(std::size_t bytes, std::size_t alignment = 256) {
    offset = (offset + alignment - 1) & ~(alignment - 1);
    auto* result = reinterpret_cast<T*>(base + offset);
    offset += bytes;
    return result;
  }
};

bool parse_digest(std::string_view value,
                  std::array<std::uint8_t, 32>* output) noexcept {
  return parse_target_moe_artifact_key(value, output);
}

TargetDenseIdentity dense_identity(
    const decode::TargetLayerNativePlan& plan) {
  TargetDenseIdentity result{{}, {}, plan.rank, plan.layer};
  if (!parse_digest(plan.artifact_key, &result.artifact_sha256) ||
      !parse_digest(plan.layout_sha256, &result.layout_sha256))
    throw std::invalid_argument("target layer MoE digest changed");
  return result;
}

TargetMoeN768StageScratch make_stage_scratch(
    void* storage, TargetMoeN640StageEvidence* host_evidence,
    TargetMoeN640StageEvidence* device_evidence) {
  Cursor c{static_cast<std::uint8_t*>(storage)};
  TargetMoeN768StageScratch s{};
  s.w13_packed = c.take<std::uint8_t>(kTargetMoeStagedW13PackedBytes);
  s.w13_packed_bytes = kTargetMoeStagedW13PackedBytes;
  s.w13_scale = c.take<std::uint8_t>(kTargetMoeStagedW13ScaleBytes);
  s.w13_scale_bytes = kTargetMoeStagedW13ScaleBytes;
  s.down_packed = c.take<std::uint8_t>(kTargetMoeStagedDownPackedBytes);
  s.down_packed_bytes = kTargetMoeStagedDownPackedBytes;
  s.down_scale = c.take<std::uint8_t>(kTargetMoeStagedDownScaleBytes);
  s.down_scale_bytes = kTargetMoeStagedDownScaleBytes;
  s.input_global_scale = c.take<float>(10 * sizeof(float), alignof(float));
  s.folded_w1_alpha = c.take<float>(10 * sizeof(float), alignof(float));
  s.w2_alpha = c.take<float>(10 * sizeof(float), alignof(float));
  s.down_input_scale = c.take<float>(10 * sizeof(float), alignof(float));
  s.source_expert_ids = c.take<std::int32_t>(
      10 * sizeof(std::int32_t), alignof(std::int32_t));
  s.compact_expert_ids = c.take<std::int32_t>(
      10 * sizeof(std::int32_t), alignof(std::int32_t));
  s.compact_routing_weights = c.take<float>(
      10 * sizeof(float), alignof(float));
  s.evidence = device_evidence;
  s.host_evidence = host_evidence;
  s.scalar_capacity = 10;
  s.route_capacity = 10;
  if (c.offset > kTargetLayerMoeStageDeviceBytes ||
      !validate_target_moe_stage_scratch(s))
    throw std::logic_error("target layer MoE stage layout changed");
  return s;
}

TargetFullMoeC1Workspace make_runtime_workspace(
    void* storage, const std::uint64_t* requested,
    const TargetMoeN768StageScratch& stage, __nv_bfloat16** output) {
  Cursor c{static_cast<std::uint8_t*>(storage)};
  TargetFullMoeC1Workspace w{};
  w.router_logits_f32 = c.take<float>(512 * sizeof(float));
  w.global_ids_i32 = c.take<std::int32_t>(10 * sizeof(std::int32_t));
  w.routing_weights_f32 = c.take<float>(10 * sizeof(float));
  w.local_ids_i32 = c.take<std::int32_t>(10 * sizeof(std::int32_t));
  w.local_weights_f32 = c.take<float>(10 * sizeof(float));
  w.source_generation = c.take<std::uint64_t>(sizeof(std::uint64_t));
  w.requested_generation = requested;
  w.route_summary = c.take<TargetMoeC1Summary>(sizeof(TargetMoeC1Summary));
  w.routed.packed_a = c.take<std::uint8_t>(140'800);
  w.routed.packed_a_scale = c.take<std::uint8_t>(225'280);
  w.routed.route_output_scratch_bf16 = c.take<void>(153'600);
  w.routed.barrier_count = c.take<std::int32_t>(4);
  w.routed.barrier_epoch = c.take<std::int32_t>(4);
  w.routed.row_counts = c.take<std::int32_t>(44);
  w.routed.active_expert_count = c.take<std::int32_t>(4);
  w.routed.weight_expert_ids = c.take<std::int32_t>(44);
  w.routed.global_to_local_expert = c.take<std::int32_t>(40);
  w.routed.virtual_route_scratch = c.take<std::int32_t>(112);
  w.routed.token_map = c.take<std::int32_t>(440);
  w.routed.token_weights = c.take<float>(440);
  w.routed_stage = stage;
  w.shared_gate_scratch_f32 = c.take<float>(160 * sizeof(float));
  w.shared_up_scratch_f32 = c.take<float>(160 * sizeof(float));
  w.shared_gate_scalar_f32 = c.take<float>(sizeof(float));
  *output = c.take<__nv_bfloat16>(
      kTargetMoeHidden * sizeof(__nv_bfloat16));
  if (c.offset > kTargetLayerMoeRuntimeBytes)
    throw std::logic_error("target layer MoE runtime layout changed");
  return w;
}

}  // namespace

struct TargetLayerMoeDeviceOwner::Bundle {
  std::shared_ptr<const model::TargetSlabLease> slab_lease;
  std::shared_ptr<TargetFullMoeOtelSink> telemetry;
  std::shared_ptr<TargetMoeStageOtelSink> stage_telemetry;
  std::uint64_t* requested_generation = nullptr;
  void* stage_storage = nullptr;
  void* runtime_storage = nullptr;
  TargetMoeN640StageEvidence* mapped_evidence_host = nullptr;
  TargetMoeN768StageScratch stage_scratch{};
  TargetFullMoeC1Workspace workspace{};
  __nv_bfloat16* rank_local_output = nullptr;
  std::unique_ptr<TargetMoeN640DeviceStage> stage;
  std::unique_ptr<TargetFullMoeC1> participant;
  std::unique_ptr<NativeTargetMoeGraph> graph;
};

bool validate_target_layer_moe_owner_plan(
    const decode::TargetLayerNativePlan& plan,
    const model::TargetSlabPublication& slab) noexcept {
  try {
    const auto weights =
        decode::bind_target_layer_native_moe_weights(plan, slab);
    return authenticate_target_moe_compact_runtime_identity(
               weights.routed_identity) &&
           weights.router.packed_e2m1 && weights.router.cutlass_sfb_e4m3 &&
           weights.router.weight_scale_2 && weights.shared.gate &&
           weights.shared.up && weights.shared.down &&
           weights.shared.shared_gate &&
           weights.routed_source.front().up_packed &&
           weights.routed_source.back().down_alpha;
  } catch (...) {
    return false;
  }
}

bool validate_target_layer_moe_owner_handoff(
    const decode::TargetLayerNativePlan& plan,
    void* accepted_loader_lease_handle) noexcept {
  const auto lease = model::TargetSlabStartupFactory::lease_from_handle(
      accepted_loader_lease_handle);
  return lease &&
         lease->lifetime() ==
             model::TargetSlabLease::Lifetime::kProcessLifetime &&
         validate_target_layer_moe_owner_plan(plan, lease->publication());
}

std::unique_ptr<TargetLayerMoeDeviceOwner>
TargetLayerMoeDeviceOwner::create(
    int device, const decode::TargetLayerNativePlan& plan,
    void* accepted_loader_lease_handle,
    std::shared_ptr<TargetFullMoeOtelSink> telemetry,
    std::shared_ptr<TargetMoeStageOtelSink> stage_telemetry,
    TargetLayerMoeConstructionStage* construction_stage,
    TargetMoeAotConstructionStage* aot_stage) {
  auto lease = model::TargetSlabStartupFactory::lease_from_handle(
      accepted_loader_lease_handle);
  if (!lease)
    throw std::invalid_argument("accepted-loader slab lease handle changed");
  return std::unique_ptr<TargetLayerMoeDeviceOwner>(
      new TargetLayerMoeDeviceOwner(
          device, plan, std::move(lease), std::move(telemetry),
          std::move(stage_telemetry), construction_stage, aot_stage));
}

TargetLayerMoeStorageBinding bind_target_layer_moe_storage(
    void* stage_storage, std::size_t stage_bytes,
    TargetMoeN640StageEvidence* evidence_host,
    TargetMoeN640StageEvidence* evidence_device,
    void* runtime_storage, std::size_t runtime_bytes,
    const std::uint64_t* requested_generation) {
  if (!stage_storage || stage_bytes != kTargetLayerMoeStageDeviceBytes ||
      !evidence_host || !evidence_device || !runtime_storage ||
      runtime_bytes != kTargetLayerMoeRuntimeBytes || !requested_generation ||
      reinterpret_cast<std::uintptr_t>(stage_storage) % 256 != 0 ||
      reinterpret_cast<std::uintptr_t>(runtime_storage) % 256 != 0)
    throw std::invalid_argument("target layer MoE storage extent changed");
  TargetLayerMoeStorageBinding result{};
  result.stage = make_stage_scratch(
      stage_storage, evidence_host, evidence_device);
  result.runtime = make_runtime_workspace(
      runtime_storage, requested_generation, result.stage,
      &result.rank_local_output);
  return result;
}

TargetLayerMoeDeviceOwner::TargetLayerMoeDeviceOwner(
    int device, const decode::TargetLayerNativePlan& plan,
    std::shared_ptr<const model::TargetSlabLease> slab_lease,
    std::shared_ptr<TargetFullMoeOtelSink> telemetry,
    std::shared_ptr<TargetMoeStageOtelSink> stage_telemetry,
    TargetLayerMoeConstructionStage* construction_stage,
    TargetMoeAotConstructionStage* aot_stage)
    : device_(device), rank_(plan.rank), layer_(plan.layer),
      bundle_(std::make_unique<Bundle>()) {
  const auto mark = [construction_stage](TargetLayerMoeConstructionStage stage) {
    if (construction_stage) *construction_stage = stage;
  };
  mark(TargetLayerMoeConstructionStage::kPlanBinder);
  bundle_->slab_lease = std::move(slab_lease);
  bundle_->telemetry = std::move(telemetry);
  bundle_->stage_telemetry = std::move(stage_telemetry);
  if (!bundle_->slab_lease || !bundle_->telemetry ||
      !bundle_->stage_telemetry)
    throw std::invalid_argument("target layer MoE owner lease changed");
  if (bundle_->slab_lease->lifetime() !=
      model::TargetSlabLease::Lifetime::kProcessLifetime)
    throw std::invalid_argument(
        "target layer MoE requires process-lifetime slab lease");
  const auto& slab = bundle_->slab_lease->publication();
  if (device != slab.device ||
      !validate_target_layer_moe_owner_plan(plan, slab))
    throw std::invalid_argument("target layer MoE owner plan changed");
  const auto bindings =
      decode::bind_target_layer_native_moe_weights(plan, slab);
  TargetMoeN640StageEvidence* device_evidence = nullptr;
  const auto release = [&] {
    if (bundle_->mapped_evidence_host)
      cudaFreeHost(bundle_->mapped_evidence_host);
    if (bundle_->runtime_storage) cudaFree(bundle_->runtime_storage);
    if (bundle_->stage_storage) cudaFree(bundle_->stage_storage);
    if (bundle_->requested_generation)
      cudaFree(bundle_->requested_generation);
    bundle_->mapped_evidence_host = nullptr;
    bundle_->runtime_storage = nullptr;
    bundle_->stage_storage = nullptr;
    bundle_->requested_generation = nullptr;
  };
  if (cudaSetDevice(device) != cudaSuccess ||
      cudaMalloc(&bundle_->stage_storage,
                 kTargetLayerMoeStageDeviceBytes) != cudaSuccess ||
      cudaMalloc(&bundle_->runtime_storage,
                 kTargetLayerMoeRuntimeBytes) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&bundle_->requested_generation),
                 sizeof(*bundle_->requested_generation)) != cudaSuccess ||
      cudaMemset(bundle_->requested_generation, 0,
                 sizeof(*bundle_->requested_generation)) != cudaSuccess ||
      cudaHostAlloc(&bundle_->mapped_evidence_host,
                    sizeof(*bundle_->mapped_evidence_host),
                    cudaHostAllocMapped) != cudaSuccess ||
      cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&device_evidence),
          bundle_->mapped_evidence_host,
          0) != cudaSuccess) {
    release();
    throw std::runtime_error("target layer MoE allocation failed");
  }
  *bundle_->mapped_evidence_host =
      {0, -1, TargetMoeOutcome::kContractError};
  try {
    const auto storage = bind_target_layer_moe_storage(
        bundle_->stage_storage, kTargetLayerMoeStageDeviceBytes,
        bundle_->mapped_evidence_host, device_evidence,
        bundle_->runtime_storage, kTargetLayerMoeRuntimeBytes,
        bundle_->requested_generation);
    bundle_->stage_scratch = storage.stage;
    bundle_->workspace = storage.runtime;
    bundle_->rank_local_output = storage.rank_local_output;
    mark(TargetLayerMoeConstructionStage::kStage);
    bundle_->stage = std::make_unique<TargetMoeN640DeviceStage>(
        device, plan.rank, plan.layer, slab.ready_event,
        bindings.routed_source);
    const auto identity = dense_identity(plan);
    mark(TargetLayerMoeConstructionStage::kAot);
    bundle_->participant = std::make_unique<TargetFullMoeC1>(
        device, identity,
        TargetFullMoeC1Weights{bindings.router, bindings.routed_identity,
                               bundle_->stage.get(),
                               bundle_->stage_scratch,
                               bundle_->stage_telemetry.get(),
                               bindings.shared}, aot_stage);
    bundle_->graph = std::make_unique<NativeTargetMoeGraph>(
        *bundle_->participant, bundle_->workspace,
        bundle_->rank_local_output, *bundle_->telemetry);
    authenticated_ = true;
  } catch (...) {
    bundle_->graph.reset();
    bundle_->participant.reset();
    bundle_->stage.reset();
    release();
    throw;
  }
}

TargetLayerMoeDeviceOwner::~TargetLayerMoeDeviceOwner() {
  if (!bundle_) return;
  const bool owning_device_selected =
      device_ >= 0 && cudaSetDevice(device_) == cudaSuccess;
  const bool graph_drained =
      owning_device_selected &&
      (!bundle_->graph || bundle_->graph->drain_for_destruction());
  if (!graph_drained) {
    if (!owning_device_selected && bundle_->telemetry)
      bundle_->telemetry->emit({TargetFullMoeComponent::kStaging,
                                TargetDenseOutcome::kCudaError,
                                bundle_->graph ? bundle_->graph->rank() : -1,
                                bundle_->graph ? bundle_->graph->layer() : layer_});
    (void)bundle_.release();
    return;
  }
  bundle_->graph.reset();
  bundle_->participant.reset();
  bundle_->stage.reset();
  if (bundle_->mapped_evidence_host)
    cudaFreeHost(bundle_->mapped_evidence_host);
  if (bundle_->runtime_storage) cudaFree(bundle_->runtime_storage);
  if (bundle_->stage_storage) cudaFree(bundle_->stage_storage);
  if (bundle_->requested_generation)
    cudaFree(bundle_->requested_generation);
}

void TargetLayerMoeDeviceOwner::wait_source(cudaStream_t stream) {
  bundle_->graph->wait_source(stream);
}

NativeTargetMoeGraph& TargetLayerMoeDeviceOwner::graph() noexcept {
  return *bundle_->graph;
}

const TargetFullMoeC1Workspace&
TargetLayerMoeDeviceOwner::workspace() const noexcept {
  return bundle_->workspace;
}

std::uint64_t* TargetLayerMoeDeviceOwner::requested_generation() noexcept {
  return bundle_->requested_generation;
}

}  // namespace rocket::qwen38::moe

// SPDX-License-Identifier: Apache-2.0
#include "moe/target_full_moe_c1.h"

#include <stdexcept>

namespace rocket::qwen38::moe {
namespace {

TargetMoeB12xIdentity routed_identity(const TargetDenseIdentity& identity) {
  TargetMoeB12xIdentity result{
      identity.artifact_sha256, {}, identity.rank, identity.layer};
  if (!target_moe_compact_layout_sha256(&result.layout_sha256))
    throw std::invalid_argument("target MoE compact layout identity changed");
  return result;
}

bool valid_workspace(const TargetFullMoeC1Workspace& workspace) noexcept {
  return workspace.router_logits_f32 && workspace.global_ids_i32 &&
         workspace.routing_weights_f32 && workspace.local_ids_i32 &&
         workspace.local_weights_f32 && workspace.source_generation &&
         workspace.requested_generation && workspace.route_summary &&
         workspace.source_generation != workspace.requested_generation &&
         validate_target_moe_stage_scratch(workspace.routed_stage) &&
         workspace.shared_gate_scratch_f32 &&
         workspace.shared_up_scratch_f32 &&
         workspace.shared_gate_scalar_f32;
}

TargetDenseOutcome map_outcome(TargetMoeOutcome outcome) noexcept {
  if (outcome == TargetMoeOutcome::kOk) return TargetDenseOutcome::kOk;
  if (outcome == TargetMoeOutcome::kContractError ||
      outcome == TargetMoeOutcome::kStaleGeneration)
    return TargetDenseOutcome::kContractError;
  return TargetDenseOutcome::kCudaError;
}

bool same_stage_scratch(const TargetMoeN768StageScratch& a,
                        const TargetMoeN768StageScratch& b) noexcept {
  return a.w13_packed == b.w13_packed && a.w13_scale == b.w13_scale &&
         a.down_packed == b.down_packed && a.down_scale == b.down_scale &&
         a.input_global_scale == b.input_global_scale &&
         a.folded_w1_alpha == b.folded_w1_alpha &&
         a.w2_alpha == b.w2_alpha &&
         a.down_input_scale == b.down_input_scale &&
         a.source_expert_ids == b.source_expert_ids &&
         a.compact_expert_ids == b.compact_expert_ids &&
         a.compact_routing_weights == b.compact_routing_weights &&
         a.evidence == b.evidence && a.host_evidence == b.host_evidence &&
         a.w13_packed_bytes == b.w13_packed_bytes &&
         a.w13_scale_bytes == b.w13_scale_bytes &&
         a.down_packed_bytes == b.down_packed_bytes &&
         a.down_scale_bytes == b.down_scale_bytes &&
         a.scalar_capacity == b.scalar_capacity &&
         a.route_capacity == b.route_capacity;
}

}  // namespace

struct TargetFullMoeC1::Impl {
  TargetDenseIdentity identity;
  TargetFullMoeC1Weights weights;
  TargetMoeB12xAot routed;
  mutable TargetMoeN768StageScratch last_stage{};
  mutable bool stage_pending = false;
  cudaStream_t source_stream = nullptr;

  Impl(int device, TargetDenseIdentity identity, TargetFullMoeC1Weights weights)
      : identity(identity), weights(weights),
        routed(device, routed_identity(identity),
               target_moe_staged_weights(weights.routed_stage_scratch)) {}
};

TargetFullMoeC1::TargetFullMoeC1(
    int device, TargetDenseIdentity identity, TargetFullMoeC1Weights weights)
    : impl_(nullptr) {
  TargetRouterC1Launch router_probe{};
  router_probe.hidden_bf16 = reinterpret_cast<const __nv_bfloat16*>(1);
  router_probe.logits_f32 = reinterpret_cast<float*>(1);
  router_probe.global_ids_i32 = reinterpret_cast<std::int32_t*>(1);
  router_probe.routing_weights_f32 = reinterpret_cast<float*>(1);
  router_probe.source_generation = reinterpret_cast<std::uint64_t*>(1);
  router_probe.requested_generation = reinterpret_cast<const std::uint64_t*>(1);
  router_probe.stream = reinterpret_cast<cudaStream_t>(1);
  TargetSharedC1Launch shared_probe{};
  shared_probe.hidden_bf16 = reinterpret_cast<const __nv_bfloat16*>(1);
  shared_probe.routed_plus_shared_bf16 = reinterpret_cast<__nv_bfloat16*>(1);
  shared_probe.gate_scratch_f32 = reinterpret_cast<float*>(1);
  shared_probe.up_scratch_f32 = reinterpret_cast<float*>(1);
  shared_probe.shared_gate_scratch_f32 = reinterpret_cast<float*>(1);
  shared_probe.stream = reinterpret_cast<cudaStream_t>(1);
  if (device < 0 ||
      !authenticate_target_moe_compact_runtime_identity(
          weights.routed_identity) ||
      weights.routed_identity.rank != identity.rank ||
      !weights.routed_stage || weights.routed_stage->rank() != identity.rank ||
      weights.routed_stage->layer() != identity.layer ||
      !weights.routed_stage_telemetry ||
      !validate_target_moe_stage_scratch(weights.routed_stage_scratch) ||
      diagnose_target_router_c1(identity, weights.router, router_probe) !=
          TargetDenseFailure::kNone ||
      diagnose_target_shared_c1(identity, weights.shared, shared_probe) !=
          TargetDenseFailure::kNone)
    throw std::invalid_argument("target full MoE identity or weights changed");
  impl_ = new Impl(device, identity, weights);
}

TargetFullMoeC1::~TargetFullMoeC1() {
  if (impl_ && impl_->stage_pending) {
    impl_->weights.routed_stage_telemetry->add_counter(
        {TargetMoeStageCounter::kLaunch, TargetMoeOutcome::kCudaError,
         impl_->identity.rank, impl_->identity.layer, 0});
    // Only the owning graph may prove a terminal fence. A standalone pending
    // participant cannot safely unload its module or release borrowed views.
    impl_ = nullptr;
    return;
  }
  delete impl_;
}

TargetDenseOutcome TargetFullMoeC1::enqueue(
    const TargetFullMoeC1Launch& launch) const noexcept {
  struct Discard final : TargetFullMoeOtelSink {
    void emit(const TargetFullMoeOtelPoint&) noexcept override {}
  } discard;
  return enqueue_with_telemetry(launch, discard);
}

TargetDenseOutcome TargetFullMoeC1::enqueue_with_telemetry(
    const TargetFullMoeC1Launch& launch,
    TargetFullMoeOtelSink& telemetry) const noexcept {
  if (!impl_ || !launch.hidden_bf16 || !launch.rank_local_partial_bf16 ||
      impl_->stage_pending || !valid_workspace(launch.workspace) ||
      !same_stage_scratch(impl_->weights.routed_stage_scratch,
                          launch.workspace.routed_stage) || !launch.stream ||
      launch.stream != impl_->source_stream)
    return TargetDenseOutcome::kContractError;
  const auto emit = [&](TargetFullMoeComponent component,
                        TargetDenseOutcome outcome) {
    telemetry.emit({component, outcome, impl_->identity.rank,
                    impl_->identity.layer});
    return outcome;
  };
  const auto& w = launch.workspace;
  impl_->last_stage = w.routed_stage;
  impl_->stage_pending = true;
  const auto pending_outcome = impl_->weights.routed_stage->
      enqueue_pending_evidence(w.requested_generation, w.routed_stage,
                               launch.stream);
  if (pending_outcome != TargetMoeOutcome::kOk)
    return emit(TargetFullMoeComponent::kStaging,
                map_outcome(pending_outcome));
  const TargetRouterC1Launch router_launch{
      launch.hidden_bf16, w.router_logits_f32, w.global_ids_i32,
      w.routing_weights_f32, w.source_generation, w.requested_generation,
      launch.stream};
  const auto router_outcome = enqueue_target_router_c1(
      impl_->identity, impl_->weights.router, router_launch);
  emit(TargetFullMoeComponent::kRouter, router_outcome);
  if (router_outcome != TargetDenseOutcome::kOk) return router_outcome;
  const TargetMoeC1RouteLaunch route_launch{
      impl_->identity.rank,
      impl_->identity.layer,
      w.global_ids_i32,
      w.routing_weights_f32,
      w.source_generation,
      w.requested_generation,
      w.local_ids_i32,
      w.local_weights_f32,
      w.route_summary,
      launch.stream};
  const auto route_outcome = enqueue_target_moe_c1_routes(route_launch);
  const auto localized_outcome = map_outcome(route_outcome);
  emit(TargetFullMoeComponent::kLocalization, localized_outcome);
  if (route_outcome != TargetMoeOutcome::kOk) return localized_outcome;
  const TargetMoeN640StageLaunch stage_launch{
      w.local_ids_i32, w.local_weights_f32, w.source_generation,
      w.requested_generation, w.routed_stage, launch.stream};
  const auto stage_outcome = impl_->weights.routed_stage->enqueue(stage_launch);
  const auto staged_dense_outcome = map_outcome(stage_outcome);
  emit(TargetFullMoeComponent::kStaging, staged_dense_outcome);
  if (stage_outcome != TargetMoeOutcome::kOk) return staged_dense_outcome;
  const TargetMoeB12xLaunch routed_launch{
      launch.hidden_bf16, w.routed_stage.compact_expert_ids,
      w.routed_stage.compact_routing_weights,
      launch.rank_local_partial_bf16, w.routed, launch.stream};
  const auto routed_outcome = impl_->routed.enqueue(routed_launch);
  const auto routed_dense_outcome = map_outcome(routed_outcome);
  emit(TargetFullMoeComponent::kRoutedExperts, routed_dense_outcome);
  if (routed_outcome != TargetMoeOutcome::kOk) return routed_dense_outcome;
  const TargetSharedC1Launch shared_launch{
      launch.hidden_bf16, launch.rank_local_partial_bf16,
      w.shared_gate_scratch_f32, w.shared_up_scratch_f32,
      w.shared_gate_scalar_f32, launch.stream};
  const auto shared_outcome = enqueue_target_shared_c1(
      impl_->identity, impl_->weights.shared, shared_launch);
  emit(TargetFullMoeComponent::kSharedExpert, shared_outcome);
  return shared_outcome;
}

const TargetDenseIdentity& TargetFullMoeC1::identity() const noexcept {
  return impl_->identity;
}

void TargetFullMoeC1::wait_source(cudaStream_t stream) {
  if (!impl_ || !impl_->weights.routed_stage || !stream ||
      impl_->source_stream)
    throw std::logic_error("target full MoE stage owner changed");
  impl_->weights.routed_stage->wait_source(stream);
  impl_->source_stream = stream;
}

TargetDenseOutcome TargetFullMoeC1::publish_after_fence(
    std::uint64_t generation, TargetFullMoeOtelSink& telemetry) noexcept {
  if (!impl_ || !impl_->stage_pending)
    return TargetDenseOutcome::kContractError;
  const auto outcome = map_outcome(validate_target_moe_stage_after_fence(
      impl_->last_stage, generation));
  export_target_moe_stage_otel_after_fence(
      *impl_->last_stage.host_evidence, generation, impl_->identity.rank,
      impl_->identity.layer, *impl_->weights.routed_stage_telemetry);
  telemetry.emit({TargetFullMoeComponent::kStaging, outcome,
                  impl_->identity.rank, impl_->identity.layer});
  impl_->stage_pending = false;
  return outcome;
}

}  // namespace rocket::qwen38::moe

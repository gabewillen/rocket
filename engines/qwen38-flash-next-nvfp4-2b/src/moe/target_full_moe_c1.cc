// SPDX-License-Identifier: Apache-2.0
#include "moe/target_full_moe_c1.h"

#include <stdexcept>

namespace rocket::qwen38::moe {
namespace {

TargetMoeB12xIdentity routed_identity(const TargetDenseIdentity& identity) {
  return {identity.artifact_sha256, identity.layout_sha256, identity.rank,
          identity.layer};
}

bool valid_workspace(const TargetFullMoeC1Workspace& workspace) noexcept {
  return workspace.router_logits_f32 && workspace.global_ids_i32 &&
         workspace.routing_weights_f32 && workspace.local_ids_i32 &&
         workspace.local_weights_f32 && workspace.source_generation &&
         workspace.requested_generation && workspace.route_summary &&
         workspace.source_generation != workspace.requested_generation &&
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

}  // namespace

struct TargetFullMoeC1::Impl {
  TargetDenseIdentity identity;
  TargetFullMoeC1Weights weights;
  TargetMoeB12xAot routed;

  Impl(int device, TargetDenseIdentity identity, TargetFullMoeC1Weights weights)
      : identity(identity),
        weights(weights),
        routed(device, routed_identity(identity), weights.routed) {}
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
      diagnose_target_router_c1(identity, weights.router, router_probe) !=
          TargetDenseFailure::kNone ||
      diagnose_target_shared_c1(identity, weights.shared, shared_probe) !=
          TargetDenseFailure::kNone)
    throw std::invalid_argument("target full MoE identity or weights changed");
  impl_ = new Impl(device, identity, weights);
}

TargetFullMoeC1::~TargetFullMoeC1() { delete impl_; }

TargetDenseOutcome TargetFullMoeC1::enqueue(
    const TargetFullMoeC1Launch& launch) const noexcept {
  if (!impl_ || !launch.hidden_bf16 || !launch.rank_local_partial_bf16 ||
      !valid_workspace(launch.workspace) || !launch.stream)
    return TargetDenseOutcome::kContractError;
  const auto& w = launch.workspace;
  const TargetRouterC1Launch router_launch{
      launch.hidden_bf16, w.router_logits_f32, w.global_ids_i32,
      w.routing_weights_f32, w.source_generation, w.requested_generation,
      launch.stream};
  const auto router_outcome = enqueue_target_router_c1(
      impl_->identity, impl_->weights.router, router_launch);
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
  if (route_outcome != TargetMoeOutcome::kOk) return map_outcome(route_outcome);
  const TargetMoeB12xLaunch routed_launch{
      launch.hidden_bf16, w.local_ids_i32, w.local_weights_f32,
      launch.rank_local_partial_bf16, w.routed, launch.stream};
  const auto routed_outcome = impl_->routed.enqueue(routed_launch);
  if (routed_outcome != TargetMoeOutcome::kOk)
    return map_outcome(routed_outcome);
  const TargetSharedC1Launch shared_launch{
      launch.hidden_bf16, launch.rank_local_partial_bf16,
      w.shared_gate_scratch_f32, w.shared_up_scratch_f32,
      w.shared_gate_scalar_f32, launch.stream};
  return enqueue_target_shared_c1(impl_->identity, impl_->weights.shared,
                                  shared_launch);
}

const TargetDenseIdentity& TargetFullMoeC1::identity() const noexcept {
  return impl_->identity;
}

}  // namespace rocket::qwen38::moe

extern "C" int rocket_qwen38_target_full_moe_c1_create(
    int device, const rocket::qwen38::moe::TargetDenseIdentity* identity,
    const rocket::qwen38::moe::TargetFullMoeC1Weights* weights,
    void** handle) noexcept {
  using namespace rocket::qwen38::moe;
  if (!identity || !weights || !handle || *handle)
    return static_cast<int>(TargetDenseOutcome::kContractError);
  try {
    *handle = new TargetFullMoeC1(device, *identity, *weights);
    return static_cast<int>(TargetDenseOutcome::kOk);
  } catch (const std::invalid_argument&) {
    *handle = nullptr;
    return static_cast<int>(TargetDenseOutcome::kContractError);
  } catch (...) {
    *handle = nullptr;
    return static_cast<int>(TargetDenseOutcome::kCudaError);
  }
}

extern "C" int rocket_qwen38_target_full_moe_c1_enqueue(
    void* handle,
    const rocket::qwen38::moe::TargetFullMoeC1Launch* launch) noexcept {
  using namespace rocket::qwen38::moe;
  if (!handle || !launch)
    return static_cast<int>(TargetDenseOutcome::kContractError);
  return static_cast<int>(
      static_cast<TargetFullMoeC1*>(handle)->enqueue(*launch));
}

extern "C" void rocket_qwen38_target_full_moe_c1_destroy(
    void* handle) noexcept {
  delete static_cast<rocket::qwen38::moe::TargetFullMoeC1*>(handle);
}

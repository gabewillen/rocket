// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_b12x_aot.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#if ROCKET_QWEN38_TARGET_MOE_B12X_AOT
#include "target_moe_b12x_c1.h"
#endif

namespace rocket::qwen38::moe {
namespace {

#if ROCKET_QWEN38_TARGET_MOE_B12X_AOT
constexpr std::array<std::uint8_t, 32> kArtifactSha256{
    0xa9, 0xfc, 0xca, 0x02, 0x6a, 0x87, 0xad, 0x12,
    0x85, 0xb9, 0x4f, 0xef, 0x19, 0x44, 0x8c, 0x51,
    0xb4, 0x2d, 0x97, 0x51, 0x6f, 0x16, 0x21, 0x1c,
    0x61, 0xae, 0x4c, 0x77, 0x0c, 0x6f, 0x0f, 0x4f};

bool valid_identity(const TargetMoeB12xIdentity& identity) noexcept {
  return identity.artifact_sha256 == kArtifactSha256 &&
         std::any_of(identity.layout_sha256.begin(), identity.layout_sha256.end(),
                     [](std::uint8_t byte) { return byte != 0; }) &&
         (identity.rank == 0 || identity.rank == 1) && identity.layer >= 0 &&
         identity.layer < 48;
}

bool valid_weights(const TargetMoeB12xWeights& weights) noexcept {
  return weights.w13_packed && weights.w13_scale && weights.down_packed &&
         weights.down_scale && weights.input_global_scale && weights.w1_alpha &&
         weights.w2_alpha && weights.down_input_scale;
}
#endif

bool valid_launch(const TargetMoeB12xLaunch& launch) noexcept {
  const auto& w = launch.workspace;
  return launch.hidden_bf16 && launch.local_expert_ids &&
         launch.local_routing_weights && launch.output_bf16 && launch.stream &&
         w.packed_a && w.packed_a_scale && w.route_output_scratch_bf16 &&
         w.barrier_count && w.barrier_epoch && w.row_counts &&
         w.active_expert_count && w.weight_expert_ids &&
         w.global_to_local_expert && w.virtual_route_scratch && w.token_map &&
         w.token_weights;
}

#if ROCKET_QWEN38_TARGET_MOE_B12X_AOT
template <class Module>
bool load_module(Module* module, int device, auto init, auto load) noexcept {
  cudaLibrary_t* library = &module->module;
  cudaError_t status = cudaSuccess;
  struct InitArgs {
    cudaLibrary_t** library;
    cudaError_t* status;
  } init_args{&library, &status};
  init(reinterpret_cast<void**>(&init_args));
  if (status != cudaSuccess) return false;
  std::int32_t selected_device = device;
  struct LoadArgs {
    cudaLibrary_t** library;
    std::int32_t* device;
    cudaError_t* status;
  } load_args{&library, &selected_device, &status};
  load(reinterpret_cast<void**>(&load_args));
  return status == cudaSuccess;
}
#endif

}  // namespace

#if ROCKET_QWEN38_TARGET_MOE_B12X_AOT
struct TargetMoeB12xAot::Impl {
  int device;
  TargetMoeB12xIdentity identity;
  TargetMoeB12xWeights weights;
  qwen38_target_moe_b12x_c1_Kernel_Module_t module{};
};

TargetMoeB12xAot::TargetMoeB12xAot(
    int device, TargetMoeB12xIdentity identity, TargetMoeB12xWeights weights)
    : impl_(nullptr) {
  if (device < 0 || !valid_identity(identity) || !valid_weights(weights))
    throw std::invalid_argument("target MoE B12X identity or weights changed");
  impl_ = new Impl{device, identity, weights};
  if (!load_module(&impl_->module, device,
                   _mlir_qwen38_target_moe_b12x_c1_cuda_init,
                   _mlir_qwen38_target_moe_b12x_c1_cuda_load_to_device)) {
    delete impl_;
    impl_ = nullptr;
    throw std::runtime_error("initialize target MoE B12X module failed");
  }
}

TargetMoeB12xAot::~TargetMoeB12xAot() {
  if (!impl_) return;
  cudaSetDevice(impl_->device);
  if (impl_->module.module) cudaLibraryUnload(impl_->module.module);
  delete impl_;
}

TargetMoeOutcome TargetMoeB12xAot::enqueue(
    const TargetMoeB12xLaunch& launch) const noexcept {
  if (!impl_ || !valid_launch(launch)) return TargetMoeOutcome::kContractError;
  // Generated memref spellings are checked by the AOT-enabled build. The
  // ordered call mirrors MoEStaticKernel.__call__; all storage is borrowed.
  qwen38_target_moe_b12x_c1_Tensor_a_input_t a{
      const_cast<void*>(launch.hidden_bf16)};
  qwen38_target_moe_b12x_c1_Tensor_topk_ids_t ids{
      const_cast<std::int32_t*>(launch.local_expert_ids)};
  qwen38_target_moe_b12x_c1_Tensor_topk_weights_t route_weights{
      const_cast<float*>(launch.local_routing_weights)};
  qwen38_target_moe_b12x_c1_Tensor_packed_a_t packed_a{
      launch.workspace.packed_a};
  qwen38_target_moe_b12x_c1_Tensor_packed_a_storage_t packed_storage{
      launch.workspace.packed_a};
  qwen38_target_moe_b12x_c1_Tensor_route_output_scratch_t route_scratch{
      launch.workspace.route_output_scratch_bf16};
  qwen38_target_moe_b12x_c1_Tensor_scale_storage_t scale_storage{
      launch.workspace.packed_a_scale};
  qwen38_target_moe_b12x_c1_Tensor_barrier_count_t barrier_count{
      launch.workspace.barrier_count};
  qwen38_target_moe_b12x_c1_Tensor_barrier_epoch_t barrier_epoch{
      launch.workspace.barrier_epoch};
  qwen38_target_moe_b12x_c1_Tensor_b_w13_t w13{
      const_cast<std::uint8_t*>(impl_->weights.w13_packed)};
  qwen38_target_moe_b12x_c1_Tensor_b_down_t down{
      const_cast<std::uint8_t*>(impl_->weights.down_packed)};
  qwen38_target_moe_b12x_c1_Tensor_row_counts_t row_counts{
      launch.workspace.row_counts};
  qwen38_target_moe_b12x_c1_Tensor_active_expert_count_t active_count{
      launch.workspace.active_expert_count};
  qwen38_target_moe_b12x_c1_Tensor_weight_expert_ids_t weight_ids{
      launch.workspace.weight_expert_ids};
  qwen38_target_moe_b12x_c1_Tensor_global_to_local_expert_t global_to_local{
      launch.workspace.global_to_local_expert};
  qwen38_target_moe_b12x_c1_Tensor_virt_route_scratch_t virt{
      launch.workspace.virtual_route_scratch};
  qwen38_target_moe_b12x_c1_Tensor_input_global_scale_t input_scale{
      const_cast<float*>(impl_->weights.input_global_scale)};
  qwen38_target_moe_b12x_c1_Tensor_alpha_t alpha{
      const_cast<float*>(impl_->weights.w1_alpha)};
  qwen38_target_moe_b12x_c1_Tensor_down_alpha_t down_alpha{
      const_cast<float*>(impl_->weights.w2_alpha)};
  qwen38_target_moe_b12x_c1_Tensor_global_scale_t global_scale{
      const_cast<float*>(impl_->weights.down_input_scale)};
  qwen38_target_moe_b12x_c1_Tensor_scatter_output_t output{launch.output_bf16};
  qwen38_target_moe_b12x_c1_Tensor_token_map_t token_map{launch.workspace.token_map};
  qwen38_target_moe_b12x_c1_Tensor_token_weights_t token_weights{
      launch.workspace.token_weights};
  const int status = cute_dsl_qwen38_target_moe_b12x_c1_wrapper(
      &impl_->module, &a, &ids, &route_weights, &packed_a,
      launch.workspace.packed_a_scale, &packed_storage, &route_scratch,
      &scale_storage, &barrier_count, &barrier_epoch, &w13,
      const_cast<std::uint8_t*>(impl_->weights.w13_scale), &down,
      const_cast<std::uint8_t*>(impl_->weights.down_scale), &row_counts,
      &active_count, &weight_ids, &global_to_local, &virt, &input_scale,
      &alpha, &down_alpha, &global_scale, &output, &token_map, &token_weights,
      launch.stream);
  return status == 0 ? TargetMoeOutcome::kOk : TargetMoeOutcome::kCudaError;
}

const TargetMoeB12xIdentity& TargetMoeB12xAot::identity() const noexcept {
  return impl_->identity;
}

bool target_moe_b12x_aot_compiled() noexcept { return true; }

#else

struct TargetMoeB12xAot::Impl {};

TargetMoeB12xAot::TargetMoeB12xAot(
    int, TargetMoeB12xIdentity, TargetMoeB12xWeights)
    : impl_(nullptr) {
  throw std::runtime_error(
      "target MoE B12X AOT object was not supplied at configure time");
}

TargetMoeB12xAot::~TargetMoeB12xAot() = default;

TargetMoeOutcome TargetMoeB12xAot::enqueue(
    const TargetMoeB12xLaunch& launch) const noexcept {
  return valid_launch(launch) ? TargetMoeOutcome::kCudaError
                              : TargetMoeOutcome::kContractError;
}

const TargetMoeB12xIdentity& TargetMoeB12xAot::identity() const noexcept {
  std::terminate();
}

bool target_moe_b12x_aot_compiled() noexcept { return false; }

#endif

}  // namespace rocket::qwen38::moe

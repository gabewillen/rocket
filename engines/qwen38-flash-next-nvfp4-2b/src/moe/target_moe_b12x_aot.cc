// SPDX-License-Identifier: Apache-2.0
#include "moe/target_moe_b12x_aot.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#if ROCKET_QWEN38_TARGET_MOE_B12X_AOT
#include "target_moe_b12x_c1.h"
#include "target_moe_artifact_key.h"
#endif

namespace rocket::qwen38::moe {
namespace {

#if ROCKET_QWEN38_TARGET_MOE_B12X_AOT
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

bool parse_target_moe_artifact_key(
    std::string_view ascii, std::array<std::uint8_t, 32>* bytes) noexcept {
  if (!bytes || ascii.size() != 64) return false;
  const auto nibble = [](char ch) -> int {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    return -1;
  };
  std::array<std::uint8_t, 32> parsed{};
  for (std::size_t i = 0; i < parsed.size(); ++i) {
    const int high = nibble(ascii[2 * i]);
    const int low = nibble(ascii[2 * i + 1]);
    if (high < 0 || low < 0) return false;
    parsed[i] = static_cast<std::uint8_t>((high << 4) | low);
  }
  *bytes = parsed;
  return true;
}

std::string_view target_moe_artifact_key_ascii() noexcept {
#if ROCKET_QWEN38_TARGET_MOE_B12X_AOT
  return kRocketQwen38TargetMoeArtifactKey;
#else
  return {};
#endif
}

TargetMoeCreateFailure diagnose_target_moe_b12x_create(
    int device, const TargetMoeB12xIdentity& identity,
    const TargetMoeB12xWeights& weights) noexcept {
  if (device < 0) return TargetMoeCreateFailure::kDevice;
#if ROCKET_QWEN38_TARGET_MOE_B12X_AOT
  std::array<std::uint8_t, 32> artifact_sha256{};
  if (!parse_target_moe_artifact_key(target_moe_artifact_key_ascii(),
                                     &artifact_sha256) ||
      identity.artifact_sha256 != artifact_sha256)
    return TargetMoeCreateFailure::kArtifactSha256;
#endif
  if (!std::any_of(identity.layout_sha256.begin(), identity.layout_sha256.end(),
                   [](std::uint8_t byte) { return byte != 0; }))
    return TargetMoeCreateFailure::kLayoutSha256;
  if (identity.rank != 0 && identity.rank != 1)
    return TargetMoeCreateFailure::kRank;
  if (identity.layer < 0 || identity.layer >= 48)
    return TargetMoeCreateFailure::kLayer;
  if (!weights.w13_packed) return TargetMoeCreateFailure::kW13Packed;
  if (!weights.w13_scale) return TargetMoeCreateFailure::kW13Scale;
  if (!weights.down_packed) return TargetMoeCreateFailure::kDownPacked;
  if (!weights.down_scale) return TargetMoeCreateFailure::kDownScale;
  if (!weights.input_global_scale)
    return TargetMoeCreateFailure::kInputGlobalScale;
  if (!weights.folded_w1_alpha)
    return TargetMoeCreateFailure::kFoldedW1Alpha;
  if (!weights.w2_alpha) return TargetMoeCreateFailure::kW2Alpha;
  if (!weights.down_input_scale)
    return TargetMoeCreateFailure::kDownInputScale;
  return TargetMoeCreateFailure::kNone;
}

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
  if (diagnose_target_moe_b12x_create(device, identity, weights) !=
      TargetMoeCreateFailure::kNone)
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
      const_cast<float*>(impl_->weights.folded_w1_alpha)};
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

extern "C" int rocket_qwen38_target_moe_b12x_create(
    int device,
    const rocket::qwen38::moe::TargetMoeB12xIdentity* identity,
    const rocket::qwen38::moe::TargetMoeB12xWeights* weights,
    void** handle) noexcept {
  if (!identity || !weights || !handle || *handle)
    return static_cast<int>(rocket::qwen38::moe::TargetMoeOutcome::kContractError);
  try {
    *handle = new rocket::qwen38::moe::TargetMoeB12xAot(
        device, *identity, *weights);
    return static_cast<int>(rocket::qwen38::moe::TargetMoeOutcome::kOk);
  } catch (const std::invalid_argument&) {
    *handle = nullptr;
    return static_cast<int>(
        rocket::qwen38::moe::TargetMoeOutcome::kContractError);
  } catch (...) {
    *handle = nullptr;
    return static_cast<int>(rocket::qwen38::moe::TargetMoeOutcome::kCudaError);
  }
}

extern "C" int rocket_qwen38_target_moe_b12x_diagnose_create(
    int device,
    const rocket::qwen38::moe::TargetMoeB12xIdentity* identity,
    const rocket::qwen38::moe::TargetMoeB12xWeights* weights) noexcept {
  using Failure = rocket::qwen38::moe::TargetMoeCreateFailure;
  if (!identity) return static_cast<int>(Failure::kArtifactSha256);
  if (!weights) return static_cast<int>(Failure::kW13Packed);
  return static_cast<int>(rocket::qwen38::moe::diagnose_target_moe_b12x_create(
      device, *identity, *weights));
}

extern "C" int rocket_qwen38_target_moe_b12x_enqueue(
    void* handle,
    const rocket::qwen38::moe::TargetMoeB12xLaunch* launch) noexcept {
  if (!handle || !launch)
    return static_cast<int>(rocket::qwen38::moe::TargetMoeOutcome::kContractError);
  return static_cast<int>(
      static_cast<rocket::qwen38::moe::TargetMoeB12xAot*>(handle)->enqueue(
          *launch));
}

extern "C" void rocket_qwen38_target_moe_b12x_destroy(void* handle) noexcept {
  delete static_cast<rocket::qwen38::moe::TargetMoeB12xAot*>(handle);
}

// SPDX-License-Identifier: Apache-2.0
#include "decode/target_gdn_layer_owner.h"

#include "hyperconnection/native_adapter.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocket::qwen38::decode {
namespace {

constexpr std::string_view kNvfp4Abi =
    "modelopt_nvfp4_group16_cutlass_sm121_sfb";

[[noreturn]] void fail(std::string_view reason) {
  throw std::invalid_argument("target GDN layer owner: " +
                              std::string(reason));
}

const TargetLayerNativeExtent& extent(
    const TargetLayerNativePlan& plan, std::string_view suffix,
    std::uint64_t bytes, std::string_view dtype, std::string_view layout,
    std::string_view abi, std::initializer_list<std::uint64_t> shape) {
  const std::string name = "model.language_model.layers." +
                           std::to_string(plan.layer) + "." +
                           std::string(suffix);
  const auto found = std::find_if(
      plan.extents.begin(), plan.extents.end(),
      [&](const auto& item) { return item.name == name; });
  if (found == plan.extents.end() || found->storage != "target_slab" ||
      found->length_bytes != bytes || found->dtype != dtype ||
      found->layout != layout || found->abi != abi ||
      found->shape != std::vector<std::uint64_t>(shape))
    fail("required extent changed: " + name);
  return *found;
}

void validate_required_extents(const TargetLayerNativePlan& plan) {
  const auto matrix = [&](std::string_view root, std::uint64_t weight_bytes,
                          std::uint64_t scale_bytes,
                          std::initializer_list<std::uint64_t> weight_shape,
                          std::initializer_list<std::uint64_t> scale_shape) {
    (void)extent(plan, std::string(root) + ".weight", weight_bytes, "U8",
                 "packed_e2m1_row_major", kNvfp4Abi, weight_shape);
    (void)extent(plan, std::string(root) + ".weight_scale", scale_bytes,
                 "F8_E4M3", "cutlass_sm121_sfb", kNvfp4Abi, scale_shape);
    (void)extent(plan, std::string(root) + ".weight_scale_2", 4, "F32",
                 "scalar", kNvfp4Abi, {1});
    (void)extent(plan, std::string(root) + ".input_scale", 4, "F32",
                 "scalar", kNvfp4Abi, {1});
  };
  matrix("linear_attn.in_proj_qkv", 6'553'600, 819'200,
         {5'120, 1'280}, {819'200});
  matrix("linear_attn.in_proj_z", 3'932'160, 491'520,
         {3'072, 1'280}, {491'520});
  matrix("linear_attn.in_proj_b", 30'720, 20'480,
         {24, 1'280}, {20'480});
  matrix("linear_attn.in_proj_a", 30'720, 20'480,
         {24, 1'280}, {20'480});
  matrix("linear_attn.out_proj", 3'932'160, 491'520,
         {2'560, 1'536}, {491'520});
  const auto native_bf16 = [&](std::string_view suffix, std::uint64_t bytes,
                               std::initializer_list<std::uint64_t> shape) {
    (void)extent(plan, suffix, bytes, "BF16", "checkpoint", "native", shape);
  };
  native_bf16("linear_attn.conv1d.weight", 40'960, {5'120, 1, 4});
  native_bf16("linear_attn.A_log", 48, {24});
  native_bf16("linear_attn.dt_bias", 48, {24});
  native_bf16("linear_attn.norm.weight", 256, {128});
  for (const std::string_view family : {"attn", "mlp"}) {
    const std::string root = std::string(family) + "_hyper_connection.";
    native_bf16(root + "hc_norm.weight", 20'480, {10'240});
    native_bf16(root + "input_mix_weight_down.weight", 6'553'600,
                {320, 10'240});
    native_bf16(root + "block_inject_weight.weight", 81'920, {4, 10'240});
    native_bf16(root + "input_mix_weight_up.weight", 6'553'600,
                {10'240, 320});
  }
}

template <class T>
const T* address(const std::uint8_t* base,
                 const TargetLayerNativeExtent& item) {
  const auto value = reinterpret_cast<std::uintptr_t>(base);
  if (!value || item.offset_bytes >
                    std::numeric_limits<std::uintptr_t>::max() - value)
    fail("device address overflow");
  return reinterpret_cast<const T*>(
      value + static_cast<std::uintptr_t>(item.offset_bytes));
}

TargetGdnNativeWeightBindings resolve(
    const TargetLayerNativePlan& plan, const std::uint8_t* base) {
  validate_required_extents(plan);
  const auto nvfp4 = [&](std::string_view root, std::uint64_t weight_bytes,
                         std::uint64_t scale_bytes,
                         std::initializer_list<std::uint64_t> weight_shape,
                         std::initializer_list<std::uint64_t> scale_shape,
                         float global) {
    return linear_attention::Nvfp4Matrix{
        address<std::uint8_t>(base, extent(
            plan, std::string(root) + ".weight", weight_bytes, "U8",
            "packed_e2m1_row_major", kNvfp4Abi, weight_shape)),
        address<std::uint8_t>(base, extent(
            plan, std::string(root) + ".weight_scale", scale_bytes,
            "F8_E4M3", "cutlass_sm121_sfb", kNvfp4Abi, scale_shape)),
        address<float>(base, extent(
            plan, std::string(root) + ".input_scale", 4, "F32",
            "scalar", kNvfp4Abi, {1})),
        global};
  };
  const auto native_bf16 = [&](std::string_view suffix, std::uint64_t bytes,
                               std::initializer_list<std::uint64_t> shape) {
    return address<__nv_bfloat16>(
        base, extent(plan, suffix, bytes, "BF16", "checkpoint", "native",
                     shape));
  };
  const auto hyper = [&](std::string_view family) {
    const std::string root = std::string(family) + "_hyper_connection.";
    return hyperconnection::Weights{
        native_bf16(root + "hc_norm.weight", 20'480, {10'240}),
        native_bf16(root + "input_mix_weight_down.weight", 6'553'600,
                    {320, 10'240}),
        native_bf16(root + "block_inject_weight.weight", 81'920,
                    {4, 10'240}),
        native_bf16(root + "input_mix_weight_up.weight", 6'553'600,
                    {10'240, 320})};
  };
  const auto& globals = plan.attention_projection_globals;
  TargetGdnNativeWeightBindings result{};
  result.attention = {
      nvfp4("linear_attn.in_proj_qkv", 6'553'600, 819'200,
            {5'120, 1'280}, {819'200}, globals[0]),
      nvfp4("linear_attn.in_proj_z", 3'932'160, 491'520,
            {3'072, 1'280}, {491'520}, globals[1]),
      nvfp4("linear_attn.in_proj_b", 30'720, 20'480,
            {24, 1'280}, {20'480}, globals[2]),
      nvfp4("linear_attn.in_proj_a", 30'720, 20'480,
            {24, 1'280}, {20'480}, globals[3]),
      nvfp4("linear_attn.out_proj", 3'932'160, 491'520,
            {2'560, 1'536}, {491'520}, globals[4]),
      native_bf16("linear_attn.conv1d.weight", 40'960, {5'120, 1, 4}),
      native_bf16("linear_attn.A_log", 48, {24}),
      native_bf16("linear_attn.dt_bias", 48, {24}),
      native_bf16("linear_attn.norm.weight", 256, {128})};
  result.attention_hyperconnection = hyper("attn");
  result.mlp_hyperconnection = hyper("mlp");
  return result;
}

void validate_publication(const TargetLayerNativePlan& plan,
                          const model::TargetSlabPublication& slab) {
  if (!slab.device_base || !slab.ready_event || slab.bytes != plan.slab_bytes ||
      slab.device < 0 || slab.rank != plan.rank ||
      slab.artifact_key != model::kTargetSlabArtifactKey ||
      slab.artifact_key != plan.artifact_key || slab.slab_key != plan.slab_key ||
      slab.manifest_sha256 != model::kTargetSlabManifestSha256 ||
      slab.layout_sha256 != plan.slab_publication_layout_sha256 ||
      slab.open_to_publish_ns == 0 ||
      slab.chunks_authenticated != model::kTargetSlabChunks ||
      slab.peak_host_pinned_bytes != model::kTargetSlabPeakPinnedBytes)
    fail("slab publication identity changed");
}

struct Cursor {
  std::uint8_t* base;
  std::size_t offset = 0;
  template <class T>
  T* take(std::size_t elements) {
    offset = (offset + 255) & ~std::size_t{255};
    auto* result = reinterpret_cast<T*>(base + offset);
    offset += elements * sizeof(T);
    return result;
  }
};

__global__ void initialize_target_gdn_state_index(std::int32_t* state_index) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *state_index = 1;
}

__global__ void publish_target_gdn_moe_generation(
    std::uint64_t* destination, std::uint64_t generation) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *destination = generation;
}

class NativeTargetGdnMoeGeneration final
    : public TargetGdnMoeGenerationPort {
 public:
  NativeTargetGdnMoeGeneration(int rank, int layer,
                               std::uint64_t* requested_generation)
      : rank_(rank), layer_(layer), requested_generation_(requested_generation) {
    if ((rank != 0 && rank != 1) || !is_linear_attention_layer(layer) ||
        !requested_generation)
      fail("MoE generation binding changed");
  }
  int rank() const noexcept override { return rank_; }
  int layer() const noexcept override { return layer_; }
  bool authenticated() const noexcept override { return true; }
  void enqueue(std::uint64_t generation, cudaStream_t stream) override {
    if (!generation || !stream) fail("MoE generation request changed");
    publish_target_gdn_moe_generation<<<1, 1, 0, stream>>>(
        requested_generation_, generation);
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("target GDN generation launch: ") +
                               cudaGetErrorString(status));
  }
 private:
  int rank_;
  int layer_;
  std::uint64_t* requested_generation_;
};

void emit(pair_reduce::OtelStageSink& telemetry, int rank,
          pair_reduce::Outcome outcome) noexcept {
  telemetry.emit_span_and_log({
      "rocket.qwen38.k0.gdn_layer_owner", "gdn-owner-init", "all36-gdn",
      (rank == 0 || rank == 1) ? rank : -1, 1, pair_reduce::kDtype, outcome,
      0, 0});
  telemetry.record_duration({(rank == 0 || rank == 1) ? rank : -1, 1,
                             pair_reduce::kDtype, outcome, 0});
}

}  // namespace

struct TargetGdnLayerDeviceOwner::Bundle {
  std::shared_ptr<const model::TargetSlabLease> slab_lease;
  std::unique_ptr<moe::TargetLayerMoeDeviceOwner> moe_owner;
  void* storage = nullptr;
  TargetGdnOwnerStorageBinding storage_binding{};
  std::unique_ptr<linear_attention::CutlassGdnGraph> graph;
  std::unique_ptr<hyperconnection::Plan> hyperconnection;
  std::unique_ptr<hyperconnection::NativeFullAttentionHyperConnection>
      hyperconnection_adapter;
  std::unique_ptr<NativeTargetGdnMoeGeneration> moe_generation;
  std::unique_ptr<TargetGdnLayer> layer;
};

bool validate_target_gdn_layer_plan(
    const TargetLayerNativePlan& plan) noexcept {
  try {
    validate_target_layer_native_plan_binding(plan);
    if (plan.attention_kind != TargetK0AttentionKind::kGdn ||
        !is_linear_attention_layer(plan.layer))
      return false;
    for (const float value : plan.attention_projection_globals)
      if (!std::isfinite(value) || value <= 0.0F) return false;
    validate_required_extents(plan);
    return true;
  } catch (...) {
    return false;
  }
}

TargetGdnNativeWeightBindings bind_target_gdn_native_weights(
    const TargetLayerNativePlan& plan,
    const model::TargetSlabPublication& slab) {
  validate_target_layer_native_plan_binding(plan);
  if (plan.attention_kind != TargetK0AttentionKind::kGdn ||
      !is_linear_attention_layer(plan.layer))
    fail("GDN topology changed");
  for (const float value : plan.attention_projection_globals)
    if (!std::isfinite(value) || value <= 0.0F)
      fail("projection scalar changed");
  validate_publication(plan, slab);
  return resolve(plan, slab.device_base);
}

TargetGdnOwnerStorageBinding bind_target_gdn_owner_storage(
    void* storage, std::size_t bytes) {
  if (!storage || bytes != kTargetGdnOwnerStorageBytes ||
      reinterpret_cast<std::uintptr_t>(storage) % 256 != 0)
    fail("device storage extent changed");
  Cursor cursor{static_cast<std::uint8_t*>(storage)};
  TargetGdnOwnerStorageBinding result{};
  result.state.convolution = cursor.take<__nv_bfloat16>(
      kTargetGdnStateSlots * kTargetGdnConvSlotElements);
  result.state.convolution_elements =
      kTargetGdnStateSlots * kTargetGdnConvSlotElements;
  result.state.recurrent = cursor.take<float>(
      kTargetGdnStateSlots * kTargetGdnRecurrentSlotElements);
  result.state.recurrent_elements =
      kTargetGdnStateSlots * kTargetGdnRecurrentSlotElements;
  result.mutable_state_index = cursor.take<std::int32_t>(1);
  result.state.state_index = result.mutable_state_index;
  result.state.state_index_elements = 1;
  result.buffers.attention_input = cursor.take<__nv_bfloat16>(kTargetK0Hidden);
  result.buffers.attention_injection =
      cursor.take<__nv_bfloat16>(kTargetK0HiddenStreams);
  result.buffers.reduced_attention = cursor.take<float>(kTargetK0Hidden);
  result.buffers.post_attention_hidden =
      cursor.take<__nv_bfloat16>(kTargetK0HyperHidden);
  result.buffers.moe_input = cursor.take<__nv_bfloat16>(kTargetK0Hidden);
  result.buffers.moe_injection =
      cursor.take<__nv_bfloat16>(kTargetK0HiddenStreams);
  result.buffers.reduced_moe = cursor.take<float>(kTargetK0Hidden);
  if (cursor.offset > bytes || !valid_target_gdn_c1_state_extent(result.state) ||
      !complete_target_gdn_c1_buffers(result.buffers))
    throw std::logic_error("target GDN storage layout changed");
  return result;
}

std::unique_ptr<TargetGdnLayerDeviceOwner>
TargetGdnLayerDeviceOwner::create(
    int device, const TargetLayerNativePlan& plan,
    void* accepted_loader_lease_handle,
    std::unique_ptr<moe::TargetLayerMoeDeviceOwner> moe_owner,
    TargetK0PairReduceSchedule& reductions,
    pair_reduce::OtelStageSink& telemetry,
    TargetGdnOwnerConstructionStage* construction_stage) {
  auto lease = model::TargetSlabStartupFactory::lease_from_handle(
      accepted_loader_lease_handle);
  if (!lease)
    throw std::invalid_argument("accepted-loader slab lease handle changed");
  return std::unique_ptr<TargetGdnLayerDeviceOwner>(
      new TargetGdnLayerDeviceOwner(
          device, plan, std::move(lease), std::move(moe_owner), reductions,
          telemetry, construction_stage));
}

TargetGdnLayerDeviceOwner::TargetGdnLayerDeviceOwner(
    int device, const TargetLayerNativePlan& plan,
    std::shared_ptr<const model::TargetSlabLease> slab_lease,
    std::unique_ptr<moe::TargetLayerMoeDeviceOwner> moe_owner,
    TargetK0PairReduceSchedule& reductions,
    pair_reduce::OtelStageSink& telemetry,
    TargetGdnOwnerConstructionStage* construction_stage)
    : device_(device), rank_(plan.rank), layer_(plan.layer),
      bundle_(std::make_unique<Bundle>()) {
  const auto mark = [construction_stage](TargetGdnOwnerConstructionStage stage) {
    if (construction_stage) *construction_stage = stage;
  };
  try {
    mark(TargetGdnOwnerConstructionStage::kLease);
    bundle_->slab_lease = std::move(slab_lease);
    bundle_->moe_owner = std::move(moe_owner);
    if (!bundle_->slab_lease ||
        bundle_->slab_lease->lifetime() !=
            model::TargetSlabLease::Lifetime::kProcessLifetime ||
        device != bundle_->slab_lease->publication().device ||
        !validate_target_gdn_layer_plan(plan) ||
        !bundle_->moe_owner || !bundle_->moe_owner->authenticated() ||
        bundle_->moe_owner->rank() != rank_ ||
        bundle_->moe_owner->layer() != layer_ || reductions.rank() != rank_)
      fail("owned dependency identity changed");
    mark(TargetGdnOwnerConstructionStage::kPlanBinder);
    const auto weights = bind_target_gdn_native_weights(
        plan, bundle_->slab_lease->publication());
    mark(TargetGdnOwnerConstructionStage::kGlobals);
    for (const float value : plan.attention_projection_globals)
      if (!std::isfinite(value) || value <= 0.0F)
        fail("projection scalar changed");
    mark(TargetGdnOwnerConstructionStage::kStorage);
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMalloc(&bundle_->storage, kTargetGdnOwnerStorageBytes) != cudaSuccess)
      throw std::runtime_error("target GDN storage allocation failed");
    bundle_->storage_binding = bind_target_gdn_owner_storage(
        bundle_->storage, kTargetGdnOwnerStorageBytes);
    cudaStream_t init_stream = nullptr;
    cudaEvent_t initialized = nullptr;
    const auto release_init = [&] {
      if (initialized) cudaEventDestroy(initialized);
      if (init_stream) cudaStreamDestroy(init_stream);
    };
    if (cudaStreamCreateWithFlags(&init_stream, cudaStreamNonBlocking) !=
            cudaSuccess ||
        cudaEventCreateWithFlags(&initialized, cudaEventDisableTiming) !=
            cudaSuccess ||
        cudaMemsetAsync(bundle_->storage, 0, kTargetGdnOwnerStorageBytes,
                        init_stream) != cudaSuccess) {
      release_init();
      throw std::runtime_error("target GDN state initialization failed");
    }
    initialize_target_gdn_state_index<<<1, 1, 0, init_stream>>>(
        bundle_->storage_binding.mutable_state_index);
    if (cudaPeekAtLastError() != cudaSuccess ||
        cudaEventRecord(initialized, init_stream) != cudaSuccess ||
        cudaEventSynchronize(initialized) != cudaSuccess) {
      release_init();
      throw std::runtime_error("target GDN state publication failed");
    }
    release_init();
    mark(TargetGdnOwnerConstructionStage::kCutlassGraph);
    bundle_->graph = std::make_unique<linear_attention::CutlassGdnGraph>(
        device, rank_, layer_, weights.attention);
    mark(TargetGdnOwnerConstructionStage::kHyperconnection);
    bundle_->hyperconnection = std::make_unique<hyperconnection::Plan>(
        device, weights.attention_hyperconnection,
        weights.mlp_hyperconnection);
    bundle_->hyperconnection_adapter = std::make_unique<
        hyperconnection::NativeFullAttentionHyperConnection>(
        *bundle_->hyperconnection);
    bundle_->moe_generation = std::make_unique<NativeTargetGdnMoeGeneration>(
        rank_, layer_, bundle_->moe_owner->requested_generation());
    mark(TargetGdnOwnerConstructionStage::kComposite);
    bundle_->layer = std::make_unique<TargetGdnLayer>(
        *bundle_->graph, bundle_->moe_owner->graph(),
        reductions.attention_port(layer_), reductions.moe_port(layer_),
        *bundle_->hyperconnection_adapter, *bundle_->moe_generation, telemetry,
        bundle_->storage_binding.state, bundle_->storage_binding.buffers);
    authenticated_ = true;
    emit(telemetry, rank_, pair_reduce::Outcome::kOk);
  } catch (...) {
    if (bundle_ && bundle_->storage) {
      if (device_ >= 0) (void)cudaSetDevice(device_);
      (void)cudaFree(bundle_->storage);
      bundle_->storage = nullptr;
    }
    emit(telemetry, rank_, pair_reduce::Outcome::kContractError);
    throw;
  }
}

TargetGdnLayerDeviceOwner::~TargetGdnLayerDeviceOwner() {
  if (!bundle_) return;
  if ((bundle_->layer && bundle_->layer->faulted()) || device_ < 0 ||
      cudaSetDevice(device_) != cudaSuccess) {
    (void)bundle_.release();
    return;
  }
  bundle_->layer.reset();
  bundle_->moe_generation.reset();
  bundle_->hyperconnection_adapter.reset();
  bundle_->hyperconnection.reset();
  bundle_->graph.reset();
  bundle_->moe_owner.reset();
  if (bundle_->storage) {
    cudaFree(bundle_->storage);
    bundle_->storage = nullptr;
  }
}

const HiddenPartialReducer*
TargetGdnLayerDeviceOwner::attention_reducer_identity() const noexcept {
  return bundle_ && bundle_->layer
             ? bundle_->layer->attention_reducer_identity()
             : nullptr;
}

const HiddenPartialReducer*
TargetGdnLayerDeviceOwner::moe_reducer_identity() const noexcept {
  return bundle_ && bundle_->layer ? bundle_->layer->moe_reducer_identity()
                                   : nullptr;
}

void TargetGdnLayerDeviceOwner::wait_source(cudaStream_t stream) {
  if (!authenticated_ || !stream)
    fail("source wait request changed");
  const auto event = bundle_->slab_lease->publication().ready_event;
  if (!event || cudaStreamWaitEvent(stream, event, 0) != cudaSuccess)
    throw std::runtime_error("target GDN slab wait failed");
  bundle_->moe_owner->wait_source(stream);
}

void TargetGdnLayerDeviceOwner::execute_row(
    std::uint64_t generation, const __nv_bfloat16* replicated_pre_layer,
    __nv_bfloat16* replicated_post_layer, cudaStream_t stream,
    TargetK0ExecutionProgress* progress) {
  if (!authenticated_ || !bundle_ || !bundle_->layer)
    fail("execution ownership changed");
  const auto result = bundle_->layer->execute(
      generation, replicated_pre_layer, replicated_post_layer,
      "k0-gdn-layer", "oracle-05ea3af", stream, progress);
  if (result.generation != generation || result.rank != rank_ ||
      result.layer != layer_ || result.post_layer != replicated_post_layer)
    throw std::logic_error("target GDN row publication changed");
}

}  // namespace rocket::qwen38::decode

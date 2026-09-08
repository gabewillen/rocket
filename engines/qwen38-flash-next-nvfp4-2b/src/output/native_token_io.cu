// SPDX-License-Identifier: Apache-2.0
#include "output/native_token_io.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "hyperconnection/hyperconnection.h"
#include "output/token_output.h"

namespace rocket::qwen38::output {
namespace {

using Clock = std::chrono::steady_clock;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("K0 token I/O ") + operation +
                             ": " + cudaGetErrorString(status));
}

void check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("K0 token I/O ") + operation +
                             " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
}

template <class T>
const T* slab_at(const model::TargetSlabPublication& publication,
                 SlabDescriptor descriptor) {
  if (descriptor.offset_bytes > publication.bytes ||
      descriptor.length_bytes > publication.bytes - descriptor.offset_bytes)
    throw std::invalid_argument("K0 token I/O slab extent changed");
  return reinterpret_cast<const T*>(publication.device_base +
                                    descriptor.offset_bytes);
}

void validate_physical_lease(
    int device, int rank,
    const std::shared_ptr<const model::TargetSlabLease>& lease) {
  if (!lease ||
      dynamic_cast<const model::ProcessLifetimeTargetSlabLease*>(lease.get()) ==
          nullptr ||
      lease->lifetime() != model::TargetSlabLease::Lifetime::kProcessLifetime)
    throw std::invalid_argument("K0 token I/O accepted slab capability changed");
  const auto& publication = lease->publication();
  if (device < 0 || (rank != 0 && rank != 1) || publication.device != device ||
      publication.rank != rank || !publication.device_base ||
      !publication.ready_event || publication.bytes != model::kTargetSlabBytes ||
      publication.artifact_key != model::kTargetSlabArtifactKey ||
      publication.manifest_sha256 != model::kTargetSlabManifestSha256 ||
      publication.slab_key != (rank == 0 ? "rank0-target" : "rank1-target") ||
      publication.layout_sha256.size() != 64 ||
      publication.chunks_authenticated != model::kTargetSlabChunks ||
      publication.peak_host_pinned_bytes != model::kTargetSlabPeakPinnedBytes ||
      publication.open_to_publish_ns == 0)
    throw std::invalid_argument("K0 token I/O slab identity changed");

  check(cudaSetDevice(device), "set device for slab probe");
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, publication.device_base) !=
      cudaSuccess)
    throw std::invalid_argument("K0 token I/O slab pointer is not CUDA memory");
  CUdeviceptr allocation_base = 0;
  std::size_t allocation_bytes = 0;
  if (cuMemGetAddressRange(
          &allocation_base, &allocation_bytes,
          reinterpret_cast<CUdeviceptr>(publication.device_base)) !=
      CUDA_SUCCESS)
    throw std::invalid_argument("K0 token I/O slab allocation is unavailable");
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != device ||
      allocation_base !=
          reinterpret_cast<CUdeviceptr>(publication.device_base) ||
      allocation_bytes != publication.bytes ||
      cudaEventQuery(publication.ready_event) != cudaSuccess)
    throw std::invalid_argument("K0 token I/O slab pointer/event probe changed");
}

__global__ void round_and_replicate(const float* reduced,
                                    __nv_bfloat16* replicated) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= kHidden) return;
  const __nv_bfloat16 value = __float2bfloat16(reduced[column]);
#pragma unroll
  for (int stream = 0; stream < kHyperConnections; ++stream)
    replicated[stream * kHidden + column] = value;
}

}  // namespace

struct NativeTokenIoOwner::Impl {
  int device = -1;
  int rank = -1;
  std::shared_ptr<const model::TargetSlabLease> slab;
  TokenIoArtifactRoots roots;
  mtp::WinnerExchangePort* winner_exchange = nullptr;
  pair_reduce::OtelStageSink* telemetry = nullptr;
  std::unique_ptr<pair_reduce::PairReduce> embedding_reduce;
  std::unique_ptr<hyperconnection::FinalPlan> final;
  cublasHandle_t head = nullptr;
  std::int32_t* token_device = nullptr;
  std::int32_t* invalid_token = nullptr;
  __nv_bfloat16* embedding_partial = nullptr;
  float* embedding_reduced = nullptr;
  float* zero_block_output = nullptr;
  __nv_bfloat16* zero_injection = nullptr;
  __nv_bfloat16* updated_hidden = nullptr;
  __nv_bfloat16* final_hidden = nullptr;
  float* local_logits = nullptr;
  Winner* local_winner = nullptr;
  Winner* rank_winners = nullptr;
  std::int32_t* global_token = nullptr;
  std::int32_t* terminal_token_host = nullptr;
  std::uint64_t last_embedding_generation = 0;
  std::uint64_t last_finish_generation = 0;

  ~Impl() {
    if (head) cublasDestroy(head);
    cudaFreeHost(terminal_token_host);
    cudaFree(global_token);
    cudaFree(rank_winners);
    cudaFree(local_winner);
    cudaFree(local_logits);
    cudaFree(final_hidden);
    cudaFree(updated_hidden);
    cudaFree(zero_injection);
    cudaFree(zero_block_output);
    cudaFree(embedding_reduced);
    cudaFree(embedding_partial);
    cudaFree(invalid_token);
    cudaFree(token_device);
  }

  void emit(std::string_view stage, pair_reduce::Outcome outcome,
            std::uint64_t duration_ns, std::uint64_t bytes) noexcept {
    telemetry->emit_span_and_log({"rocket.qwen38.token_io.lifecycle", stage,
                                  "k0-token-io", rank, 1,
                                  pair_reduce::kDtype, outcome, duration_ns,
                                  bytes});
    telemetry->record_duration(
        {rank, 1, pair_reduce::kDtype, outcome, duration_ns});
  }
};

NativeTokenIoOwner::NativeTokenIoOwner(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::unique_ptr<NativeTokenIoOwner> NativeTokenIoOwner::create(
    int device, int rank, void* accepted_loader_lease_handle,
    pair_reduce::Transport& embedding_transport,
    mtp::WinnerExchangePort& winner_exchange,
    pair_reduce::OtelStageSink& telemetry, TokenIoArtifactRoots roots) {
  const auto started = Clock::now();
  auto impl = std::make_unique<Impl>();
  impl->device = device;
  impl->rank = rank;
  impl->telemetry = &telemetry;
  try {
    const auto authenticated_roots = authenticate_token_io_artifact_roots(
        roots.tokenizer, roots.oracle_capture);
    if (authenticated_roots.tokenizer_identity_sha256 !=
            roots.tokenizer_identity_sha256 ||
        authenticated_roots.oracle_manifest_sha256 !=
            roots.oracle_manifest_sha256)
      throw std::invalid_argument("K0 token I/O artifact identity changed");
    impl->slab = model::TargetSlabStartupFactory::lease_from_handle(
        accepted_loader_lease_handle);
    validate_physical_lease(device, rank, impl->slab);
    if (embedding_transport.rank() != rank ||
        embedding_transport.world_size() != kTpSize)
      throw std::invalid_argument("K0 embedding transport identity changed");
    impl->roots = authenticated_roots;
    impl->winner_exchange = &winner_exchange;
    impl->embedding_reduce = std::make_unique<pair_reduce::PairReduce>(
        embedding_transport, telemetry);
    const auto& publication = impl->slab->publication();
    impl->final = std::make_unique<hyperconnection::FinalPlan>(
        device, slab_at<__nv_bfloat16>(publication, kFinalNorm),
        slab_at<__nv_bfloat16>(publication, kFinalDown),
        slab_at<__nv_bfloat16>(publication, kFinalUp));
    auto allocate = [](auto** pointer, std::size_t count) {
      check(cudaMalloc(pointer, count * sizeof(**pointer)), "allocate arena");
    };
    allocate(&impl->token_device, 1);
    allocate(&impl->invalid_token, 1);
    allocate(&impl->embedding_partial, kHidden);
    allocate(&impl->embedding_reduced, kHidden);
    allocate(&impl->zero_block_output, kHidden);
    allocate(&impl->zero_injection, kHyperConnections);
    allocate(&impl->updated_hidden, kHyperHidden);
    allocate(&impl->final_hidden, kHidden);
    allocate(&impl->local_logits, kLocalVocab);
    allocate(&impl->local_winner, 1);
    allocate(&impl->rank_winners, kTpSize);
    allocate(&impl->global_token, 1);
    check(cudaHostAlloc(&impl->terminal_token_host, sizeof(std::int32_t),
                        cudaHostAllocDefault),
          "allocate terminal token");
    *impl->terminal_token_host = -1;
    check(cudaMemset(impl->zero_block_output, 0, kHidden * sizeof(float)),
          "initialize final block zero");
    check(cudaMemset(impl->zero_injection, 0,
                     kHyperConnections * sizeof(__nv_bfloat16)),
          "initialize final injection zero");
    check(cublasCreate(&impl->head), "create lm_head handle");
    const auto duration = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                             started)
            .count());
    impl->emit("construct", pair_reduce::Outcome::kOk, duration, 0);
    return std::unique_ptr<NativeTokenIoOwner>(
        new NativeTokenIoOwner(std::move(impl)));
  } catch (const std::invalid_argument&) {
    impl->emit("construct", pair_reduce::Outcome::kContractError, 0, 0);
    throw;
  } catch (...) {
    impl->emit("construct", pair_reduce::Outcome::kCudaError, 0, 0);
    throw;
  }
}

NativeTokenIoOwner::~NativeTokenIoOwner() = default;

int NativeTokenIoOwner::rank() const noexcept { return impl_->rank; }

bool NativeTokenIoOwner::authenticated() const noexcept {
  return impl_->slab && impl_->roots.oracle_manifest_sha256 ==
                            kTokenIoOracleManifestSha256 &&
         impl_->roots.tokenizer_identity_sha256.size() == 64;
}

TokenIoArenaView NativeTokenIoOwner::arena() const noexcept {
  return {impl_->final_hidden, impl_->local_logits, impl_->local_winner,
          impl_->rank_winners, impl_->global_token};
}

void NativeTokenIoOwner::wait_source(cudaStream_t stream) {
  if (!stream) throw std::invalid_argument("K0 token I/O stream changed");
  check(cudaStreamWaitEvent(stream, impl_->slab->publication().ready_event, 0),
        "wait for slab publication");
}

void NativeTokenIoOwner::embed_row(std::int32_t token,
                                   std::uint64_t generation,
                                   __nv_bfloat16* replicated_hidden,
                                   cudaStream_t stream) {
  const auto started = Clock::now();
  try {
    if (!replicated_hidden || !stream || token < 0 || token >= kVocab ||
        generation == 0 || generation <= impl_->last_embedding_generation)
      throw std::invalid_argument("K0 embedding row contract changed");
    check(cudaMemcpyAsync(impl_->token_device, &token, sizeof(token),
                          cudaMemcpyHostToDevice, stream),
          "stage token id");
    check(cudaMemsetAsync(impl_->invalid_token, 0, sizeof(std::int32_t), stream),
          "clear invalid-token flag");
    check(embedding_lookup_rank(
              impl_->token_device,
              slab_at<__nv_bfloat16>(impl_->slab->publication(), kEmbedding),
              impl_->embedding_partial, impl_->invalid_token, 1, impl_->rank,
              stream),
          "launch embedding lookup");
    impl_->embedding_reduce->reduce(impl_->embedding_partial,
                                    impl_->embedding_reduced, 1,
                                    "k0-embedding", "k0-token-io", stream);
    round_and_replicate<<<(kHidden + 255) / 256, 256, 0, stream>>>(
        impl_->embedding_reduced, replicated_hidden);
    check(cudaPeekAtLastError(), "launch embedding replication");
    impl_->last_embedding_generation = generation;
    const auto duration = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                             started)
            .count());
    impl_->emit("embedding", pair_reduce::Outcome::kOk, duration,
                kHidden * sizeof(__nv_bfloat16));
  } catch (const std::invalid_argument&) {
    impl_->emit("embedding", pair_reduce::Outcome::kContractError, 0, 0);
    throw;
  } catch (const pair_reduce::PairReduceContractError&) {
    impl_->emit("embedding", pair_reduce::Outcome::kContractError, 0, 0);
    throw;
  } catch (const pair_reduce::PairReduceTransportError&) {
    impl_->emit("embedding", pair_reduce::Outcome::kTransportError, 0, 0);
    throw;
  } catch (const pair_reduce::PairReduceCudaError&) {
    impl_->emit("embedding", pair_reduce::Outcome::kCudaError, 0, 0);
    throw;
  } catch (...) {
    impl_->emit("embedding", pair_reduce::Outcome::kCudaError, 0, 0);
    throw;
  }
}

decode::TargetK0TokenOutput NativeTokenIoOwner::finish_prefill(
    const __nv_bfloat16* replicated_post_layer, std::uint64_t generation,
    cudaStream_t stream, decode::TargetK0ExecutionProgress* progress) {
  const auto started = Clock::now();
  try {
    if (!replicated_post_layer || !stream || generation == 0 ||
        generation != impl_->last_embedding_generation ||
        generation <= impl_->last_finish_generation)
      throw std::invalid_argument("K0 final token row contract changed");
    decode::target_k0_enter_stage(
        progress, decode::TargetK0ExecutionStage::kFinalNorm);
    impl_->final->combine_and_collapse(
        replicated_post_layer, impl_->zero_block_output,
        impl_->zero_injection, impl_->updated_hidden, impl_->final_hidden, 1,
        stream);
    decode::target_k0_enter_stage(progress,
                                  decode::TargetK0ExecutionStage::kLmHead);
    check(lm_head(impl_->head, impl_->final_hidden,
                  slab_at<__nv_bfloat16>(impl_->slab->publication(), kLmHead),
                  impl_->local_logits, 1, impl_->rank, stream),
          "enqueue lm_head");
    check(local_argmax(impl_->local_logits, impl_->local_winner, 1,
                       impl_->rank, stream),
          "enqueue local argmax");
    decode::target_k0_enter_stage(
        progress, decode::TargetK0ExecutionStage::kWinnerExchange);
    impl_->winner_exchange->enqueue(impl_->local_winner, impl_->rank_winners,
                                    1, impl_->rank, stream);
    check(global_greedy(impl_->rank_winners, impl_->global_token, 1, 0.0F,
                        1.0F, stream),
          "enqueue global greedy");
    check(cudaMemcpyAsync(impl_->terminal_token_host, impl_->global_token,
                          sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream),
          "stage terminal token");
    decode::target_k0_enter_stage(
        progress, decode::TargetK0ExecutionStage::kTerminalFence);
    check(cudaStreamSynchronize(stream), "terminal token fence");
    impl_->winner_exchange->validate_after_fence();
    const std::int32_t token = *impl_->terminal_token_host;
    if (token < 0 || token >= kVocab)
      throw std::runtime_error("K0 global greedy publication changed");
    impl_->last_finish_generation = generation;
    const auto duration = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                             started)
            .count());
    impl_->emit("final", pair_reduce::Outcome::kOk, duration,
                kHidden * sizeof(__nv_bfloat16) +
                    kLocalVocab * sizeof(float) + sizeof(std::int32_t));
    return {impl_->final_hidden, impl_->local_logits, token};
  } catch (const std::invalid_argument&) {
    impl_->emit("final", pair_reduce::Outcome::kContractError, 0, 0);
    throw;
  } catch (...) {
    impl_->emit("final", pair_reduce::Outcome::kCudaError, 0, 0);
    throw;
  }
}

}  // namespace rocket::qwen38::output

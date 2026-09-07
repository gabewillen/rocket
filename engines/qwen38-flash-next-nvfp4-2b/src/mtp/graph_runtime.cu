// SPDX-License-Identifier: Apache-2.0
#include "mtp/graph_runtime.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include "hyperconnection/hyperconnection.h"

namespace rocket::qwen38::mtp {
namespace {

constexpr int kBuckets[] = {1, 2, 4, 8, 16};
constexpr std::uint64_t kNormEmbeddingBytes = 5'120;
constexpr std::uint64_t kNormHiddenBytes = 20'480;
constexpr std::uint64_t kProjectionBytes = 6'553'600;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("MTP graph runtime ") + operation +
                             ": " + cudaGetErrorString(status));
}

bool valid_digest(const std::array<std::uint8_t, 32>& digest) {
  return std::any_of(digest.begin(), digest.end(),
                     [](std::uint8_t byte) { return byte != 0; });
}

bool valid_extent(TensorExtent extent, std::uint64_t expected,
                  std::size_t slab_bytes) {
  return extent.offset_bytes % 256 == 0 && extent.length_bytes == expected &&
         extent.offset_bytes <= slab_bytes &&
         expected <= slab_bytes - extent.offset_bytes;
}

int bucket_index(int m) {
  for (int index = 0; index < 5; ++index)
    if (kBuckets[index] == m) return index;
  throw std::invalid_argument("MTP graph bucket changed");
}

template <typename T>
T* at(const std::byte* base, TensorExtent extent) {
  return reinterpret_cast<T*>(const_cast<std::byte*>(base + extent.offset_bytes));
}

}  // namespace

struct MtpGraphRuntime::Impl {
  GraphRuntimeBinding binding{};
  GraphArenaView arena{};
  std::unique_ptr<InputFusionPlan> input;
  std::unique_ptr<hyperconnection::FinalPlan> final;
  cublasHandle_t head_handle = nullptr;
  cudaStream_t capture_stream = nullptr;
  std::array<cudaGraphExec_t, 5> input_local{};
  std::array<cudaGraphExec_t, 5> input_finish{};
  std::array<cudaGraphExec_t, 5> final_local{};

  ~Impl() {
    for (cudaGraphExec_t graph : final_local)
      if (graph) cudaGraphExecDestroy(graph);
    for (cudaGraphExec_t graph : input_finish)
      if (graph) cudaGraphExecDestroy(graph);
    for (cudaGraphExec_t graph : input_local)
      if (graph) cudaGraphExecDestroy(graph);
    if (capture_stream) cudaStreamDestroy(capture_stream);
    if (head_handle) cublasDestroy(head_handle);
    cudaFree(arena.local_winners);
    cudaFree(arena.rank_winners);
    cudaFree(arena.proposal_tokens);
    cudaFree(arena.rank_logits);
    cudaFree(arena.token_hidden);
    cudaFree(arena.updated_multi_hidden);
    cudaFree(arena.final_injection);
    cudaFree(arena.reduced_moe_output);
    cudaFree(arena.fused_multi_hidden);
    cudaFree(arena.reduced_hidden);
    cudaFree(arena.reduced_embedding);
    cudaFree(arena.hidden_partial);
    cudaFree(arena.embedding_partial);
    cudaFree(arena.multi_hidden);
    cudaFree(arena.embedding);
  }
};

MtpGraphRuntime::MtpGraphRuntime(GraphRuntimeBinding binding)
    : impl_(new Impl) {
  const auto& layout = binding.layout;
  if (binding.device < 0 || (binding.rank != 0 && binding.rank != 1) ||
      !binding.target_rank_slab ||
      binding.target_rank_slab_bytes < output::kLmHead.length_bytes ||
      !binding.mtp_rank_slab || !valid_digest(binding.source_contract_digest) ||
      !valid_extent(layout.pre_fc_norm_embedding, kNormEmbeddingBytes,
                    binding.mtp_rank_slab_bytes) ||
      !valid_extent(layout.pre_fc_norm_hidden, kNormHiddenBytes,
                    binding.mtp_rank_slab_bytes) ||
      !valid_extent(layout.fc_embedding, kProjectionBytes,
                    binding.mtp_rank_slab_bytes) ||
      !valid_extent(layout.fc_hidden, kProjectionBytes,
                    binding.mtp_rank_slab_bytes) ||
      !valid_extent(layout.final_hc_norm, kNormHiddenBytes,
                    binding.mtp_rank_slab_bytes) ||
      !valid_extent(layout.final_hc_down, kProjectionBytes,
                    binding.mtp_rank_slab_bytes) ||
      !valid_extent(layout.final_hc_up, kProjectionBytes,
                    binding.mtp_rank_slab_bytes)) {
    delete impl_;
    impl_ = nullptr;
    throw std::invalid_argument("authenticated MTP graph binding changed");
  }
  try {
    impl_->binding = binding;
    check(cudaSetDevice(binding.device), "set device");
    auto allocate = [](auto** pointer, std::size_t elements) {
      check(cudaMalloc(pointer, elements * sizeof(**pointer)), "allocate arena");
    };
    allocate(&impl_->arena.embedding, 16 * kFusionHidden);
    allocate(&impl_->arena.multi_hidden, 16 * kFusionHyperHidden);
    allocate(&impl_->arena.embedding_partial, 16 * kFusionHidden);
    allocate(&impl_->arena.hidden_partial, 16 * kFusionHyperHidden);
    allocate(&impl_->arena.reduced_embedding, 16 * kFusionHidden);
    allocate(&impl_->arena.reduced_hidden, 16 * kFusionHyperHidden);
    allocate(&impl_->arena.fused_multi_hidden, 16 * kFusionHyperHidden);
    allocate(&impl_->arena.reduced_moe_output, 16 * kFusionHidden);
    allocate(&impl_->arena.final_injection, 16 * kFusionStreams);
    allocate(&impl_->arena.updated_multi_hidden, 16 * kFusionHyperHidden);
    allocate(&impl_->arena.token_hidden, 16 * kFusionHidden);
    allocate(&impl_->arena.rank_logits,
             16 * static_cast<std::size_t>(output::kLocalVocab));
    allocate(&impl_->arena.local_winners, 16);
    allocate(&impl_->arena.rank_winners, 16 * output::kTpSize);
    allocate(&impl_->arena.proposal_tokens, 16);
    check(cudaMemset(impl_->arena.token_hidden, 0,
                     16 * kFusionHidden * sizeof(__nv_bfloat16)),
          "initialize head warmup input");

    impl_->input = std::make_unique<InputFusionPlan>(
        binding.device, binding.rank,
        InputFusionWeights{
            at<const __nv_bfloat16>(binding.mtp_rank_slab,
                                    layout.pre_fc_norm_embedding),
            at<const __nv_bfloat16>(binding.mtp_rank_slab,
                                    layout.pre_fc_norm_hidden),
            at<const __nv_bfloat16>(binding.mtp_rank_slab,
                                    layout.fc_embedding),
            at<const __nv_bfloat16>(binding.mtp_rank_slab, layout.fc_hidden)});
    impl_->final = std::make_unique<hyperconnection::FinalPlan>(
        binding.device,
        at<const __nv_bfloat16>(binding.mtp_rank_slab, layout.final_hc_norm),
        at<const __nv_bfloat16>(binding.mtp_rank_slab, layout.final_hc_down),
        at<const __nv_bfloat16>(binding.mtp_rank_slab, layout.final_hc_up));
    if (cublasCreate(&impl_->head_handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("MTP graph runtime create head handle failed");
    check(cudaStreamCreateWithFlags(&impl_->capture_stream,
                                    cudaStreamNonBlocking),
          "create capture stream");

    // Initialize the cuBLAS head path before capture. Cold construction is the
    // only synchronization point owned by this runtime.
    if (output::lm_head(
            impl_->head_handle, impl_->arena.token_hidden,
            reinterpret_cast<const __nv_bfloat16*>(binding.target_rank_slab +
                                                    output::kLmHead.offset_bytes),
            impl_->arena.rank_logits, 2, binding.rank,
            impl_->capture_stream) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("MTP graph runtime head warmup failed");
    check(cudaStreamSynchronize(impl_->capture_stream), "complete head warmup");

    auto capture = [&](cudaGraphExec_t& executable, auto launch) {
      cudaGraph_t graph = nullptr;
      check(cudaStreamBeginCapture(impl_->capture_stream,
                                   cudaStreamCaptureModeThreadLocal),
            "begin capture");
      launch();
      check(cudaStreamEndCapture(impl_->capture_stream, &graph), "end capture");
      try {
        check(cudaGraphInstantiate(&executable, graph, 0), "instantiate graph");
      } catch (...) {
        cudaGraphDestroy(graph);
        throw;
      }
      cudaGraphDestroy(graph);
    };
    for (int index = 0; index < 5; ++index) {
      const int m = kBuckets[index];
      capture(impl_->input_local[index], [&] {
        impl_->input->local_project(
            impl_->arena.embedding, impl_->arena.multi_hidden,
            impl_->arena.embedding_partial, impl_->arena.hidden_partial, m,
            impl_->capture_stream);
      });
      capture(impl_->input_finish[index], [&] {
        impl_->input->finish(impl_->arena.reduced_embedding,
                             impl_->arena.reduced_hidden,
                             impl_->arena.fused_multi_hidden, m,
                             impl_->capture_stream);
      });
      capture(impl_->final_local[index], [&] {
        impl_->final->combine_and_collapse(
            impl_->arena.fused_multi_hidden,
            impl_->arena.reduced_moe_output, impl_->arena.final_injection,
            impl_->arena.updated_multi_hidden, impl_->arena.token_hidden, m,
            impl_->capture_stream);
        if (output::lm_head(
                impl_->head_handle, impl_->arena.token_hidden,
                reinterpret_cast<const __nv_bfloat16*>(
                    binding.target_rank_slab + output::kLmHead.offset_bytes),
                impl_->arena.rank_logits, m, binding.rank,
                impl_->capture_stream) != CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("MTP graph runtime capture head failed");
        check(output::local_argmax(impl_->arena.rank_logits,
                                   impl_->arena.local_winners, m, binding.rank,
                                   impl_->capture_stream),
              "capture local argmax");
      });
    }
  } catch (...) {
    delete impl_;
    impl_ = nullptr;
    throw;
  }
}

MtpGraphRuntime::~MtpGraphRuntime() { delete impl_; }

GraphArenaView MtpGraphRuntime::arena() const noexcept { return impl_->arena; }

void MtpGraphRuntime::launch_input_local(int m, cudaStream_t stream) {
  if (!stream) throw std::invalid_argument("MTP input-local stream changed");
  check(cudaGraphLaunch(impl_->input_local[bucket_index(m)], stream),
        "launch input local");
}

void MtpGraphRuntime::launch_input_finish(int m, cudaStream_t stream) {
  if (!stream) throw std::invalid_argument("MTP input-finish stream changed");
  check(cudaGraphLaunch(impl_->input_finish[bucket_index(m)], stream),
        "launch input finish");
}

void MtpGraphRuntime::launch_final_local(int m, cudaStream_t stream) {
  if (!stream) throw std::invalid_argument("MTP final-local stream changed");
  check(cudaGraphLaunch(impl_->final_local[bucket_index(m)], stream),
        "launch final local");
}

const std::int32_t* MtpGraphRuntime::enqueue_winner_exchange_and_greedy(
    WinnerExchangePort& exchange, int m, cudaStream_t stream) {
  if (!stream) throw std::invalid_argument("MTP winner-exchange stream changed");
  bucket_index(m);
  exchange.enqueue(impl_->arena.local_winners, impl_->arena.rank_winners, m,
                   impl_->binding.rank, stream);
  check(output::global_greedy(impl_->arena.rank_winners,
                              impl_->arena.proposal_tokens, m, 0.0F, 1.0F,
                              stream),
        "launch global greedy");
  return impl_->arena.proposal_tokens;
}

}  // namespace rocket::qwen38::mtp

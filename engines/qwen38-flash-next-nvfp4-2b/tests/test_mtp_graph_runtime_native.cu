#include "mtp/graph_runtime.h"
#include "mtp/nccl_winner_exchange.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace mtp = rocket::qwen38::mtp;

namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}
std::uint64_t append(mtp::TensorExtent& extent, std::uint64_t offset,
                     std::uint64_t bytes) {
  offset = (offset + 255) & ~std::uint64_t{255};
  extent = {offset, bytes};
  return offset + bytes;
}

__global__ void order_winners(const rocket::qwen38::output::Winner* local,
                              rocket::qwen38::output::Winner* ordered, int m,
                              int rank, float peer_value) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m) return;
  ordered[row * 2 + rank] = local[row];
  ordered[row * 2 + 1 - rank] =
      {peer_value, (1 - rank) * rocket::qwen38::output::kLocalVocab};
}

class FakeWinnerExchange final : public mtp::WinnerExchangePort {
 public:
  explicit FakeWinnerExchange(float peer_value) : peer_value_(peer_value) {}
  void enqueue(const rocket::qwen38::output::Winner* local,
               rocket::qwen38::output::Winner* ordered, int m, int rank,
               cudaStream_t stream) override {
    if (fault_) throw std::runtime_error("injected winner exchange fault");
    order_winners<<<(m + 31) / 32, 32, 0, stream>>>(local, ordered, m, rank,
                                                    peer_value_);
    cuda_check(cudaPeekAtLastError(), "enqueue fake winner exchange");
  }
  void validate_after_fence() override {
    if (fault_) throw std::runtime_error("injected winner exchange fault");
  }
  float peer_value_;
  bool fault_ = false;
};
}  // namespace

int main() {
  try {
    bool missing_nccl_rejected = false;
    try {
      mtp::NcclWinnerExchange unavailable(
          reinterpret_cast<void*>(1), 0, "/rocket/missing/libnccl.so");
    } catch (const std::runtime_error&) {
      missing_nccl_rejected = true;
    }
    check(missing_nccl_rejected, "missing NCCL library was accepted");
    mtp::NonexpertLayout layout{};
    std::uint64_t mtp_bytes = 0;
    mtp_bytes = append(layout.pre_fc_norm_embedding, mtp_bytes, 5'120);
    mtp_bytes = append(layout.pre_fc_norm_hidden, mtp_bytes, 20'480);
    mtp_bytes = append(layout.fc_embedding, mtp_bytes, 6'553'600);
    mtp_bytes = append(layout.fc_hidden, mtp_bytes, 6'553'600);
    mtp_bytes = append(layout.final_hc_norm, mtp_bytes, 20'480);
    mtp_bytes = append(layout.final_hc_down, mtp_bytes, 6'553'600);
    mtp_bytes = append(layout.final_hc_up, mtp_bytes, 6'553'600);
    mtp_bytes = (mtp_bytes + 255) & ~std::uint64_t{255};

    std::byte* target_slab = nullptr;
    std::byte* mtp_slab = nullptr;
    cuda_check(cudaMalloc(&target_slab,
                          rocket::qwen38::output::kLmHead.length_bytes),
               "allocate target slab");
    cuda_check(cudaMalloc(&mtp_slab, mtp_bytes), "allocate MTP slab");
    cuda_check(cudaMemset(target_slab, 0,
                          rocket::qwen38::output::kLmHead.length_bytes),
               "clear target slab");
    cuda_check(cudaMemset(mtp_slab, 0, mtp_bytes), "clear MTP slab");
    std::array<std::uint8_t, 32> digest{};
    digest[0] = 1;
    mtp::MtpGraphRuntime runtime({
        0,
        0,
        target_slab,
        rocket::qwen38::output::kLmHead.length_bytes,
        mtp_slab,
        static_cast<std::size_t>(mtp_bytes),
        digest,
        layout,
    });
    bool rejected = false;
    try {
      runtime.launch_input_local(3, nullptr);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check(rejected, "invalid graph bucket was accepted");
    const auto arena = runtime.arena();
    cuda_check(cudaMemset(arena.embedding, 0,
                          16 * mtp::kFusionHidden * sizeof(__nv_bfloat16)),
               "clear embedding");
    cuda_check(cudaMemset(arena.multi_hidden, 0,
                          16 * mtp::kFusionHyperHidden * sizeof(__nv_bfloat16)),
               "clear hidden");
    cuda_check(cudaMemset(arena.reduced_embedding, 0,
                          16 * mtp::kFusionHidden * sizeof(float)),
               "clear reduced embedding");
    cuda_check(cudaMemset(arena.reduced_hidden, 0,
                          16 * mtp::kFusionHyperHidden * sizeof(float)),
               "clear reduced hidden");
    cuda_check(cudaMemset(arena.reduced_moe_output, 0,
                          16 * mtp::kFusionHidden * sizeof(float)),
               "clear reduced MoE");
    cuda_check(cudaMemset(arena.final_injection, 0,
                          16 * mtp::kFusionStreams * sizeof(__nv_bfloat16)),
               "clear final injection");
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create stream");
    for (const int m : {1, 16}) {
      runtime.launch_input_local(m, stream);
      runtime.launch_input_finish(m, stream);
      runtime.launch_final_local(m, stream);
      runtime.launch_logits_local(m, stream);
      FakeWinnerExchange ties(0.0F);
      runtime.enqueue_winner_exchange_and_greedy(ties, m, stream);
      cuda_check(cudaStreamSynchronize(stream), "complete runtime graphs");
    }
    std::array<rocket::qwen38::output::Winner, 16> winners{};
    cuda_check(cudaMemcpy(winners.data(), arena.local_winners, sizeof(winners),
                          cudaMemcpyDeviceToHost),
               "copy winners");
    for (const auto winner : winners)
      check(winner.token == 0 && winner.value == 0.0F,
            "zero-weight local winner drift");
    std::array<std::int32_t, 16> tokens{};
    cuda_check(cudaMemcpy(tokens.data(), arena.proposal_tokens, sizeof(tokens),
                          cudaMemcpyDeviceToHost),
               "copy tie tokens");
    for (const int token : tokens)
      check(token == 0, "global greedy tie ordering drift");
    FakeWinnerExchange better_peer(1.0F);
    runtime.enqueue_winner_exchange_and_greedy(better_peer, 16, stream);
    cuda_check(cudaStreamSynchronize(stream), "complete better-peer greedy");
    cuda_check(cudaMemcpy(tokens.data(), arena.proposal_tokens, sizeof(tokens),
                          cudaMemcpyDeviceToHost),
               "copy better-peer tokens");
    for (const int token : tokens)
      check(token == rocket::qwen38::output::kLocalVocab,
            "global greedy rank ordering drift");
    cuda_check(cudaMemset(arena.proposal_tokens, 0xff, sizeof(tokens)),
               "poison proposal tokens");
    FakeWinnerExchange fault(0.0F);
    fault.fault_ = true;
    bool exchange_failed = false;
    try {
      runtime.enqueue_winner_exchange_and_greedy(fault, 16, stream);
    } catch (const std::runtime_error&) {
      exchange_failed = true;
    }
    check(exchange_failed, "winner exchange fault was accepted");
    cuda_check(cudaMemcpy(tokens.data(), arena.proposal_tokens, sizeof(tokens),
                          cudaMemcpyDeviceToHost),
               "copy unpublished tokens");
    for (const int token : tokens)
      check(token == -1, "exchange fault published proposal tokens");
    std::printf("qwen38_mtp_graph_runtime graphs=input_local,input_finish,final_local,logits_local winner_exchange=device buckets=1,16 result=match\n");
    cudaStreamDestroy(stream);
    cudaFree(mtp_slab);
    cudaFree(target_slab);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

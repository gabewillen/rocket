// SPDX-License-Identifier: Apache-2.0
#include "mtp/native_executor.h"
#include <cuda_runtime.h>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#undef assert
#define assert(x) do { if (!(x)) std::abort(); } while (false)
namespace mtp = rocket::qwen38::mtp;
namespace output = rocket::qwen38::output;

namespace {
std::uint64_t append(mtp::TensorExtent& e, std::uint64_t o, std::uint64_t n) {
  o = (o + 255) & ~std::uint64_t{255}; e = {o, n}; return o + n;
}
__global__ void order(const output::Winner* local, output::Winner* both,
                      int m, int rank) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) { both[2*i + rank] = local[i];
    both[2*i + 1-rank] = {0.0F, (1-rank)*output::kLocalVocab}; }
}
class Exchange final : public mtp::WinnerExchangePort { public:
  void enqueue(const output::Winner* local, output::Winner* both, int m,
               int rank, cudaStream_t s) override {
    order<<<1, 32, 0, s>>>(local, both, m, rank);
  }
  void validate_after_fence() override {}
};
class Middle final : public mtp::MtpMiddleStagePort { public:
  Middle(int sequences, int depth) : sequences_(sequences), depth_(depth) {
    assert(cudaMalloc(&tokens_, sequences * (depth + 1) * sizeof(std::int32_t)) == cudaSuccess);
    assert(cudaMalloc(&routes_, sequences * 8 * sizeof(std::int32_t)) == cudaSuccess);
    assert(cudaMemset(tokens_, 0, sequences * (depth + 1) * sizeof(std::int32_t)) == cudaSuccess);
    assert(cudaMemset(routes_, 0, sequences * 8 * sizeof(std::int32_t)) == cudaSuccess);
  }
  ~Middle() { cudaFree(routes_); cudaFree(tokens_); }
  const std::int32_t* prepare(mtp::GraphArenaView a, mtp::StateArena&,
                              mtp::GraphKey, cudaStream_t s) override {
    cudaMemsetAsync(a.embedding, 0, sequences_ * mtp::kFusionHidden * 2, s);
    cudaMemsetAsync(a.multi_hidden, 0, sequences_ * mtp::kFusionHyperHidden * 2, s);
    cudaMemsetAsync(a.final_injection, 0, sequences_ * mtp::kFusionStreams * 2, s);
    return tokens_;
  }
  void reduce_input(mtp::GraphArenaView a, int, mtp::GraphKey, cudaStream_t s) override {
    cudaMemsetAsync(a.reduced_embedding, 0, sequences_ * mtp::kFusionHidden * 4, s);
    cudaMemsetAsync(a.reduced_hidden, 0, sequences_ * mtp::kFusionHyperHidden * 4, s);
  }
  void stage_attention(mtp::GraphArenaView,
                       rocket::qwen38::attention::MtpQsaWriteView state,
                       int step, mtp::GraphKey key, cudaStream_t) override {
    assert(state.rows == sequences_);
    assert(step >= 0 && step < depth_);
    assert(key.query_tokens == 300);
    ++attention_calls_;
  }
  void reduce_attention(mtp::GraphArenaView, int, mtp::GraphKey,
                        cudaStream_t) override {}
  void stage_moe(mtp::GraphArenaView, int, mtp::GraphKey, cudaStream_t) override {}
  void reduce_moe(mtp::GraphArenaView a, int, mtp::GraphKey, cudaStream_t s) override {
    cudaMemsetAsync(a.reduced_moe_output, 0,
                    sequences_ * mtp::kFusionHidden * sizeof(float), s);
  }
  const std::int32_t* router_expert_ids(int) const noexcept override {
    return fault_routes_ ? nullptr : routes_;
  }
  void advance(mtp::GraphArenaView, mtp::StateArena&, int step, mtp::GraphKey,
               const std::int32_t* proposals, cudaStream_t s) override {
    cudaMemcpyAsync(tokens_ + (step + 1) * sequences_, proposals,
                    sequences_ * sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s);
  }
  int sequences_, depth_; std::int32_t *tokens_ = nullptr, *routes_ = nullptr;
  int attention_calls_ = 0;
  bool fault_routes_ = false;
};
class Sink final : public mtp::TelemetrySink { public:
  void record_phase(const mtp::PhaseMetric& m) noexcept override { phases.push_back(m); }
  void record_expert_usage(const mtp::ExpertUsageMetric& m) noexcept override { experts.push_back(m); }
  std::vector<mtp::PhaseMetric> phases; std::vector<mtp::ExpertUsageMetric> experts;
};
}

int main() {
  static_assert(mtp::allowed_graph_key({4, 16, 300}));
  static_assert(mtp::allowed_graph_key({4, 16, 8192}));
  static_assert(!mtp::allowed_graph_key({4, 16, 299}));
  static_assert(!mtp::allowed_graph_key({4, 16, 8193}));
  static_assert(!mtp::allowed_graph_key({5, 16, 300}));
  constexpr int sequences = 2, depth = 2;
  mtp::NonexpertLayout l{}; std::uint64_t bytes = 0;
  bytes=append(l.pre_fc_norm_embedding,bytes,5120); bytes=append(l.pre_fc_norm_hidden,bytes,20480);
  bytes=append(l.fc_embedding,bytes,6553600); bytes=append(l.fc_hidden,bytes,6553600);
  bytes=append(l.final_hc_norm,bytes,20480); bytes=append(l.final_hc_down,bytes,6553600);
  bytes=append(l.final_hc_up,bytes,6553600); bytes=(bytes+255)&~std::uint64_t{255};
  std::byte *target=nullptr,*slab=nullptr,*inactive=nullptr;
  assert(cudaMalloc(&target, output::kLmHead.length_bytes)==cudaSuccess);
  assert(cudaMalloc(&slab,bytes)==cudaSuccess); cudaMemset(target,0,output::kLmHead.length_bytes); cudaMemset(slab,0,bytes);
  std::array<std::uint8_t,32> digest{}; digest[0]=1;
  mtp::MtpGraphRuntime runtime({0,0,target,output::kLmHead.length_bytes,slab,
                                static_cast<std::size_t>(bytes),digest,l});
  mtp::StateArena state(sequences,depth,false); Middle middle(sequences,depth);
  Exchange exchange; Sink sink; cudaStream_t stream=nullptr; cudaStreamCreate(&stream);
  assert(cudaMalloc(&inactive,state.transaction_bytes())==cudaSuccess);
  cudaMemset(state.prefix(0).multi_hidden,0x11,sequences*mtp::kMtpMultiHidden*2);
  cudaMemset(state.prefix(1).multi_hidden,0x22,sequences*mtp::kMtpMultiHidden*2);
  {
    mtp::StateArena failed_state(sequences,depth,false);
    Middle failed_middle(sequences,depth); failed_middle.fault_routes_=true;
    mtp::NativeExecutor failed({depth, sequences, 300}, runtime, failed_middle, exchange,
                               failed_state,sink,stream);
    bool rejected=false;
    try { failed.draft(1); } catch (const mtp::NativeExecutorError&) { rejected=true; }
    assert(rejected && failed.phase()==mtp::ExecutorPhase::kFaulted);
    failed.commit(1);
    assert(failed.phase()==mtp::ExecutorPhase::kFaulted);
  }
  mtp::NativeExecutor executor(
      {depth, sequences, 300}, runtime, middle, exchange, state, sink, stream);
  bool stale_rejected = false;
  try { static_cast<void>(executor.draft(2)); }
  catch (const mtp::NativeExecutorError&) { stale_rejected = true; }
  assert(stale_rejected && middle.attention_calls_ == 0);
  const auto draft=executor.draft(1); assert(draft.depth==depth && draft.sequences==sequences);
  assert(middle.attention_calls_ == depth);
  std::array<std::int32_t,sequences> widths{1,3}; std::int32_t* dwidths=nullptr;
  cudaMalloc(&dwidths,sizeof(widths)); cudaMemcpy(dwidths,widths.data(),sizeof(widths),cudaMemcpyHostToDevice);
  executor.stage_accept(1,inactive,dwidths,{sequences,depth+1},stream);
  assert(cudaStreamSynchronize(stream)==cudaSuccess); executor.commit(1);
  executor.export_telemetry_after_fence(1); assert(sink.experts.size()==depth);
  assert(executor.phase()==mtp::ExecutorPhase::kReady);
  cudaFree(dwidths); cudaFree(inactive); cudaStreamDestroy(stream); cudaFree(slab); cudaFree(target);
  return 0;
}

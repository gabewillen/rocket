#include "decode/full_attention_graph.h"

#include <concepts>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace decode = rocket::qwen38::decode;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class Trace final : public decode::FullAttentionGraphOtelSink {
 public:
  void emit(const decode::FullAttentionGraphOtelRecord& record) noexcept override {
    records.push_back(record);
  }
  std::vector<decode::FullAttentionGraphOtelRecord> records;
};

struct Program {
  std::uint64_t* generation = nullptr;
  const __nv_bfloat16* output = nullptr;
  decode::FullAttentionLaunchShape last_shape{};
  std::vector<std::int32_t> accepted;
  bool fail_stage = false;
  bool staged = false;

  static int stage(void* opaque, const __nv_bfloat16* input,
                   decode::FullAttentionLaunchShape shape, cudaStream_t) {
    auto& self = *static_cast<Program*>(opaque);
    self.last_shape = shape;
    self.staged = input != nullptr;
    return self.fail_stage ? 1 : 0;
  }
  static int accept(void* opaque, const std::int32_t* lengths, int count,
                    std::uint64_t generation, cudaStream_t) {
    auto& self = *static_cast<Program*>(opaque);
    self.accepted.assign(lengths, lengths + count);
    *self.generation = generation;
    self.staged = false;
    return 0;
  }
  static int reset(void* opaque, cudaStream_t) {
    static_cast<Program*>(opaque)->staged = false;
    return 0;
  }
  static const __nv_bfloat16* projected(void* opaque) {
    return static_cast<Program*>(opaque)->output;
  }
  static const char* error(void*) { return "injected native failure"; }
};

decode::FullAttentionDeviceExtent address(std::uintptr_t value,
                                          std::uint64_t bytes) {
  return {reinterpret_cast<void*>(value), bytes};
}

decode::FullAttentionDeviceBindings bindings_for(Program& program,
                                                  std::uint64_t& generation) {
  decode::FullAttentionDeviceBindings value{};
  std::uintptr_t pointer = 0x1000;
  auto next = [&](std::uint64_t bytes) {
    pointer += 0x1000;
    return address(pointer, bytes);
  };
  value.q_weight = next(7864320);
  value.q_scale = next(983040);
  value.k_weight = next(327680);
  value.k_scale = next(40960);
  value.v_weight = next(327680);
  value.v_scale = next(40960);
  value.o_weight = next(3932160);
  value.o_scale = next(491520);
  value.q_norm = next(512);
  value.k_norm = next(512);
  value.index_qk_weight_first = next(1638400);
  value.index_qk_weight_second = next(1638400);
  value.index_q_norm = next(256);
  value.index_k_norm = next(256);
  value.main_state = next(decode::kFullAttentionMainStateBytes);
  value.raw_state = next(decode::kFullAttentionRawStateBytes);
  value.compressed_state = next(decode::kFullAttentionCompressedStateBytes);
  value.projected_output = next(655360);
  value.rope_positions = next(3072);
  value.logical_positions = next(1024);
  value.sequence_lengths = next(64);
  value.token_to_request = next(512);
  value.query_start_offsets = next(68);
  value.active_state_generation = &generation;
  program.generation = &generation;
  program.output = static_cast<const __nv_bfloat16*>(value.projected_output.pointer);
  value.program = {&program, Program::stage, Program::accept, Program::reset,
                   Program::projected, Program::error};
  return value;
}

decode::FullAttentionIdentity identity_for(int rank, int layer,
                                           std::uint64_t generation) {
  return {
      .revision = "fc694b54fb0174e0913e6adf86691ef85a4ead47",
      .artifact_key =
          "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4",
      .weight_chunk_sha256 =
          "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      .peer_index_chunk_sha256 =
          "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0",
      .state_commit_sha256 =
          "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
      .rank = rank,
      .layer = layer,
      .state_generation = generation,
  };
}

}  // namespace

int main() {
  try {
    static_assert(std::derived_from<decode::RankLocalFullAttentionGraph,
                                    decode::FullAttentionGraph>);
    check(decode::is_full_attention_layer(3) &&
              decode::is_full_attention_layer(47) &&
              !decode::is_full_attention_layer(2),
          "fixed full-attention layer set drift");

    decode::FullAttentionDeviceBindings missing{};
    bool rejected = false;
    try {
      decode::RankLocalFullAttentionGraph graph(
          identity_for(0, 3, 7), missing, nullptr);
    } catch (const decode::FullAttentionGraphContractError& error) {
      rejected = std::string_view(error.what()).find("device bindings") !=
                 std::string_view::npos;
    }
    check(rejected, "missing concrete graph bindings did not fail closed");

    std::uint64_t generation = 7;
    Program program;
    Trace trace;
    decode::RankLocalFullAttentionGraph graph(
        identity_for(1, 47, generation), bindings_for(program, generation),
        &trace);
    const auto stream = reinterpret_cast<cudaStream_t>(0x1230);
    const auto* input = reinterpret_cast<const __nv_bfloat16*>(0x5670);
    graph.launch(input, {16, 5, 80}, stream);
    check(graph.staged() && graph.projected_output() == program.output &&
              program.last_shape.token_rows == 80,
          "K4 verifier stage did not preserve the graph shape");
    const std::int32_t accepted[16] = {5, 4, 3, 2, 1, 0, 5, 4,
                                       3, 2, 1, 0, 5, 4, 3, 2};
    bool stream_rejected = false;
    try {
      graph.accept(accepted, 16, 8, reinterpret_cast<cudaStream_t>(0x1240));
    } catch (const decode::FullAttentionGraphContractError&) {
      stream_rejected = true;
    }
    check(stream_rejected && graph.staged(),
          "accepted state crossed the staged CUDA stream");
    graph.accept(accepted, 16, 8, stream);
    check(!graph.staged() && graph.state_generation() == 8 && generation == 8 &&
              program.accepted.size() == 16,
          "accepted prefixes did not publish one state generation");

    graph.launch(input, {4, 8, 32}, stream);
    graph.reset(stream);
    check(!graph.staged() && generation == 8,
          "rejected verifier rows changed active state generation");

    program.fail_stage = true;
    bool terminal = false;
    try {
      graph.launch(input, {1, 1, 1}, stream);
    } catch (const decode::FullAttentionGraphContractError& error) {
      terminal = std::string_view(error.what()).find("injected native failure") !=
                 std::string_view::npos;
    }
    check(terminal && graph.faulted() && graph.projected_output() == nullptr,
          "native uncertainty did not fault the concrete graph");
    check(trace.records.size() == 6 &&
              trace.records.front().stage ==
                  decode::FullAttentionGraphStage::kValidate &&
              trace.records.back().outcome ==
                  decode::FullAttentionGraphOutcome::kError,
          "bounded graph telemetry contract drift");

    std::puts("qwen38 concrete full-attention graph contract passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

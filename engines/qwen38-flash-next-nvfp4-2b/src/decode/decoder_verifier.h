// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "decode/execution.h"
#include "linear_attention/gdn_verifier.h"
#include "pair_reduce/otel.h"

namespace rocket::qwen38::decode {

inline constexpr int kDecoderLayers = 48;
inline constexpr int kDecoderGdnLayers = 36;
inline constexpr int kDecoderQsaLayers = 12;
inline constexpr int kDecoderMaxSequences = 16;

[[nodiscard]] constexpr bool is_qsa_layer(int layer) noexcept {
  return layer >= 0 && layer < kDecoderLayers && layer % 4 == 3;
}

struct DecoderVerifierShape {
  int sequences;
  int verify_width;
  [[nodiscard]] constexpr int token_rows() const noexcept {
    return sequences * verify_width;
  }
};

struct GdnInactiveState {
  __nv_bfloat16* convolution;
  float* recurrent;
  const std::int32_t* authenticated_slots;
};

struct VerificationOutput {
  std::array<std::int32_t, kDecoderMaxSequences> tokens{};
  std::array<std::int32_t, kDecoderMaxSequences> accepted_prefixes{};
  int sequences = 0;
  // Runtime-owned device result used by same-stream accepted-state publishers.
  // Its values are identical to accepted_prefixes and include the target token.
  const std::int32_t* accepted_prefixes_device = nullptr;
};

class DecoderVerifierError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Exact adapter over linear_attention::GdnVerifier. Production construction
// binds one instance per GDN layer. Tests may substitute this port without
// claiming kernel execution.
class GdnVerifierPort {
 public:
  virtual ~GdnVerifierPort() = default;
  virtual void stage(const __nv_bfloat16* position_major_input,
                     GdnInactiveState inactive, DecoderVerifierShape shape,
                     std::string_view trace_id, std::string_view request_id,
                     cudaStream_t stream) = 0;
  virtual const __nv_bfloat16* staged_output() const noexcept = 0;
  virtual void accept(const std::int32_t* accepted_prefixes,
                      std::string_view trace_id, std::string_view request_id,
                      cudaStream_t stream) = 0;
  virtual void reset(std::string_view trace_id,
                     std::string_view request_id) noexcept = 0;
};

class NativeGdnVerifierPort final : public GdnVerifierPort {
 public:
  explicit NativeGdnVerifierPort(
      linear_attention::GdnVerifier& verifier) noexcept
      : verifier_(verifier) {}
  void stage(const __nv_bfloat16* position_major_input,
             GdnInactiveState inactive, DecoderVerifierShape shape,
             std::string_view trace_id, std::string_view request_id,
             cudaStream_t stream) override;
  const __nv_bfloat16* staged_output() const noexcept override;
  void accept(const std::int32_t* accepted_prefixes,
              std::string_view trace_id, std::string_view request_id,
              cudaStream_t stream) override;
  void reset(std::string_view trace_id,
             std::string_view request_id) noexcept override;

 private:
  linear_attention::GdnVerifier& verifier_;
};

// Owns inactive state storage and the one atomic active-pointer publication.
// begin() returns a private token. publish() is noexcept and is the only method
// allowed to change active_generation(). discard() never changes active state.
class DecoderStateTransaction {
 public:
  virtual ~DecoderStateTransaction() = default;
  virtual std::uint64_t active_generation() const noexcept = 0;
  virtual void* begin(std::uint64_t generation,
                      DecoderVerifierShape shape) = 0;
  virtual GdnInactiveState gdn_state(void* transaction, int layer) = 0;
  virtual std::byte* mtp_state(void*, std::size_t) { return nullptr; }
  virtual void publish(void* transaction) noexcept = 0;
  virtual void discard(void* transaction) noexcept = 0;
};

class AcceptedStateParticipant {
 public:
  virtual ~AcceptedStateParticipant() = default;
  virtual std::size_t state_bytes_per_sequence() const noexcept = 0;
  virtual void stage_accept(std::uint64_t generation, std::byte* inactive_state,
                            const std::int32_t* accepted_widths_device,
                            DecoderVerifierShape shape,
                            cudaStream_t stream) = 0;
  virtual void commit(std::uint64_t generation) noexcept = 0;
  virtual void discard(std::uint64_t generation) noexcept = 0;
};

// Native CUDA work surface. Every method is synchronous enqueue onto the bound
// stream; synchronize() is the sole fence. QSA accepted state must target the
// same inactive transaction and remain invisible until transaction publication.
class DecoderStepRuntime {
 public:
  virtual ~DecoderStepRuntime() = default;
  virtual const __nv_bfloat16* embed(const std::int32_t* token_ids,
                                     DecoderVerifierShape shape,
                                     cudaStream_t stream) = 0;
  virtual const __nv_bfloat16* gdn_input(int layer,
                                        const __nv_bfloat16* hidden,
                                        DecoderVerifierShape shape,
                                        cudaStream_t stream) = 0;
  virtual const __nv_bfloat16* consume_attention(
      int layer, const __nv_bfloat16* attention_output,
      DecoderVerifierShape shape, cudaStream_t stream) = 0;
  virtual const __nv_bfloat16* stage_qsa(
      int layer, const __nv_bfloat16* hidden, void* inactive_transaction,
      DecoderVerifierShape shape, cudaStream_t stream) = 0;
  virtual const __nv_bfloat16* pair_reduce(
      int layer, ReductionKind kind, const __nv_bfloat16* partial,
      DecoderVerifierShape shape, cudaStream_t stream) = 0;
  virtual const __nv_bfloat16* stage_moe(
      int layer, const __nv_bfloat16* hidden, DecoderVerifierShape shape,
      cudaStream_t stream) = 0;
  virtual void produce_logits(const __nv_bfloat16* hidden,
                              DecoderVerifierShape shape,
                              cudaStream_t stream) = 0;
  virtual VerificationOutput sample_and_verify(
      DecoderVerifierShape shape, cudaStream_t stream) = 0;
  virtual void accept_qsa(void* inactive_transaction,
                          const std::int32_t* accepted_prefixes,
                          DecoderVerifierShape shape,
                          cudaStream_t stream) = 0;
  virtual void reset_qsa(void* inactive_transaction) noexcept = 0;
  virtual void synchronize(cudaStream_t stream) = 0;
};

enum class DecoderVerifierPhase : std::uint8_t { kReady, kActive, kFaulted };

class DecoderVerifier final {
 public:
  // gdn_by_layer has non-null entries exactly at the 36 non-QSA positions.
  // All dependencies and the stream are borrowed for this object's lifetime.
  DecoderVerifier(std::array<GdnVerifierPort*, kDecoderLayers> gdn_by_layer,
                  DecoderStepRuntime& runtime,
                  DecoderStateTransaction& state,
                  pair_reduce::OtelStageSink& telemetry,
                  cudaStream_t stream,
                  AcceptedStateParticipant* accepted_state = nullptr);

  VerificationOutput step(
      std::uint64_t generation, const std::int32_t* token_ids,
      DecoderVerifierShape shape, std::string_view trace_id,
      std::string_view request_id);

  [[nodiscard]] DecoderVerifierPhase phase() const noexcept { return phase_; }

 private:
  void emit(std::string_view stage, pair_reduce::Outcome outcome,
            DecoderVerifierShape shape, std::string_view trace_id,
            std::string_view request_id, std::uint64_t duration_ns) noexcept;

  std::array<GdnVerifierPort*, kDecoderLayers> gdn_;
  DecoderStepRuntime& runtime_;
  DecoderStateTransaction& state_;
  pair_reduce::OtelStageSink& telemetry_;
  cudaStream_t stream_;
  AcceptedStateParticipant* accepted_state_;
  DecoderVerifierPhase phase_ = DecoderVerifierPhase::kReady;
};

}  // namespace rocket::qwen38::decode

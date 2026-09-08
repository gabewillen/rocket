// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_full_layer.h"

namespace rocket::qwen38::decode {

inline constexpr int kTargetLayer3OracleRows = 35;
inline constexpr int kTargetLayer3HiddenStreams = 4;

struct TargetLayer3RowBuffers {
  __nv_bfloat16* attention_input;
  __nv_bfloat16* attention_injection;
  float* reduced_attention;
  __nv_bfloat16* post_attention_hidden;
  __nv_bfloat16* moe_input;
  __nv_bfloat16* moe_injection;
  float* reduced_moe;
};

class TargetLayer3GenerationOwner {
 public:
  virtual ~TargetLayer3GenerationOwner() = default;
  virtual int rank() const noexcept = 0;
  virtual int layer() const noexcept = 0;
  virtual bool authenticated() const noexcept = 0;
  // Host-only state lookup. The executor validates it before enqueue_prepare.
  virtual const attention::TargetQsaStateView& view(
      int row, std::uint64_t generation) const = 0;
  virtual void enqueue_prepare(
      int row, std::uint64_t generation, cudaStream_t stream) = 0;
};

class TargetLayer3RowPort {
 public:
  virtual ~TargetLayer3RowPort() = default;
  virtual int rank() const noexcept = 0;
  virtual int layer() const noexcept = 0;
  virtual bool authenticated() const noexcept = 0;
  virtual TargetFullLayerResult execute_row(
      std::uint64_t generation, const attention::TargetQsaStateView& state,
      const __nv_bfloat16* replicated_pre_layer,
      const TargetLayer3RowBuffers& buffers,
      __nv_bfloat16* replicated_post_layer, cudaStream_t stream) = 0;
};

class TargetLayer3Comparator {
 public:
  virtual ~TargetLayer3Comparator() = default;
  virtual bool authenticated() const noexcept = 0;
  virtual bool compare_row34(const __nv_bfloat16* replicated_post_layer,
                             cudaStream_t stream) = 0;
};

class TargetLayer3ReductionGeneration {
 public:
  virtual ~TargetLayer3ReductionGeneration() = default;
  virtual int rank() const noexcept = 0;
  virtual bool authenticated() const noexcept = 0;
  virtual void begin_row(int row, std::uint64_t generation) = 0;
  virtual bool complete() const noexcept = 0;
};

class NativeTargetLayer3RowPort final : public TargetLayer3RowPort {
 public:
  explicit NativeTargetLayer3RowPort(TargetFullLayer& layer) noexcept
      : layer_(layer), rank_(layer.rank()) {}
  int rank() const noexcept override { return rank_; }
  int layer() const noexcept override { return 3; }
  bool authenticated() const noexcept override { return true; }
  TargetFullLayerResult execute_row(
      std::uint64_t generation, const attention::TargetQsaStateView& state,
      const __nv_bfloat16* replicated_pre_layer,
      const TargetLayer3RowBuffers& buffers,
      __nv_bfloat16* replicated_post_layer, cudaStream_t stream) override;
 private:
  TargetFullLayer& layer_;
  int rank_ = -1;
};

class TargetLayer3Prefill final {
 public:
  TargetLayer3Prefill(int rank, TargetLayer3RowPort& rows,
                      TargetLayer3GenerationOwner& generations,
                      TargetLayer3ReductionGeneration& reductions,
                      TargetLayer3Comparator& comparator,
                      pair_reduce::OtelStageSink& telemetry);
  const __nv_bfloat16* execute(
      const __nv_bfloat16* replicated_layer02,
      __nv_bfloat16* replicated_layer03,
      const TargetLayer3RowBuffers& buffers, cudaStream_t stream);
 private:
  int rank_;
  TargetLayer3RowPort& rows_;
  TargetLayer3GenerationOwner& generations_;
  TargetLayer3ReductionGeneration& reductions_;
  TargetLayer3Comparator& comparator_;
  pair_reduce::OtelStageSink& telemetry_;
  bool completed_ = false;
  bool faulted_ = false;
};

}  // namespace rocket::qwen38::decode

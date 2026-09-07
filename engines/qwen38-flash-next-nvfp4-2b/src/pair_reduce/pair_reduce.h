#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include "pair_reduce/otel.h"
#include "pair_reduce/transport.h"

namespace rocket::qwen38::pair_reduce {

inline constexpr int kWorldSize = 2;
inline constexpr int kHidden = 2'560;
inline constexpr std::size_t kPageBytes = 65'536;
inline constexpr int kAllowedM[] = {1, 2, 4, 8, 16};
inline constexpr std::string_view kDtype = "bf16_fp32";

class PairReduceError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
class PairReduceContractError final : public PairReduceError {
 public:
  using PairReduceError::PairReduceError;
};
class PairReduceCudaError final : public PairReduceError {
 public:
  using PairReduceError::PairReduceError;
};
class PairReduceTransportError final : public PairReduceError {
 public:
  using PairReduceError::PairReduceError;
};

struct alignas(64) WireHeader {
  std::uint64_t magic;
  std::uint64_t sequence;
  std::uint32_t schema;
  std::uint32_t rank;
  std::uint32_t world_size;
  std::uint32_t page_bytes;
  std::uint32_t hidden;
  std::uint32_t m;
  std::uint32_t dtype;
  std::uint32_t payload_bytes;
  std::uint64_t reserved[2];
};
static_assert(sizeof(WireHeader) == 64);

// Owns one four-page anonymous cudaHostAlloc region registered with Transport.
// reduce() borrows device input/output only until it returns. The object is
// single-threaded and non-reentrant. This header is the canonical contract,
// owned by this Qwen engine with no cross-engine compatibility promise.
// Success leaves FP32 [M,2560] on output; both ranks perform rank-0 then rank-1
// addition. Contract, transport, and CUDA failures use the corresponding typed
// PairReduceError subtype and emit a failed OTEL stage; output is then unspecified.
class PairReduce final {
 public:
  PairReduce(Transport& transport, OtelStageSink& telemetry);
  ~PairReduce();
  PairReduce(const PairReduce&) = delete;
  PairReduce& operator=(const PairReduce&) = delete;

  void reduce(const __nv_bfloat16* input, float* output, int m,
              std::string_view trace_id, std::string_view request_id,
              cudaStream_t stream = nullptr);

  static constexpr std::size_t slot_bytes() noexcept { return 2 * kPageBytes; }
  static constexpr std::size_t region_bytes() noexcept { return 2 * slot_bytes(); }
  static constexpr std::size_t peer_offset() noexcept { return slot_bytes(); }

 private:
  Transport& transport_;
  OtelStageSink& telemetry_;
  void* region_ = nullptr;
  void* device_region_ = nullptr;
  int region_handle_ = -1;
  std::uint64_t last_sequence_ = 0;
};

}  // namespace rocket::qwen38::pair_reduce

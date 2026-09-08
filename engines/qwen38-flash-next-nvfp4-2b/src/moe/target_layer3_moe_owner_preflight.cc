// SPDX-License-Identifier: Apache-2.0
#include "moe/target_layer3_moe_owner.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>

namespace decode = rocket::qwen38::decode;
namespace moe = rocket::qwen38::moe;

namespace {

thread_local std::string last_error;

class FullSink final : public moe::TargetFullMoeOtelSink {
 public:
  void emit(const moe::TargetFullMoeOtelPoint&) noexcept override {
    ++records;
  }
  std::uint64_t records = 0;
};

class StageSink final : public moe::TargetMoeStageOtelSink {
 public:
  void add_counter(const moe::TargetMoeStageOtelPoint&) noexcept override {
    ++records;
  }
  std::uint64_t records = 0;
};

}  // namespace

// Startup-only physical proof. It authenticates the descriptor and accepted
// loader capability, allocates the fixed E10/E11 owner storage, constructs the
// generated AOT participant/module, and destroys it without wait/enqueue.
extern "C" __attribute__((visibility("default"))) int
qwen38_target_layer3_moe_owner_preflight(
    int device, int rank, const char* descriptor_path,
    void* accepted_loader_lease_handle, std::uint64_t* stage_bytes,
    std::uint64_t* runtime_bytes, std::uint64_t* full_otel_records,
    std::uint64_t* stage_otel_records) noexcept {
  last_error.clear();
  if (!descriptor_path || !accepted_loader_lease_handle || !stage_bytes ||
      !runtime_bytes || !full_otel_records || !stage_otel_records) {
    last_error = "preflight argument contract changed";
    return 1;
  }
  try {
    auto plan = decode::load_target_layer3_native_plan(descriptor_path);
    if (rank != plan.rank || device < 0)
      throw std::invalid_argument("preflight rank/device identity changed");
    auto full = std::make_shared<FullSink>();
    auto stage = std::make_shared<StageSink>();
    {
      auto owner = moe::TargetLayer3MoeDeviceOwner::create(
          device, plan, accepted_loader_lease_handle, full, stage);
      if (!owner || !owner->requested_generation() ||
          !owner->workspace().routed_stage.w13_packed)
        throw std::runtime_error("preflight owner construction incomplete");
      *stage_bytes = moe::kTargetMoeStageScratchBytes;
      *runtime_bytes = moe::kTargetLayer3MoeRuntimeBytes;
    }
    *full_otel_records = full->records;
    *stage_otel_records = stage->records;
    return 0;
  } catch (const std::exception& error) {
    last_error = std::string(error.what()).substr(0, 384);
  } catch (...) {
    last_error = "unknown native preflight failure";
  }
  return 1;
}

extern "C" __attribute__((visibility("default"))) const char*
qwen38_target_layer3_moe_owner_preflight_last_error() noexcept {
  return last_error.c_str();
}

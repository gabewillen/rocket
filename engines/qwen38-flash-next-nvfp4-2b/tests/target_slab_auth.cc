// SPDX-License-Identifier: Apache-2.0
#include "model/target_slab_owner.h"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace model = rocket::qwen38::model;

struct Sink final : model::TargetSlabTelemetrySink {
  void emit(const model::TargetSlabTelemetryRecord& value) noexcept override {
    ++records;
    last = value;
  }
  int records = 0;
  model::TargetSlabTelemetryRecord last{};
};

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    const std::string_view rank_text(argv[1]);
    if (rank_text != "0" && rank_text != "1")
      throw std::invalid_argument("rank argument changed");
    const int rank = rank_text == "0" ? 0 : 1;
    const auto metadata = model::authenticate_target_slab_metadata(argv[2], rank);
    if (!metadata.chunks || metadata.chunks->size() != model::kTargetSlabChunks)
      throw std::logic_error("chunk publication changed");
    Sink sink;
    try {
      (void)model::TargetSlabDeviceOwner::load(-1, rank, argv[2], sink);
      throw std::logic_error("invalid-device load unexpectedly succeeded");
    } catch (const std::invalid_argument&) {
    }
    if (sink.records != 1 || sink.last.success || sink.last.rank != rank ||
        sink.last.phase != model::TargetSlabLoadPhase::kValidate ||
        sink.last.failure != model::TargetSlabFailureClass::kContract)
      throw std::logic_error("terminal failure telemetry changed");
    std::cout << "target_slab_auth rank=" << metadata.rank
              << " bytes=" << model::kTargetSlabBytes
              << " chunks=" << metadata.chunks->size()
              << " ring_depth=" << model::kTargetSlabRingDepth
              << " failure_telemetry=" << sink.records
              << " valid=1 complete=1\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "target_slab_auth valid=0 complete=0 phase=metadata reason="
              << error.what() << '\n';
    return 1;
  }
}

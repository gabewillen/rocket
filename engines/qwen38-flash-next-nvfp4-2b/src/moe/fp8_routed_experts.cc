// SPDX-License-Identifier: Apache-2.0
#include "moe/fp8_routed_experts.h"

namespace rocket::qwen38::moe {

RoutedExpertOutcome validate_routed_expert_summary(
    const RoutedExpertValidation& validation) noexcept {
  const auto& summary = validation.summary;
  if (!allowed_shape(validation.shape) || validation.requested_generation == 0)
    return RoutedExpertOutcome::kContractError;
  if (summary.outcome != RoutedExpertOutcome::kOk) return summary.outcome;
  if (summary.generation != validation.requested_generation)
    return RoutedExpertOutcome::kStaleGeneration;
  if (summary.active_experts < 0 || summary.active_experts > kLocalExperts ||
      summary.active_routes < 0 ||
      summary.active_routes > validation.shape.rows * kTopK ||
      summary.active_weight_bytes !=
          static_cast<std::uint64_t>(summary.active_experts) *
              kFp8BytesPerExpert ||
      summary.fc1_tiles != static_cast<std::uint32_t>(summary.active_routes) * 5 ||
      summary.fc2_tiles !=
          static_cast<std::uint32_t>(summary.active_routes) * 20)
    return RoutedExpertOutcome::kContractError;
  return RoutedExpertOutcome::kOk;
}

void export_routed_expert_otel_after_fence(
    const RoutedExpertDeviceSummary& snapshot,
    std::uint64_t requested_generation, RouteCompactionShape shape,
    RoutedExpertOtelSink& sink) noexcept {
  const auto outcome = validate_routed_expert_summary(
      {.summary = snapshot,
       .requested_generation = requested_generation,
       .shape = shape});
  const bool valid = outcome == RoutedExpertOutcome::kOk;
  const RoutedExpertOtelPoint base{.counter = RoutedExpertCounter::kActiveExperts,
                                  .outcome = outcome,
                                  .rank = allowed_shape(shape) ? shape.rank : 0,
                                  .sequence_bucket =
                                      allowed_shape(shape) ? shape.sequences : 1,
                                  .value = 0};
  auto emit = [&](RoutedExpertCounter counter, std::uint64_t value) {
    auto point = base;
    point.counter = counter;
    point.value = valid ? value : 0;
    sink.add_routed_expert_counter(point);
  };
  emit(RoutedExpertCounter::kActiveExperts, snapshot.active_experts);
  emit(RoutedExpertCounter::kActiveRoutes, snapshot.active_routes);
  emit(RoutedExpertCounter::kActiveWeightBytes, snapshot.active_weight_bytes);
  emit(RoutedExpertCounter::kFc1Tiles, snapshot.fc1_tiles);
  emit(RoutedExpertCounter::kFc2Tiles, snapshot.fc2_tiles);
}

}  // namespace rocket::qwen38::moe

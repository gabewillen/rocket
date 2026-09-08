// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace rocket::qwen38::decode {

// Caller-owned, allocation-free construction evidence. Values are closed and
// bounded; exception text and dependency identities never cross the C ABI.
enum class TargetK0StartupConstructionStage : std::uint8_t {
  kNone,
  kDependencyValidation,
  kTokenIoPairReduceRegistration,
  kTokenizerReauthentication,
  kExecutorOwnershipValidation,
  kFinalPublication,
};

inline void target_k0_enter_startup_construction(
    TargetK0StartupConstructionStage* progress,
    TargetK0StartupConstructionStage stage) noexcept {
  if (progress) *progress = stage;
}

}  // namespace rocket::qwen38::decode

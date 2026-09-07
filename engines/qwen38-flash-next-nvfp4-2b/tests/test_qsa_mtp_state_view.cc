// SPDX-License-Identifier: Apache-2.0
#include "attention/qsa_mtp_state_view.h"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace attention = rocket::qwen38::attention;

namespace {
void check(bool condition) {
  if (!condition) throw std::runtime_error("QSA MTP state-view contract failed");
}
}  // namespace

int main() {
  static_assert(attention::allowed_mtp_qsa_rows(1));
  static_assert(attention::allowed_mtp_qsa_rows(16));
  static_assert(!attention::allowed_mtp_qsa_rows(3));
  static_assert(!attention::allowed_mtp_qsa_rows(128));
  constexpr int rows = 16;
  std::array<__nv_bfloat16, rows * attention::kMtpQsaMainWidth> main_key{};
  std::array<__nv_bfloat16, rows * attention::kMtpQsaMainWidth> main_value{};
  std::array<__nv_bfloat16, rows * attention::kMtpQsaIndexerWidth> raw_key{};
  std::array<__nv_bfloat16, rows * attention::kMtpQsaIndexerWidth>
      compressed_key{};
  std::array<std::int64_t, rows * attention::kMtpQsaMropeAxes> rope{};
  std::array<std::int32_t, rows> main_slots{}, raw_slots{}, compressed_slots{},
      compressed_valid{};

  attention::MtpQsaPrefixStateView state{
      main_key.data(),          main_value.data(), raw_key.data(),
      compressed_key.data(),    rope.data(),       main_slots.data(),
      raw_slots.data(),         compressed_slots.data(),
      compressed_valid.data(),  rows,              true};
  const auto bound = attention::bind_mtp_qsa_write_view(state, rows, true);
  check(bound.key == main_key.data());
  check(bound.value == main_value.data());
  check(bound.raw_key == raw_key.data());
  check(bound.compressed_key == compressed_key.data());
  check(bound.rope_positions == rope.data());
  check(bound.main_slots == main_slots.data());
  check(bound.raw_slots == raw_slots.data());
  check(bound.compressed_slots == compressed_slots.data());
  check(bound.compressed_valid == compressed_valid.data());
  check(bound.rows == rows);

  bool row_drift_rejected = false;
  try {
    static_cast<void>(attention::bind_mtp_qsa_write_view(state, rows - 1, true));
  } catch (const attention::MtpQsaStateViewError&) {
    row_drift_rejected = true;
  }
  check(row_drift_rejected);

  bool missing_family_rejected = false;
  state.compressed_key = nullptr;
  try {
    static_cast<void>(attention::bind_mtp_qsa_write_view(state, rows, true));
  } catch (const attention::MtpQsaStateViewError&) {
    missing_family_rejected = true;
  }
  check(missing_family_rejected);

  state.compressed_key = compressed_key.data();
  bool mrope_drift_rejected = false;
  try {
    static_cast<void>(attention::bind_mtp_qsa_write_view(state, rows, false));
  } catch (const attention::MtpQsaStateViewError&) {
    mrope_drift_rejected = true;
  }
  check(mrope_drift_rejected);

  state.rope_positions = nullptr;
  state.uses_mrope = false;
  const auto ordinary =
      attention::bind_mtp_qsa_write_view(state, rows, false);
  check(ordinary.rope_positions == nullptr);
  return 0;
}

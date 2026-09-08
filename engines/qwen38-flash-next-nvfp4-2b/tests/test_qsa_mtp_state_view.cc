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
  static_assert(attention::allowed_mtp_qsa_rows(2));
  static_assert(attention::allowed_mtp_qsa_rows(4));
  static_assert(attention::allowed_mtp_qsa_rows(8));
  static_assert(attention::allowed_mtp_qsa_rows(16));
  static_assert(!attention::allowed_mtp_qsa_rows(3));
  static_assert(!attention::allowed_mtp_qsa_rows(128));
  static_assert(attention::allowed_mtp_qsa_query_tokens(300));
  static_assert(attention::allowed_mtp_qsa_query_tokens(8192));
  static_assert(!attention::allowed_mtp_qsa_query_tokens(299));
  static_assert(!attention::allowed_mtp_qsa_query_tokens(8193));
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

  rocket::qwen38::mtp::PrefixStateView arena_state{
      nullptr,                  main_key.data(),       main_value.data(),
      raw_key.data(),           compressed_key.data(), rope.data(),
      main_slots.data(),        raw_slots.data(),      compressed_slots.data(),
      compressed_valid.data()};
  const attention::MtpQsaArenaIdentity identity{
      rows, 4, 300, 2, true, 17, 17};
  const auto arena_bound =
      attention::bind_mtp_qsa_write_view(arena_state, identity);
  check(arena_bound.key == arena_state.main_key);
  check(arena_bound.value == arena_state.main_value);
  check(arena_bound.raw_key == arena_state.raw_key);
  check(arena_bound.compressed_key == arena_state.compressed_key);
  check(arena_bound.rope_positions == arena_state.rope_positions);
  check(arena_bound.main_slots == arena_state.main_slots);
  check(arena_bound.raw_slots == arena_state.raw_slots);
  check(arena_bound.compressed_slots == arena_state.compressed_slots);
  check(arena_bound.compressed_valid == arena_state.compressed_valid);

  for (const auto invalid : std::array<attention::MtpQsaArenaIdentity, 7>{
           attention::MtpQsaArenaIdentity{3, 4, 300, 2, true, 17, 17},
           attention::MtpQsaArenaIdentity{rows, 0, 300, 0, true, 17, 17},
           attention::MtpQsaArenaIdentity{rows, 4, 299, 2, true, 17, 17},
           attention::MtpQsaArenaIdentity{rows, 4, 8193, 2, true, 17, 17},
           attention::MtpQsaArenaIdentity{rows, 4, 300, 4, true, 17, 17},
           attention::MtpQsaArenaIdentity{rows, 4, 300, 2, true, 0, 0},
           attention::MtpQsaArenaIdentity{rows, 4, 300, 2, true, 17, 18}}) {
    bool identity_rejected = false;
    try {
      static_cast<void>(
          attention::bind_mtp_qsa_write_view(arena_state, invalid));
    } catch (const attention::MtpQsaStateViewError&) {
      identity_rejected = true;
    }
    check(identity_rejected);
  }

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

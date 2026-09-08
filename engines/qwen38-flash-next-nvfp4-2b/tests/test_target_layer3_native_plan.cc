// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_native_plan.h"

#include <charconv>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string_view>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s PLAN RANK\n", argv[0]);
    return 2;
  }
  try {
    const auto plan =
        rocket::qwen38::decode::load_target_layer3_native_plan(argv[1]);
    int rank = -1;
    const std::string_view rank_text(argv[2]);
    const auto [end, error] =
        std::from_chars(rank_text.data(), rank_text.data() + rank_text.size(), rank);
    if (error != std::errc{} || end != rank_text.data() + rank_text.size() ||
        (rank != 0 && rank != 1))
      throw std::invalid_argument("expected rank must be exactly 0 or 1");
    if (plan.rank != rank || plan.peer_rank != 1 - rank || plan.layer != 3 ||
        plan.extents.size() != 3108 || plan.descriptor_sha256.size() != 64)
      throw std::runtime_error("native plan publication changed");
    for (float value : plan.qsa_projection_globals)
      if (!(value > 0.0F))
        throw std::runtime_error("native QSA projection scalar changed");
    std::printf("rank=%d extents=%zu descriptor_sha256=%s\n", plan.rank,
                plan.extents.size(), plan.descriptor_sha256.c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

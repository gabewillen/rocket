// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

namespace rocket::qwen38::linear_attention {

inline constexpr int kGdnB12xInputWidth = 2'560;
inline constexpr int kGdnB12xQkvzWidth = 8'192;
inline constexpr int kGdnB12xBaWidth = 48;

struct GdnB12xLaunch {
  const std::uint8_t* activation;
  const std::uint8_t* activation_scale;
  const std::uint8_t* weight;
  const std::uint8_t* weight_scale;
  void* output;
  const float* alpha;
  int tokens;
  int output_width;
  cudaStream_t stream;
};

// Owns the four pinned CuTe DSL modules. Activation, scale, weight, output,
// scalar, and stream storage remain caller-owned and device-resident.
class GdnB12xAot final {
 public:
  explicit GdnB12xAot(int device);
  ~GdnB12xAot();
  GdnB12xAot(const GdnB12xAot&) = delete;
  GdnB12xAot& operator=(const GdnB12xAot&) = delete;

  void launch(const GdnB12xLaunch& launch) const;

 private:
  struct Impl;
  Impl* impl_;
};

[[nodiscard]] bool gdn_b12x_aot_compiled() noexcept;

}  // namespace rocket::qwen38::linear_attention

// SPDX-License-Identifier: Apache-2.0
#include "linear_attention/gdn_b12x_aot.h"

#include <stdexcept>
#include <string>

#if ROCKET_QWEN38_GDN_B12X_AOT
#include "t300_ba.h"
#include "t300_qkvz.h"
#include "t8192_ba.h"
#include "t8192_qkvz.h"
#endif

namespace rocket::qwen38::linear_attention {
namespace {

void require_launch(const GdnB12xLaunch& launch) {
  if (!launch.activation || !launch.activation_scale || !launch.weight ||
      !launch.weight_scale || !launch.output || !launch.alpha ||
      !launch.stream ||
      (launch.tokens != 300 && launch.tokens != 8'192) ||
      (launch.output_width != kGdnB12xQkvzWidth &&
       launch.output_width != kGdnB12xBaWidth)) {
    throw std::invalid_argument("fixed GDN B12X launch contract changed");
  }
}

#if ROCKET_QWEN38_GDN_B12X_AOT
template <class Module>
void load_module(Module* module, int device, auto init, auto load) {
  cudaLibrary_t* library = &module->module;
  cudaError_t status = cudaSuccess;
  struct InitArgs {
    cudaLibrary_t** library;
    cudaError_t* status;
  } init_args{&library, &status};
  init(reinterpret_cast<void**>(&init_args));
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("initialize GDN B12X module: ") +
                             cudaGetErrorString(status));
  std::int32_t selected_device = device;
  struct LoadArgs {
    cudaLibrary_t** library;
    std::int32_t* device;
    cudaError_t* status;
  } load_args{&library, &selected_device, &status};
  load(reinterpret_cast<void**>(&load_args));
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("load GDN B12X module: ") +
                             cudaGetErrorString(status));
}
#endif

}  // namespace

#if ROCKET_QWEN38_GDN_B12X_AOT
struct GdnB12xAot::Impl {
  int device;
  qwen38_gdn_t300_qkvz_Kernel_Module_t t300_qkvz{};
  qwen38_gdn_t300_ba_Kernel_Module_t t300_ba{};
  qwen38_gdn_t8192_qkvz_Kernel_Module_t t8192_qkvz{};
  qwen38_gdn_t8192_ba_Kernel_Module_t t8192_ba{};
};

GdnB12xAot::GdnB12xAot(int device) : impl_(new Impl{device}) {
  if (device < 0) {
    delete impl_;
    impl_ = nullptr;
    throw std::invalid_argument("GDN B12X device is invalid");
  }
  try {
    load_module(&impl_->t300_qkvz, device,
                _mlir_qwen38_gdn_t300_qkvz_cuda_init,
                _mlir_qwen38_gdn_t300_qkvz_cuda_load_to_device);
    load_module(&impl_->t300_ba, device,
                _mlir_qwen38_gdn_t300_ba_cuda_init,
                _mlir_qwen38_gdn_t300_ba_cuda_load_to_device);
    load_module(&impl_->t8192_qkvz, device,
                _mlir_qwen38_gdn_t8192_qkvz_cuda_init,
                _mlir_qwen38_gdn_t8192_qkvz_cuda_load_to_device);
    load_module(&impl_->t8192_ba, device,
                _mlir_qwen38_gdn_t8192_ba_cuda_init,
                _mlir_qwen38_gdn_t8192_ba_cuda_load_to_device);
  } catch (...) {
    delete impl_;
    impl_ = nullptr;
    throw;
  }
}

GdnB12xAot::~GdnB12xAot() {
  if (!impl_) return;
  cudaSetDevice(impl_->device);
  if (impl_->t8192_ba.module) cudaLibraryUnload(impl_->t8192_ba.module);
  if (impl_->t8192_qkvz.module) cudaLibraryUnload(impl_->t8192_qkvz.module);
  if (impl_->t300_ba.module) cudaLibraryUnload(impl_->t300_ba.module);
  if (impl_->t300_qkvz.module) cudaLibraryUnload(impl_->t300_qkvz.module);
  delete impl_;
}

#define ROCKET_B12X_CALL(prefix, module_field)                               \
  do {                                                                       \
    prefix##_Tensor_mA_t a{const_cast<std::uint8_t*>(launch.activation),     \
                            {launch.tokens, kGdnB12xInputWidth / 2},          \
                            {kGdnB12xInputWidth / 2}};                        \
    prefix##_Tensor_mB_t b{const_cast<std::uint8_t*>(launch.weight),         \
                            {launch.output_width, kGdnB12xInputWidth / 2},    \
                            {kGdnB12xInputWidth / 2}};                        \
    prefix##_Tensor_mC_t c{launch.output,                                    \
                            {launch.tokens, launch.output_width},             \
                            {launch.output_width}};                           \
    prefix##_Tensor_alpha_tensor_t alpha{const_cast<float*>(launch.alpha)};  \
    const int status = cute_dsl_##prefix##_wrapper(                          \
        &impl_->module_field, &a, &b, &c, (launch.tokens + 127) / 128,       \
        (launch.output_width + 127) / 128, 40,                               \
        const_cast<std::uint8_t*>(launch.activation_scale),                  \
        const_cast<std::uint8_t*>(launch.weight_scale), &alpha,              \
        launch.stream);                                                       \
    if (status != 0) throw std::runtime_error("GDN B12X launch failed");    \
  } while (false)

void GdnB12xAot::launch(const GdnB12xLaunch& launch) const {
  require_launch(launch);
  if (launch.tokens == 300 && launch.output_width == kGdnB12xQkvzWidth) {
    ROCKET_B12X_CALL(qwen38_gdn_t300_qkvz, t300_qkvz);
  } else if (launch.tokens == 300) {
    ROCKET_B12X_CALL(qwen38_gdn_t300_ba, t300_ba);
  } else if (launch.output_width == kGdnB12xQkvzWidth) {
    ROCKET_B12X_CALL(qwen38_gdn_t8192_qkvz, t8192_qkvz);
  } else {
    ROCKET_B12X_CALL(qwen38_gdn_t8192_ba, t8192_ba);
  }
}
#undef ROCKET_B12X_CALL

bool gdn_b12x_aot_compiled() noexcept { return true; }

#else

struct GdnB12xAot::Impl {};

GdnB12xAot::GdnB12xAot(int) : impl_(nullptr) {
  throw std::runtime_error(
      "GDN B12X AOT objects were not supplied at configure time");
}

GdnB12xAot::~GdnB12xAot() = default;

void GdnB12xAot::launch(const GdnB12xLaunch& launch) const {
  require_launch(launch);
  throw std::runtime_error("GDN B12X AOT backend is unavailable");
}

bool gdn_b12x_aot_compiled() noexcept { return false; }

#endif

}  // namespace rocket::qwen38::linear_attention

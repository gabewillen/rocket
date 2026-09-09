// SPDX-License-Identifier: Apache-2.0
#include "attention/layer3_rope_owner.h"

#include <memory>

namespace {
struct C1RopeHandle {
  std::unique_ptr<rocket::qwen38::attention::Layer3RopeC1DeviceOwner> owner;
};
}  // namespace

extern "C" int qwen38_layer3_rope_c1_create(int device, int rank, int layer,
                                             void** handle) noexcept {
  if (!handle || *handle) return 1;
  try {
    auto result = std::make_unique<C1RopeHandle>();
    result->owner = std::make_unique<
        rocket::qwen38::attention::Layer3RopeC1DeviceOwner>(
        device,
        rocket::qwen38::attention::target_qsa_c1_rope_identity(rank, layer));
    *handle = result.release();
    return 0;
  } catch (...) {
    return 2;
  }
}

extern "C" int qwen38_layer3_rope_c1_view(
    void* handle, const __nv_bfloat16** cos_sin, cudaEvent_t* ready,
    int* rows, int* columns) noexcept {
  if (!handle || !cos_sin || !ready || !rows || !columns) return 1;
  const auto view = static_cast<C1RopeHandle*>(handle)->owner->view();
  if (!view.cos_sin || !view.ready ||
      view.payload_sha256 !=
          rocket::qwen38::attention::kLayer3RopeC1PayloadSha256 ||
      view.rows != rocket::qwen38::attention::kLayer3RopeC1Rows ||
      view.columns != rocket::qwen38::attention::kLayer3RopeColumns ||
      view.row_stride != rocket::qwen38::attention::kLayer3RopeColumns)
    return 2;
  *cos_sin = view.cos_sin;
  *ready = view.ready;
  *rows = view.rows;
  *columns = view.columns;
  return 0;
}

extern "C" int qwen38_layer3_rope_c1_wait(void* handle,
                                           cudaStream_t stream) noexcept {
  if (!handle || !stream) return 1;
  try {
    static_cast<C1RopeHandle*>(handle)->owner->wait(stream);
    return 0;
  } catch (...) {
    return 2;
  }
}

extern "C" int qwen38_layer3_rope_c1_destroy(void* handle) noexcept {
  delete static_cast<C1RopeHandle*>(handle);
  return 0;
}

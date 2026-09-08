// SPDX-License-Identifier: Apache-2.0
#include "decode/target_gdn_layer_owner.h"

#include <cstdlib>
#include <stdexcept>

namespace decode = rocket::qwen38::decode;
namespace linear = rocket::qwen38::linear_attention;

namespace {
void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
}  // namespace

int main() {
  static_assert(decode::kTargetGdnStateSlots == 2);
  static_assert(decode::kTargetGdnConvSlotElements == 30'720);
  static_assert(decode::kTargetGdnRecurrentSlotElements == 393'216);
  static_assert(decode::kTargetGdnOwnerStorageBytes == 3'320'576);
  static_assert(linear::gdn_bucket_index(1) == 0);
  static_assert(linear::gdn_bucket_index(2) == 1);
  static_assert(linear::gdn_bucket_index(4) == 2);
  static_assert(linear::gdn_bucket_index(8) == 3);
  static_assert(linear::gdn_bucket_index(16) == 4);
  static_assert(linear::gdn_bucket_index(3) == -1);
  static_assert(linear::gdn_bucket_rows(0) == 1);
  static_assert(linear::gdn_bucket_rows(1) == 2);
  static_assert(linear::gdn_bucket_rows(2) == 4);
  static_assert(linear::gdn_bucket_rows(3) == 8);
  static_assert(linear::gdn_bucket_rows(4) == 16);
  static_assert(linear::gdn_bucket_rows(-1) == 0);
  static_assert(linear::gdn_bucket_rows(5) == 0);
  static_assert(linear::CutlassGdnGraph::decode_workspace_count() == 1);
  static_assert(linear::gdn_quantizer_scale(0.001331147737801075F) ==
                751.2314453125F);
  static_assert(linear::gdn_quantizer_scale(0.00039527530316263437F) ==
                2529.88232421875F);
  static_assert(
      linear::gdn_projection_alpha(0.001331147737801075F,
                                   0.0003022693563F) > 4.02e-7F);
  static_assert(
      linear::gdn_projection_alpha(0.001331147737801075F,
                                   0.0003022693563F) < 4.03e-7F);
  static_assert(
      linear::gdn_projection_alpha(0.001331147737801075F,
                                   0.0003022693563F) !=
      linear::gdn_projection_alpha(0.001331147737801075F,
                                   0.0002615792328F));

  void* storage = nullptr;
  if (posix_memalign(&storage, 256, decode::kTargetGdnOwnerStorageBytes) != 0)
    return 2;
  try {
    const auto binding = decode::bind_target_gdn_owner_storage(
        storage, decode::kTargetGdnOwnerStorageBytes);
    check(decode::valid_target_gdn_c1_state_extent(binding.state),
          "GDN state binding changed");
    check(decode::complete_target_gdn_c1_buffers(binding.buffers),
          "GDN scratch binding changed");
    check(binding.state.state_index == binding.mutable_state_index,
          "GDN state-index ownership changed");
    check(binding.state.convolution !=
              reinterpret_cast<__nv_bfloat16*>(binding.state.recurrent) &&
              binding.buffers.attention_input !=
                  binding.buffers.post_attention_hidden,
          "GDN storage aliases changed");
    bool rejected = false;
    try {
      (void)decode::bind_target_gdn_owner_storage(
          storage, decode::kTargetGdnOwnerStorageBytes - 1);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check(rejected, "short GDN storage was accepted");
  } catch (...) {
    free(storage);
    throw;
  }
  free(storage);
  return 0;
}

// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_generation.h"

#include <stdexcept>

namespace attention = rocket::qwen38::attention;
namespace decode = rocket::qwen38::decode;

int main() {
  __nv_bfloat16 bf16{};
  std::int32_t i32{};
  std::int64_t i64{};
  std::uint64_t requested{};
  attention::TargetQsaStateView state{
      .main_key_cache=&bf16, .main_value_cache=&bf16,
      .raw_key_cache=&bf16, .compressed_key_cache=&bf16,
      .positions=&i64, .main_slot_mapping=&i32, .main_block_table=&i32,
      .raw_slot_mapping=&i32, .raw_block_table=&i32,
      .compressed_slot_mapping=&i32, .compressed_block_table=&i32,
      .query_start_locations=&i32, .logical_positions=&i64,
      .sequence_lengths=&i32, .token_to_request=&i32,
      .compression_work=&i32, .main_blocks=1, .compressed_blocks=1,
      .compression_work_items=1, .rows=1, .rank=1, .layer=3,
      .uses_mrope=false,
      .main_kv_dtype=attention::TargetQsaServingDtype::kBfloat16,
      .side_cache_dtype=attention::TargetQsaServingDtype::kBfloat16};
  decode::NativeTargetQsaGenerationOwner owner(1, 3, state, &requested, 35);
  if (!owner.authenticated() || owner.rank() != 1 || owner.layer() != 3 ||
      owner.view(0, 1).generation != 1 ||
      owner.view(34, 35).expected_generation != 35)
    return 1;
  try {
    owner.view(34, 0);
    return 2;
  } catch (const std::logic_error&) {
  }
  try {
    owner.enqueue_prepare(35, 36, reinterpret_cast<cudaStream_t>(0x10));
    return 3;
  } catch (const std::logic_error&) {
  }
  return 0;
}

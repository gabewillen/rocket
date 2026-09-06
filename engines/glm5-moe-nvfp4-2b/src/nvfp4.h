// NVFP4 as this checkpoint stores it, and the scale-factor layout CUTLASS
// wants it in.
//
// On disk each quantized projection is a triplet:
//   <p>.weight         U8      [N, K/2]    packed e2m1, low nibble is column 2j
//   <p>.weight_scale   F8_E4M3 [N, K/16]   one block scale per 16 columns
//   <p>.weight_scale_2 F32     scalar      one global scale for the tensor
// and the dequantized value is
//   w[n,k] = e2m1(nibble) * e4m3(block_scale[n, k/16]) * global
//
// The global is the ModelOpt per-tensor scale amax/(448*6); on layer 10 expert
// 0 gate_proj it reads 4.650297705666162e-05, and 4.650297705666162e-05*2688
// is exactly 0.125, which is that tensor's amax. Block scales are stored plain
// row-major here (safetensors never stores a kernel-specific layout), so the
// loader has to reshuffle them before CUTLASS sees them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "safetensors.h"

namespace rocket::fuel {

// --- element codecs, host side, no CUDA -----------------------------------
// The reference path in the test uses these, so they stay independent of any
// device intrinsic the kernel under test might share a bug with.

float e4m3_to_float(std::uint8_t bits);
std::uint8_t float_to_e4m3(float v);   // round-to-nearest-even, saturates at +/-448
float e2m1_to_float(std::uint8_t nibble);  // low 4 bits
std::uint8_t float_to_e2m1(float v);       // round-to-nearest-even, saturates at +/-6
float bf16_to_float(std::uint16_t bits);
std::uint16_t float_to_bf16(float v);

// --- CUTLASS block-scaled scale-factor layout ------------------------------
//
// Derived from third_party/cutlass/include/cutlass/detail/sm100_blockscaled_layout.hpp:
//
//   line 51-57  Blk_MN = _128, Blk_SF = _4, and
//               SfKMajorAtom = Layout<Shape <Shape<_32,_4>, Shape<SFVecSize,_4>>,
//                                     Stride<Stride<_16,_4>, Stride<       _0,_1>>>
//   line 93/108 tile_atom_to_shape_SFA/SFB tile that atom over (M,K,L)/(N,K,L)
//               with Step<_2,_1,_3>, so the K tile index varies fastest.
//
// One atom therefore holds 128 rows x 4 scale factors = 512 bytes, and inside
// an atom the byte offset for row m (0..127) and scale index s (0..3) is
//   (m % 32) * 16 + ((m % 128) / 32) * 4 + s
// which is the Shape<_32,_4> / Stride<_16,_4> decomposition of the MN mode
// with the K mode contributing s at stride 1 (the SFVecSize mode has stride 0
// because all SFVecSize elements of a block share one scale).
//
// This is the SM100 header, but it is also what sm_121 runs: the SM120
// collectives alias it directly, e.g.
//   include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp:128
//   include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp:133
//   include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl:164
// all say `using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>`.
// Only SM103 substitutes a different config (Sm103BlockScaledConfig). The test
// checks this layout against the instantiated CUTLASS type rather than
// trusting the derivation.
struct SfLayout {
  std::int64_t mn = 0;        // rows of the operand: M for A, N for B
  std::int64_t k = 0;         // contraction extent in elements
  int sf_vec = 16;            // elements per block scale
  std::int64_t k_sf = 0;      // k / sf_vec
  std::int64_t mn_tiles = 0;  // ceil(mn / 128)
  std::int64_t k_tiles = 0;   // ceil(k_sf / 4)

  std::size_t bytes() const;
  // Byte offset of the scale for row mn_idx covering columns
  // [sf_idx*sf_vec, (sf_idx+1)*sf_vec).
  std::size_t offset(std::int64_t mn_idx, std::int64_t sf_idx) const;
};

// Throws std::runtime_error if k is not a multiple of sf_vec.
SfLayout sf_layout(std::int64_t mn, std::int64_t k, int sf_vec = 16);

// Reshuffle row-major [mn, k/sf_vec] block scales into the CUTLASS layout.
// `out` must hold layout.bytes(); padding bytes are zeroed. Pure byte
// permutation, no numeric change.
void swizzle_block_scales(const std::uint8_t* linear, const SfLayout& layout, std::uint8_t* out);

// --- checkpoint access -----------------------------------------------------

// One quantized projection of one routed expert. Pointers alias the mmap.
struct ExpertProj {
  std::string name;               // tensor prefix, e.g. ...experts.0.down_proj
  std::int64_t n = 0;             // out_features
  std::int64_t k = 0;             // in_features
  const std::uint8_t* packed = nullptr;       // [n, k/2]
  const std::uint8_t* block_scale = nullptr;  // [n, k/16], checkpoint-linear
  float global_scale = 0.0f;

  std::size_t packed_bytes() const { return static_cast<std::size_t>(n) * static_cast<std::size_t>(k) / 2; }
  std::size_t scale_bytes() const { return static_cast<std::size_t>(n) * static_cast<std::size_t>(k) / 16; }
  // Fully dequantized weight value, the definition the reference matmul uses.
  float value(std::int64_t row, std::int64_t col) const;
};

// proj is one of "gate_proj", "up_proj", "down_proj".
// Throws std::runtime_error if the triplet is missing or malformed.
ExpertProj load_expert_proj(Checkpoint& ckpt, int layer, int expert, std::string_view proj);

}  // namespace rocket::fuel

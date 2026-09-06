#include "nvfp4.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace rocket::fuel {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("rocket::fuel: " + what); }

constexpr int kBlkMN = 128;   // Sm1xxBlockScaledBasicChunk::Blk_MN
constexpr int kBlkSF = 4;     // Sm1xxBlockScaledBasicChunk::Blk_SF
constexpr int kAtomBytes = kBlkMN * kBlkSF;  // 512

// e2m1 magnitudes, index = low 3 bits, bit 3 is sign.
constexpr float kE2M1Mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

}  // namespace

// ---------------------------------------------------------------------------
// codecs
// ---------------------------------------------------------------------------

float e4m3_to_float(std::uint8_t bits) {
  const std::uint32_t sign = (bits & 0x80u) ? 0x80000000u : 0u;
  std::uint32_t exp = (bits >> 3) & 0x0Fu;
  std::uint32_t mant = bits & 0x07u;
  if (exp == 0x0F && mant == 0x07) return std::nanf("");  // e4m3 has no inf; 0x7F/0xFF are NaN
  float out;
  if (exp == 0) {
    if (mant == 0) {
      std::uint32_t z = sign;
      std::memcpy(&out, &z, 4);
      return out;
    }
    // subnormal: 2^-6 * mant/8
    float v = static_cast<float>(mant) * (1.0f / 8.0f) * 0.015625f;
    return (sign != 0u) ? -v : v;
  }
  const std::uint32_t f32 = sign | ((exp + 127u - 7u) << 23) | (mant << 20);
  std::memcpy(&out, &f32, 4);
  return out;
}

std::uint8_t float_to_e4m3(float v) {
  if (std::isnan(v)) return 0x7F;
  const std::uint8_t sign = std::signbit(v) ? 0x80 : 0x00;
  float a = std::fabs(v);
  if (a >= 464.0f) return static_cast<std::uint8_t>(sign | 0x7E);  // saturate to 448
  if (a < 0.001953125f * 0.5f) return sign;                        // below half of min subnormal
  // Search is fine here: this runs on 4096-element activation blocks in a test
  // and on the loader's cold path, never per token.
  std::uint8_t best = sign;
  float best_err = a;
  for (std::uint32_t bits = 0; bits <= 0x7E; ++bits) {
    const float c = e4m3_to_float(static_cast<std::uint8_t>(bits));
    const float err = std::fabs(a - c);
    if (err < best_err || (err == best_err && (bits & 1u) == 0u)) {
      best_err = err;
      best = static_cast<std::uint8_t>(sign | bits);
    }
  }
  return best;
}

float e2m1_to_float(std::uint8_t nibble) {
  const float mag = kE2M1Mag[nibble & 0x07u];
  return (nibble & 0x08u) ? -mag : mag;
}

std::uint8_t float_to_e2m1(float v) {
  const std::uint8_t sign = std::signbit(v) ? 0x08 : 0x00;
  const float a = std::fabs(v);
  int best = 0;
  float best_err = a;
  for (int i = 0; i < 8; ++i) {
    const float err = std::fabs(a - kE2M1Mag[i]);
    // ties go to the even code, matching round-to-nearest-even on the mantissa
    if (err < best_err || (err == best_err && (i & 1) == 0)) {
      best_err = err;
      best = i;
    }
  }
  return static_cast<std::uint8_t>(sign | best);
}

float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t f32 = static_cast<std::uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &f32, 4);
  return out;
}

std::uint16_t float_to_bf16(float v) {
  std::uint32_t f32;
  std::memcpy(&f32, &v, 4);
  const std::uint32_t rounding = 0x7FFFu + ((f32 >> 16) & 1u);
  return static_cast<std::uint16_t>((f32 + rounding) >> 16);
}

// ---------------------------------------------------------------------------
// scale-factor layout
// ---------------------------------------------------------------------------

SfLayout sf_layout(std::int64_t mn, std::int64_t k, int sf_vec) {
  if (mn <= 0 || k <= 0) fail("sf_layout needs positive extents");
  if (sf_vec <= 0 || k % sf_vec != 0)
    fail("sf_layout: k=" + std::to_string(k) + " is not a multiple of sf_vec=" + std::to_string(sf_vec));
  SfLayout l;
  l.mn = mn;
  l.k = k;
  l.sf_vec = sf_vec;
  l.k_sf = k / sf_vec;
  l.mn_tiles = (mn + kBlkMN - 1) / kBlkMN;
  l.k_tiles = (l.k_sf + kBlkSF - 1) / kBlkSF;
  return l;
}

std::size_t SfLayout::bytes() const {
  return static_cast<std::size_t>(mn_tiles) * static_cast<std::size_t>(k_tiles) * kAtomBytes;
}

std::size_t SfLayout::offset(std::int64_t mn_idx, std::int64_t sf_idx) const {
  const std::int64_t mn_tile = mn_idx / kBlkMN;
  const std::int64_t mn_in = mn_idx % kBlkMN;
  const std::int64_t k_tile = sf_idx / kBlkSF;
  const std::int64_t s = sf_idx % kBlkSF;
  const std::int64_t atom = mn_tile * k_tiles + k_tile;
  return static_cast<std::size_t>(atom) * kAtomBytes +
         static_cast<std::size_t>((mn_in % 32) * 16 + (mn_in / 32) * 4 + s);
}

void swizzle_block_scales(const std::uint8_t* linear, const SfLayout& layout, std::uint8_t* out) {
  if (linear == nullptr || out == nullptr) fail("swizzle_block_scales got a null pointer");
  std::memset(out, 0, layout.bytes());
  // Inside one atom the four scale factors s=0..3 of a row land on four
  // consecutive bytes (the K mode has stride 1 and Blk_SF is 4), so a whole
  // Blk_SF group moves as one 4-byte store. Source is contiguous too, which
  // makes the inner loop a strided 4-byte copy instead of a byte scatter.
  const std::int64_t full_tiles = layout.k_sf / kBlkSF;
  const std::int64_t tail = layout.k_sf % kBlkSF;
  for (std::int64_t row = 0; row < layout.mn; ++row) {
    const std::uint8_t* src =
        linear + static_cast<std::size_t>(row) * static_cast<std::size_t>(layout.k_sf);
    for (std::int64_t t = 0; t < full_tiles; ++t)
      std::memcpy(out + layout.offset(row, t * kBlkSF), src + t * kBlkSF, kBlkSF);
    for (std::int64_t s = full_tiles * kBlkSF; s < full_tiles * kBlkSF + tail; ++s)
      out[layout.offset(row, s)] = src[s];
  }
}

// ---------------------------------------------------------------------------
// checkpoint access
// ---------------------------------------------------------------------------

float ExpertProj::value(std::int64_t row, std::int64_t col) const {
  const std::size_t byte_idx = static_cast<std::size_t>(row) * static_cast<std::size_t>(k / 2) +
                               static_cast<std::size_t>(col / 2);
  const std::uint8_t byte = packed[byte_idx];
  const std::uint8_t nib = (col & 1) ? static_cast<std::uint8_t>(byte >> 4)
                                     : static_cast<std::uint8_t>(byte & 0x0F);
  const std::uint8_t sf = block_scale[static_cast<std::size_t>(row) * static_cast<std::size_t>(k / 16) +
                                      static_cast<std::size_t>(col / 16)];
  return e2m1_to_float(nib) * e4m3_to_float(sf) * global_scale;
}

ExpertProj load_expert_proj(Checkpoint& ckpt, int layer, int expert, std::string_view proj) {
  const std::string prefix = "model.language_model.layers." + std::to_string(layer) +
                             ".mlp.experts." + std::to_string(expert) + "." + std::string(proj);

  const TensorView& w = ckpt.tensor(prefix + ".weight");
  const TensorView& s = ckpt.tensor(prefix + ".weight_scale");
  const TensorView& g = ckpt.tensor(prefix + ".weight_scale_2");

  if (w.dtype != DType::kU8 || w.shape.size() != 2) fail(prefix + ".weight is not a 2-D U8 tensor");
  if (s.dtype != DType::kF8E4M3 || s.shape.size() != 2)
    fail(prefix + ".weight_scale is not a 2-D F8_E4M3 tensor");
  if (g.dtype != DType::kF32 || g.numel() != 1) fail(prefix + ".weight_scale_2 is not an F32 scalar");

  ExpertProj p;
  p.name = prefix;
  p.n = w.shape[0];
  p.k = w.shape[1] * 2;  // two packed e2m1 values per byte
  if (s.shape[0] != p.n || s.shape[1] * 16 != p.k)
    fail(prefix + ": weight_scale shape does not match a block size of 16 over weight's K");
  p.packed = w.data;
  p.block_scale = s.data;
  std::memcpy(&p.global_scale, g.data, 4);
  return p;
}

}  // namespace rocket::fuel

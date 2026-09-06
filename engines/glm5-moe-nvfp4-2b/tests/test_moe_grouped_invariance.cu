// Properties the grouped-GEMM MoE path (model.cu::run_moe_grouped,
// moe_grouped.h) relies on but that no other test isolates directly:
//
//   1. a row's CUTLASS output does not depend on what other rows share its
//      group (the property that makes grouped-M1 vs grouped-M8 token parity
//      possible at all, tests/test_batch_parity.cu)
//   2. concatenating two independently-swizzled [2048,4096] SFB buffers
//      (gate then up) is bit-identical to swizzling the fused [4096,4096]
//      matrix directly (weights.h::ExpertDev's w13_scale layout)
//   3. CUTLASS's grouped GEMM is deterministic run to run for identical
//      inputs (no atomic-reduction nondeterminism this path could inherit)
//
// Synthetic random NVFP4 buffers, no checkpoint. These are the checks that
// ruled out a CUTLASS-level bug when tests/test_batch_parity.cu first failed
// during development (blog/posts/runtime/2026-09-07-grouped-gemm-beats-gemv-
// at-every-m/): the real cause was a stale test binary, not a batch-
// invariance bug, but the properties below are worth guarding permanently
// since a future CUTLASS upgrade could change any of them silently.
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "moe_grouped.h"
#include "nvfp4.h"

namespace {

using rocket::engine::bf16;
using rocket::engine::GroupedGemmGroup;
using rocket::engine::grouped_gemm_nvfp4;
using namespace rocket::fuel;

int failures = 0;

void check(const std::string& what, bool ok) {
  std::printf("  %-60s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

std::mt19937 rng(20260907);

uint8_t* to_dev(const std::vector<uint8_t>& h) {
  uint8_t* d = nullptr;
  cudaMalloc(&d, h.size());
  cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice);
  return d;
}

std::vector<uint16_t> from_dev_bf16(const bf16* d, std::size_t n) {
  std::vector<uint16_t> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(bf16), cudaMemcpyDeviceToHost);
  return h;
}

// ------------------------------------------------------- row invariance
void test_row_invariance() {
  const int N = 4096, K = 4096;
  const int k2 = K / 2;
  std::uniform_int_distribution<int> byte_d(0, 255);
  std::uniform_int_distribution<int> sf_d(0x30, 0x40);

  std::vector<uint8_t> b(static_cast<std::size_t>(N) * k2);
  std::vector<uint8_t> b_sf(static_cast<std::size_t>(N) / 128 * 64 * 512);
  for (auto& v : b) v = byte_d(rng);
  for (auto& v : b_sf) v = sf_d(rng);
  uint8_t* dB = to_dev(b);
  uint8_t* dBSF = to_dev(b_sf);

  SfLayout layout2 = sf_layout(2, K, 16);
  std::vector<uint8_t> sf_lin(2 * (K / 16));
  for (auto& v : sf_lin) v = sf_d(rng);
  std::vector<uint8_t> sf_sw2(layout2.bytes());
  swizzle_block_scales(sf_lin.data(), layout2, sf_sw2.data());

  SfLayout layout1 = sf_layout(1, K, 16);
  std::vector<uint8_t> sf_sw_row0(layout1.bytes()), sf_sw_row1(layout1.bytes());
  swizzle_block_scales(sf_lin.data(), layout1, sf_sw_row0.data());
  swizzle_block_scales(sf_lin.data() + (K / 16), layout1, sf_sw_row1.data());

  std::vector<uint8_t> a_row0(k2), a_row1(k2);
  for (auto& v : a_row0) v = byte_d(rng);
  for (auto& v : a_row1) v = byte_d(rng);

  // Paired: rows 0 and 1 in one Mg=2 group.
  std::vector<uint8_t> a_pair(2 * static_cast<std::size_t>(k2));
  std::copy(a_row0.begin(), a_row0.end(), a_pair.begin());
  std::copy(a_row1.begin(), a_row1.end(), a_pair.begin() + k2);
  uint8_t* dA_pair = to_dev(a_pair);
  uint8_t* dSFA_pair = to_dev(sf_sw2);
  bf16* d_paired = nullptr;
  cudaMalloc(&d_paired, static_cast<std::size_t>(2) * N * sizeof(bf16));
  {
    GroupedGemmGroup g;
    g.m = 2;
    g.a_packed = dA_pair;
    g.a_scale = dSFA_pair;
    g.b_packed = dB;
    g.b_scale = dBSF;
    g.d_out = d_paired;
    std::vector<GroupedGemmGroup> gs = {g};
    grouped_gemm_nvfp4(gs, N, K, nullptr);
  }

  // Alone: each row its own Mg=1 group.
  auto run_alone = [&](const std::vector<uint8_t>& a_row, const std::vector<uint8_t>& sf_row) {
    uint8_t* dA = to_dev(a_row);
    uint8_t* dSFA = to_dev(sf_row);
    bf16* d_out = nullptr;
    cudaMalloc(&d_out, static_cast<std::size_t>(N) * sizeof(bf16));
    GroupedGemmGroup g;
    g.m = 1;
    g.a_packed = dA;
    g.a_scale = dSFA;
    g.b_packed = dB;
    g.b_scale = dBSF;
    g.d_out = d_out;
    std::vector<GroupedGemmGroup> gs = {g};
    grouped_gemm_nvfp4(gs, N, K, nullptr);
    return d_out;
  };
  bf16* d_alone0 = run_alone(a_row0, sf_sw_row0);
  bf16* d_alone1 = run_alone(a_row1, sf_sw_row1);
  cudaDeviceSynchronize();

  const auto h_paired = from_dev_bf16(d_paired, static_cast<std::size_t>(2) * N);
  const auto h_alone0 = from_dev_bf16(d_alone0, N);
  const auto h_alone1 = from_dev_bf16(d_alone1, N);
  bool row0_ok = true, row1_ok = true;
  for (int i = 0; i < N; ++i) {
    if (h_paired[i] != h_alone0[i]) row0_ok = false;
    if (h_paired[N + i] != h_alone1[i]) row1_ok = false;
  }
  check("row 0: alone == paired with row 1 (Mg=1 vs Mg=2)", row0_ok);
  check("row 1: alone == paired with row 0 (Mg=1 vs Mg=2)", row1_ok);
}

// --------------------------------------------- fused-w13 concatenation
void test_fused_w13_concatenation() {
  const int M = 3, H = 4096, MI = 2048;
  std::uniform_int_distribution<int> byte_d(0, 255);
  std::uniform_int_distribution<int> sf_d(0x30, 0x40);

  std::vector<uint8_t> a_packed(static_cast<std::size_t>(M) * H / 2);
  for (auto& v : a_packed) v = byte_d(rng);
  SfLayout a_layout = sf_layout(M, H, 16);
  std::vector<uint8_t> a_sf(a_layout.bytes(), 0x38);
  uint8_t* dA = to_dev(a_packed);
  uint8_t* dASF = to_dev(a_sf);

  std::vector<uint8_t> gate_packed(static_cast<std::size_t>(MI) * H / 2);
  std::vector<uint8_t> up_packed(static_cast<std::size_t>(MI) * H / 2);
  for (auto& v : gate_packed) v = byte_d(rng);
  for (auto& v : up_packed) v = byte_d(rng);
  std::vector<uint8_t> gate_sf_lin(static_cast<std::size_t>(MI) * H / 16);
  std::vector<uint8_t> up_sf_lin(static_cast<std::size_t>(MI) * H / 16);
  for (auto& v : gate_sf_lin) v = sf_d(rng);
  for (auto& v : up_sf_lin) v = sf_d(rng);
  SfLayout half_layout = sf_layout(MI, H, 16);
  std::vector<uint8_t> gate_sf_sw(half_layout.bytes()), up_sf_sw(half_layout.bytes());
  swizzle_block_scales(gate_sf_lin.data(), half_layout, gate_sf_sw.data());
  swizzle_block_scales(up_sf_lin.data(), half_layout, up_sf_sw.data());

  // (a) fused: concatenated packed + concatenated swizzled scale, one
  // N=2*MI grouped GEMM call, matching weights.h::ExpertDev's cache layout.
  std::vector<uint8_t> fused_packed(gate_packed.size() + up_packed.size());
  std::copy(gate_packed.begin(), gate_packed.end(), fused_packed.begin());
  std::copy(up_packed.begin(), up_packed.end(), fused_packed.begin() + gate_packed.size());
  std::vector<uint8_t> fused_sf(gate_sf_sw.size() + up_sf_sw.size());
  std::copy(gate_sf_sw.begin(), gate_sf_sw.end(), fused_sf.begin());
  std::copy(up_sf_sw.begin(), up_sf_sw.end(), fused_sf.begin() + gate_sf_sw.size());
  uint8_t* dFusedB = to_dev(fused_packed);
  uint8_t* dFusedSF = to_dev(fused_sf);
  bf16* d_fused_out = nullptr;
  cudaMalloc(&d_fused_out, static_cast<std::size_t>(M) * 2 * MI * sizeof(bf16));
  {
    GroupedGemmGroup g;
    g.m = M;
    g.a_packed = dA;
    g.a_scale = dASF;
    g.b_packed = dFusedB;
    g.b_scale = dFusedSF;
    g.d_out = d_fused_out;
    std::vector<GroupedGemmGroup> gs = {g};
    grouped_gemm_nvfp4(gs, 2 * MI, H, nullptr);
  }

  // (b) ground truth: two separate N=MI GEMMs with each half's own swizzle.
  uint8_t* dGateB = to_dev(gate_packed);
  uint8_t* dGateSF = to_dev(gate_sf_sw);
  uint8_t* dUpB = to_dev(up_packed);
  uint8_t* dUpSF = to_dev(up_sf_sw);
  bf16* d_gate_out = nullptr;
  bf16* d_up_out = nullptr;
  cudaMalloc(&d_gate_out, static_cast<std::size_t>(M) * MI * sizeof(bf16));
  cudaMalloc(&d_up_out, static_cast<std::size_t>(M) * MI * sizeof(bf16));
  {
    GroupedGemmGroup g;
    g.m = M;
    g.a_packed = dA;
    g.a_scale = dASF;
    g.b_packed = dGateB;
    g.b_scale = dGateSF;
    g.d_out = d_gate_out;
    std::vector<GroupedGemmGroup> gs = {g};
    grouped_gemm_nvfp4(gs, MI, H, nullptr);
  }
  {
    GroupedGemmGroup g;
    g.m = M;
    g.a_packed = dA;
    g.a_scale = dASF;
    g.b_packed = dUpB;
    g.b_scale = dUpSF;
    g.d_out = d_up_out;
    std::vector<GroupedGemmGroup> gs = {g};
    grouped_gemm_nvfp4(gs, MI, H, nullptr);
  }
  cudaDeviceSynchronize();

  const auto h_fused = from_dev_bf16(d_fused_out, static_cast<std::size_t>(M) * 2 * MI);
  const auto h_gate = from_dev_bf16(d_gate_out, static_cast<std::size_t>(M) * MI);
  const auto h_up = from_dev_bf16(d_up_out, static_cast<std::size_t>(M) * MI);
  bool gate_ok = true, up_ok = true;
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < MI; ++c) {
      if (h_fused[static_cast<std::size_t>(r) * 2 * MI + c] != h_gate[static_cast<std::size_t>(r) * MI + c])
        gate_ok = false;
      if (h_fused[static_cast<std::size_t>(r) * 2 * MI + MI + c] !=
          h_up[static_cast<std::size_t>(r) * MI + c])
        up_ok = false;
    }
  }
  check("fused w13 gate half == standalone gate-only GEMM", gate_ok);
  check("fused w13 up half == standalone up-only GEMM", up_ok);
}

// ------------------------------------------------------------ determinism
void test_determinism() {
  const int N = 4096, K = 4096, M = 3;
  std::uniform_int_distribution<int> byte_d(0, 255);
  std::vector<uint8_t> a(static_cast<std::size_t>(M) * K / 2);
  std::vector<uint8_t> b(static_cast<std::size_t>(N) * K / 2);
  std::vector<uint8_t> asf(64 * 512);
  std::vector<uint8_t> bsf(static_cast<std::size_t>(N) / 128 * 64 * 512);
  for (auto& v : a) v = byte_d(rng);
  for (auto& v : b) v = byte_d(rng);
  for (auto& v : asf) v = 0x30 + (byte_d(rng) % 16);
  for (auto& v : bsf) v = 0x30 + (byte_d(rng) % 16);
  uint8_t* dA = to_dev(a);
  uint8_t* dB = to_dev(b);
  uint8_t* dASF = to_dev(asf);
  uint8_t* dBSF = to_dev(bsf);
  bf16* d_out = nullptr;
  cudaMalloc(&d_out, static_cast<std::size_t>(M) * N * sizeof(bf16));

  std::vector<std::vector<uint16_t>> runs;
  for (int r = 0; r < 5; ++r) {
    cudaMemset(d_out, 0, static_cast<std::size_t>(M) * N * sizeof(bf16));
    GroupedGemmGroup g;
    g.m = M;
    g.a_packed = dA;
    g.a_scale = dASF;
    g.b_packed = dB;
    g.b_scale = dBSF;
    g.d_out = d_out;
    std::vector<GroupedGemmGroup> gs = {g};
    grouped_gemm_nvfp4(gs, N, K, nullptr);
    cudaDeviceSynchronize();
    runs.push_back(from_dev_bf16(d_out, static_cast<std::size_t>(M) * N));
  }
  bool ok = true;
  for (std::size_t i = 0; i < runs[0].size() && ok; ++i)
    for (int r = 1; r < 5; ++r)
      if (runs[r][i] != runs[0][i]) ok = false;
  check("5 repeated calls, identical inputs, bit-identical output", ok);
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("no CUDA device\n");
    return 77;
  }
  std::printf("row invariance (Mg=1 vs Mg=2)\n");
  test_row_invariance();
  std::printf("fused-w13 concatenation\n");
  test_fused_w13_concatenation();
  std::printf("determinism\n");
  test_determinism();
  std::printf("\n%s: %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}

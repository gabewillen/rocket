// Component tests for the decode kernels, each against a CPU reference written
// from the reference modeling code's own arithmetic rather than from the
// kernel it checks. No checkpoint is needed: every input is synthetic.
#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "kernels.h"
#include "nvfp4.h"

namespace {

using rocket::engine::bf16;

int failures = 0;

float bf(float v) {  // round a float through BF16, as the device buffers do
  return __bfloat162float(__float2bfloat16(v));
}

void check(const std::string& what, double err, double tol) {
  const bool ok = std::isfinite(err) && err <= tol;
  std::printf("  %-34s max rel err %-12.3g %s\n", what.c_str(), err, ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  double scale = 1e-6;
  for (const float v : b) scale = std::max(scale, static_cast<double>(std::fabs(v)));
  for (std::size_t i = 0; i < a.size(); ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]) / scale);
  return worst;
}

template <typename T>
T* to_dev(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <typename T>
std::vector<T> to_host(const T* d, std::size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}
std::vector<bf16> as_bf16(const std::vector<float>& f) {
  std::vector<bf16> o(f.size());
  for (std::size_t i = 0; i < f.size(); ++i) o[i] = __float2bfloat16(f[i]);
  return o;
}
std::vector<float> as_float(const std::vector<bf16>& b) {
  std::vector<float> o(b.size());
  for (std::size_t i = 0; i < b.size(); ++i) o[i] = __bfloat162float(b[i]);
  return o;
}

std::mt19937 rng(20260907);
std::vector<float> randn(std::size_t n, float s = 1.0f) {
  std::normal_distribution<float> d(0.0f, s);
  std::vector<float> v(n);
  for (float& x : v) x = bf(d(rng));
  return v;
}

float sigmoidf_h(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ------------------------------------------------------------------ gemv
void test_gemv() {
  const int N = 257, K = 512;
  const std::vector<float> w = randn(N * K), x = randn(K);
  std::vector<float> ref(N);
  for (int n = 0; n < N; ++n) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += w[n * K + k] * x[k];
    ref[n] = acc;
  }
  auto dw = to_dev(as_bf16(w));
  auto dx = to_dev(as_bf16(x));
  float* dy = nullptr;
  cudaMalloc(&dy, N * sizeof(float));
  rocket::engine::gemv_bf16_f32(dy, dw, dx, N, K, nullptr);
  cudaDeviceSynchronize();
  check("gemv_bf16_f32", rel_err(to_host(dy, N), ref), 2e-3);
  cudaFree(dw); cudaFree(dx); cudaFree(dy);
}

// ------------------------------------------------- hyper-connections (mHC)
// Reference: Glm5NextTextHyperConnection.forward.
void test_hyperconnection() {
  const int hc = 4, hidden = 256, mix_n = (2 + hc) * hc, flat = hc * hidden;
  const float eps = 1e-5f, hc_eps = 1e-6f;
  const int iters = 20;
  const std::vector<float> streams = randn(flat), fn = randn(mix_n * flat, 0.05f);
  const std::vector<float> base = randn(mix_n, 0.5f), scale = randn(3, 0.5f);
  const std::vector<float> y = randn(hidden), residual = randn(flat);

  // --- CPU reference ---
  double ss = 0.0;
  for (const float v : streams) ss += static_cast<double>(v) * v;
  const float inv = 1.0f / std::sqrt(static_cast<float>(ss / flat) + eps);
  std::vector<float> mix(mix_n);
  for (int r = 0; r < mix_n; ++r) {
    float acc = 0.0f;
    for (int i = 0; i < flat; ++i) acc += fn[r * flat + i] * (streams[i] * inv);
    mix[r] = acc;
  }
  std::vector<float> pre(hc), post(hc), comb(hc * hc);
  for (int i = 0; i < hc; ++i) pre[i] = sigmoidf_h(mix[i] * scale[0] + base[i]) + hc_eps;
  for (int i = 0; i < hc; ++i) post[i] = 2.0f * sigmoidf_h(mix[hc + i] * scale[1] + base[hc + i]);
  for (int r = 0; r < hc; ++r) {
    float mx = -1e30f;
    for (int c = 0; c < hc; ++c) {
      comb[r * hc + c] = mix[2 * hc + r * hc + c] * scale[2] + base[2 * hc + r * hc + c];
      mx = std::max(mx, comb[r * hc + c]);
    }
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) { comb[r * hc + c] = std::exp(comb[r * hc + c] - mx); sum += comb[r * hc + c]; }
    for (int c = 0; c < hc; ++c) comb[r * hc + c] = comb[r * hc + c] / sum + hc_eps;
  }
  for (int c = 0; c < hc; ++c) {
    float s = 0.0f;
    for (int r = 0; r < hc; ++r) s += comb[r * hc + c];
    for (int r = 0; r < hc; ++r) comb[r * hc + c] /= (s + hc_eps);
  }
  for (int it = 0; it < iters - 1; ++it) {
    for (int r = 0; r < hc; ++r) {
      float s = 0.0f;
      for (int c = 0; c < hc; ++c) s += comb[r * hc + c];
      for (int c = 0; c < hc; ++c) comb[r * hc + c] /= (s + hc_eps);
    }
    for (int c = 0; c < hc; ++c) {
      float s = 0.0f;
      for (int r = 0; r < hc; ++r) s += comb[r * hc + c];
      for (int r = 0; r < hc; ++r) comb[r * hc + c] /= (s + hc_eps);
    }
  }
  std::vector<float> collapsed(hidden), out(flat);
  for (int d = 0; d < hidden; ++d) {
    float acc = 0.0f;
    for (int i = 0; i < hc; ++i) acc += pre[i] * streams[i * hidden + d];
    collapsed[d] = bf(acc);
  }
  for (int i = 0; i < hc; ++i)
    for (int d = 0; d < hidden; ++d) {
      float acc = post[i] * y[d];
      for (int j = 0; j < hc; ++j) acc += comb[j * hc + i] * residual[j * hidden + d];
      out[i * hidden + d] = acc;
    }

  // A doubly-stochastic comb is the point of the Sinkhorn projection.
  double worst_row = 0.0, worst_col = 0.0;
  for (int r = 0; r < hc; ++r) {
    double s = 0.0;
    for (int c = 0; c < hc; ++c) s += comb[r * hc + c];
    worst_row = std::max(worst_row, std::fabs(s - 1.0));
  }
  for (int c = 0; c < hc; ++c) {
    double s = 0.0;
    for (int r = 0; r < hc; ++r) s += comb[r * hc + c];
    worst_col = std::max(worst_col, std::fabs(s - 1.0));
  }

  // --- device ---
  auto dstreams = to_dev(as_bf16(streams));
  auto dfn = to_dev(as_bf16(fn));
  auto dbase = to_dev(base);
  auto dscale = to_dev(scale);
  auto dy = to_dev(as_bf16(y));
  auto dres = to_dev(as_bf16(residual));
  float *dmix, *dpost, *dcomb;
  bf16 *dcol, *dout;
  cudaMalloc(&dmix, mix_n * sizeof(float));
  cudaMalloc(&dpost, hc * sizeof(float));
  cudaMalloc(&dcomb, hc * hc * sizeof(float));
  cudaMalloc(&dcol, hidden * sizeof(bf16));
  cudaMalloc(&dout, flat * sizeof(bf16));
  rocket::engine::hc_mix_gemv(dmix, dfn, dstreams, mix_n, hc, hidden, eps, nullptr);
  rocket::engine::hc_split(dpost, dcomb, dcol, dmix, dbase, dscale, dstreams, hc, hidden, hc_eps,
                           iters, nullptr);
  rocket::engine::hc_combine(dout, dpost, dy, dcomb, dres, hc, hidden, nullptr);
  cudaDeviceSynchronize();

  check("mHC mix logits", rel_err(to_host(dmix, mix_n), mix), 3e-3);
  check("mHC post weights", rel_err(to_host(dpost, hc), post), 1e-4);
  check("mHC comb (Sinkhorn)", rel_err(to_host(dcomb, hc * hc), comb), 1e-4);
  check("mHC collapsed stream", rel_err(as_float(to_host(dcol, hidden)), collapsed), 5e-3);
  check("mHC combine", rel_err(as_float(to_host(dout, flat)), out), 5e-3);
  std::printf("  %-34s row %.2e col %.2e %s\n", "mHC comb doubly stochastic", worst_row, worst_col,
              (worst_row < 1e-3 && worst_col < 1e-3) ? "ok" : "FAIL");
  if (!(worst_row < 1e-3 && worst_col < 1e-3)) ++failures;
}

// ------------------------------------------------------------------- KDA
// Reference: recurrent_kimi_delta_attention, plus the conv update, the forget
// gate, and the gated output norm around it.
void test_kda() {
  const int heads = 8, hd = 128, qkv = heads * hd, kernel = 4, taps = 3;
  const float lower = -5.0f, eps = 1e-5f;
  const std::vector<float> in = randn(3 * qkv), cw = randn(3 * qkv * kernel, 0.3f);
  std::vector<float> cstate = randn(3 * qkv * taps);
  const std::vector<float> fb = randn(qkv), dt = randn(qkv), alog = randn(heads, 0.3f);
  const std::vector<float> beta_raw = randn(heads);
  std::vector<float> state = randn(heads * hd * hd, 0.1f);

  // conv + silu
  std::vector<float> conv_ref(3 * qkv);
  std::vector<float> cstate_ref = cstate;
  for (int c = 0; c < 3 * qkv; ++c) {
    float acc = 0.0f;
    for (int t = 0; t < taps; ++t) acc += cw[c * kernel + t] * cstate[c * taps + t];
    acc += cw[c * kernel + taps] * in[c];
    for (int t = 0; t < taps - 1; ++t) cstate_ref[c * taps + t] = cstate[c * taps + t + 1];
    cstate_ref[c * taps + taps - 1] = bf(in[c]);
    conv_ref[c] = bf(acc * sigmoidf_h(acc));
  }
  // l2 norm on q and k, q scaled
  std::vector<float> q(conv_ref.begin(), conv_ref.begin() + qkv);
  std::vector<float> k(conv_ref.begin() + qkv, conv_ref.begin() + 2 * qkv);
  const std::vector<float> v(conv_ref.begin() + 2 * qkv, conv_ref.end());
  const float qs = 1.0f / std::sqrt(static_cast<float>(hd));
  for (int h = 0; h < heads; ++h) {
    float sq = 0.0f, sk = 0.0f;
    for (int i = 0; i < hd; ++i) { sq += q[h * hd + i] * q[h * hd + i]; sk += k[h * hd + i] * k[h * hd + i]; }
    const float iq = 1.0f / std::sqrt(sq + 1e-6f), ik = 1.0f / std::sqrt(sk + 1e-6f);
    for (int i = 0; i < hd; ++i) {
      q[h * hd + i] = bf(q[h * hd + i] * iq * qs);
      k[h * hd + i] = bf(k[h * hd + i] * ik);
    }
  }
  // forget gate and beta
  std::vector<float> g(qkv), beta(heads);
  for (int i = 0; i < qkv; ++i)
    g[i] = bf(lower * sigmoidf_h(std::exp(alog[i / hd]) * (fb[i] + dt[i])));
  for (int h = 0; h < heads; ++h) beta[h] = bf(sigmoidf_h(beta_raw[h]));
  // recurrence
  std::vector<float> o(qkv);
  std::vector<float> state_ref = state;
  for (int h = 0; h < heads; ++h) {
    float* S = state_ref.data() + static_cast<std::size_t>(h) * hd * hd;
    std::vector<float> decay(hd);
    for (int i = 0; i < hd; ++i) decay[i] = std::exp(g[h * hd + i]);
    for (int i = 0; i < hd; ++i)
      for (int j = 0; j < hd; ++j) S[i * hd + j] *= decay[i];
    std::vector<float> kv(hd, 0.0f);
    for (int j = 0; j < hd; ++j) {
      float acc = 0.0f;
      for (int i = 0; i < hd; ++i) acc += S[i * hd + j] * k[h * hd + i];
      kv[j] = acc;
    }
    for (int j = 0; j < hd; ++j) {
      const float d = beta[h] * (v[h * hd + j] - kv[j]);
      for (int i = 0; i < hd; ++i) S[i * hd + j] += k[h * hd + i] * d;
    }
    for (int j = 0; j < hd; ++j) {
      float acc = 0.0f;
      for (int i = 0; i < hd; ++i) acc += S[i * hd + j] * q[h * hd + i];
      o[h * hd + j] = acc;
    }
  }

  auto din = to_dev(as_bf16(in));
  auto dcw = to_dev(as_bf16(cw));
  auto dcs = to_dev(as_bf16(cstate));
  bf16* dconv = nullptr;
  cudaMalloc(&dconv, 3 * qkv * sizeof(bf16));
  rocket::engine::kda_conv_update(dconv, dcs, din, dcw, 3 * qkv, kernel, nullptr);
  cudaDeviceSynchronize();
  check("kda conv + silu", rel_err(as_float(to_host(dconv, 3 * qkv)), conv_ref), 5e-3);
  check("kda conv state advance", rel_err(as_float(to_host(dcs, 3 * qkv * taps)), cstate_ref), 1e-6);

  rocket::engine::kda_norm_qk(dconv, dconv + qkv, heads, hd, nullptr);
  cudaDeviceSynchronize();
  auto got_q = as_float(to_host(dconv, qkv));
  auto got_k = as_float(to_host(dconv + qkv, qkv));
  check("kda q l2norm + scale", rel_err(got_q, q), 5e-3);
  check("kda k l2norm", rel_err(got_k, k), 5e-3);

  auto dfb = to_dev(as_bf16(fb));
  auto ddt = to_dev(dt);
  auto dal = to_dev(alog);
  bf16* dg = nullptr;
  cudaMalloc(&dg, qkv * sizeof(bf16));
  rocket::engine::kda_forget_gate(dg, dfb, ddt, dal, heads, hd, lower, nullptr);
  cudaDeviceSynchronize();
  check("kda forget gate", rel_err(as_float(to_host(dg, qkv)), g), 5e-3);

  auto dstate = to_dev(state);
  auto dbeta = to_dev(as_bf16(beta));
  bf16* dout = nullptr;
  cudaMalloc(&dout, qkv * sizeof(bf16));
  rocket::engine::kda_recurrent_step(dstate, dout, dconv, dconv + qkv, dconv + 2 * qkv, dg, dbeta,
                                     heads, hd, nullptr);
  cudaDeviceSynchronize();
  check("kda recurrent readout", rel_err(as_float(to_host(dout, qkv)), o), 2e-2);
  check("kda recurrent state", rel_err(to_host(dstate, heads * hd * hd), state_ref), 2e-2);
}

// ------------------------------------------------------------- MoE router
// Reference: Glm5NextTextTopkRouter.forward with n_group == 1.
void test_router() {
  const int E = 288, K = 8;
  const float scale = 2.5f;
  const std::vector<float> logits = randn(E), bias = randn(E, 0.2f);
  std::vector<float> scores(E), choice(E);
  for (int i = 0; i < E; ++i) {
    scores[i] = sigmoidf_h(logits[i]);
    choice[i] = scores[i] + bias[i];
  }
  std::vector<int> ref_idx;
  std::vector<float> tmp = choice;
  for (int t = 0; t < K; ++t) {
    int best = 0;
    for (int i = 1; i < E; ++i)
      if (tmp[i] > tmp[best]) best = i;
    ref_idx.push_back(best);
    tmp[best] = -1e30f;
  }
  float sum = 0.0f;
  for (const int i : ref_idx) sum += scores[i];
  std::vector<float> ref_w;
  for (const int i : ref_idx) ref_w.push_back(scores[i] / (sum + 1e-20f) * scale);

  auto dl = to_dev(logits);
  auto db = to_dev(bias);
  int* di = nullptr;
  float* dw = nullptr;
  cudaMalloc(&di, K * sizeof(int));
  cudaMalloc(&dw, K * sizeof(float));
  rocket::engine::moe_router(di, dw, dl, db, E, K, true, scale, nullptr);
  cudaDeviceSynchronize();
  const auto got_i = to_host(di, K);
  bool idx_ok = true;
  for (int t = 0; t < K; ++t) idx_ok = idx_ok && (got_i[t] == ref_idx[t]);
  std::printf("  %-34s %s\n", "moe router top-8 indices", idx_ok ? "ok" : "FAIL");
  if (!idx_ok) ++failures;
  check("moe router weights", rel_err(to_host(dw, K), ref_w), 1e-5);

  double wsum = 0.0;
  for (const float v : to_host(dw, K)) wsum += v;
  std::printf("  %-34s %.6f (expect %.4f) %s\n", "moe router weight sum", wsum, scale,
              std::fabs(wsum - scale) < 1e-4 ? "ok" : "FAIL");
  if (std::fabs(wsum - scale) >= 1e-4) ++failures;
}

// ---------------------------------------------------------- clamped SwiGLU
void test_swiglu() {
  const int n = 4096;
  const float limit = 10.0f;
  const std::vector<float> g = randn(n, 8.0f), u = randn(n, 8.0f);
  std::vector<float> ref(n);
  for (int i = 0; i < n; ++i) {
    const float gc = std::min(g[i], limit);
    const float uc = std::min(std::max(u[i], -limit), limit);
    ref[i] = gc * sigmoidf_h(gc) * uc;
  }
  auto dg = to_dev(as_bf16(g));
  auto du = to_dev(as_bf16(u));
  bf16* dout = nullptr;
  cudaMalloc(&dout, n * sizeof(bf16));
  rocket::engine::swiglu_clamped(dout, dg, du, n, limit, nullptr);
  cudaDeviceSynchronize();
  check("clamped swiglu (limit 10)", rel_err(as_float(to_host(dout, n)), ref), 5e-3);
}

// -------------------------------------------------------------- NVFP4 GEMV
// Reference is the host codec pair in nvfp4.cc, which the loader test already
// checks against CUTLASS, so this compares the kernel to an audited decoder.
void test_nvfp4_gemv() {
  const int N = 64, K = 512;
  std::uniform_int_distribution<int> nib(0, 255);
  std::vector<std::uint8_t> packed(static_cast<std::size_t>(N) * K / 2);
  std::vector<std::uint8_t> sf(static_cast<std::size_t>(N) * K / 16);
  for (auto& v : packed) v = static_cast<std::uint8_t>(nib(rng));
  for (auto& v : sf) v = static_cast<std::uint8_t>(0x30 + (nib(rng) % 24));  // sane e4m3 range
  const float global = 4.65e-05f;
  const std::vector<float> x = randn(K);

  std::vector<float> ref(N);
  for (int n = 0; n < N; ++n) {
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
      const std::uint8_t byte = packed[static_cast<std::size_t>(n) * (K / 2) + k / 2];
      const std::uint8_t nb = (k & 1) ? (byte >> 4) : (byte & 0x0F);
      const float w = rocket::fuel::e2m1_to_float(nb) *
                      rocket::fuel::e4m3_to_float(sf[static_cast<std::size_t>(n) * (K / 16) + k / 16]);
      acc += static_cast<double>(w) * x[k];
    }
    ref[n] = static_cast<float>(acc * global);
  }

  auto dp = to_dev(packed);
  auto ds = to_dev(sf);
  auto dx = to_dev(as_bf16(x));
  bf16* dy = nullptr;
  cudaMalloc(&dy, N * sizeof(bf16));
  rocket::engine::gemv_nvfp4(dy, dp, ds, global, dx, N, K, nullptr);
  cudaDeviceSynchronize();
  check("nvfp4 gemv vs host codec", rel_err(as_float(to_host(dy, N)), ref), 1e-2);
}

// ------------------------------------------------------------- DSA indexer
// Pool compression is a per-channel softmax over the kpool members, and the
// selection must return the true top-k when the candidate set is larger than
// the budget (the radix path).
void test_indexer() {
  const int kpool = 4, hd = 128, n_pools = 37, heads = 32;
  const std::vector<float> keys = randn(n_pools * kpool * hd);
  const std::vector<float> gates = randn(n_pools * kpool * hd);
  const std::vector<float> ape = randn(kpool * hd, 0.5f);
  std::vector<float> ref(n_pools * hd);
  for (int p = 0; p < n_pools; ++p)
    for (int c = 0; c < hd; ++c) {
      float mx = -1e30f;
      float l[8];
      for (int i = 0; i < kpool; ++i) {
        l[i] = gates[((p * kpool + i) * hd) + c] + ape[i * hd + c];
        mx = std::max(mx, l[i]);
      }
      float sum = 0.0f;
      for (int i = 0; i < kpool; ++i) { l[i] = std::exp(l[i] - mx); sum += l[i]; }
      float acc = 0.0f;
      for (int i = 0; i < kpool; ++i) acc += (l[i] / sum) * keys[((p * kpool + i) * hd) + c];
      ref[p * hd + c] = acc;
    }
  auto dk = to_dev(as_bf16(keys));
  auto dg = to_dev(as_bf16(gates));
  auto da = to_dev(as_bf16(ape));
  bf16* dpk = nullptr;
  cudaMalloc(&dpk, n_pools * hd * sizeof(bf16));
  rocket::engine::indexer_pool_keys(dpk, dk, dg, da, n_pools, kpool, hd, nullptr);
  cudaDeviceSynchronize();
  check("indexer pool compression", rel_err(as_float(to_host(dpk, n_pools * hd)), ref), 5e-3);

  // scores: relu inside the per-head dot, then the weighted sum over heads
  const std::vector<float> q = randn(heads * hd), hw = randn(heads);
  std::vector<float> sref(n_pools, 0.0f);
  const float sc = 1.0f / std::sqrt(static_cast<float>(hd));
  const std::vector<float> pk = as_float(to_host(dpk, n_pools * hd));
  for (int p = 0; p < n_pools; ++p) {
    float total = 0.0f;
    for (int h = 0; h < heads; ++h) {
      float dot = 0.0f;
      for (int c = 0; c < hd; ++c) dot += q[h * hd + c] * pk[p * hd + c];
      total += hw[h] * std::max(dot * sc, 0.0f);
    }
    sref[p] = total;
  }
  auto dq = to_dev(as_bf16(q));
  auto dhw = to_dev(hw);
  float* dsc = nullptr;
  cudaMalloc(&dsc, n_pools * sizeof(float));
  rocket::engine::indexer_scores(dsc, dq, dpk, dhw, n_pools, heads, hd, nullptr);
  cudaDeviceSynchronize();
  check("indexer scores (relu inside)", rel_err(to_host(dsc, n_pools), sref), 1e-2);

  // selection, budget smaller than the candidate set: exercises the radix path
  const int select_k = 12;
  int *dsel = nullptr, *dn = nullptr;
  cudaMalloc(&dsel, select_k * sizeof(int));
  cudaMalloc(&dn, sizeof(int));
  rocket::engine::indexer_select(dsel, dn, dsc, n_pools, select_k, nullptr);
  cudaDeviceSynchronize();
  const int n_sel = to_host(dn, 1)[0];
  const auto sel = to_host(dsel, select_k);
  std::vector<float> sorted = sref;
  std::sort(sorted.begin(), sorted.end(), std::greater<float>());
  const float cut = sorted[select_k - 1];
  bool sel_ok = (n_sel == select_k);
  for (int i = 0; i < n_sel; ++i) sel_ok = sel_ok && sel[i] >= 0 && sel[i] < n_pools && sref[sel[i]] >= cut;
  std::printf("  %-34s %d selected, all >= rank-%d score %s\n", "indexer radix select", n_sel,
              select_k, sel_ok ? "ok" : "FAIL");
  if (!sel_ok) ++failures;

  // budget larger than the candidate set: everything is selected
  rocket::engine::indexer_select(dsel, dn, dsc, 8, select_k, nullptr);
  cudaDeviceSynchronize();
  const bool all_ok = to_host(dn, 1)[0] == 8;
  std::printf("  %-34s %s\n", "indexer select takes all when P<=k", all_ok ? "ok" : "FAIL");
  if (!all_ok) ++failures;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("no CUDA device\n");
    return 77;
  }
  std::printf("gemv\n");            test_gemv();
  std::printf("hyper-connections\n"); test_hyperconnection();
  std::printf("kda\n");             test_kda();
  std::printf("moe router\n");      test_router();
  std::printf("swiglu\n");          test_swiglu();
  std::printf("nvfp4 gemv\n");      test_nvfp4_gemv();
  std::printf("dsa indexer\n");     test_indexer();
  std::printf("\n%s: %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}

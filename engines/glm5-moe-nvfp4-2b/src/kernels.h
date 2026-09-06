// Decode-step kernels for glm5-moe-288e-top8-kda-dsa, batch 1.
//
// Every launcher takes a stream and returns immediately. Shapes are the fuel's
// shapes, passed as arguments rather than compiled in, so one build serves the
// whole layer stack; the values come from ModelConfig, which is itself read off
// attention.yaml and config.json.
//
// Activations are BF16. Reductions accumulate in FP32, which is what the
// reference does (it casts to float inside every norm, the router, and the
// KDA recurrence).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace rocket::engine {

using bf16 = __nv_bfloat16;

// y[n] = sum_k W[n,k] * x[k].  W is row-major [n_rows, k].
void gemv_bf16(bf16* y, const bf16* w, const bf16* x, int n_rows, int k, cudaStream_t s);
// Same, accumulating into an FP32 destination instead of BF16.
void gemv_bf16_f32(float* y, const bf16* w, const bf16* x, int n_rows, int k, cudaStream_t s);

// out = weight * (x * rsqrt(mean(x^2) + eps)), FP32 interior.
void rmsnorm(bf16* out, const bf16* x, const bf16* weight, int n, float eps, cudaStream_t s);

// --- manifold-constrained hyper-connections -------------------------------
// mix = fn @ rmsnorm_unweighted(flatten(streams)), then the split into
// pre / post / comb, the Sinkhorn projection of comb, and the stream collapse.
// `mix` is scratch of hc_mix floats; `post` is hc floats; `comb` is hc*hc.
void hc_mix_gemv(float* mix, const bf16* fn, const bf16* streams, int hc_mix, int hc, int hidden,
                 float eps, cudaStream_t s);
void hc_split(float* post, float* comb, bf16* collapsed, const float* mix, const float* base,
              const float* scale, const bf16* streams, int hc, int hidden, float hc_eps,
              int sinkhorn_iters, cudaStream_t s);
// streams_out[i] = post[i] * y + sum_j comb[j][i] * residual[j]
void hc_combine(bf16* streams_out, const float* post, const bf16* y, const float* comb,
                const bf16* residual, int hc, int hidden, cudaStream_t s);
// out = mean over the hc stream axis.
void hc_head_mean(bf16* out, const bf16* streams, int hc, int hidden, cudaStream_t s);

// --- KDA ------------------------------------------------------------------
// Depthwise causal conv over the 3 stored taps plus the new position, then
// silu. `state` is [channels, taps] and is advanced in place. qkv is
// [channels] in, [channels] out.
void kda_conv_update(bf16* out, bf16* state, const bf16* in, const bf16* weight, int channels,
                     int kernel, cudaStream_t s);
// g = lower_bound * sigmoid(exp(A_log[h]) * (fb + dt_bias)), per (head, channel).
void kda_forget_gate(bf16* g, const bf16* fb, const float* dt_bias, const float* a_log, int heads,
                     int head_dim, float lower_bound, cudaStream_t s);
// L2-normalise each head row (eps inside the sqrt, as FLA does) and scale q.
void kda_norm_qk(bf16* q, bf16* k, int heads, int head_dim, cudaStream_t s);
void kda_sigmoid(bf16* out, const bf16* in, int n, cudaStream_t s);
// One recurrent step: S = diag(exp(g)) S; delta = beta*(v - S^T k); S += k delta^T; o = S^T q.
void kda_recurrent_step(float* state, bf16* o, const bf16* q, const bf16* k, const bf16* v,
                        const bf16* g, const bf16* beta, int heads, int head_dim, cudaStream_t s);
// RMSNorm over head_dim with weight, then multiplied by sigmoid(gate).
void kda_gated_norm(bf16* out, const bf16* x, const bf16* gate, const bf16* weight, int heads,
                    int head_dim, float eps, cudaStream_t s);

// --- MLA, absorbed ---------------------------------------------------------
// q_absorbed[h][c] = sum_d Wk[h][d][c] * q[h][d], with Wk the k_nope half of
// kv_b_proj. kv_b is [heads*(nope+v), kv_lora].
void mla_absorb_q(float* q_abs, const bf16* kv_b, const bf16* q, int heads, int nope, int v_dim,
                  int kv_lora, cudaStream_t s);
// scores over the selected token set, softmax, then context in latent space.
void mla_scores(float* scores, const float* q_abs, const bf16* latent, const int* sel, int n_sel,
                int heads, int kv_lora, float scaling, cudaStream_t s);
void mla_softmax(float* scores, int heads, int n_sel, cudaStream_t s);
void mla_context(float* ctx, const float* scores, const bf16* latent, const int* sel, int n_sel,
                 int heads, int kv_lora, cudaStream_t s);
// out[h][v] = sum_c Wv[h][v][c] * ctx[h][c]
void mla_expand_v(bf16* out, const bf16* kv_b, const float* ctx, int heads, int nope, int v_dim,
                  int kv_lora, cudaStream_t s);

// --- DSA indexer -----------------------------------------------------------
// LayerNorm (weight and bias, eps 1e-6) over the indexer key.
void layernorm(bf16* out, const bf16* x, const bf16* w, const bf16* b, int n, float eps,
               cudaStream_t s);
// pool_key[p][c] = sum_i softmax_i(gate[4p+i][c] + ape[i][c]) * key[4p+i][c]
void indexer_pool_keys(bf16* pool_keys, const bf16* keys, const bf16* gates, const bf16* ape,
                       int n_pools, int kpool, int head_dim, cudaStream_t s);
// score[p] = sum_h w[h] * relu(q[h] . pool_key[p] * head_dim^-0.5)
void indexer_scores(float* scores, const bf16* q, const bf16* pool_keys, const float* head_w,
                    int n_pools, int heads, int head_dim, cudaStream_t s);
// Selects up to select_k pool ids by score. Writes the count to n_sel_out
// (device int). Uses an 8-bit radix pass when the candidate set is larger than
// the budget, and takes everything when it is not.
void indexer_select(int* sel_pools, int* n_sel_out, const float* scores, int n_pools, int select_k,
                    cudaStream_t s);
// Expands selected pools into token ids and appends the incomplete tail.
void indexer_expand(int* sel_tokens, int* n_tokens_out, const int* sel_pools, const int* n_sel,
                    int kpool, int n_tokens, cudaStream_t s);

// --- MoE -------------------------------------------------------------------
// sigmoid(logits), + correction bias, top-k, gather unbiased scores, optional
// renormalise, then scale. One block.
void moe_router(int* topk_idx, float* topk_w, const float* logits, const float* bias, int n_experts,
                int top_k, bool norm_topk, float scale, cudaStream_t s);
// clamped SwiGLU: silu(min(gate, limit)) * clamp(up, -limit, limit)
void swiglu_clamped(bf16* out, const bf16* gate, const bf16* up, int n, float limit, cudaStream_t s);

// NVFP4 GEMV. Packed weight is [n_rows, k/2] with the low nibble holding the
// even column; block scales are the checkpoint's own row-major [n_rows, k/16]
// e4m3, not the CUTLASS swizzle, because a decode-shaped GEMV reads them
// linearly and a swizzle would only add a pass over every streamed expert.
void gemv_nvfp4(bf16* y, const std::uint8_t* packed, const std::uint8_t* block_scale,
                float global_scale, const bf16* x, int n_rows, int k, cudaStream_t s);

// acc += w * v, elementwise, BF16 accumulator with an FP32 interior.
void axpy_bf16(bf16* acc, const bf16* v, const float* w, int widx, int n, cudaStream_t s);
void add_bf16(bf16* acc, const bf16* v, int n, cudaStream_t s);

// argmax over the vocabulary, plus the max logit, for greedy decode.
void argmax_f32(int* idx_out, float* val_out, const float* x, int n, cudaStream_t s);
// sum of squares, used for the per-layer hidden-state RMS check.
void sumsq_bf16(float* out, const bf16* x, int n, cudaStream_t s);

}  // namespace rocket::engine

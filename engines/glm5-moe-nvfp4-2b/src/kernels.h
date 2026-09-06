// Decode-step kernels for glm5-moe-288e-top8-kda-dsa, M concurrent streams.
//
// Every launcher takes a stream and returns immediately. Shapes are the fuel's
// shapes, passed as arguments rather than compiled in, so one build serves the
// whole layer stack; the values come from ModelConfig, which is itself read off
// attention.yaml and config.json.
//
// Activations are BF16 and batched: an activation of width n is [batch, n]
// row-major. Every kernel replicates its M=1 block over blockIdx.y, so the
// reduction order inside one stream does not depend on the batch size and a
// stream decodes bit-for-bit the same at any M. The routed-expert path is the
// one exception (grouped GEMM, moe_grouped.h).
//
// Reductions accumulate in FP32, which is what the reference does (it casts to
// float inside every norm, the router, and the KDA recurrence).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace rocket::engine {

using bf16 = __nv_bfloat16;

// Paged KV for the sparse-MLA layers. One page carries `page_tokens` token
// slots for every MLA layer, so a stream's page list is shared by all layers
// and the block table is [batch, max_pages]. Token t of stream s sits in
// physical page table[s * max_pages + t / page_tokens] at slot t % page_tokens.
//
// Per token per layer: kv_lora latent (1024 B) plus the indexer key and gate
// (256 B each, 512 B together, fuels/glm-5.3-flash/attention.yaml).
struct KvPages {
  bf16* latent = nullptr;  // [pages][layers][page_tokens][kv_lora]
  bf16* key = nullptr;     // [pages][layers][page_tokens][index_head_dim]
  bf16* gate = nullptr;    // [pages][layers][page_tokens][index_head_dim]
  const int* table = nullptr;
  int max_pages = 0;
  int page_tokens = 0;
  int layers = 0;
  int kv_lora = 0;
  int index_head_dim = 0;
};

// --- batched dense GEMM ----------------------------------------------------
// y[m, n_rows] = x[m, k] . W[n_rows, k]^T.  W is row-major [n_rows, k] and is
// shared by every stream.
void gemm_bf16(bf16* y, const bf16* w, const bf16* x, int batch, int n_rows, int k, cudaStream_t s);
// Same, accumulating into an FP32 destination instead of BF16.
void gemm_bf16_f32(float* y, const bf16* w, const bf16* x, int batch, int n_rows, int k,
                   cudaStream_t s);

// --- norms -----------------------------------------------------------------
// out = weight * (x * rsqrt(mean(x^2) + eps)), FP32 interior, per row of batch.
void rmsnorm(bf16* out, const bf16* x, const bf16* weight, int batch, int n, float eps,
             cudaStream_t s);
// LayerNorm (weight and bias) over each row of batch.
void layernorm(bf16* out, const bf16* x, const bf16* w, const bf16* b, int batch, int n, float eps,
               cudaStream_t s);

// --- embedding -------------------------------------------------------------
// Every hyper-connection stream of every batch row starts as a copy of that
// row's token embedding. `tokens` is a device array of `batch` ids.
void embed_streams(bf16* streams, const bf16* embed, const int* tokens, int batch, int hc,
                   int hidden, cudaStream_t s);

// --- manifold-constrained hyper-connections --------------------------------
// mix = fn @ rmsnorm_unweighted(flatten(streams)), then the split into
// pre / post / comb, the Sinkhorn projection of comb, and the stream collapse.
// `mix` is [batch, hc_mix]; `post` is [batch, hc]; `comb` is [batch, hc*hc].
void hc_mix_gemv(float* mix, const bf16* fn, const bf16* streams, int batch, int hc_mix, int hc,
                 int hidden, float eps, cudaStream_t s);
void hc_split(float* post, float* comb, bf16* collapsed, const float* mix, const float* base,
              const float* scale, const bf16* streams, int batch, int hc, int hidden, float hc_eps,
              int sinkhorn_iters, cudaStream_t s);
// streams_out[i] = post[i] * y + sum_j comb[j][i] * residual[j]
void hc_combine(bf16* streams_out, const float* post, const bf16* y, const float* comb,
                const bf16* residual, int batch, int hc, int hidden, cudaStream_t s);
// out = mean over the hc stream axis.
void hc_head_mean(bf16* out, const bf16* streams, int batch, int hc, int hidden, cudaStream_t s);

// --- KDA -------------------------------------------------------------------
// Depthwise causal conv over the 3 stored taps plus the new position, then
// silu. `state` is [batch, channels, taps] and is advanced in place.
void kda_conv_update(bf16* out, bf16* state, const bf16* in, const bf16* weight, int batch,
                     int channels, int kernel, cudaStream_t s);
// g = lower_bound * sigmoid(exp(A_log[h]) * (fb + dt_bias)), per (head, channel).
void kda_forget_gate(bf16* g, const bf16* fb, const float* dt_bias, const float* a_log, int batch,
                     int heads, int head_dim, float lower_bound, cudaStream_t s);
// L2-normalise each head row (eps inside the sqrt, as FLA does) and scale q.
void kda_norm_qk(bf16* q, bf16* k, int batch, int heads, int head_dim, int row_stride,
                 cudaStream_t s);
void kda_sigmoid(bf16* out, const bf16* in, int n, cudaStream_t s);
// One recurrent step: S = diag(exp(g)) S; delta = beta*(v - S^T k); S += k delta^T; o = S^T q.
// `state` is [batch, heads, head_dim, head_dim].
void kda_recurrent_step(float* state, bf16* o, const bf16* q, const bf16* k, const bf16* v,
                        const bf16* g, const bf16* beta, int batch, int heads, int head_dim,
                        int row_stride, cudaStream_t s);
// RMSNorm over head_dim with weight, then multiplied by sigmoid(gate).
void kda_gated_norm(bf16* out, const bf16* x, const bf16* gate, const bf16* weight, int batch,
                    int heads, int head_dim, float eps, cudaStream_t s);

// --- paged KV writes -------------------------------------------------------
// Append this step's latent / indexer key / indexer gate for every stream at
// its own position. `pos` is a device array of `batch` positions.
void kv_write_latent(const KvPages& kv, const bf16* src, const int* pos, int batch, int layer_slot,
                     cudaStream_t s);
void kv_write_index(const KvPages& kv, const bf16* key_src, const bf16* gate_src, const int* pos,
                    int batch, int layer_slot, cudaStream_t s);

// --- MLA, absorbed ---------------------------------------------------------
// q_absorbed[h][c] = sum_d Wk[h][d][c] * q[h][d], with Wk the k_nope half of
// kv_b_proj. kv_b is [heads*(nope+v), kv_lora].
void mla_absorb_q(float* q_abs, const bf16* kv_b, const bf16* q, int batch, int heads, int nope,
                  int v_dim, int kv_lora, cudaStream_t s);
// scores over each stream's own selected token set, read through the page
// table. `n_sel` is a device array of per-stream counts; `sel_stride` is the
// padded width of both the selection and the score rows.
// `n_sel_max` is the host's bound on n_sel over the batch and only sizes the
// launch; no kernel reads past a stream's own n_sel, so the padding in a row
// never reaches the softmax.
void mla_scores(float* scores, const float* q_abs, const KvPages& kv, const int* sel,
                const int* n_sel, int n_sel_max, int sel_stride, int batch, int layer_slot,
                int heads, int kv_lora, float scaling, cudaStream_t s);
void mla_softmax(float* scores, const int* n_sel, int sel_stride, int batch, int heads,
                 cudaStream_t s);
void mla_context(float* ctx, const float* scores, const KvPages& kv, const int* sel,
                 const int* n_sel, int n_sel_max, int sel_stride, int batch, int layer_slot,
                 int heads, int kv_lora, cudaStream_t s);
// out[h][v] = sum_c Wv[h][v][c] * ctx[h][c]
void mla_expand_v(bf16* out, const bf16* kv_b, const float* ctx, int batch, int heads, int nope,
                  int v_dim, int kv_lora, cudaStream_t s);

// --- DSA indexer -----------------------------------------------------------
// pool_key[p][c] = sum_i softmax_i(gate[4p+i][c] + ape[i][c]) * key[4p+i][c],
// over each stream's own pool count (n_pools[m] = tokens[m] / kpool).
// `n_pools_max` is the host's bound on n_pools over the batch; it only sizes
// the launch, so no kernel reads past a stream's own n_pools.
void indexer_pool_keys(bf16* pool_keys, const KvPages& kv, const bf16* ape, const int* n_pools,
                       int n_pools_max, int pool_stride, int batch, int layer_slot, int kpool,
                       int head_dim, cudaStream_t s);
// score[p] = sum_h w[h] * relu(q[h] . pool_key[p] * head_dim^-0.5)
void indexer_scores(float* scores, const bf16* q, const bf16* pool_keys, const float* head_w,
                    const int* n_pools, int n_pools_max, int pool_stride, int batch, int heads,
                    int head_dim, cudaStream_t s);
// Selects up to select_k pool ids by score for each stream. Writes the counts
// to n_sel_out. Uses an 8-bit radix pass when the candidate set is larger than
// the budget, and takes everything when it is not.
void indexer_select(int* sel_pools, int* n_sel_out, const float* scores, const int* n_pools,
                    int n_pools_max, int pool_stride, int sel_pool_stride, int batch, int select_k,
                    cudaStream_t s);
// Expands selected pools into token ids and appends the incomplete tail.
void indexer_expand(int* sel_tokens, int* n_tokens_out, const int* sel_pools, const int* n_sel,
                    const int* n_tokens, int sel_pool_stride, int sel_stride, int batch, int kpool,
                    cudaStream_t s);

// --- MoE -------------------------------------------------------------------
// sigmoid(logits), + correction bias, top-k, gather unbiased scores, optional
// renormalise, then scale. One block per stream.
void moe_router(int* topk_idx, float* topk_w, const float* logits, const float* bias, int batch,
                int n_experts, int top_k, bool norm_topk, float scale, cudaStream_t s);
// clamped SwiGLU: silu(min(gate, limit)) * clamp(up, -limit, limit)
void swiglu_clamped(bf16* out, const bf16* gate, const bf16* up, int n, float limit, cudaStream_t s);
// Same, over the fused w13 output of the grouped path: `gu` is [rows, 2*inter]
// with the gate half first, and each half carries its own NVFP4 per-tensor
// scale, which the grouped GEMM leaves out of its epilogue (alpha is one
// scalar per launch and the two halves do not share a scale).
void swiglu_grouped(bf16* out, const bf16* gu, const float* gate_global, const float* up_global,
                    const int* group_of_row, int rows, int inter, float limit, cudaStream_t s);

// NVFP4 GEMV, the M=1 expert path, called once per (stream, selected expert)
// pair: stage 1 keeps the routed path unbatched. Packed weight is [n_rows,
// k/2] with the low nibble holding the even column; block scales are the
// checkpoint's own row-major [n_rows, k/16] e4m3, not the CUTLASS swizzle,
// because a decode-shaped GEMV reads them linearly and a swizzle would only
// add a pass over every streamed expert. Stage 2's grouped GEMM path swizzles
// the cache instead (see nvfp4_quantize_rows below), which is a change to
// weights.cu's expert cache layout that stage 1 does not make.
void gemv_nvfp4(bf16* y, const std::uint8_t* packed, const std::uint8_t* block_scale,
                float global_scale, const bf16* x, int n_rows, int k, cudaStream_t s);

// Quantize gathered BF16 activations to NVFP4 for the grouped GEMM: one e4m3
// block scale per 16 contiguous columns, written in the CUTLASS SFA swizzle,
// and nibble-packed data. The per-tensor A scale is 1, so the block scale
// alone carries the row's range.
//
// `rows` activations are already sorted into groups (one group per active
// expert this step; see model.cu::run_moe). Packed data is globally
// contiguous in call order, row `i` at `packed + i*(k/2)`, since it carries
// no swizzle. Scale factors are swizzled *within a group*, because CUTLASS's
// SFA layout is sized to that group's own M (nvfp4.h::SfLayout): row `i`
// belongs to group `group_of_row[i]`, is the `row_in_group[i]`-th row of
// that group, and its group's swizzled scale slab starts at byte
// `group_sf_base[group_of_row[i]]` of `sf_swizzled`. The caller owns slab
// sizing and zeroing (a group's tail padding above its own M must be zero,
// same as fuel::swizzle_block_scales).
void nvfp4_quantize_rows(std::uint8_t* packed, std::uint8_t* sf_swizzled, const bf16* x,
                         const int* row_in_group, const int* group_of_row,
                         const long long* group_sf_base, int rows, int k, cudaStream_t s);

// acc[row] += w[row*k_stride + widx] * v[row], elementwise.
void axpy_bf16(bf16* acc, const bf16* v, const float* w, int widx, int w_stride, int batch, int n,
               cudaStream_t s);
void add_bf16(bf16* acc, const bf16* v, int n, cudaStream_t s);
// Scatter expert output rows into the batched accumulator:
// acc[row_of[i]] += weight_of[i] * y[i], for i in [0, rows). One expert row can
// land on the same accumulator row as another only for different experts, so
// the adds are ordered by i to keep the sum reproducible.
void moe_scatter_add(bf16* acc, const bf16* y, const int* row_of, const float* weight_of, int rows,
                     int batch, int n, cudaStream_t s);
// Scatter a contiguous [rows, n] block back into the rows of `out` named by
// `row_of`: out[row_of[i]] = y[i]. The inverse of moe_gather_rows, used by the
// two-booster split to place the peer's returned expert-output rows at their
// true positions when this rank's expert set is not a contiguous id range
// (src/fabric/expert_balance.h).
void moe_scatter_rows(bf16* out, const bf16* y, const int* row_of, int rows, int n, cudaStream_t s);
// Gather the rows of `x` named by `row_of` into a contiguous [rows, n] block.
void moe_gather_rows(bf16* out, const bf16* x, const int* row_of, int rows, int n, cudaStream_t s);

// argmax over the vocabulary for every stream, plus the max logit.
void argmax_f32(int* idx_out, float* val_out, const float* x, int batch, int n, cudaStream_t s);
// sum of squares per batch row, used for the per-layer hidden-state RMS check.
void sumsq_bf16(float* out, const bf16* x, int batch, int n, cudaStream_t s);
// max(|x|) per batch row, used for the telemetry activation-absmax hook.
void absmax_bf16(float* out, const bf16* x, int batch, int n, cudaStream_t s);

}  // namespace rocket::engine

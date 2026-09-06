# GLM-5.3-Flash weight artifact inventory

Machine-checked. Local = gx10-5e36. Peer = gx10-2a13 (192.168.100.11).
Commands run 2026-09-06. No cache modified or deleted.

## 1. Inventory

### 1.1 NVFP4 (from yesterday's NIM deployment)

Path (identical on both nodes):

```
~/.cache/nim-glm53-2.1.2/ngc/hub/models--nim--zai-org--glm-5.3-flash/snapshots/nim-aa28e1f-nvfp4/
```

- Format: sharded safetensors, `model-00001-of-00120.safetensors` .. `model-00120-of-00120.safetensors`, plus `model.safetensors.index.json`, `config.json`, `tokenizer_config.json`, `chat_template.jinja`, `checksums.blake3`.
- On-disk size (`du -shL`, follows the HF-cache blob symlinks): **182 GiB** on both nodes.
- Files are symlinks into `../../blobs/<hash>`; `find` alone (no `-L`) undercounts, `du` without `-L` reports near-zero.
- A second snapshot, `nim-3f1971b-fp8`, exists at the same path level but contains only `config.json` (no weight shards on either node) — an unused/aborted FP8 variant reference, not a usable checkpoint.
- Peer node: byte-identical directory name and size (182 GiB), confirming both nodes hold the same NVFP4 checkpoint content, consistent with the nnodes=2/tp=1 deployment needing matching weights per rank.

```
$ du -shL ~/.cache/nim-glm53-2.1.2/.../nim-aa28e1f-nvfp4
182G
$ ssh 192.168.100.11 'du -shL ~/.cache/nim-glm53-2.1.2/.../nim-aa28e1f-nvfp4'
182G
```

Note: the task mentioned a "~137 GiB persistent cache." That figure does not match anything found here — the on-disk NVFP4 checkpoint measures 182 GiB on both nodes (confirmed by both `du -shL` and a full tensor-byte accounting, see 1.4). The only "137" found in this repo tree is an unrelated `exit 137` (OOM kill) in `spark-stack/GLM-5.3-Flash-NVFP4-2x-DGX-Sparks/TRIAL.md`. Treat "~137 GiB" as unverified/stale until its source is identified.

### 1.2 EXL3 4bpw (Mia-AiLab)

Path (identical on both nodes):

```
~/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/25a44fdbf16862a46b7cc9921142c6c81350af2f/
```

- Format: sharded safetensors, same 120-shard split as the NVFP4 checkpoint (`model-00001-of-00120.safetensors` .. `model-00120-of-00120.safetensors`), plus `quantization_config.json`, `materialization-receipt.json`, `PROVENANCE.md`, `MIRROR.json`, `provenance/source-model-revision.json`.
- On-disk size: **164 GiB** on both nodes, same snapshot hash `25a44fdbf16862a46b7cc9921142c6c81350af2f` present on both, so this is one checkpoint replicated to both nodes, not two different downloads.
- `provenance/source-model-revision.json` names the source explicitly: `zai-org/GLM-5.3-Flash-BF16`, revision `a6c167b62691b2bac901344b65cb651a70f53e43`, `weight_dtype: bfloat16`. This BF16 source checkpoint is **not present on either node** (see 1.5) — the EXL3 quantization was done elsewhere and only the 4bpw result was mirrored here.
- `quantization_config.json`: `quant_method: exl3`, `bits: 4`, `codebook: mcg`, `scope: glm53_routed_experts_only`, `non_routed_dtype_policy: official_source_native`. Only routed-expert MLP weights are EXL3-quantized; everything else (attention, dense layers 0-2, norms, embed/lm_head, hc_* mixing tensors) stays in the source's native dtype (BF16).

### 1.3 Other artifacts found

| artifact | path | size | notes |
| --- | --- | --- | --- |
| DFlash2 external drafter | `~/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2` | 2.2 GiB | present on local only (not checked further, out of scope: not the target fuel's own weights) |
| RedHatAI/GLM-5.3-Flash-NVFP4 | `~/.cache/huggingface/hub/.locks/models--RedHatAI--GLM-5.3-Flash-NVFP4` | lock file only | **no snapshot data on either node** — download was started/reserved and never completed |
| LibertAIDAI/GLM-5.3-Flash-NVFP4 | `~/.cache/huggingface/hub/.locks/models--LibertAIDAI--GLM-5.3-Flash-NVFP4` | lock file only | same — lock only, no data, on either node |
| BF16 source (`zai-org/GLM-5.3-Flash-BF16`) | — | absent | not found anywhere on either node under any cache root searched (`~/.cache/huggingface`, `~/.cache/nim-glm53-2.1.2`) |

### 1.4 Docker images (both stacks still present)

Local (gx10-5e36):

```
ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3        20.9 GB
nim/glm-5.3-flash:2.1.2-variant-modelopt64k-r1             25.7 GB
nim/glm-5.3-flash:2.1.2-variant-modelopt64k-r2             25.7 GB
nim/glm-5.3-flash:2.1.2-variant-modelopt64k-r3             25.7 GB
nvcr.io/nim/zai-org/glm-5.3-flash:2.1.2-variant             25.7 GB
```
`docker system df`: Images total 72.86 GB disk, 46.58 GB reclaimable (dangling/duplicate layers across the 3 modelopt64k tags + the nvcr tag, all built from the same base).

Peer (gx10-2a13) has the same EXL3 and NIM images plus an unrelated set of `sglang`/`vllm` DeepSeek-V4 images (not GLM-5.3-Flash). `docker system df`: Images total 94.65 GB, 94.64 GB reclaimable (peer is mid-experiment, containers stopped).

## 2. NVFP4 tensor layout (safetensors headers only, no tensor data loaded)

Read via Python `struct`/`json` on the 8-byte length prefix + JSON header of each shard; tensor bytes never touched.

Global config (`config.json`): `hidden_size=4096`, `intermediate_size=12288` (dense MLP), MoE `moe_intermediate_size=2048`/`n_routed_experts=288`/`num_experts_per_tok=8` (from the EXL3 `quantization_config.json` trellis shapes and prior blog entry), `first_k_dense_replace=3` (layers 0-2 dense, MoE from layer 3 with `deepseek_sparse_attention` every 4th layer).

Per-expert tensor triplet, e.g. `model.language_model.layers.10.mlp.experts.0.down_proj`:

```
.weight            U8       [4096, 1024]   # packed FP4, 2 values/byte, out=4096 x in/2=1024 (in=2048)
.weight_scale      F8_E4M3  [4096, 128]    # block scale, block_size=16 along the packed-in axis (1024*2/16=128)
.weight_scale_2    F32      []             # single global per-tensor scale (scalar)
```

Same triplet shape for `gate_proj`/`up_proj` (in=4096, so `.weight` is `[2048, 2048]` U8, `.weight_scale` is `[2048,256]`... not verified per-tensor beyond down_proj/gate_proj/up_proj naming; shapes follow directly from `in_features/2` for the packed axis and `in_features/16` for the block-scale axis).

Non-expert tensors (embeddings, `lm_head`, `input_layernorm`, `self_attn.*`, dense `mlp.*` in layers 0-2, `hc_attn_fn`/`hc_ffn_fn` mixing weights) are plain `BF16`; per-layer scalars (`hc_attn_base`, `hc_attn_scale`, `A_log`, `dt_bias`) are `F32`.

Full-checkpoint byte accounting (120 shards, all headers scanned, tensor byte size = shape product x dtype size, no tensor data read):

| dtype | role | GiB |
| --- | --- | --- |
| BF16 | dense/attn/embed/lm_head/hc_* | 18.01 |
| U8 | packed FP4 expert weights | 145.12 |
| F8_E4M3 | block scales (block=16) | 18.14 |
| F32 | per-tensor global scales + layer scalars | ~0.00 |
| **total** | | **181.28** (matches `du -shL` 182 GiB modulo rounding) |

`F8_E4M3` bytes / `U8` bytes = 18.14/145.12 = 1/8.00, exactly consistent with a **block size of 16** (1 scale byte per 16 packed-nibble weights = 8 packed bytes per scale byte).

## 3. Direct consumability for CUTLASS NVFP4

The referenced bench (`blog/posts/kernels/2026-09-06-cutlass-nvfp4-sm121/index.qmd`) states the CUTLASS grouped GEMM used "**block scales are FP8 e4m3 over 16 elements**" (`SF vector size: 16`) and reads expert dims `hidden_size=4096`, `moe_intermediate_size=2048`, `n_routed_experts=288`, `num_experts_per_tok=8` from the EXL3 checkpoint config — the same architecture as the NIM NVFP4 checkpoint.

The NIM NVFP4 checkpoint's on-disk layout matches this exactly: packed-U8 FP4 weight + FP8-E4M3 per-16-element block scale + one F32 global scale per weight tensor. This is the standard NVIDIA ModelOpt/TensorRT-LLM NVFP4 export format (also what `vllm`'s `cutlass_scaled_fp4_mm` / `run_cutlass_moe_fp4` consume, per `spark-stack/GLM-5.3-Flash-NVFP4-2x-DGX-Sparks/TRIAL.md` line 685-686 referencing those exact op names against this checkpoint).

**Caveat, not verified here:** the bench notes that its verification ran "with all block scales set to 1.0, which makes the swizzled CUTLASS scale layout and the plain fallback layout agree" — i.e. CUTLASS's blockscaled GEMM expects scales in a specific *swizzled* memory layout, and the bench sidestepped checking that swizzle against real data. This checkpoint's `weight_scale` tensors are stored row-major `[out, in/16]` in the safetensors file (no swizzle applied at rest — HF safetensors never store CUTLASS-specific layouts). A loader must apply CUTLASS's SF (scale-factor) swizzle transform when staging scales into the kernel's expected layout; this is a **layout reshuffle of existing bytes**, not a requantization, and does not require access to BF16 source weights.

**Net: the NIM NVFP4 weights are directly consumable in value** (FP4 nibbles + FP8 block scales + FP32 global scale, block=16, matching CUTLASS's documented SF vector size). **A loader still needs to implement the SF swizzle** (packing/unpacking, no numeric change) before handing scale tensors to the CUTLASS grouped-GEMM API. No conversion from BF16 or from EXL3 is needed to obtain FP4 values — the NIM checkpoint already contains them.

## 4. What a conversion would require, if one were needed

- BF16 original (`zai-org/GLM-5.3-Flash-BF16`) is **not present on either node**. If the NIM checkpoint's FP4 values were ever found unusable (e.g. block size or global-scale convention mismatch discovered later), a from-scratch NVFP4 requantization would need this BF16 source, which would have to be fetched fresh (not derivable from either quantized checkpoint on disk: EXL3's trellis codebook format and NVFP4's block-scaled format are both lossy and not round-trippable to each other or back to BF16 bit-exactly).
- EXL3 is **not** a valid conversion source for NVFP4: its `mcg`/trellis codebook (`.mcg`, `.suh`, `.svh`, `.trellis` int16 tensors) has no simple linear map to FP4 nibbles + block scales; converting would mean dequantizing EXL3 to floating point first, which reintroduces EXL3's own quantization error before requantizing to FP4 — strictly worse than starting from BF16.
- Conclusion: current inventory supports loading the NIM NVFP4 checkpoint as-is (pending swizzle work in the loader). No conversion path is needed given what's on disk; if one is needed, it has to start from BF16, which requires a new download.

## 5. Disk budget

| node | filesystem | size | used | avail |
| --- | --- | --- | --- | --- |
| local (gx10-5e36) | `/dev/nvme0n1p2` on `/` | 916G | 543G | 327G |
| peer (gx10-2a13) | `/dev/nvme0n1p2` on `/` | 916G | 569G | 301G |

Current GLM-5.3-Flash footprint (per node, both nodes hold the same set):

| item | GiB |
| --- | --- |
| NVFP4 checkpoint (NIM cache) | 182 |
| EXL3 4bpw checkpoint (HF cache) | 164 |
| DFlash2 drafter (local only) | 2.2 |
| Docker images (GLM-5.3-Flash only, local) | ~124 (5 images, 46.6 GB reclaimable per `docker system df`) |
| **subtotal, local** | **~472** of 543 used |

327 GiB free locally, 301 GiB free on peer. Both the 182 GiB NVFP4 and 164 GiB EXL3 checkpoints already coexist on both nodes today (346 GiB combined) without deletion, so **the loading path can keep both on disk simultaneously** and still leaves ~300 GiB headroom per node for a build tree, a new engine's own working cache, or a fresh BF16 fetch (BF16 GLM-5.3-Flash would be roughly 2x the NVFP4 U8+scale weight bytes, i.e. very roughly 290-310 GiB based on the FP8-native `total_bytes_gib: 306` figure already recorded in `fuels/glm-5.3-flash/fuel.yaml` — BF16 is the same order of magnitude as FP8, not computed fresh here). A full BF16 fetch would not fit alongside both existing checkpoints on either node without freeing something first (reclaiming the 46-95 GB of dangling docker layers gets partway there but not all the way).

**Recommendation:** load NVFP4 directly from the existing NIM cache path (`~/.cache/nim-glm53-2.1.2/ngc/hub/models--nim--zai-org--glm-5.3-flash/snapshots/nim-aa28e1f-nvfp4/`) on both nodes. Write the loader to read the safetensors shards + index, apply the CUTLASS SF swizzle to `*.weight_scale` tensors at load time (in memory, not on disk), and leave `*.weight` (packed FP4) and `*.weight_scale_2` (global scale) untouched. This needs no new downloads, no conversion pipeline, and no disk reclamation.

## Commands used

```bash
# local inventory
find / -iname "*glm*5.3*" -o -iname "*glm-5.3*" 2>/dev/null | grep -v "^/proc"
du -shL <snapshot-dir>              # follow HF-cache symlinks, du alone undercounts
stat -L -c '%s %n' <shard>
python3 -c 'read 8-byte length prefix + json header from each .safetensors shard'
docker images ; docker system df
df -h /

# peer inventory (passwordless ssh)
ssh 192.168.100.11 'hostname; df -h /; docker images; docker system df; find ... '
```

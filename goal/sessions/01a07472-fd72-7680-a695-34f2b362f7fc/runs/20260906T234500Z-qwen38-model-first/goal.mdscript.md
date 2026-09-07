---
active: true
status: pursuing
conversation_id: 01a07472-fd72-7680-a695-34f2b362f7fc
run_id: 20260906T234500Z-qwen38-model-first
proof_kind: default
resume_heading: pursue-goal
skip_hooks: true
loop_driver: harness-goal
objective: "Inspect and optimize nvidia/Qwen3.8-Flash-Next-NVFP4 revision fc694b54fb0174e0913e6adf86691ef85a4ead47 for quality and minimum weight bytes per generated token on the two-GB10 pair, then build a specialized engine around the measured precision and state map."
completion_gate:
  - exact checkpoint-header inventory covers every text-path tensor, active expert set, recurrent state, KV state, MTP, PLE, and ignored BF16 family
  - baseline quality and token traces exist before any precision change
  - every candidate model precision or residency change is isolated and measured for bytes/token, projection/logit drift, token parity, and evaluation quality
  - selected model map minimizes bytes per generated token subject to the quality gate and fits the two-GB10 pair at agent concurrency with 262144 context
  - only after the model map is accepted, a new qwen38-flash-next-nvfp4 engine is designed and measured against the hardware limits and general-engine baselines
  - the engine carries forward adaptive lazy MTP whose draft depth tapers from measured acceptance and session phase, with accepted-token telemetry proving the policy
  - the engine restores and evicts complete resumable prefix state directly from NVMe at an atomic shared-token boundary, including full-attention QSA state, linear-attention recurrent and convolution state, PLE state, and any MTP state required for parity
  - the NVMe tier uses no nvidia-fs, GDS, or cuFile dependency and never registers a regular-file mmap; all I/O and staging honor the 65536-byte host-page contract
  - every measured decision is logged under blog/posts with terminal verdict and the blog renders before push
current_state:
  - source repository nvidia/Qwen3.8-Flash-Next-NVFP4 pinned at fc694b54fb0174e0913e6adf86691ef85a4ead47
  - metadata reports 132639846394 bytes and 119602003859 parameters across BF16, U8, and F8_E4M3 tensors
  - config reports 48 text layers, 36 linear attention, 12 full attention, 512 experts top-10, hidden 2560, context 262144, native hybrid MTP, PLE at layer 2, and FP32 recurrent state
  - NVIDIA quantizes routed experts to NVFP4 and excludes attention, shared experts, routers, embeddings, lm_head, and hyper-connections
  - full checkpoint download is live; engine scaffold is forbidden until the model traffic and quality map is accepted
  - GLM-5.3-Flash run is preserved as stopped history and supplies reusable NVFP4 codec, telemetry, cache, and fabric lessons only
  - exact checkpoint now loads end to end on both 64 KiB nodes after cloning each selected safetensors tensor before its CUDA copy; 11/11 shards loaded in 598.34 s, model memory is 62.72 GiB/node, and the endpoint is healthy
  - the live baseline has speculative_config=None and reaches 19.82, 29.65, 61.43, 94.72, and 163.40 aggregate tok/s at c1, c2, c4, c8, and c16 on a 64 KiB shared-prefix workload
  - c16 is 11.78 tok/s per stream, 41.1% below the greater-than-20 target; published 207 tok/s c8 uses MTP3 and remains a separate comparison
  - scripts/baseline/openai-forked-prefix.py and scripts/runtime/patch-vllm-64k-loader.py make the baseline and 64 KiB loader fix reproducible
  - accepted entries supersede the stale 15-minute loader cutoff and record the non-speculative pair baseline; quarto render passes
  - pre-change greedy quality baseline passes all 10 deterministic reasoning tasks plus 13,088-token and 104,088-token needle retrieval; the negation case requires 299 completion tokens, including 279 reasoning tokens, and only appeared to fail under the temporary 256-token cap
  - NVIDIA's checkpoint contract is W4A4 NVFP4 group-size 16, and every existing expert projection carries weight, weight_scale, weight_scale_2, and input_scale; a valid dense-family experiment therefore requires activation calibration rather than weight-only packing
  - model-first lever order remains base_linear_attention first: projected removal is 2,999,358,792 bytes per c16 step and raises the read-only ceiling from 281.92 to 317.13 aggregate tok/s before quality and compute costs
  - activation telemetry is committed and pushed at 7abe989; it captures monotonic maxima for in_proj_qkvz, in_proj_ba, and out_proj on every linear-attention layer and rejects captures missing any of 108 expected channels
  - the first telemetry launch loaded 11/11 shards and captured all 36 qkvz/ba channels but only two out_proj channels before Python hook synchronization invalidated CUDA graph capture; this is a reproduced calibration-mode failure
  - parser tab handling is fixed and pushed at cb5d2fd; the incomplete warmup capture correctly rejects 72/108 channels
  - corrected eager calibration completed on 2026-09-07 under kernel 6.17.13-rocket64k with 65536-byte pages on both nodes; all 11 shards loaded in 624.63 s on head and 241.75 s on worker without OOM, at 62.72 GiB model memory per node
  - the calibration reducer accepts all 108 required channels across 36 linear-attention layers; in_proj_qkvz, in_proj_ba, and out_proj are complete
  - calibration-mode greedy quality passes 12/12 against /tmp/qwen38-baseline-quality.json, including 13,088-token and 104,088-token needles, with no regressions
  - vLLM reported 36.99 GiB KV cache and 19.76 full-262144-token concurrency for its own pair configuration; known vLLM reporting and capture bugs make both figures engine-local baselines only, never inputs to Rocket capacity, cache geometry, or memory budgeting
  - Rocket capacity must be established independently from measured CUDA-usable memory, byte-counted allocations for every resident state family, and real allocation/restore tests at concurrency
  - the next telemetry revision must cover full-attention boundaries, recurrent-state drift, PLE, router interactions, and MTP availability before any precision family is accepted
  - expanded telemetry v2 is live-proven by /home/glwillen/calibration/qwen38-expanded-mtp3-20260907-03 on both 64 KiB nodes; its v2-only gate covers all 36 linear-attention layers and 108 input plus 108 output projection channels, all 12 full-attention layers and both projection families, all 36 recurrent-state layers, PLE layer 2, and all 48 routers
  - engine carryovers are mandatory: adaptive tapering MTP and direct NVMe prefix-state load/unload; nvidia-fs, GDS, cuFile, and registered regular-file mappings are forbidden
  - startup diagnosis on the pinned checkpoint counts 299545 tensors across 11 safetensors files; 149531 tensors are at most 65536 bytes and 297194 are at most 1 MiB
  - the 598.34 s vLLM load averages 1.997 ms per checkpoint tensor, while O_DIRECT reads one 3115991696-byte shard in 0.67 s (about 4.65 GB/s); storage bandwidth is not the ten-minute bottleneck
  - every safetensors payload is contiguous within its shard (zero inter-tensor gap), so the structural startup target is a prepacked final-layout slab loaded in large 65536-byte-aligned reads instead of roughly 300000 per-tensor clone/copy dispatches
  - the successful expanded MTP3 launch traverses the checkpoint twice because target and MTP name filtering occurs downstream of get_tensor(name).clone(); head target/MTP passes took 625.24/328.95 s and worker passes took 237.06/152.38 s, consistent with the measured 242288619520 read bytes against a 132639846394-byte checkpoint
  - head readiness took 1320.66 s: 977.24 s model loading, about 204.71 s profiling/KV/kernel warmup on the critical path, 14.23 s cached FlashInfer autotuning, and 28.64 s API multimodal warmup; calibration hooks inflate profiling and must be separated from production startup
  - startup phase benchmark and accepted public verdict are committed and pushed at e07644a; cached 1 GiB bulk phases reach median 19.598 GiB/s preadv, 16.669 GiB/s mmap+copy, and 26.197 GiB/s hot reused memcpy, confirming per-tensor dispatch rather than bulk movement owns the observed loader time
  - reviewed startup artifact contract is four immutable rank-local data slabs (target and MTP per rank) with compact runtime manifests, offline full tensor provenance, pre-swizzled CUTLASS scales, and one-pass chunk verification; exact TP2/EP2/PLE slicing remains to be proved before materialization
  - PLE TP2 placement is source-proven as a transform: the checkpoint's 20000000-row source vocabulary is split across 128 source tensors, then 16 n-gram heads produce 320001446 logical FP8 rows padded to 320001536; rank 0 owns [0,160000768), rank 1 owns [160000768,320001446) plus 90 padding rows, while the global BF16 scale and PLE dense projections, convolution, norms, and generated head metadata are replicated
  - MTP routed experts are FP8_PB_WO block-128 weights and must use a separate slab transform/kernel ABI from the target model's NVFP4 expert records
  - target expert EP2 ownership is live-log proven with linear placement: rank 0 owns global experts 0 through 255 and rank 1 owns 256 through 511; each rank reports 256 local of 512 global experts
  - scripts/runtime/qwen38-rank-slab-plan.py now accounts for all 299545 tensors and 132639846394 source bytes, emits four target/MTP rank-local slabs with exact TP2/EP2/PLE transforms and separate target NVFP4 versus MTP block-FP8 ABIs, assigns shared payload I/O once, excludes only the guarded 897862112-byte vision family, and passes 13 focused tests against the real checkpoint
  - linear-attention TP2 source slicing is source-proven for all 36 identical layers: Q, K, and V are independently halved before packing with local Z; B and A are independently halved; out_proj slices input dimension; convolution slices each Q/K/V segment; A_log and dt_bias split by value head; only norm.weight replicates
  - a language-model-only Rocket may omit all 333 model.visual tensors totaling 897862112 bytes, provided construction omits the vision tower and the request path rejects multimodal embeddings; the vLLM calibration baseline did not use this contract
  - scripts/numerics/qwen38-precision-family-ranking.py joins pinned headers to the complete v2 trace and selects the whole base_linear_attention BF16-to-FP8 move: 180 matrices, 3.883667 GiB source, 1.941833 GiB removed per c16 step, 288 interaction channels including recurrent outputs, with high observed outlier risk requiring calibrated scaling and the unchanged quality gate
  - the next model mutation is an immutable source-hash-keyed offline FP8 overlay for all 36 base linear-attention layers using an existing NVIDIA runtime ABI; unsupported ABI evidence must fail closed before any expensive model reload
  - immutable FP8 artifact dbefeae04f00118080ce821909786b0c84941b3ac39f1854941a2d2bf4cd516d is materialized and preflight-proven: 180 source matrices become exactly 540 tensors in a 2085102408-byte safetensors overlay, source hashing is bounded to 4210871410 bytes, the quant config preserves the patched MTP block-FP8 route and selects 180 dense FP8 methods, and runtime preflight verifies all 180 replacements before first yield
  - corrected slab packing uses 256-byte tensor alignment and 65536-byte bulk I/O boundaries: each rank carries 65258494618 target payload bytes plus 1370113536 MTP payload bytes, with 18646374 tensor-padding bytes and 39424 I/O-tail bytes total, for 66647293952 resident slab bytes and under 0.029 percent layout overhead
  - production materialization uses bounded vectorized torch F8_E4M3 conversion in the pinned image with bit-for-bit scalar-oracle parity; 256 MiB converts at 0.326 GB/s, while the real artifact completed without mutating the base checkpoint
  - the immutable FP8 artifact passes the complete two-node launcher preflight, including image, revision, page-size, artifact-key, checksum, and all 180 source-replacement checks; 25 focused integration tests pass at commit f15c6e2
  - first live FP8 launch /home/glwillen/calibration/qwen38-linear-fp8-live-20260907-03 fails before checkpoint streaming at the first selected MergedColumnParallelLinear load with AttributeError because the runtime loader receives the module object where linear.py expects a parameter carrying data
  - the failure excludes memory pressure, fabric, storage, and 64 KiB paging; a container-level red reproduction and ABI-selection repair are required before another expensive reload
  - commit ba2e916 adds Qwen packed-family resolution and proves the previous generated runtime resolves in_proj_qkvz and in_proj_ba as unquantized while the repaired runtime resolves both as FP8; 30 focused tests and full launcher preflight pass
  - live rerun /home/glwillen/calibration/qwen38-linear-fp8-live-20260907-04 still fails in MergedColumnParallelLinear.weight_loader with the same module-object symptom, proving quant-method selection was necessary but insufficient; exact post-mapper offending tensor name and shard metadata must be captured before another repair
  - bounded load tracing in run -07 proves the first failure is layer 0 in_proj_ba.weight_scale shard 1 with UnquantizedLinearMethod and only bias/weight registered
  - root cause is the runtime precedence of config.json quantization_config over hf_quant_config.json; the launcher mounted the overlay sidecar while config_patched.json retained 36 linear-attention exclusions
  - commit a30b0a6 embeds the verified 180-projection FP8 policy into config_fp8_patched.json, preserves ModelOpt identity, rejects partial policies, and mounts that effective config only for FP8 runs
  - live run /home/glwillen/calibration/qwen38-linear-fp8-live-20260907-08 clears the prior boundary: layers 0 and 1 use ModelOptFp8LinearMethod with weight, weight_scale, and input_scale registered and correct QKVZ/BA shard metadata; the same launcher process remains live loading the checkpoint
  - live run -08 completed end to end: target and MTP loading took 604.53 and 278.29 seconds on head, model residency is 63.29 GiB per node versus 64.30 GiB for the matched BF16-plus-MTP telemetry run, and vLLM allocated 36.66 GiB of engine-local KV cache
  - expanded FP8 calibration passes all 14 executable mechanism cases with zero failures and complete two-rank coverage of 36 linear-attention layers, 12 full-attention layers, 36 recurrent-state layers, PLE layer 2, and 48 routers
  - runtime logs prove MTP3 despite the metadata endpoint's false unavailable report: 2581 of 4626 drafted tokens were accepted, total acceptance is 55.793 percent, and mean accepted length is 2.729 across 30 workload-window records
  - the unchanged greedy quality gate passes 12/12 with no BF16-baseline regressions, including the 13088-token and 104088-token needles; 11 of 12 stored final-answer tails are byte-identical and the changed negation explanation remains correct
  - an instrumented eager c16 control reaches 33.30 aggregate tok/s and 4.77 tok/s per stream with 39.18-second mean TTFT; it proves 16-stream stability but is excluded from production performance because calibration hooks synchronize Python and eager mode disables the fast path
  - the loaded endpoint must next be replaced once by a production FP8 launch with telemetry and eager mode removed, then measured across the complete concurrency ladder before accepting the model map
  - scripts/numerics/qwen38-compare-precision.py reproducibly compares all 494 paired telemetry channels and 12 quality cases; selected projection outputs reach 1.559x RMS and 0.2865 histogram TV, while downstream router logits remain 0.977x to 1.024x RMS with at most 0.0059 TV and the quality gate has zero regressions
  - production mode is committed and pushed at 6ba463e; it removes calibration environment, load tracing, the telemetry model mount, and enforce-eager before launching, then automatically runs c1/c2/c4/c8/c16
  - production run /home/glwillen/calibration/qwen38-linear-fp8-production-20260907-09 is live under launcher PID 655344 and is loading shard 4 of the first 11-shard target pass; do not restart it
  - the head clock incident is superseded by a public power-renegotiation entry at commit 82250bb; idle clocks and advertised maximum now match, and loaded c16 clock parity remains the closing measurement
  - production run -09 completes the full ladder at c1/c2/c4/c8/c16 aggregate 30.54/46.20/48.51/62.43/65.81 tok/s and per-stream 30.54/23.56/19.81/13.97/7.81 tok/s, but is rejected as a healthy-hardware baseline because the head remained capped
  - synchronized active inference samples prove the head at median/max 721/728 MHz and 10.61/13.02 W while the peer reaches 2457/2489 MHz and 36.45/42.34 W; both reach 96 percent utilization, so display-power renegotiation did not remove the cap
  - rejected power-fix evidence is rendered, committed, and pushed at 21ed1b7; reboot the head, then run the exact cheap static-fire clock reproduction before another model load
  - a warm reboot left the head capped: exact static fire measured 1688.6 ms/step and 728 MHz; the peer measured 1423.0 ms/step and 2411 MHz on the same current binary and 20 GiB configuration
  - a full cold cycle with every head power source removed clears the cap: head reaches 2418 MHz and 1438.4 ms/step, within 1.1 percent of the peer; accepted superseding evidence is rendered and pushed at 606ebe7
  - healthy-clock production rerun /home/glwillen/calibration/qwen38-linear-fp8-production-20260907-10 is live under launcher PID 5842 with automatic c1/c2/c4/c8/c16 and synchronized head/peer clock capture armed; do not restart it
  - healthy-clock FP8 production run -10 completed at 44.51, 73.22, 72.92, 98.28, and 101.40 aggregate tok/s for c1/c2/c4/c8/c16; c16 is 11.58 tok/s per stream while both nodes hold 95-96 percent utilization and 2450-2483 MHz median clocks
  - immutable whole-family NVFP4 artifact 64539e4a5a4b533aa143d20e055ae73ebd10b21f05dbd422847a9be30e56c4e5 contains exactly 720 ModelOpt ABI tensors for the same 180 projections; 4170055680 BF16 source bytes become 1172829600 payload bytes and remove 2997226080 bytes per c16 step
  - NVFP4 materialization uses the installed scaled_fp4_quant kernel with unswizzled group-16 checkpoint scales, source-hash identity, activation global scales from the accepted v2 trace, and bounded one-matrix GPU staging
  - 11 focused tests pass, including actual ModelOptNvFp4LinearMethod construction and fused QKV shard loading; full two-node preflight /home/glwillen/calibration/qwen38-linear-nvfp4-preflight-20260907-01 validates all 180 replacements before first yield
  - the next expensive residency is one expanded-telemetry NVFP4 launch; run the unchanged quality and interaction gates, then retain NVFP4 wholesale or bisect qkv, z, ba, and out projection families from that single structural result
  - first NVFP4 live load reached all 11 target shards in 113 seconds, then failed because runtime MTP layer 48 did not resolve checkpoint MTP layer 0 and inherited the target expert NVFP4 method
  - the corrected mixed-precision overlay resolves mtp.layers.48.mlp.experts to NVIDIA FP8_BLOCK_SCALES with a 128x128 block while retaining NVFP4 for all 180 target linear-attention projections; 28 focused tests and pinned preflight pass
next_owner: root orchestrator
---

<!-- mdscript: use the mdscript-exec skill -->

## Pursue Goal

* verify the pinned checkpoint download and parse every safetensors header
* derive exact active bytes per generated token and concurrent-step upper bounds from the actual tensor map
* derive FP32 recurrent state and paged KV/PLE/MTP state per stream at 262144 context
* capture baseline quality and telemetry using an existing engine before changing precision
* benchmark the 64 KiB loader phases and define a prepacked final-layout slab manifest that preserves tensor identity while reducing load dispatches by orders of magnitude
* eliminate the duplicate target/MTP checkpoint pass by assigning each prepacked slab to its consumer before payload I/O; do not filter after clone or CUDA placement
* rank model changes by quality loss per GiB/token removed and test large structural changes first
* create no engine code until the accepted model map fixes its traffic and state contract
* keep experiments append-only in blog/posts and render before push

## Resume Goal

* execute [Pursue Goal](#pursue-goal)

## Complete Goal

* require direct proof for every completion gate and a matching Proven-for review verdict

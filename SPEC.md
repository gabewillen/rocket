# Wildfire — Inference Engine Specification

Status: draft 0.1 · 2026-09-05 · target hardware verified on both nodes

Wildfire is a purpose-built inference engine for Flash-class sparse-MoE models
on DGX Spark clusters. The first optimization target is **GLM-5.3-Flash** on
**two DGX Sparks** running the **64 KiB-page kernel**.

Wildfire is not a general-purpose serving framework. It exists because the
general-purpose frameworks make three assumptions that are all false here: that
host pages are 4 KiB, that GPU memory is a discrete pool distinct from host RAM,
and that a layer's reusable state is fully captured by an attention KV cache.
On GB10 with a hybrid linear/sparse-attention MoE, each of those costs real
throughput. See section 12 for the measured bar to beat.

---

## 1. Target hardware (measured on both nodes, 2026-09-05)

| Property | Value |
| --- | --- |
| SoC | NVIDIA GB10, compute capability 12.1 (sm_121), driver 580.173.02 |
| Kernel | 6.17.0-1031-nvidia-64k |
| Page size | **65536 B**; hugepage size 512 MiB |
| MemTotal | 129,735,232 kB = **123.73 GiB** per node (unified CPU+GPU) |
| Memory bandwidth | ~273 GB/s LPDDR5x (datasheet; **must be measured in M0**) |
| CPU | 20 cores |
| Storage | 916 GB NVMe root, ~544 GB free |
| Interconnect | **2x 200 Gb/s RoCE** per node, both links up: 192.168.100.0/24 and 192.168.101.0/24; peer RTT ~0.55 ms |

Cluster total: **247.5 GiB** unified memory, ~546 GB/s aggregate memory
bandwidth, ~50 GB/s theoretical dual-rail fabric between nodes.

### 1.1 GB10 constraints that shape the design

These are load-bearing. Each has already cost a bringup cycle in the prior
stacks under ~/spark-stack.

1. **C2C cannot CPU-read cudaMalloc memory.** Reading device-allocated memory
   from the host segfaults; managed memory works. Any buffer the CPU must touch
   (offload staging, descriptor rings, sampling scratch) is managed or host
   memory, never plain device memory.
2. **Shared memory is ~99 KB per block.** Kernels asking for more fail at
   launch; one DSA kernel requested 169,984 B and died. Every kernel declares
   its dynamic smem budget and is validated against the ceiling.
3. **sm_121 is outside several vendor kernel matrices.** TensorRT-LLM's FMHA
   runner rejects the architecture outright. Wildfire owns its attention
   kernels rather than depending on a dispatch table with no sm_121 tactic.
4. **Memory overcommit kills the OS.** One prior boot at 0.87 utilization plus
   an 8.58 GB pinned tier OOM-killed *both* operating systems. Wildfire never
   discovers its budget by allocating until failure.

---

## 2. Target model — GLM-5.3-Flash

| Property | Value |
| --- | --- |
| Parameters | ~321B total, **18B active per token** |
| Layers | 45 text layers + 24-layer vision encoder |
| Attention | Hybrid: **KDA linear attention** on ~34 layers, **NoPE sparse MLA / DSA** on every 4th layer (~11 layers), top-k 2048 token selection |
| MoE | **288 routed experts, top-8 per token** |
| Context | 1,048,576 tokens |
| Speculation | Native **MTP draft layer**; DFlash2 external drafter also available |
| Checkpoints | FP8 native (~306 GiB), NVFP4 expert variant, EXL3 4bpw |

Cached locally today: Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw (164 GB) and
incoai/GLM-5.3-Flash-DFlash2 (2.2 GB drafter).

### 2.1 Capacity budget

| Weight format | Size | Verdict |
| --- | --- | --- |
| FP8 native | ~306 GiB | **Does not fit** in 247.5 GiB |
| EXL3 4bpw | ~164 GiB | Fits, ~83 GiB cluster headroom |
| NVFP4 (4-bit experts) | ~182 GiB | Fits, ~65 GiB headroom |

Wildfire targets the **4-bit weight regime**, with ~82 GiB resident weights per
node as the design point. KV, KDA state, activations, drafter, vision encoder,
runtime and OS all share the remaining ~41 GiB per node.

### 2.2 Bandwidth ceiling

Decode is bandwidth-bound. 18B active parameters at ~4.25 bits is about
**9.6 GB read per token** when every active weight is re-read.

- One node: 9.6 / 273 gives a **~28 tok/s** single-stream ceiling.
- Both nodes on the same token (TP/EP=2): ~4.8 GB each in parallel, **~57 tok/s**.
- Pipeline parallelism does **not** raise the single-stream ceiling, because
  its stages are serial for any one token.

This is the whole reason for the parallelism choice in section 4: only a layout
where both nodes touch the same token concurrently can beat ~28 tok/s per
stream. Batching raises aggregate throughput by amortizing expert reads across
tokens that share experts, which is what section 6.2 exploits.

---

## 3. The 64 KiB page regime

Wildfire **requires** the 64K-page kernel, installed on both nodes. It buys
~2.10 GiB per node: boot-time reserved memory drops from 6.09 GiB to 4.00 GiB,
and MemTotal rises from 121.63 GiB to 123.73 GiB. On memory-bound boxes that is
a ~2% capacity gain for free.

It is not free in practice, and a prior A/B **rejected** this kernel for the
vLLM stack. The failure was not the kernel; it was framework assumptions.
Wildfire encodes the following as hard rules.

### 3.1 Never size a pool from MemTotal

The rejected A/B failed exactly here: the larger reported total drove the
framework's desired pool to 100.84 GiB at utilization 0.815, while CUDA exposed
only 97.77-98.94 GiB after transfer registration. The admission check failed
before the model loaded. Dropping clean page cache did not close the gap.

**Rule.** The memory manager takes an *absolute byte budget*, never a fraction
of MemTotal:

```
usable = measured_cuda_free_after_registration
       - registration_reserve      (probed at startup, not estimated)
       - os_floor                  (hard minimum 5 GiB, from the OOM incidents)
```

Startup refuses to proceed if requested residency exceeds `usable`. MemTotal may
appear in logs; it may never appear in an arithmetic path that sizes an
allocation.

### 3.2 64 KiB is the universal alignment quantum

Every offset, stride, chunk and block boundary is a multiple of 65536: O_DIRECT
NVMe I/O, mmap offsets, KV block rows, offload copy descriptors, shared-region
layout. Any constant inherited from 4K-page code is treated as wrong until
re-derived. Note the interaction with the model: DSA forces a logical attention
page of 64 tokens, so KV row strides must satisfy both the 64-token logical page
and the 64 KiB physical page.

### 3.3 Host registration is the dangerous operation

cudaHostRegister is where the 64K regime bites. The prior stacks converged on a
specific safe pattern, which wildfire adopts natively:

- **Serialize across ranks.** Concurrent whole-region registration from every
  rank contends in the driver on the same pages; at 48 GiB the ioctl stalled
  past the engine-ready timeout and deadlocked startup. Take an exclusive file
  lock and register one rank at a time.
- **Register in bounded, row-aligned chunks.** Chunk size is a whole multiple of
  the row stride. A copy descriptor targets one block row, and batched copies
  reject ranges spanning two separately-registered regions
  (CUDA_ERROR_INVALID_VALUE).
- **Roll back to unpinned on failure.** Registration is an optimization, not a
  correctness requirement. On any chunk failure, unregister what succeeded and
  continue with pageable DMA.
- **Unregister explicitly at teardown.** Large registered buffers reclaimed by
  the kernel during SIGKILL stall teardown in uninterruptible sleep for tens of
  seconds. Unregistration is idempotent and runs on every exit path.
- **Never register a regular-file mapping.** Pinning a file-backed mmap pins its
  page cache and turns an NVMe-backed tier into unreclaimable host memory, the
  exact opposite of the intent. Only Device-DAX character mappings get the
  registered fast path.

### 3.4 Mixed-page clusters are supported

A 4K head and a 64K worker interoperated correctly over RoCE, including
cross-node collectives. Wildfire does not require both nodes to share a page
size, but each node advertises its page size in the cluster handshake, and no
node assumes a peer's alignment.

---

## 4. Parallelism layout

**Default: attention TP=2 plus expert-parallel EP=2 (144 experts per node) over
dual-rail RoCE.**

The fabric makes this affordable. Per MoE layer, dispatch and combine move on
the order of 20 KB per token; across 44 MoE layers that is under 1 MB per token,
roughly 35 us of wire time at 25 GB/s. The real cost is latency: ~88 sequential
hops at ~10 us is ~0.9 ms per token against a ~17 ms compute budget at the
bandwidth ceiling. About 5% overhead to double the per-token bandwidth available
to a single stream.

Pipeline parallelism is kept as a fallback since it nearly eliminates cross-node
traffic, but it is not the default: it cannot beat the ~28 tok/s single-stream
ceiling.

Both rails must be used. Single-rail operation is a degraded mode that wildfire
reports explicitly rather than silently absorbing.

---

## 5. Hybrid state cache — the differentiating subsystem

GLM-5.3-Flash's reusable per-sequence state has two parts:

1. **Sparse-MLA KV** on ~11 layers, growing with sequence length, pageable.
2. **KDA recurrent state** on ~34 layers, fixed size per sequence, and *not*
   reconstructible from KV.

Every prior attempt at NVMe prefix caching on this model failed for one reason:
the storage tier held KV pages only. Restoring KV without the matching KDA state
forces a full re-prefill, so reads never hit. One stack verified writes to NVMe
on both nodes and still never landed a single restore.

**Wildfire treats the (KV pages, KDA state) pair as one indivisible cache
entry**, anchored at the same token boundary, written and restored atomically.
This subsystem is what justifies building an engine instead of patching one.

Requirements:

- A prefix entry is valid only if both components are present and their anchor
  token offsets agree exactly.
- Anchors sit on 64-token DSA page boundaries so KV and KDA state truncate to a
  common prefix.
- Two tiers: unified-memory hot tier and NVMe cold tier. **No CPU-pinned middle
  tier** — it is the tier most likely to OOM the OS, and a pinned tier is what
  pushed a node over the edge before.
- Writes are event-fenced device-to-NVMe through a fixed bounce ring in
  **managed** memory (section 1.1, constraint 1), then fsync plus atomic rename.
- TTL sweep on the cold tier. The arena is a fixed byte budget, never "the rest
  of the disk".

Acceptance: a cold prefill of a large prompt followed by a full engine restart
must replay from cache with cached_tokens non-zero. The bar is **5.8x**
(54.98 s cold to 9.54 s replayed at 44,236 tokens).

### 5.1 KDA state is the concurrency limiter

KDA state dominates the concurrency ceiling far more than KV does. Prior tuning
measured ~70 MB per request at 5 slots per request, capping concurrency at 6.
Dropping to 4 slots per request with lazy buffering and a tuned convolution
width lifted the same box to 32 concurrent requests. Wildfire treats **KDA slots
per request** as a primary, explicitly tuned scheduler parameter, not an
implementation detail.

---

## 6. Scheduler

### 6.1 Decode priority over cold prefill

Mixed traffic is the failure mode that matters for coding-agent workloads.
Packing large uncached prefills into the same step as active decoders collapsed
a cached stream from 39.4 tok/s to 1.2-1.4 tok/s in a prior stack, and a static
chunk cap did not fix it.

Admission rule: standalone prefills may use the full batch; cached suffixes
below a threshold stay eligible; large *uncached* prefill chunks are deferred
while any peer is actively decoding. Target: under 2% degradation of an active
decode stream when a cold prefill arrives.

### 6.2 Expert-affinity batching

Decode is bandwidth-bound and only 8 of 288 experts fire per token, so the
scheduler groups tokens within a step by routed expert set. Each expert's
weights are read once and amortized across every token needing them. This is the
main lever for aggregate throughput above the single-stream ceiling.

### 6.3 Speculative decoding

The MTP draft layer is the default path, with DFlash2 as an alternative. Prior
evidence is mixed and must be re-measured: one stack ran DFlash2 at k=7
productively, another disabled MTP entirely because the drafter's memory cost
would have capped concurrency near 10. Speculation depth is a runtime knob with
an explicit memory charge in the section 3.1 budget, and the drafter gets its
own KV group.

---

## 7. Numerics

- Experts: 4-bit. NVFP4 preferred on Blackwell; EXL3 4bpw supported because it
  is the checkpoint already on disk.
- Attention, KDA and dense paths: FP8 where kernels permit, BF16 otherwise.
- Router logits and normalization: BF16 minimum. With 288 experts, routing
  precision is a correctness concern, not a performance knob.
- KV dtype configurable. FP8 KV requires Blackwell and must pass a quality gate
  before becoming default.

A quality gate (perplexity plus a coding/agentic task subset) runs on every
numerics change. Throughput regressions are recoverable; silent quality
regressions are not.

---

## 8. Kernels

Wildfire owns: KDA linear attention, DSA sparse attention with the top-2048
indexer, fused MoE grouped GEMM for 4-bit experts, and the MTP draft head.

Every kernel declares its dynamic shared-memory requirement and is rejected at
build time above the 99 KB GB10 ceiling. The prior stack's working DSA
configuration on sm_12x — block_I=32, num_stages=1, threads=128 — is the
starting point, not a guess.

---

## 9. Observability

All runtime telemetry goes through **OpenTelemetry**. Required signals: per-step
time split (attention / MoE / all-to-all / sampling), bytes read per token,
expert cache hit rate, KDA slot occupancy, prefix hit rate and restore latency,
registration reserve versus measured CUDA free, and per-rail RoCE utilization.

**Cardinality is bounded by construction.** Request IDs, prompt hashes and
expert IDs are never metric dimensions. Expert-level detail is exported as
bounded histograms; per-request detail lives on spans, not metrics.

---

## 10. API

OpenAI-compatible /v1/chat/completions and /v1/completions with streaming, tool
calling and reasoning-content separation. Multimodal image input in scope; video
deferred. A native endpoint exposes wildfire-specific controls: speculation
depth, residency hints, cache anchor policy.

---

## 11. Milestones

| Milestone | Content | Exit criterion |
| --- | --- | --- |
| **M0** | Hardware characterization: measured memory bandwidth, registration-reserve probe, dual-rail RoCE bandwidth and latency, 64 KiB alignment audit | Section 1 estimates replaced by measurements; budget calculator *refuses* a deliberate over-allocation instead of being OOM-killed |
| **M1** | Weight loading, 4-bit expert path, single-node forward parity against reference | Logit parity within tolerance on a short prompt |
| **M2** | TP=2 plus EP=2 across both nodes, dual rail | Correct output; single-stream tok/s measured against the ~57 ceiling |
| **M3** | Hybrid state cache (KV + KDA) with NVMe tier | Restart-survival restore beats 5.8x replay speedup |
| **M4** | Scheduler: decode priority, expert-affinity batching, continuous batching | c32 aggregate above 132.8 tok/s; decode degradation under cold prefill below 2% |
| **M5** | MTP speculation, quality gates, OTEL, OpenAI surface | Full c1-c32 bench matrix plus quality gate green |

---

## 12. Prior art and the bar to beat

Measured on this exact hardware pair, recorded under ~/spark-stack. Aggregate
tok/s, 512-token non-streaming.

| Stack | c1 | c8 | c32 | TTFT |
| --- | --- | --- | --- | --- |
| SGLang NVFP4 (patched, TP=2) | 13.5 | 56.9 | **132.0-132.8** | ~0.24 s |
| vLLM EXL3 4bpw + DFlash2 k=7 | **23.7-25.4** | 59.6 | not measured | 0.28-0.35 s |

Wildfire must beat the better of these at every concurrency level, on the 64K
kernel, with prefix restore actually hitting.

---

## 13. Open questions

1. **Implementation language and runtime.** Rust host with CUDA/Triton kernels,
   or C++? This decides everything downstream and should be settled first.
2. **Weight format of record.** EXL3 4bpw is on disk and proven; NVFP4 is the
   native Blackwell path with better kernel support. Support both, or commit?
3. **Vision encoder scope.** Inside M1-M5, or a later phase?
4. **What is the real motivation for 64K pages?** The 2.10 GiB capacity gain is
   modest against the registration complexity. Reduced TLB pressure across an
   82 GiB weight working set is likely the stronger argument, and M0 should
   measure it rather than assume it.

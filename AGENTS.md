# Wildfire — agent instructions

Read [SPEC.md](SPEC.md) before proposing or writing anything. It is the source
of truth for scope, hardware facts, and design doctrine. If a change contradicts
the spec, either the change is wrong or the spec needs updating first — say
which.

## What this project is

A single-model inference engine: **GLM-5.3-Flash** on **two DGX Sparks (GB10,
sm_121)** running the **64 KiB-page kernel**. It is an offline model compiler
plus a thin static runtime. It is not a serving framework and will not grow into
one.

## Doctrine (SPEC section 3)

These are review-blocking, not preferences.

- No dynamic allocation in the step loop. Memory comes from the compiled plan.
- No runtime dtype or shape dispatch. No autotuning. No kernel fallback paths.
- No Python in the serving path. A reference implementation exists only as an
  offline parity oracle.
- Batch buckets are a closed compiled set. There is no shape wildcard.
- Generality is not a feature. Any proposal to support a second model, format,
  dtype, or parallelism degree must argue against the throughput it costs.

## Hardware constraints that will bite you

Each of these has already cost a bringup cycle in a prior stack. Do not
rediscover them.

1. **C2C cannot CPU-read `cudaMalloc` memory** — it segfaults. Managed memory
   works. Anything the host touches is managed or host memory.
2. **Shared memory ceiling is ~99 KB per block.** Kernels asking for more fail
   at launch. Declare and check the dynamic smem budget for every kernel.
3. **`sm_121` is missing from several vendor kernel matrices** (TensorRT-LLM's
   FMHA runner rejects it outright). Do not assume a vendor tactic exists.
4. **Never size an allocation from `MemTotal`** or from a utilization fraction.
   Use the measured post-registration CUDA free budget, minus the probed
   registration reserve, minus a hard 5 GiB OS floor. See SPEC section 4.1.
5. **64 KiB is the universal alignment quantum.** Every offset, stride, chunk,
   and block boundary is a multiple of 65536. Constants inherited from 4K-page
   code are wrong until re-derived. DSA additionally forces a 64-token logical
   attention page.
6. **`cudaHostRegister` is the dangerous call.** Serialize across ranks with a
   file lock, register in bounded row-aligned chunks, roll back to unpinned on
   failure, unregister explicitly at teardown, and never register a regular-file
   mapping.

## Memory safety on these boxes

Unified memory, no meaningful swap. **Over-allocating has OOM-killed both
operating systems**, not just the process. Before running anything that
allocates at scale:

- Check `MemAvailable` in `/proc/meminfo` and respect the 5 GiB floor.
- Check whether a serving process is already resident (`ps -eo pid,rss,comm
  --sort=-rss | head`). A live server can hold ~100 GiB. **Do not kill or
  restart another engine to free memory without asking the user first.**
- Prefer refusing to start over discovering the limit by allocating.

## Environment

- Nodes: this host plus peer over dual-rail RoCE at `192.168.100.11` and
  `192.168.101.11`. Both run `6.17.0-1031-nvidia-64k`, 123.73 GiB each.
  Passwordless ssh to the peer works.
- Checkpoints: `~/.cache/huggingface/hub` — `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
  (164 GB) and `incoai/GLM-5.3-Flash-DFlash2` (2.2 GB drafter).
- Prior art: `~/spark-stack` — **read-only reference**, not a dependency. It
  holds the measured baselines, bringup logs, and the incident records this spec
  cites. Do not modify it.

## Working expectations

- **Measure, do not assume.** Every performance claim in this repo names how it
  was measured. Datasheet numbers are labeled as such until replaced.
- **Beat the recorded bar or explain why not.** SPEC section 13 has the numbers
  from the general-purpose stacks on this exact hardware pair.
- Runtime behavior is instrumented through OpenTelemetry. Metric dimensions are
  bounded by construction: no request IDs, prompt hashes, or expert IDs as
  labels.
- Quality gates run on every numerics change. Throughput regressions are
  recoverable; silent quality regressions are not.
- Keep commits atomic: one logical change per commit.

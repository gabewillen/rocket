---
name: glm53-radix-nvme-goal
description: Implements and proves cross-request radix prefix sharing with bounded-memory NVMe persistence for GLM-5.3 target MLA, FP32 KDA, and DFlash2 state.
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Goal Contract

* set `{{repo}}` to `/home/glwillen/Development/rocket`
* set `{{engine}}` to `{{repo}}/engines/glm5-moe-nvfp4-2b`
* set `{{fuel}}` to `/home/glwillen/.cache/rocket-fuels/glm-5.3-flash-nvfp4`
* set `{{cache_dir}}` to `/home/glwillen/.cache/rocket-kv-offload`
* set `{{peer}}` to `192.168.100.11`
* set `{{page_bytes}}` to `65536`
* set `{{staging_bytes}}` to `134217728`
* set `{{logical_capacity}}` to `512GiB`
* set `{{objective}}` to `ship cross-request radix prefix sharing and configurable bounded-memory NVMe persistence without reducing current decode correctness or throughput`
* require exact restore of target MLA latent/indexer pages, FP32 KDA state, and DFlash2 state at one common prefix boundary
* forbid `mmap`, pageable bulk buffers, unbounded pinned memory, Linux page-cache dependence, per-page files, and cuFile compatibility mode
* require `O_DIRECT` with 64 KiB-aligned offsets, lengths, files, and buffers
* require the total pinned host staging allocation to stay at or below `{{staging_bytes}}` per node
* require rank-local data and a rank-0 admission decision only after both ranks report the same restorable prefix boundary
* require DFlash2 verification width and current c8 decode kernels to remain unchanged
* require every published measurement to use committed scripts and literal commands
* [Inspect Reference And Current State](#inspect-reference-and-current-state)

## Inspect Reference And Current State

* read `{{repo}}/AGENTS.md`
* read `{{engine}}/src/kv/page_pool.h` completely
* read `{{engine}}/src/kv/page_pool.cc` completely
* read `{{engine}}/src/kv/kv_arena.h` completely
* read `{{engine}}/src/kv/kv_arena.cu` completely
* read the detach, resume, KDA pack, and KDA unpack paths in `{{engine}}/src/model.cu`
* read the DFlash2 context and state ownership paths in `{{engine}}/src/dflash2_engine.cu`
* inspect SGLang `python/sglang/srt/mem_cache/radix_cache.py`
* inspect SGLang `python/sglang/srt/mem_cache/hicache_storage.py`
* record SGLang lessons: page-aligned longest-prefix match, protected root-to-node paths, prompt-boundary priority, multi-pool hit intersection, and batched storage operations
* run `/usr/local/cuda-13.0/gds/tools/gdscheck -p`
* record that GB10 reports `Model Not Supported` and that the implementation must use bounded `O_DIRECT`
* search `{{repo}}/blog/posts` for prior prefix-cache and KV-offload verdicts
* if an existing accepted implementation already satisfies `{{objective}}`
  * verify it through [Run Full Proof](glm53-radix-nvme-proof.mdscript.md#run-full-proof)
* [Define Restorable Prefix](#define-restorable-prefix)

## Define Restorable Prefix

* define a radix edge as one full 128-token page under its parent chain hash
* define the cache namespace from checkpoint fingerprint, serving-weight fingerprint, tokenizer fingerprint, attention geometry, numerical execution shape, rank, and cache format version
* forbid merging independently computed pages until M1, M8, and M16 state-byte invariance is proven
* allow fork sharing because it references the original bytes
* allow NVMe restore because it restores the original bytes
* define a checkpoint node as a radix node carrying target pages plus one exact terminal FP32 KDA snapshot and one exact terminal DFlash2 snapshot
* select checkpoints at reusable request or prompt boundaries instead of every page
* define the usable hit as the deepest ancestor checkpoint present and valid on both ranks
* define unmatched tokens after that checkpoint as the prefill tail
* [Capture Baseline](#capture-baseline)

## Capture Baseline

* create `{{cache_dir}}` if missing
* record free bytes on both nodes
* refuse to reserve more than free bytes minus 64 GiB safety headroom
* accept `{{logical_capacity}}` as configuration even when the current node cannot physically reserve it
* run the current c8 256-token decode gate with prefix caching disabled
* record token IDs, acceptance, median round time, aggregate throughput, and per-stream throughput
* run a two-request trace with a shared 2k-token prefix and record the second request prefill time
* repeat the trace at 8k tokens
* repeat the trace at 32k tokens
* store baseline outputs in a committed script or inline in the eventual log entry
* [Implement Storage Format](#implement-storage-format)

## Implement Storage Format

* add a rank-local NVMe prefix store under `{{engine}}/src/kv`
* use append-only segment files sized from a configurable segment-byte limit
* align every record start and record length to `{{page_bytes}}`
* include a versioned 64 KiB record header
* store chained key, parent key, token count, component offsets, component lengths, and checksums in the header
* store target MLA latent pages in final `KvArena` layout
* store target indexer key and gate pages in final `KvArena` layout
* store one exact terminal FP32 KDA snapshot at each admitted checkpoint
* store the DFlash2 state required to continue proposals exactly at that checkpoint
* add an append-only manifest journal
* publish a record only after payload writes and checksums complete
* recover by ignoring incomplete or uncommitted records
* compact metadata without rewriting live payload extents
* if any record format lacks a deterministic validation failure
  * fix the format and [Implement Storage Format](#implement-storage-format)
* [Implement Bounded Direct IO](#implement-bounded-direct-io)

## Implement Bounded Direct IO

* open segment files with `O_DIRECT`, `O_CLOEXEC`, and `O_NOFOLLOW`
* allocate a fixed ring totaling no more than `{{staging_bytes}}` with 64 KiB alignment
* register only that fixed ring with CUDA
* issue direct reads and writes with bounded queue depth
* pipeline NVMe reads, checksum verification, and `cudaMemcpyAsync` on explicit streams and events
* pipeline `cudaMemcpyAsync`, checksum generation, and direct writes for eviction
* keep cache metadata in ordinary small CPU allocations
* expose counters for read bytes, write bytes, hit pages, miss pages, restore time, writeback time, checksum failures, and rejected records
* keep every metric free of request IDs, prompt hashes, and page hashes as dimensions
* if measured pinned allocation exceeds `{{staging_bytes}}`
  * fix the allocator and [Implement Bounded Direct IO](#implement-bounded-direct-io)
* [Wire Radix Admission](#wire-radix-admission)

## Wire Radix Admission

* expose a request-admission API that accepts token IDs and a stream slot
* walk the existing `PrefixTree` for the longest page-aligned target prefix
* reduce the candidate to the deepest node with valid target, KDA, and DFlash2 components
* exchange each rank's candidate boundary before loading payloads
* let rank 0 choose the minimum common valid boundary
* reserve one GPU physical page per unique restored target page
* coalesce concurrent restores of the same radix node into one IO operation
* install restored page IDs into every matching stream page table by reference
* restore the terminal FP32 KDA state into each admitted stream slot
* restore the terminal DFlash2 state into each admitted stream slot
* prefill only tokens after the common restored boundary
* lock every active root-to-node path against eviction
* unlock the path when its final stream detaches or advances to a private branch
* split prompt and generated-output boundaries so prompt nodes receive higher retention priority
* if either rank fails restore validation
  * discard the partial reservation on both ranks and prefill from the previous valid checkpoint
* run [Prove And Publish](glm53-radix-nvme-proof.mdscript.md#prove-state-invariance)

## Reduce Decode Bytes

* preserve the completed radix/NVMe implementation and its proof gates
* read `/home/glwillen/.agents/skills/glm53-bandwidth-optimizer/SKILL.md` and execute from `Baseline Bench`
* profile c4, c8, and c16 before changing a weight family or state representation
* test NVFP4 serving representations for every still-BF16/FP32 duplicated weight family, dequantizing during tensor-core computation because decode is bandwidth-bound
* keep source weight dtype independent from serving weight dtype
* isolate attention, router, gates, norms, recurrent projections, dense MLP, shared expert, embedding, and LM-head families
* for each family record activation ranges, projection output error, logit drift, token divergence, bytes per generated token, and interactions with every accepted quantized attention family
* retain FP32 KDA as the exact reference and test specialized compressed, delta, or on-load-dequantized representations without accepting BF16 recurrence
* profile KDA fusion that retains intermediates on chip
* measure identical-stream shared execution until branch, without merging independently computed state
* profile expert routing locality and predictive residency before changing expert representation
* accept a change only when quality gates pass and c16 median improves with no stage regression above five percent
* recurse until eight iterations, a hardware floor, or less than three percent improvement with no remaining kernel gap
* publish accepted and rejected family results through committed scripts
* run the full radix proof again after the final decode-byte change
* [Complete](glm53-radix-nvme-proof.mdscript.md#complete)

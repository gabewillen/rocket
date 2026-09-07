# Qwen3.8 Flash Next NVFP4 TP2 engine

Stage A freezes the rank-local weight ABI for checkpoint
`fc694b54fb0174e0913e6adf86691ef85a4ead47` and overlay artifact
`23d2c39e9c2cf36a832cb1750f6542fa46f11f1befe195d80c082051a5a772b4`.

| Contract | Value |
|---|---:|
| topology | TP2/EP2 |
| slabs | rank 0/1 target plus rank 0/1 MTP |
| bulk I/O alignment | 65,536 bytes |
| tensor alignment | 256 bytes |
| target NVFP4 scale ABI | ModelOpt group-16, CUTLASS SM121 SFB pre-swizzled |
| MTP expert ABI | FP8 E4M3, 128 x 128 block scales |
| shared target/MTP data | target-owned, MTP reference, one payload read |
| production input | compact manifest plus slab files |
| forbidden production paths | safetensors traversal, regular-file mmap, GDS, cuFile, nvidia-fs |

The materializer is offline. It validates all source provenance before publishing an
immutable content-addressed directory. The production loader requires an
OpenTelemetry tracer and Linux `O_DIRECT`; either missing contract fails closed.
The production plan is pinned at 545,726,297 bytes and SHA-256
`8035c520827bece63756138c820593cad91b68ac3388424d7482c184f19b49d6`.
Its deterministic identity replaces a generic JSON allocation ceiling.

Stage B adds one fixed `PairReduce` for TP2 hidden partials. Each rank owns a
four-page anonymous pinned region, publishes a versioned BF16 wire message with
an unsignaled RC write followed by a signaled sequence doorbell, validates the
peer header, then accumulates rank 0 followed by rank 1 into FP32 `[M, 2560]`.
Each rank publishes a second sequence doorbell after accumulation and waits for
the peer acknowledgment before reusing the single receive slot.
Only M 1, 2, 4, 8, and 16 is accepted. The embedder must supply an OpenTelemetry
sink. Metric labels are limited to rank, M bucket, dtype, and outcome. Trace and
request IDs are confined to spans and logs.

`qwen38-pair-reduce-bench` samples NVML SM clock and GPU utilization on an
owned thread during each timed M window. Each result carries UTC nanosecond
bounds so the two rank logs can be intersected without assuming simultaneous
process launch. `--timeout-ms` sets the matching 100..120000 ms peer and send
completion budget on both ranks. A physical silent-peer proof uses matching
`--fault-stall-rank` and `--fault-stall-ms` flags; the stall must exceed the
timeout. A timed-out instance is discarded because its remote writes may have
committed.

```bash
cmake -S engines/qwen38-flash-next-nvfp4-2b \
  -B engines/qwen38-flash-next-nvfp4-2b/build -DCMAKE_BUILD_TYPE=Release
cmake --build engines/qwen38-flash-next-nvfp4-2b/build -j
ctest --test-dir engines/qwen38-flash-next-nvfp4-2b/build --output-on-failure
```

```bash
PYTHONPATH=engines/qwen38-flash-next-nvfp4-2b/src python3 -m qwen38_slab.materialize \
  --plan /path/to/qwen38-rank-slab-plan.json \
  --checkpoint /home/glwillen/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --overlay /home/glwillen/calibration/qwen38-all-eligible-nvfp4-artifacts/23d2c39e9c2cf36a832cb1750f6542fa46f11f1befe195d80c082051a5a772b4 \
  --output-root /path/to/qwen38-rank-slabs
```

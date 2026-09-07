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

```bash
PYTHONPATH=engines/qwen38-flash-next-nvfp4-2b/src python3 -m qwen38_slab.materialize \
  --plan /path/to/qwen38-rank-slab-plan.json \
  --checkpoint /home/glwillen/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --overlay /home/glwillen/calibration/qwen38-all-eligible-nvfp4-artifacts/23d2c39e9c2cf36a832cb1750f6542fa46f11f1befe195d80c082051a5a772b4 \
  --output-root /path/to/qwen38-rank-slabs
```

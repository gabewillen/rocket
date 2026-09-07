#!/usr/bin/env python3
"""Benchmark pinned FlashInfer b12x MoE backends at the Qwen3.8 shape.

The payload is generated directly in the checkpoint's ModelOpt NVFP4 ABI. It
does not traverse or rewrite the 60 GiB rank slabs. The benchmark covers the
global 512-expert reference geometry; Rocket's physical rank slabs retain 256
owner-local experts and therefore need a pointer-table adapter before this
kernel can consume them without a resident repack.

Harness structure is adapted from FlashInfer's Apache-2.0
``benchmarks/bench_b12x_mxfp4_moe.py`` at ``FLASHINFER_COMMIT``: native MMA
scale views, deterministic packed weights, ``B12xMoEWrapper``, and
``bench_gpu_time`` with CUDA graph replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

FLASHINFER_COMMIT = "91bda04c66f7cb851e1ab3b78b9fecea644b9844"
SLAB_ARTIFACT_KEY = "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
SLAB_ABI = "modelopt_nvfp4_group16_cutlass_sm121_sfb"
HIDDEN = 2560
INTERMEDIATE = 640
EXPERTS = 512
TOP_K = 10
BUCKETS = (1, 2, 4, 8, 16)
BACKENDS = ("direct_micro", "micro", "static", "dynamic")
DIRECT_MICRO_MAX_INTERMEDIATE = 512
MICRO_MAX_TOKENS = 8
MICRO_MULTI_TOPK_CUTOVER_PAIRS = 40
MEASURED_GB10_READ_GBPS = 241.3
REFERENCE_SHA256 = {
    "LICENSE": "cb67c224f503e0a063908950b12f89a7280c6e527dcffac972aa114e4bf3c5de",
    "benchmarks/bench_b12x_mxfp4_moe.py": "82ea42d2bb83d405d84b7a8433bbe240dd8e70ccf3165502fe3bf8ab27dbb54a",
    "flashinfer/fused_moe/cute_dsl/b12x_moe.py": "c09065511e93c0ccf40e4d99f9215157aac9c26e0796ab41d1c52d396b45a2ff",
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py": "c518e65d6bfd7f08db1e5261e20795fd020e82e681699729171bc2fd5331239a",
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_direct_micro_kernel.py": "9b18f54588c3f24cab74dff07545f323e2af8235a4532233655a5a75498e89cc",
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py": "ccb6f65a22314961693493242f78f62ca58f79a319ecd0cb51bf6d7d8e7125c6",
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py": "c7b6f24b94d7939cc0eb917ab15cef4c34f3dc12bf75e15fc9dc316ee7327f3f",
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py": "7532e1338df8e80c4b8f0bd6473c2ffeda9a5028018c77775c4e9fe2fa47b072",
}


def natural_backend(tokens: int) -> str:
    """Return the pinned dispatch choice for this exact NVFP4 shape."""

    if tokens not in BUCKETS:
        raise ValueError("tokens must be a Qwen graph bucket")
    if tokens <= MICRO_MAX_TOKENS and tokens * TOP_K <= MICRO_MULTI_TOPK_CUTOVER_PAIRS:
        return "micro"
    return "static"


def backend_eligibility(backend: str, tokens: int) -> tuple[bool, str]:
    """Fail closed on forced backends outside the pinned kernel's domain."""

    if backend not in BACKENDS or tokens not in BUCKETS:
        raise ValueError("unknown backend or token bucket")
    if backend == "direct_micro":
        return False, (
            f"intermediate={INTERMEDIATE} exceeds direct_micro maximum "
            f"{DIRECT_MICRO_MAX_INTERMEDIATE}"
        )
    if backend == "micro" and tokens > MICRO_MAX_TOKENS:
        return False, f"tokens={tokens} exceeds micro maximum {MICRO_MAX_TOKENS}"
    return True, "eligible"


def modelopt_storage_bytes(experts: int = EXPERTS) -> dict[str, int]:
    """Return payload bytes for the checkpoint's packed group-16 ABI."""

    if isinstance(experts, bool) or not isinstance(experts, int) or experts <= 0:
        raise ValueError("experts must be a positive integer")
    return _storage_bytes(experts, INTERMEDIATE)


def _storage_bytes(experts: int, intermediate: int) -> dict[str, int]:
    if intermediate <= 0 or intermediate % 128:
        raise ValueError("intermediate must be a positive multiple of 128")
    w1 = experts * 2 * intermediate * (HIDDEN // 2)
    w2 = experts * HIDDEN * (intermediate // 2)
    w1_sf = experts * 2 * intermediate * (HIDDEN // 16)
    w2_sf = experts * HIDDEN * (intermediate // 16)
    # Gate and up share checkpoint scales. The four runtime planes are FC1
    # weight/input scales and FC2 weight/input scales.
    scalars = experts * 4 * 4
    return {
        "w1_weight": w1,
        "w2_weight": w2,
        "w1_weight_scale": w1_sf,
        "w2_weight_scale": w2_sf,
        "runtime_scalars": scalars,
        "total": w1 + w2 + w1_sf + w2_sf + scalars,
    }


def touched_bytes(tokens: int) -> int:
    """Lower bound for unique-expert payload plus input/output/routing bytes."""

    if tokens not in BUCKETS:
        raise ValueError("tokens must be a Qwen graph bucket")
    active = min(tokens * TOP_K, EXPERTS)
    per_expert = modelopt_storage_bytes(1)["total"]
    activations = tokens * HIDDEN * 2 * 2
    routing = tokens * TOP_K * (4 + 4)
    return active * per_expert + activations + routing


def kernel_touched_bytes(tokens: int, backend: str) -> tuple[int, int]:
    """Return physical N and its traffic bound after reference padding."""

    eligible, _reason = backend_eligibility(backend, tokens)
    if not eligible:
        raise ValueError("backend is ineligible for this Qwen shape")
    physical_intermediate = 768 if backend in ("micro", "static") else INTERMEDIATE
    active = min(tokens * TOP_K, EXPERTS)
    per_expert = _storage_bytes(1, physical_intermediate)["total"]
    activations = tokens * HIDDEN * 2 * 2
    routing = tokens * TOP_K * (4 + 4)
    return physical_intermediate, active * per_expert + activations + routing


def _component_contract(projection: str, leaf: str) -> tuple[str, tuple[int, ...], int, str]:
    matrix = {
        "gate_proj": (640, 2560),
        "up_proj": (640, 2560),
        "down_proj": (2560, 640),
    }.get(projection)
    if matrix is None:
        raise ValueError("unknown expert projection")
    rows, columns = matrix
    if leaf == "weight":
        return "U8", (rows, columns // 2), rows * columns // 2, "checkpoint"
    if leaf == "weight_scale":
        return "F8_E4M3", (rows * columns // 16,), rows * columns // 16, "cutlass_sm121_sfb"
    if leaf in ("weight_scale_2", "input_scale"):
        return "F32", (), 4, "checkpoint"
    raise ValueError("unknown expert component")


def validate_slab_manifest(path: Path) -> dict[str, object]:
    """Bind the exact owner-local slab extents without reading payload bytes."""

    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise RuntimeError("rank slab manifest must be an object")
    claimed = manifest.get("artifact_key")
    digest_input = dict(manifest)
    digest_input.pop("artifact_key", None)
    observed = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed != SLAB_ARTIFACT_KEY or observed != claimed:
        raise RuntimeError("rank slab manifest identity mismatch")
    if manifest.get("tp_size") != 2:
        raise RuntimeError("rank slab TP contract mismatch")
    ranks = []
    for rank in (0, 1):
        slab = manifest.get("slabs", {}).get(f"rank{rank}-target")
        if not isinstance(slab, dict) or not isinstance(slab.get("entries"), list):
            raise RuntimeError(f"rank{rank} target inventory is absent")
        entries = {entry.get("name"): entry for entry in slab["entries"]}
        first_expert, last_expert = (0, 255) if rank == 0 else (256, 511)
        payload_bytes = 0
        offsets = []
        for expert in range(first_expert, last_expert + 1):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for leaf in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                    name = (
                        f"model.language_model.layers.0.mlp.experts.{expert}."
                        f"{projection}.{leaf}"
                    )
                    entry = entries.get(name)
                    expected_dtype, expected_shape, expected_length, expected_layout = (
                        _component_contract(projection, leaf)
                    )
                    if (
                        not isinstance(entry, dict)
                        or entry.get("abi") != SLAB_ABI
                        or entry.get("dtype") != expected_dtype
                        or tuple(entry.get("shape", ())) != expected_shape
                        or entry.get("length_bytes") != expected_length
                        or entry.get("layout") != expected_layout
                    ):
                        raise RuntimeError(f"rank slab expert ABI mismatch: {name}")
                    offset = entry.get("offset_bytes")
                    if (
                        isinstance(offset, bool)
                        or not isinstance(offset, int)
                        or offset < 0
                        or offset % 256
                    ):
                        raise RuntimeError(f"rank slab expert extent mismatch: {name}")
                    payload_bytes += expected_length
                    offsets.append(offset)
        ranks.append(
            {
                "rank": rank,
                "first_expert": first_expert,
                "last_expert": last_expert,
                "expert_count": 256,
                "layer0_payload_bytes": payload_bytes,
                "first_extent_offset": min(offsets),
                "last_extent_offset": max(offsets),
                "slab_bytes": slab.get("bytes"),
            }
        )
    return {
        "artifact_key": claimed,
        "abi": SLAB_ABI,
        "tensor_alignment_bytes": manifest.get("tensor_alignment_bytes"),
        "ranks": ranks,
    }


def _validate_reference(repo: Path) -> None:
    for relative, expected in REFERENCE_SHA256.items():
        try:
            observed = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(f"FlashInfer reference path is absent: {relative}") from exc
        if observed != expected:
            raise RuntimeError(f"FlashInfer reference identity mismatch: {relative}")
    license_text = (repo / "LICENSE").read_text()
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise RuntimeError("FlashInfer license identity mismatch")


def _make_weights(torch):
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    generator = torch.Generator(device="cuda")
    generator.manual_seed(0x38F1)
    w1 = torch.empty(
        (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8, device="cuda"
    )
    w2 = torch.empty(
        (EXPERTS, HIDDEN, INTERMEDIATE // 2), dtype=torch.uint8, device="cuda"
    )
    w1.random_(0, 256, generator=generator)
    w2.random_(0, 256, generator=generator)
    # 0x38 is E4M3 1.0. The flat physical order is the same SFB order stored
    # by the rank-slab planner; the returned six-dimensional objects are views.
    w1_sf_storage = torch.full(
        (EXPERTS * 2 * INTERMEDIATE * (HIDDEN // 16),),
        0x38,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e4m3fn)
    w2_sf_storage = torch.full(
        (EXPERTS * HIDDEN * (INTERMEDIATE // 16),),
        0x38,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e4m3fn)
    w1_sf = convert_sf_to_mma_layout(
        w1_sf_storage, m=2 * INTERMEDIATE, k=HIDDEN, num_groups=EXPERTS
    )
    w2_sf = convert_sf_to_mma_layout(
        w2_sf_storage, m=HIDDEN, k=INTERMEDIATE, num_groups=EXPERTS
    )
    w1_alpha = torch.full((EXPERTS,), 8.0e-5, dtype=torch.float32, device="cuda")
    w2_alpha = torch.full((EXPERTS,), 1.0e-4, dtype=torch.float32, device="cuda")
    input_scale = torch.ones((EXPERTS,), dtype=torch.float32, device="cuda")
    fc2_input_scale = torch.ones((EXPERTS,), dtype=torch.float32, device="cuda")
    return w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha, input_scale, fc2_input_scale


def _make_case(torch, tokens: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0xC16 + tokens)
    x = torch.randn(
        (tokens, HIDDEN), dtype=torch.bfloat16, device="cuda", generator=generator
    ) / 16
    # Distinct experts make the c16 traffic accounting independent of cache
    # reuse between routed pairs.
    ids = torch.arange(tokens * TOP_K, dtype=torch.int32, device="cuda").view(
        tokens, TOP_K
    )
    weights = torch.full(
        (tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device="cuda"
    )
    return x, ids, weights


def run(args: argparse.Namespace) -> dict[str, object]:
    repo = args.flashinfer_repo.resolve()
    _validate_reference(repo)
    slab_layout = validate_slab_manifest(args.manifest.resolve())
    import torch
    from flashinfer import B12xMoEWrapper
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.testing.utils import bench_gpu_time

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 1):
        raise RuntimeError(f"benchmark requires SM121, got SM{props.major}{props.minor}")
    weights = _make_weights(torch)
    cases: list[dict[str, object]] = []
    references: dict[int, object] = {}
    observed_by_backend: dict[tuple[int, str], object] = {}
    try:
        for tokens in args.tokens:
            x, ids, routing = _make_case(torch, tokens)
            for backend in args.backends:
                eligible, reason = backend_eligibility(backend, tokens)
                if not eligible:
                    cases.append(
                        {"tokens": tokens, "backend": backend, "status": "rejected", "reason": reason}
                    )
                    continue
                moe_dispatch._FORCED_BACKEND = backend
                print(
                    f"compile/run tokens={tokens} backend={backend}",
                    file=sys.stderr,
                    flush=True,
                )
                wrapper = B12xMoEWrapper(
                    num_experts=EXPERTS,
                    top_k=TOP_K,
                    hidden_size=HIDDEN,
                    intermediate_size=INTERMEDIATE,
                    use_cuda_graph=True,
                    max_num_tokens=tokens,
                    quant_mode="nvfp4",
                    source_format="modelopt",
                )

                def call():
                    return wrapper.run(
                        x=x,
                        w1_weight=weights[0],
                        w1_weight_sf=weights[1],
                        w1_alpha=weights[2],
                        w2_weight=weights[3],
                        w2_weight_sf=weights[4],
                        w2_alpha=weights[5],
                        input_global_scale=weights[6],
                        fc2_input_scale=weights[7],
                        token_selected_experts=ids,
                        token_final_scales=routing,
                    )

                try:
                    call()
                    measurements = bench_gpu_time(
                        call,
                        dry_run_iters=args.warmup,
                        repeat_iters=args.iterations,
                        sleep_after_run=False,
                        use_cuda_graph=True,
                        cold_l2_cache=False,
                    )
                    samples = [float(value) for value in measurements]
                    latency_ms = statistics.median(samples)
                    latency_std_ms = statistics.pstdev(samples)
                    replay_a = call().clone()
                    replay_b = call().clone()
                    replay_error = float(
                        (replay_a.float() - replay_b.float()).abs().max().item()
                    )
                    observed_output = replay_b
                    observed_by_backend[(tokens, backend)] = observed_output.clone()
                    if backend == "static":
                        references[tokens] = observed_output.clone()
                    logical_byte_count = touched_bytes(tokens)
                    physical_intermediate, byte_count = kernel_touched_bytes(
                        tokens, backend
                    )
                    roofline_ms = byte_count / (MEASURED_GB10_READ_GBPS * 1.0e9) * 1.0e3
                    cases.append(
                        {
                            "tokens": tokens,
                            "backend": backend,
                            "status": "measured",
                            "latency_ms": latency_ms,
                            "latency_std_ms": latency_std_ms,
                            "replay_max_abs": replay_error,
                            "static_parity": None,
                            "logical_touched_bytes_lower_bound": logical_byte_count,
                            "physical_intermediate": physical_intermediate,
                            "touched_bytes_lower_bound": byte_count,
                            "effective_gbps_lower_bound": byte_count / (latency_ms * 1.0e6),
                            "measured_read_roofline_ms": roofline_ms,
                            "roofline_fraction": roofline_ms / latency_ms,
                        }
                    )
                    print(
                        f"measured tokens={tokens} backend={backend} "
                        f"latency_ms={latency_ms:.6f}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as exc:
                    cases.append(
                        {
                            "tokens": tokens,
                            "backend": backend,
                            "status": "runtime_rejected",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(
                        f"rejected tokens={tokens} backend={backend}: "
                        f"{type(exc).__name__}",
                        file=sys.stderr,
                        flush=True,
                    )
            # Static is intentionally measured before parity is finalized for
            # earlier backends. Fill those comparisons after all paths finish.
            reference = references.get(tokens)
            if reference is not None:
                for case in cases:
                    key = (tokens, str(case.get("backend")))
                    observed_output = observed_by_backend.get(key)
                    if observed_output is not None:
                        delta = observed_output.float() - reference.float()
                        denom = max(float(reference.float().norm().item()), 1.0e-30)
                        case["static_parity"] = {
                            "max_abs": float(delta.abs().max().item()),
                            "relative_l2": float(delta.norm().item()) / denom,
                        }
    finally:
        moe_dispatch._FORCED_BACKEND = None
    if not all(any(c["tokens"] == m and c["status"] == "measured" for c in cases) for m in args.tokens):
        raise RuntimeError("at least one Qwen bucket has no measured backend")
    return {
        "result": "qwen38_flashinfer_b12x_reference",
        "flashinfer_commit": FLASHINFER_COMMIT,
        "flashinfer_license": "Apache-2.0",
        "device": props.name,
        "geometry": {
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "experts": EXPERTS,
            "top_k": TOP_K,
            "model_buckets": BUCKETS,
            "measured_buckets": tuple(args.tokens),
        },
        "source_format": "modelopt_nvfp4_group16_cutlass_sm121_sfb",
        "slab_layout": slab_layout,
        "storage_bytes": modelopt_storage_bytes(),
        "natural_dispatch": {str(m): natural_backend(m) for m in args.tokens},
        "resident_layout_gap": (
            "reference expects contiguous [E,projection-row,packed-K]; Rocket slabs are "
            "owner-local E=256 with individually addressed projection extents"
        ),
        "cases": cases,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flashinfer-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokens", nargs="+", type=int, choices=BUCKETS, default=list(BUCKETS))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    if "static" not in args.backends:
        parser.error("static is required as the parity reference")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

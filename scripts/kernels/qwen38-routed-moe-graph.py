#!/usr/bin/env python3
"""Live production-slab proof for Rocket's fixed rank-local Qwen MoE graph.

Kernel structure and tensor ABI follow FlashInfer Apache-2.0 commit
91bda04c66f7cb851e1ab3b78b9fecea644b9844, specifically
flashinfer/fused_moe/cute_dsl/b12x_moe.py and the SM12x static/dynamic
dispatch kernels. This harness never starts a vLLM server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from qwen38_slab.routed_moe import (
    GLOBAL_EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    LOCAL_EXPERTS,
    SHARED_INTERMEDIATE,
    TOP_K,
    FlashInferRoutedMoeBackend,
    MoeShape,
    RoutedMoeGraph,
    load_owner_local_moe,
    materialize_flashinfer_weights,
    selector_for,
)

READ_GBPS = 238.0
SHAPES = (
    MoeShape(1), MoeShape(2), MoeShape(4), MoeShape(8), MoeShape(16),
    MoeShape(16, 2), MoeShape(16, 4), MoeShape(16, 5), MoeShape(16, 8),
)


class _Span:
    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def set_attribute(self, _key, _value): return None
    def record_exception(self, _exception): return None


class _Tracer:
    def start_as_current_span(self, _name): return _Span()


def _tensor_hash(tensor) -> str:
    import torch
    raw = tensor.detach().contiguous()
    payload = raw.view(torch.uint16).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _error_metrics(observed, reference) -> dict[str, float]:
    delta = observed.float() - reference.float()
    rms = float(delta.square().mean().sqrt().item())
    reference_rms = float(reference.float().square().mean().sqrt().item())
    return {
        "max_abs": float(delta.abs().max().item()),
        "rms": rms,
        "relative_rms": rms / reference_rms if reference_rms else 0.0,
    }


def _traffic_bytes(rows: int, unique_experts: int) -> int:
    routed_per_expert = (
        2 * INTERMEDIATE * (HIDDEN // 2)
        + HIDDEN * (INTERMEDIATE // 2)
        + 2 * INTERMEDIATE * (HIDDEN // 16)
        + HIDDEN * (INTERMEDIATE // 16)
        + 16
    )
    shared = 3 * (SHARED_INTERMEDIATE // 2) * HIDDEN * 2 + HIDDEN * 2
    activations = rows * HIDDEN * 2 * 2 + rows * TOP_K * 8
    return unique_experts * routed_per_expert + shared + activations


def _case(torch, rows: int, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(0xE256 + rows)
    hidden = torch.randn((rows, HIDDEN), dtype=torch.bfloat16, device=device, generator=generator) / 16
    pair = torch.arange(rows * TOP_K, dtype=torch.int32, device=device)
    ids = ((pair // 2) % LOCAL_EXPERTS + (pair % 2) * LOCAL_EXPERTS).view(rows, TOP_K)
    weights = torch.full((rows, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
    return hidden, ids, weights


def _measure(torch, call, iterations: int, rows: int, replay_atol: float):
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        output = call()
    graph.replay(); torch.cuda.synchronize()
    first = output.clone()
    graph.replay(); torch.cuda.synchronize()
    second = output.clone()
    replay_error = _error_metrics(first, second)
    error = replay_error["max_abs"]
    if error > replay_atol:
        raise RuntimeError(f"CUDA graph replay rows={rows} exceeds tolerance: max_abs={error}")
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record(); graph.replay(); stop.record(); stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return output.clone(), samples, replay_error, _tensor_hash(first), _tensor_hash(second)


def run(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 1):
        raise RuntimeError(f"SM121 required, got SM{props.major}{props.minor}")
    slab = load_owner_local_moe(args.artifact.resolve(), args.rank, args.layer)
    weights = materialize_flashinfer_weights(slab, torch_api=torch, device="cuda")
    backend = FlashInferRoutedMoeBackend(weights, torch_api=torch)
    graph = RoutedMoeGraph(slab, backend, _Tracer())
    cases = []
    shapes = tuple(shape for shape in SHAPES if not args.rows or shape.token_rows in args.rows)
    for shape in shapes:
        rows = shape.token_rows
        hidden, ids, routing = _case(torch, rows, weights.w1_weight.device)
        first = args.rank * LOCAL_EXPERTS
        owned = ids[(ids >= first) & (ids < first + LOCAL_EXPERTS)]
        local_routes = int(owned.numel())
        unique_experts = int(torch.unique(owned - first).numel())
        # Compile and allocate the selected path outside capture.
        graph.launch(hidden, ids, routing, shape)
        routed_a = backend.routed_only(
            hidden, ids, routing, rank=args.rank, shape=shape, selector=selector_for(shape)
        ).clone()
        routed_b = backend.routed_only(
            hidden, ids, routing, rank=args.rank, shape=shape, selector=selector_for(shape)
        ).clone()
        routed_replay_error = _error_metrics(routed_a, routed_b)
        if routed_replay_error["max_abs"] > args.replay_atol:
            raise RuntimeError(f"routed replay rows={rows} exceeds tolerance: max_abs={routed_replay_error['max_abs']}")
        shared_a = backend.shared_partial(hidden, args.rank).clone()
        shared_b = backend.shared_partial(hidden, args.rank).clone()
        shared_replay_error = _error_metrics(shared_a, shared_b)
        if shared_replay_error["max_abs"] > args.replay_atol:
            raise RuntimeError(f"shared replay rows={rows} exceeds tolerance: max_abs={shared_replay_error['max_abs']}")
        output, samples, replay_error, replay_hash_a, replay_hash_b = _measure(
            torch, lambda: graph.launch(hidden, ids, routing, shape), args.iterations,
            rows, args.replay_atol,
        )
        selected = selector_for(shape)
        control = backend.launch(
            hidden, ids, routing, rank=args.rank, shape=shape, selector="static"
        ).clone()
        reference_error = _error_metrics(output, control)
        if reference_error["max_abs"] > args.reference_atol:
            raise RuntimeError(f"selected/static parity failed for rows={rows}: {reference_error['max_abs']}")
        latency = statistics.median(samples)
        byte_count = _traffic_bytes(rows, unique_experts)
        roof_ms = byte_count / (READ_GBPS * 1.0e9) * 1.0e3
        cases.append({
            "sequences": shape.sequences,
            "verify_width": shape.verify_width,
            "rows": rows,
            "selector": selected,
            "latency_ms": latency,
            "latency_min_ms": min(samples),
            "latency_max_ms": max(samples),
            "latency_pstdev_ms": statistics.pstdev(samples),
            "touched_bytes_lower_bound": byte_count,
            "local_routes": local_routes,
            "unique_local_experts": unique_experts,
            "routes_per_unique_expert": local_routes / unique_experts,
            "effective_gbps_lower_bound": byte_count / (latency * 1.0e6),
            "roofline_fraction": roof_ms / latency,
            "full_48_layer_contribution_ms": latency * 48,
            "selected_vs_static_error": reference_error,
            "routed_replay_error": routed_replay_error,
            "shared_replay_error": shared_replay_error,
            "graph_replay_error": replay_error,
            "graph_replay_sha256": (replay_hash_a, replay_hash_b),
            "output_sha256": _tensor_hash(output),
        })
    shared = None
    if args.shared_parity:
        hidden, _, _ = _case(torch, 16, weights.w1_weight.device)
        left = backend.shared_partial(hidden, 0)
        right = backend.shared_partial(hidden, 1)
        full = backend.shared_reference(hidden)
        # Both partials use the same replicated checkpoint tensors. Their sum
        # is the full N320 control before PairReduce.
        shared_error = float(((left.float() + right.float()) - full.float()).abs().max().item())
        if shared_error > args.shared_atol:
            raise RuntimeError(f"shared EP2 shard parity failed: {shared_error}")
        shared = {"rows": 16, "max_abs": shared_error, "atol": args.shared_atol}
    return {
        "result": "qwen38_rank_local_routed_moe",
        "rank": args.rank,
        "layer": args.layer,
        "artifact_key": slab.artifact_key,
        "layout_sha256": slab.layout_sha256,
        "chunk_sha256": slab.chunk_sha256,
        "flashinfer_commit": "91bda04c66f7cb851e1ab3b78b9fecea644b9844",
        "read_roof_gbps": READ_GBPS,
        "shared_parity": shared,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("rank", type=int, choices=(0, 1))
    parser.add_argument("--layer", type=int, default=0, choices=range(48))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--shared-parity", action="store_true")
    parser.add_argument("--shared-atol", type=float, default=0.125)
    parser.add_argument("--replay-atol", type=float, default=0.01)
    parser.add_argument("--reference-atol", type=float, default=0.01)
    parser.add_argument("--rows", type=int, nargs="*", choices=tuple(shape.token_rows for shape in SHAPES))
    args = parser.parse_args()
    if args.iterations < 5:
        parser.error("--iterations must be at least 5")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()

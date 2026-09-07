#!/usr/bin/env python3
"""Measure the fixed production-width Qwen3.8 c16 CUTLASS Q/K/V graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines/qwen38-flash-next-nvfp4-2b/src"))

from qwen38_slab.cutlass_qkv import CutlassQkvRuntime  # noqa: E402
from qwen38_slab.projection import (  # noqa: E402
    PROJECTION_FAMILY_ROWS,
    PROJECTION_K,
    load_full_projection_payload,
    load_projection_payload,
    load_rank0_layer3_projection,
    reference_cutlass_projection,
)


def run(rank_slab: Path, library: Path, device: int, iterations: int) -> dict[str, object]:
    descriptor = load_rank0_layer3_projection(rank_slab)
    compact = load_projection_payload(descriptor)
    payload = load_full_projection_payload(descriptor)
    with CutlassQkvRuntime(payload, device=device, library=library) as runtime:
        runtime.launch()
        runtime.finish()
        first = runtime.output_bytes()
        actual = runtime.selected_outputs()
        runtime.launch()
        runtime.finish()
        second = runtime.output_bytes()
        expected = reference_cutlass_projection(compact, 16)
        timing = dict(runtime.benchmark(iterations))
        graph_nodes = runtime.graph_nodes
    if first != second:
        raise RuntimeError("CUTLASS QKV graph replay is not bit-exact")
    max_abs_error = max(abs(a - b) for a, b in zip(actual, expected))
    max_abs_reference = max(abs(value) for value in expected)
    if max_abs_error > max(2.0e-5, max_abs_reference * 0.02):
        raise RuntimeError(f"CUTLASS QKV error {max_abs_error} exceeds BF16 tolerance")

    outputs = sum(PROJECTION_FAMILY_ROWS)
    weight_bytes = outputs * (PROJECTION_K // 2 + PROJECTION_K // 16)
    activation_bytes = 16 * PROJECTION_K * 2
    output_bytes = 16 * outputs * 2
    projection_bytes = weight_bytes + 16 * PROJECTION_K // 2 + 16 * (PROJECTION_K // 16) + output_bytes
    requant_bytes = activation_bytes + 16 * PROJECTION_K // 2 + 16 * (PROJECTION_K // 16)
    flops = 2 * 16 * outputs * PROJECTION_K
    projection_gbps = projection_bytes / (timing["projection_ms"] * 1.0e6)
    projection_tflops = flops / (timing["projection_ms"] * 1.0e9)
    requant_gbps = requant_bytes / (timing["requant_ms"] * 1.0e6)
    scalar_control_gbps = 26.7613
    if projection_gbps <= scalar_control_gbps:
        raise RuntimeError("CUTLASS production-width projection did not beat scalar bandwidth")
    local_roof_gbps = 238.0
    aggregate_roof_gbps = 476.0
    base_step_ms = 1000.0 * 16 / 330.835
    return {
        "schema": "rocket.qwen38-cutlass-qkv-smoke.v1",
        "artifact_key": descriptor.artifact_key,
        "shape": [16, outputs, PROJECTION_K],
        "family_widths": list(PROJECTION_FAMILY_ROWS),
        "graph_nodes": graph_nodes,
        "deterministic_replay": "bit_exact",
        "selected_outputs": 16 * 12,
        "max_abs_error": max_abs_error,
        "max_abs_reference": max_abs_reference,
        "requant": {
            "ms": timing["requant_ms"],
            "effective_gbps": requant_gbps,
            "fraction_of_238_gbps_rank_local_roof": requant_gbps / local_roof_gbps,
        },
        "projection": {
            "ms": timing["projection_ms"],
            "graph_with_requant_ms": timing["graph_ms"],
            "effective_gbps": projection_gbps,
            "tflops": projection_tflops,
            "fraction_of_238_gbps_rank_local_roof": projection_gbps / local_roof_gbps,
            "scalar_control_gbps": scalar_control_gbps,
            "effective_bandwidth_speedup": projection_gbps / scalar_control_gbps,
        },
        "two_rank_composition": {
            "aggregate_traffic_roof_gbps": aggregate_roof_gbps,
            "traffic_only_early_ceiling_tok_s": 330.835,
            "requant_24_layer_ceiling_tok_s": 16.0 / ((base_step_ms + 24 * timing["requant_ms"]) / 1000.0),
            "requant_278_map_ceiling_tok_s": 16.0 / ((base_step_ms + 278 * timing["requant_ms"]) / 1000.0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-slab-artifact", type=Path, required=True)
    parser.add_argument(
        "--library", type=Path,
        default=ROOT / "engines/qwen38-flash-next-nvfp4-2b/build/libqwen38_cutlass_qkv.so",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(run(args.rank_slab_artifact, args.library, args.device, args.iterations), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

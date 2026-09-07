#!/usr/bin/env python3
"""Replay Qwen3.8 K0 metadata plus target-prologue graphs on CUDA 13."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0, str(ROOT / "engines/qwen38-flash-next-nvfp4-2b/src")
)

from qwen38_slab.decode import Depth, DepthZeroDecodeExecutor, StreamStep  # noqa: E402
from qwen38_slab.device_decode import (  # noqa: E402
    BUFFER_LAYOUT,
    DEFAULT_CUDART,
    Cuda13GraphRuntime,
    K0DeviceBinding,
)
from qwen38_slab.k0_target import (  # noqa: E402
    DEFAULT_CUDA_DRIVER,
    DEFAULT_NVRTC,
    K0_TARGET_SCHEMA,
    TARGET_OUTPUT_BYTES,
    TARGET_ROWS,
    K0TargetRow,
)
from qwen38_slab.projection import (  # noqa: E402
    PROJECTION_K,
    PROJECTION_OUTPUTS,
    load_projection_payload,
    load_rank0_layer3_projection,
    reference_projection,
)


class Span:
    def __init__(self):
        self.attributes = {}

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, exception): return None


class RecordingTracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, name):
        span = Span(); span.name = name; self.spans.append(span); return span


def expected_target_rows(prepared) -> tuple[K0TargetRow, ...]:
    result = []
    buffers = prepared.buffers
    for row in range(TARGET_ROWS):
        request = buffers.token_to_req[row] if row < prepared.lease.graph_batch else -1
        live = (
            0 <= request < prepared.lease.graph_batch
            and buffers.query_start_loc[request] == row
            and buffers.query_start_loc[request + 1] == row + 1
        )
        if not live:
            result.append(K0TargetRow(-1, -1, -1, -1))
            continue
        result.append(
            K0TargetRow(
                buffers.stream_slots[request],
                buffers.logical_positions[row],
                (buffers.seq_lens[request] << 32)
                | (buffers.raw_ring_offsets[row] & 0xFFFF_FFFF),
                buffers.compressed_positions[row],
            )
        )
    return tuple(result)


def run(
    device: int, cudart: Path, nvrtc: Path, driver: Path, rank_slab: Path
) -> dict[str, object]:
    tracer = RecordingTracer()
    executor = DepthZeroDecodeExecutor(tracer)
    projection = load_projection_payload(load_rank0_layer3_projection(rank_slab))
    with Cuda13GraphRuntime(
        device=device,
        library=cudart,
        nvrtc=nvrtc,
        driver=driver,
        projection=projection,
    ) as runtime:
        binding = K0DeviceBinding(runtime, tracer)
        first = executor.prepare(
            [StreamStep(slot, 262_143 - slot, Depth.K0) for slot in range(16)]
        )
        first_publication = binding.upload_and_launch(first)
        first_verified = all(
            runtime.read_active(name) == bytes(getattr(first.buffers, name))
            for name, _length in BUFFER_LAYOUT
        )
        first_target_verified = (
            runtime.read_target_rows() == expected_target_rows(first)
        )
        first_qkv = runtime.read_qkv_projection()
        reference_qkv = reference_projection(projection, first.lease.actual_batch)
        qkv_max_abs_error = max(
            abs(actual - expected)
            for actual, expected in zip(first_qkv, reference_qkv)
        )
        timing = dict(runtime.benchmark_projection())
        first_pointers = dict(runtime.active_device_pointers)

        second = executor.prepare(
            [StreamStep(slot, 64 + slot, Depth.K0) for slot in range(5)]
        )
        second_publication = binding.upload_and_launch(second)
        second_verified = all(
            runtime.read_active(name) == bytes(getattr(second.buffers, name))
            for name, _length in BUFFER_LAYOUT
        )
        second_target_verified = (
            runtime.read_target_rows() == expected_target_rows(second)
        )
        second_target_rows = runtime.read_target_rows()
        runtime_qkv_second = runtime.read_qkv_projection()
        second_pointers = dict(runtime.active_device_pointers)

        third = executor.prepare(
            [StreamStep(slot, 64 + slot, Depth.K0) for slot in range(5)]
        )
        third_publication = binding.upload_and_launch(third)
        third_verified = all(
            runtime.read_active(name) == bytes(getattr(third.buffers, name))
            for name, _length in BUFFER_LAYOUT
        )
        third_target_rows = runtime.read_target_rows()
        third_target_verified = third_target_rows == expected_target_rows(third)
        third_qkv = runtime.read_qkv_projection()
        third_pointers = dict(runtime.active_device_pointers)

    if not first_verified or not second_verified or not third_verified:
        raise RuntimeError("CUDA graph metadata copyback mismatch")
    if not (
        first_target_verified and second_target_verified and third_target_verified
    ):
        raise RuntimeError("CUDA graph target-prologue output mismatch")
    if (
        set(first_pointers.values()) & set(second_pointers.values())
        or set(second_pointers.values()) & set(third_pointers.values())
    ):
        raise RuntimeError("CUDA metadata publications did not alternate banks")
    if third_target_rows != second_target_rows:
        raise RuntimeError("CUDA target-prologue repeat was not bit-exact")
    if qkv_max_abs_error > 1.0e-6:
        raise RuntimeError("CUDA Q/K/V projection exceeded scalar reference error")
    if third_qkv != runtime_qkv_second:
        raise RuntimeError("CUDA Q/K/V projection repeat was not bit-exact")
    projection_flops = 16 * PROJECTION_OUTPUTS * PROJECTION_K * 2
    projection_bytes = 16 * PROJECTION_OUTPUTS * (
        PROJECTION_K // 2
        + PROJECTION_K // 16 * 4
        + PROJECTION_K // 2
        + PROJECTION_K // 16
        + 8
    )
    requant_bytes = 16 * (PROJECTION_K * 2 + PROJECTION_K // 2 + PROJECTION_K // 16 * 4)
    graph_gbps = projection_bytes / (timing["graph_ms"] * 1.0e6)
    graph_tflops = projection_flops / (timing["graph_ms"] * 1.0e9)
    requant_gbps = requant_bytes / (timing["requant_ms"] * 1.0e6)
    traffic_ceiling = 476.0
    base_step_ms = 1000.0 * 16 / 330.835
    full_attention_requants = 24
    final_map_requants = 278
    return {
        "schema": "rocket.qwen38-k0-cuda-smoke.v2",
        "device": device,
        "cuda_runtime": str(cudart),
        "cuda_driver": str(driver),
        "nvrtc": str(nvrtc),
        "captured_graphs": 10,
        "nodes_per_graph": runtime.graph_nodes_per_exec,
        "deterministic_repeat": "bit_exact",
        "target_schema": K0_TARGET_SCHEMA,
        "target_output_bytes": TARGET_OUTPUT_BYTES,
        "projection": {
            "schema": projection.descriptor.schema,
            "artifact_key": projection.descriptor.artifact_key,
            "layer": projection.descriptor.layer,
            "rank": projection.descriptor.rank,
            "shape": [16, PROJECTION_OUTPUTS, PROJECTION_K],
            "max_abs_error": qkv_max_abs_error,
            "deterministic_repeat": "bit_exact",
            "graph_ms": timing["graph_ms"],
            "effective_gbps": graph_gbps,
            "tflops": graph_tflops,
            "fraction_of_476_gbps_early_ceiling": graph_gbps / traffic_ceiling,
        },
        "activation_requant": {
            "c16_k2560_ms": timing["requant_ms"],
            "effective_gbps": requant_gbps,
            "fraction_of_composed_graph_time": timing["requant_ms"] / timing["graph_ms"],
            "k0_full_attention_24_call_ceiling_tok_s": 16
            / ((base_step_ms + full_attention_requants * timing["requant_ms"]) / 1000),
            "k0_equal_shape_278_call_ceiling_tok_s": 16
            / ((base_step_ms + final_map_requants * timing["requant_ms"]) / 1000),
            "traffic_only_k0_early_ceiling_tok_s": 330.835,
        },
        "metadata_fields": len(BUFFER_LAYOUT),
        "metadata_bytes": sum(length for _name, length in BUFFER_LAYOUT),
        "publications": [
            {
                "generation": first_publication.generation,
                "graph_batch": first_publication.graph_batch,
                "bank": first_publication.bank,
                "copyback": "match",
                "target_prologue": "match",
            },
            {
                "generation": second_publication.generation,
                "graph_batch": second_publication.graph_batch,
                "bank": second_publication.bank,
                "copyback": "match",
                "target_prologue": "match",
            },
            {
                "generation": third_publication.generation,
                "graph_batch": third_publication.graph_batch,
                "bank": third_publication.bank,
                "copyback": "match",
                "target_prologue": "match",
            },
        ],
        "otel_spans": len(tracer.spans),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cudart", type=Path, default=DEFAULT_CUDART)
    parser.add_argument("--nvrtc", type=Path, default=DEFAULT_NVRTC)
    parser.add_argument("--driver", type=Path, default=DEFAULT_CUDA_DRIVER)
    parser.add_argument("--rank-slab-artifact", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.device,
                args.cudart,
                args.nvrtc,
                args.driver,
                args.rank_slab_artifact,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

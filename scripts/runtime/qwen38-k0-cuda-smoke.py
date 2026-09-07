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
    device: int, cudart: Path, nvrtc: Path, driver: Path
) -> dict[str, object]:
    tracer = RecordingTracer()
    executor = DepthZeroDecodeExecutor(tracer)
    with Cuda13GraphRuntime(
        device=device, library=cudart, nvrtc=nvrtc, driver=driver
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
    return {
        "schema": "rocket.qwen38-k0-cuda-smoke.v1",
        "device": device,
        "cuda_runtime": str(cudart),
        "cuda_driver": str(driver),
        "nvrtc": str(nvrtc),
        "captured_graphs": 10,
        "nodes_per_graph": runtime.graph_nodes_per_exec,
        "deterministic_repeat": "bit_exact",
        "target_schema": K0_TARGET_SCHEMA,
        "target_output_bytes": TARGET_OUTPUT_BYTES,
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
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.device, args.cudart, args.nvrtc, args.driver), sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

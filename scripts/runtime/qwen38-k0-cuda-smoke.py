#!/usr/bin/env python3
"""Replay the Qwen3.8 K0 metadata graph twice on one CUDA 13 device."""

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


def run(device: int, cudart: Path) -> dict[str, object]:
    tracer = RecordingTracer()
    executor = DepthZeroDecodeExecutor(tracer)
    with Cuda13GraphRuntime(device=device, library=cudart) as runtime:
        binding = K0DeviceBinding(runtime, tracer)
        first = executor.prepare(
            [StreamStep(slot, 262_143 - slot, Depth.K0) for slot in range(16)]
        )
        first_publication = binding.upload_and_launch(first)
        first_verified = all(
            runtime.read_active(name) == bytes(getattr(first.buffers, name))
            for name, _length in BUFFER_LAYOUT
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
        second_pointers = dict(runtime.active_device_pointers)

    if not first_verified or not second_verified:
        raise RuntimeError("CUDA graph metadata copyback mismatch")
    if set(first_pointers.values()) & set(second_pointers.values()):
        raise RuntimeError("CUDA metadata publications did not alternate banks")
    return {
        "schema": "rocket.qwen38-k0-cuda-smoke.v1",
        "device": device,
        "cuda_runtime": str(cudart),
        "captured_graphs": 10,
        "metadata_fields": len(BUFFER_LAYOUT),
        "metadata_bytes": sum(length for _name, length in BUFFER_LAYOUT),
        "publications": [
            {
                "generation": first_publication.generation,
                "graph_batch": first_publication.graph_batch,
                "bank": first_publication.bank,
                "copyback": "match",
            },
            {
                "generation": second_publication.generation,
                "graph_batch": second_publication.graph_batch,
                "bank": second_publication.bank,
                "copyback": "match",
            },
        ],
        "otel_spans": len(tracer.spans),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cudart", type=Path, default=DEFAULT_CUDART)
    args = parser.parse_args()
    print(json.dumps(run(args.device, args.cudart), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

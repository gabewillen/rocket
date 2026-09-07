#!/usr/bin/env python3
"""Run one exact layer-3 rank-0 attention transition over real TP2 PairReduce."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engines/qwen38-flash-next-nvfp4-2b"
sys.path.insert(0, str(ENGINE / "src"))

from qwen38_slab.decode import Depth, DepthZeroDecodeExecutor, StreamStep  # noqa: E402
from qwen38_slab.device_decode import (  # noqa: E402
    DEFAULT_CUDART, Cuda13GraphRuntime, K0DeviceBinding, _Cuda13Api,
)
from qwen38_slab.full_attention_layer import (  # noqa: E402
    FullAttentionLayerExecutor, RdmaPairReduceRuntime,
)
from qwen38_slab.hyperconnection import (  # noqa: E402
    HC_HIDDEN, HC_LOW_RANK, HC_STREAMS, HC_WIDTH,
    CudaHyperConnectionRuntime, load_layer3_hyperconnection,
)
from qwen38_slab.projection import (  # noqa: E402
    OUTPUT_PROJECTION_K, OUTPUT_PROJECTION_N, PROJECTION_FAMILY_ROWS,
    PROJECTION_K, load_full_projection_payload, load_rank0_layer3_projection,
)


class Span:
    def __init__(self): self.attributes = {}
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, exception): self.exception = type(exception).__name__


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, name):
        span = Span(); span.name = name; self.spans.append(span); return span


def hc_logical_bytes(m: int) -> int:
    weights = HC_WIDTH * 2 + HC_LOW_RANK * HC_WIDTH * 2
    weights += HC_STREAMS * HC_WIDTH * 2 + HC_WIDTH * HC_LOW_RANK * 2
    mix_tensors = m * (HC_WIDTH * 2 * 4 + HC_HIDDEN * 2 + HC_LOW_RANK * 4)
    combine_extra = m * (HC_HIDDEN * 4 + HC_STREAMS * 2 + HC_WIDTH * 2)
    return 2 * weights + 2 * mix_tensors + combine_extra


def run_rank0(args) -> dict[str, object]:
    descriptor = load_rank0_layer3_projection(args.rank_slab_artifact)
    projection = load_full_projection_payload(descriptor)
    hc_payload = load_layer3_hyperconnection(args.rank_slab_artifact)
    tracer = Tracer()
    scheduler = DepthZeroDecodeExecutor(tracer)
    with Cuda13GraphRuntime(
        device=0, library=args.cudart, full_projection=projection
    ) as runtime:
        binding = K0DeviceBinding(runtime, tracer)
        with CudaHyperConnectionRuntime(
            hc_payload, args.hyperconnection_library, runtime.stream_pointer,
            device=0, cudart=args.cudart,
        ) as hc:
            with RdmaPairReduceRuntime(
                0, args.pair_reduce_library, args.bootstrap_host,
                args.port, args.timeout_ms,
            ) as reducer:
                layer = FullAttentionLayerExecutor(
                    runtime, binding, hc, reducer, tracer
                )
                prepared = scheduler.prepare([
                    StreamStep(slot, 262_143 - slot, Depth.K0)
                    for slot in range(args.m)
                ])
                publication = layer.execute(prepared, verify_reference=True)
                hc_timing = dict(hc.benchmark(prepared.lease.graph_batch))
                qsa_timing = dict(runtime.benchmark_qsa_indexer(20))
                pair_counts = dict(reducer.counts)
                logical_bytes = hc_logical_bytes(prepared.lease.graph_batch)
                hc_ms = hc_timing["mix_ms"] + hc_timing["combine_mix_ms"]
                cold_hc_ms = (
                    publication.stage_ms["attn_hc_mix"]
                    + publication.stage_ms["mlp_hc_combine_mix"]
                )
                full_outputs = sum(PROJECTION_FAMILY_ROWS)
                projection_bytes = (
                    full_outputs * (PROJECTION_K // 2 + PROJECTION_K // 16)
                    + prepared.lease.graph_batch
                    * (PROJECTION_K // 2 + PROJECTION_K // 16 + full_outputs * 2)
                )
                score_bytes = prepared.lease.graph_batch * 65_536 * (128 * 2 + 4)
                sparse_attention_bytes = (
                    prepared.lease.graph_batch * 2_051 * 256 * 2 * 2
                )
                output_projection_bytes = (
                    OUTPUT_PROJECTION_N
                    * (OUTPUT_PROJECTION_K // 2 + OUTPUT_PROJECTION_K // 16)
                    + prepared.lease.graph_batch
                    * (OUTPUT_PROJECTION_K + OUTPUT_PROJECTION_N) * 2
                )
                minimum_layer_bytes = (
                    logical_bytes + projection_bytes + score_bytes
                    + sparse_attention_bytes + output_projection_bytes
                )
                composed_ms = sum(publication.stage_ms.values())
                stages = {
                    span.attributes.get("stage") for span in tracer.spans
                    if span.name == "rocket.qwen38.decode.full_attention_layer"
                }
                return {
                    "result": "qwen38_layer3_full_attention_rank0",
                    "artifact_key": descriptor.artifact_key,
                    "layer": descriptor.layer,
                    "rank": descriptor.rank,
                    "actual_m": args.m,
                    "graph_batch": publication.graph_batch,
                    "attention_graph_nodes": runtime.graph_nodes_per_exec,
                    "hc_graph_nodes": {
                        f"{operation}_m{m}": nodes
                        for (operation, m), nodes in hc.graph_nodes.items()
                    },
                    "intermediate_hashes": dict(publication.intermediate_hashes),
                    "reference_hashes": "exact_match",
                    "stage_ms": dict(publication.stage_ms),
                    "hc_kernel_ms": hc_timing,
                    "hc_logical_bytes": logical_bytes,
                    "hc_hot_cache_effective_gbps": logical_bytes / (hc_ms * 1.0e6),
                    "hc_cold_composed_effective_gbps": (
                        logical_bytes / (cold_hc_ms * 1.0e6)
                    ),
                    "hc_cold_fraction_of_238_gbps_rank_local_roof": (
                        logical_bytes / (cold_hc_ms * 1.0e6) / 238.0
                    ),
                    "minimum_layer_logical_bytes": minimum_layer_bytes,
                    "composed_stage_total_ms": composed_ms,
                    "minimum_layer_effective_gbps": (
                        minimum_layer_bytes / (composed_ms * 1.0e6)
                    ),
                    "minimum_layer_fraction_of_238_gbps_rank_local_roof": (
                        minimum_layer_bytes / (composed_ms * 1.0e6) / 238.0
                    ),
                    "qsa_ms": qsa_timing,
                    "pair_reduce_otel": pair_counts,
                    "layer_stage_labels": sorted(stages),
                    "layer_stage_label_cardinality": len(stages),
                    "otel_spans": len(tracer.spans),
                }


def run_rank1(args) -> dict[str, object]:
    api = _Cuda13Api(args.cudart)
    stream, source, output = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    api.call("cudaSetDevice", 0)
    api.call("cudaStreamCreateWithFlags", ctypes.byref(stream), 1)
    size = args.m * HC_HIDDEN * 2
    api.call("cudaMalloc", ctypes.byref(source), size)
    api.call("cudaMalloc", ctypes.byref(output), args.m * HC_HIDDEN * 4)
    api.call("cudaMemsetAsync", source, 0, size, stream)
    api.call("cudaStreamSynchronize", stream)
    try:
        with RdmaPairReduceRuntime(
            1, args.pair_reduce_library, args.bootstrap_host,
            args.port, args.timeout_ms,
        ) as reducer:
            reducer.reduce(source.value, output.value, args.m, stream.value)
            raw = ctypes.create_string_buffer(args.m * HC_HIDDEN * 4)
            api.call("cudaMemcpy", ctypes.addressof(raw), output, len(raw), 2)
            return {
                "result": "qwen38_layer3_full_attention_rank1_zero_peer",
                "rank": 1,
                "m": args.m,
                "output_sha256": hashlib.sha256(raw.raw).hexdigest(),
                "pair_reduce_otel": dict(reducer.counts),
            }
    finally:
        api.call("cudaFree", output)
        api.call("cudaFree", source)
        api.call("cudaStreamDestroy", stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--m", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--bootstrap-host", required=True)
    parser.add_argument("--port", type=int, default=18839)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--rank-slab-artifact", type=Path)
    parser.add_argument("--cudart", type=Path, default=DEFAULT_CUDART)
    parser.add_argument(
        "--hyperconnection-library", type=Path,
        default=ENGINE / "build/libqwen38_hyperconnection.so",
    )
    parser.add_argument(
        "--pair-reduce-library", type=Path,
        default=ENGINE / "build/libqwen38_layer_pair_reduce.so",
    )
    args = parser.parse_args()
    if args.rank == 0 and args.rank_slab_artifact is None:
        parser.error("rank 0 requires --rank-slab-artifact")
    print(json.dumps(run_rank0(args) if args.rank == 0 else run_rank1(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

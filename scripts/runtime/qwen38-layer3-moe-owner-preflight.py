#!/usr/bin/env python3
"""Load one accepted rank slab and construct the layer3 MoE owner, no enqueue."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines/qwen38-flash-next-nvfp4-2b/src"))

from qwen38_slab.cuda_slab_loader import (  # noqa: E402
    CudaRankSlabLoader, CudaSlabCleanupIncompleteError, CudaSlabLoadError,
    CudaSlabPublicationError,
)
from qwen38_slab.layer3_factory import (  # noqa: E402
    CtypesNativeTargetSlabLeaseFactory, Layer3FactoryError,
    NativeTargetSlabFinalizeError, native_target_slab_handoff,
)

SCHEMA = "rocket.qwen38.layer3-moe-owner-preflight.v1"
ARTIFACT_KEY = "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"


class _Owner:
    def __init__(self): self.publication = None
    def publish_rank_slabs(self, rank, slabs):
        if self.publication is not None:
            raise RuntimeError("duplicate slab publication")
        self.publication = (rank, slabs)


def _typed_cause_chain(error: BaseException, phase: str) -> tuple[dict[str, str], ...]:
    """Return at most four bounded failure classifications, never messages."""

    chain = []
    current: BaseException | None = error
    while current is not None and len(chain) < 4:
        if isinstance(current, CudaSlabPublicationError):
            kind, stage = "slab_publication", "python_publish"
        elif isinstance(current, CudaSlabCleanupIncompleteError):
            kind, stage = "slab_cleanup", "cuda_cleanup"
        elif isinstance(current, CudaSlabLoadError):
            kind, stage = "slab_load", "accepted_loader"
        elif isinstance(current, NativeTargetSlabFinalizeError):
            kind, stage = "native_finalize", current.stage
        elif isinstance(current, Layer3FactoryError):
            kind, stage = "layer3_factory", "native_finalize"
        elif isinstance(current, OSError):
            kind, stage = "io", "artifact_io"
        elif isinstance(current, TimeoutError):
            kind, stage = "timeout", "supervisor"
        elif isinstance(current, (ValueError, TypeError)):
            kind, stage = "contract", phase
        elif isinstance(current, RuntimeError):
            kind, stage = "runtime", phase
        else:
            kind, stage = "unknown", phase
        chain.append({"class": kind, "stage": stage})
        current = current.__cause__
    return tuple(chain)


def _emit_failure(counter, rank: int, phase: str,
                  terminal: dict[str, str]) -> None:
    """Best-effort bounded OTEL publication for one terminal failure."""

    try:
        counter.add(1, {
            "rank": rank, "phase": phase, "outcome": "failure",
            "failure.class": terminal["class"],
            "failure.stage": terminal["stage"],
        })
    except BaseException:
        pass


def _descriptor(path: Path, rank: int) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "rocket.qwen38.layer3-native-plan.v1"
        or value.get("rank") != rank
        or value.get("artifact_key") != ARTIFACT_KEY
        or value.get("slab_key") != f"rank{rank}-target"
        or not isinstance(value.get("slab_publication_layout_sha256"), str)
        or len(value["slab_publication_layout_sha256"]) != 64
    ):
        raise ValueError("native descriptor identity changed")
    return value


def _native_preflight(library: Path, descriptor: Path, rank: int,
                      device_index: int, lease) -> dict[str, int]:
    native = ctypes.CDLL(str(library))
    call = native.qwen38_target_layer3_moe_owner_preflight
    call.argtypes = (
        ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
    )
    call.restype = ctypes.c_int
    error = native.qwen38_target_layer3_moe_owner_preflight_last_error
    error.restype = ctypes.c_char_p
    values = [ctypes.c_uint64() for _ in range(4)]
    result = call(
        device_index, rank, os.fsencode(descriptor), lease,
        *(ctypes.byref(value) for value in values),
    )
    if result:
        reason = (error() or b"native preflight rejected").decode("utf-8", "replace")
        raise RuntimeError(reason[:384])
    return dict(zip(("stage_bytes", "runtime_bytes", "full_otel_records",
                     "stage_otel_records"), (value.value for value in values)))


def worker(args: argparse.Namespace) -> int:
    from opentelemetry import metrics, trace

    started_ns = time.perf_counter_ns()
    phase = "descriptor"
    tracer = trace.get_tracer("rocket.qwen38.layer3_moe_owner_preflight")
    meter = metrics.get_meter("rocket.qwen38.layer3_moe_owner_preflight")
    outcome_counter = meter.create_counter(
        "rocket.qwen38.layer3_moe_owner_preflight", unit="{preflight}"
    )
    try:
        descriptor = _descriptor(args.native_plan, args.rank)
        phase = "load"
        import torch
        owner = _Owner()
        finalizer = CtypesNativeTargetSlabLeaseFactory(args.library)
        loaded = CudaRankSlabLoader(
            args.artifact, rank=args.rank, owner=owner, tracer=tracer,
            meter=meter, torch_api=torch, device=f"cuda:{args.device_index}",
            native_target_finalizer=finalizer,
            target_layout_sha256=descriptor["slab_publication_layout_sha256"],
        ).load()
        phase = "handoff"
        handoff = native_target_slab_handoff(descriptor, loaded)
        phase = "owner_construct"
        evidence = _native_preflight(
            args.library, args.native_plan, args.rank, args.device_index,
            handoff.native_lease,
        )
        outcome_counter.add(1, {
            "rank": args.rank, "phase": "complete", "outcome": "success",
        })
        print(json.dumps({
            "schema": SCHEMA, "valid": True, "complete": True,
            "phase": "complete", "rank": args.rank,
            "artifact_key": descriptor["artifact_key"],
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "receipt_sha256": handoff.receipt_sha256,
            "target_bytes": handoff.bytes,
            "load_to_publish_ns": loaded.receipt.load_to_publish_ns,
            "elapsed_ns": time.perf_counter_ns() - started_ns,
            "kernel_launches": 0, "source_waits": 0, **evidence,
        }, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        cause_chain = _typed_cause_chain(exc, phase)
        terminal = cause_chain[-1]
        _emit_failure(outcome_counter, args.rank, phase, terminal)
        print(json.dumps({
            "schema": SCHEMA, "valid": False, "complete": False,
            "phase": phase, "rank": args.rank,
            "failure_class": terminal["class"],
            "failure_stage": terminal["stage"],
            "cause_chain": cause_chain,
            "elapsed_ns": time.perf_counter_ns() - started_ns,
            "kernel_launches": 0,
        }, sort_keys=True), flush=True)
        return 1


def supervise(args: argparse.Namespace) -> int:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--rank", str(args.rank), "--artifact", str(args.artifact),
               "--native-plan", str(args.native_plan), "--library", str(args.library),
               "--device-index", str(args.device_index),
               "--timeout-seconds", str(args.timeout_seconds)]
    try:
        result = subprocess.run(command, text=True, capture_output=True,
                                timeout=args.timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        try:
            from opentelemetry import metrics
            metrics.get_meter("rocket.qwen38.layer3_moe_owner_preflight").create_counter(
                "rocket.qwen38.layer3_moe_owner_preflight", unit="{preflight}"
            ).add(1, {"rank": args.rank, "phase": "timeout", "outcome": "failure"})
        except BaseException:
            pass
        print(json.dumps({"schema": SCHEMA, "valid": False, "complete": False,
                          "phase": "timeout", "rank": args.rank,
                          "failure_class": "timeout", "kernel_launches": 0},
                         sort_keys=True))
        return 124
    records = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if len(records) != 1:
        print(json.dumps({
            "schema": SCHEMA, "valid": False, "complete": False,
            "phase": "child_result", "rank": args.rank,
            "failure_class": "contract", "kernel_launches": 0,
        }, sort_keys=True))
        return 1
    print(json.dumps(json.loads(records[0]), sort_keys=True))
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device-index", type=int, choices=range(0, 16), default=0)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--native-plan", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=90,
                        choices=range(30, 301), metavar="[30-300]")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())

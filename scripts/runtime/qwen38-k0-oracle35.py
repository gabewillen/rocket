#!/usr/bin/env python3
"""Run one rank of the authenticated Rocket oracle35 K0 path."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engines/qwen38-flash-next-nvfp4-2b"
sys.path.insert(0, str(ENGINE / "src"))

from qwen38_slab.contract import canonical_bytes  # noqa: E402
from qwen38_slab.cuda_slab_loader import CudaRankSlabLoader  # noqa: E402
from qwen38_slab.layer3_factory import (  # noqa: E402
    CtypesNativeTargetSlabLeaseFactory, native_target_slab_handoff,
)
from qwen38_slab.target_layer_descriptor import (  # noqa: E402
    target_layer_descriptor,
)

SCHEMA = "rocket.qwen38.k0-oracle35-run.v1"
EXPECTED_TOKEN = 248_046


class _Owner:
    def __init__(self) -> None:
        self.publication = None

    def publish_rank_slabs(self, rank: int, slabs: object) -> None:
        if self.publication is not None:
            raise RuntimeError("duplicate slab publication")
        self.publication = (rank, slabs)


class _NativeResult(ctypes.Structure):
    _fields_ = (
        ("token", ctypes.c_int32), ("rows", ctypes.c_int32),
        ("final_generation", ctypes.c_uint64),
        ("lifecycle_outcomes", ctypes.c_uint64 * 4),
        ("moe_components", ctypes.c_uint64 * 5),
        ("stage_counters", ctypes.c_uint64 * 4),
        ("state_outcomes", ctypes.c_uint64 * 4),
        ("nccl_stages", ctypes.c_uint64 * 9),
        ("nccl_outcomes", ctypes.c_uint64 * 7),
        ("duration_samples", ctypes.c_uint64),
        ("total_bytes", ctypes.c_uint64),
    )


def _secret(path: Path) -> bytes:
    value = path.read_bytes()
    if len(value) != 32:
        raise ValueError("bootstrap secret extent changed")
    return value


def _native_symbols(library: Path) -> CtypesNativeTargetSlabLeaseFactory:
    native = ctypes.CDLL(str(library))
    getattr(native, "qwen38_target_k0_oracle35_run")
    return CtypesNativeTargetSlabLeaseFactory(
        library, retain_symbol="qwen38_target_k0_retain_accepted_loader",
    )


def _descriptors(artifact: Path, sidecar: Path, output: Path) -> dict[str, object]:
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    selected = None
    expected = {f"rank{rank}-layer{layer}.json"
                for rank in (0, 1) for layer in range(48)}
    observed = {item.name for item in output.iterdir() if item.is_file()}
    if observed - expected:
        raise ValueError("descriptor directory contains unknown files")
    for rank in (0, 1):
        for layer in range(48):
            descriptor = target_layer_descriptor(artifact, sidecar, rank, layer)
            payload = canonical_bytes(descriptor) + b"\n"
            destination = output / f"rank{rank}-layer{layer}.json"
            if destination.exists() and destination.read_bytes() != payload:
                raise ValueError("existing descriptor identity changed")
            if not destination.exists():
                fd, temporary = tempfile.mkstemp(dir=output, prefix=".descriptor-")
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, destination)
                except BaseException:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
                    raise
            if rank == 0 and layer == 0:
                selected = descriptor
    if selected is None or {item.name for item in output.iterdir()} != expected:
        raise ValueError("descriptor inventory changed")
    return selected


def _native_run(args: argparse.Namespace, lease: object,
                sessions: tuple[bytes, bytes, bytes, bytes]) -> _NativeResult:
    native = ctypes.CDLL(str(args.library))
    call = native.qwen38_target_k0_oracle35_run
    byte_pointer = ctypes.POINTER(ctypes.c_uint8)
    call.argtypes = (
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint32, byte_pointer, byte_pointer, byte_pointer, byte_pointer,
        ctypes.POINTER(_NativeResult),
    )
    call.restype = ctypes.c_int
    arrays = tuple((ctypes.c_uint8 * 32).from_buffer_copy(item)
                   for item in sessions)
    result = _NativeResult()
    status = call(
        args.device_index, args.rank, lease,
        os.fsencode(args.descriptor_directory), os.fsencode(args.sidecar_payload),
        os.fsencode(args.tokenizer), os.fsencode(args.oracle_capture),
        args.bootstrap_host.encode("ascii"), args.layer_port,
        args.embedding_port, args.nccl_port, args.timeout_ms,
        *arrays, ctypes.byref(result),
    )
    if status:
        raise RuntimeError(f"native_status_{status}")
    return result


def _snapshot(result: _NativeResult) -> dict[str, object]:
    return {
        "token": result.token, "rows": result.rows,
        "final_generation": result.final_generation,
        "lifecycle_outcomes": list(result.lifecycle_outcomes),
        "moe_components": list(result.moe_components),
        "stage_counters": list(result.stage_counters),
        "state_outcomes": list(result.state_outcomes),
        "nccl_stages": list(result.nccl_stages),
        "nccl_outcomes": list(result.nccl_outcomes),
        "duration_samples": result.duration_samples,
        "total_bytes": result.total_bytes,
    }


def worker(args: argparse.Namespace) -> int:
    from opentelemetry import metrics, trace
    import torch

    tracer = trace.get_tracer("rocket.qwen38.k0_oracle35")
    meter = metrics.get_meter("rocket.qwen38.k0_oracle35")
    terminal = meter.create_counter("rocket.qwen38.k0_oracle35", unit="{run}")
    phase = "descriptor"
    started = time.perf_counter_ns()
    try:
        sessions = tuple(_secret(path) for path in (
            args.layer_session_file, args.embedding_session_file,
            args.nccl_session_file, args.nccl_authentication_key_file,
        ))
        finalizer = _native_symbols(args.library)
        descriptor = _descriptors(args.artifact, args.sidecar,
                                  args.descriptor_directory)
        rank_descriptor = json.loads((args.descriptor_directory /
                                      f"rank{args.rank}-layer0.json").read_bytes())
        phase = "load"
        owner = _Owner()
        loaded = CudaRankSlabLoader(
            args.artifact, rank=args.rank, owner=owner, tracer=tracer,
            meter=meter, torch_api=torch, device=f"cuda:{args.device_index}",
            native_target_finalizer=finalizer,
            target_layout_sha256=rank_descriptor["slab_publication_layout_sha256"],
        ).load()
        phase = "handoff"
        handoff = native_target_slab_handoff(rank_descriptor, loaded)
        phase = "native"
        result = _native_run(args, handoff.native_lease, sessions)
        evidence = _snapshot(result)
        if result.token != EXPECTED_TOKEN or result.rows != 35:
            raise RuntimeError("native oracle result changed")
        terminal.add(1, {"rank": args.rank, "phase": "complete",
                         "outcome": "success"})
        print(json.dumps({"schema": SCHEMA, "valid": True, "complete": True,
                          "rank": args.rank, "phase": "complete",
                          "elapsed_ns": time.perf_counter_ns() - started,
                          **evidence}, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        try:
            terminal.add(1, {"rank": args.rank, "phase": phase,
                             "outcome": "failure"})
        except BaseException:
            pass
        print(json.dumps({"schema": SCHEMA, "valid": False, "complete": False,
                          "rank": args.rank, "phase": phase,
                          "failure_class": type(error).__name__[:64],
                          "elapsed_ns": time.perf_counter_ns() - started},
                         sort_keys=True), flush=True)
        return 1


def supervise(args: argparse.Namespace) -> int:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    for key, value in vars(args).items():
        if key in ("worker", "timeout_seconds"):
            continue
        option = "--" + key.replace("_", "-")
        command.extend((option, str(value)))
    command.extend(("--timeout-seconds", str(args.timeout_seconds)))
    try:
        process = subprocess.run(command, text=True, capture_output=True,
                                 timeout=args.timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        print(json.dumps({"schema": SCHEMA, "valid": False, "complete": False,
                          "rank": args.rank, "phase": "timeout",
                          "failure_class": "timeout"}, sort_keys=True))
        return 124
    records = [line for line in process.stdout.splitlines()
               if line.startswith("{") and line.endswith("}")]
    if len(records) != 1:
        print(json.dumps({"schema": SCHEMA, "valid": False, "complete": False,
                          "rank": args.rank, "phase": "child_result",
                          "failure_class": "contract"}, sort_keys=True))
        return 1
    print(records[0])
    return process.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device-index", type=int, choices=(0,), default=0)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--sidecar-payload", type=Path, required=True)
    parser.add_argument("--descriptor-directory", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--oracle-capture", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bootstrap-host", required=True)
    parser.add_argument("--layer-port", type=int, required=True)
    parser.add_argument("--embedding-port", type=int, required=True)
    parser.add_argument("--nccl-port", type=int, required=True)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--timeout-seconds", type=int, default=240,
                        choices=range(60, 601))
    parser.add_argument("--layer-session-file", type=Path, required=True)
    parser.add_argument("--embedding-session-file", type=Path, required=True)
    parser.add_argument("--nccl-session-file", type=Path, required=True)
    parser.add_argument("--nccl-authentication-key-file", type=Path, required=True)
    args = parser.parse_args()
    ports = (args.layer_port, args.embedding_port, args.nccl_port)
    if len(set(ports)) != 3 or any(not 1 <= value <= 65535 for value in ports):
        parser.error("three distinct bounded bootstrap ports are required")
    return args


if __name__ == "__main__":
    raise SystemExit(worker(parse_args()) if "--worker" in sys.argv
                     else supervise(parse_args()))

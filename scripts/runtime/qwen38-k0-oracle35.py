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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engines/qwen38-flash-next-nvfp4-2b"
sys.path.insert(0, str(ENGINE / "src"))

from qwen38_slab.contract import canonical_bytes  # noqa: E402
from qwen38_slab.contract import SlabError  # noqa: E402
from qwen38_slab.cuda_slab_loader import (  # noqa: E402
    CudaRankSlabLoader, CudaSlabCleanupIncompleteError, CudaSlabLoadError,
    CudaSlabPublicationError,
)
from qwen38_slab.layer3_factory import (  # noqa: E402
    CtypesNativeTargetSlabLeaseFactory, Layer3FactoryError,
    NativeTargetSlabFinalizeError, native_target_slab_handoff,
)
from qwen38_slab.target_layer_descriptor import (  # noqa: E402
    authenticate_descriptor_identity, descriptor_identity,
    load_descriptor_allowlist,
)

SCHEMA = "rocket.qwen38.k0-oracle35-run.v1"
EXPECTED_TOKEN = 248_046
NATIVE_STATUS_STAGES = {
    10: "validation", 20: "layer_pair_reduce_bootstrap",
    21: "embedding_pair_reduce_bootstrap", 22: "nccl_bootstrap",
    30: "token_io_construction", 31: "physical_layer_construction",
    32: "comparator_startup_construction", 40: "source_waits",
    41: "prompt_execution", 42: "terminal_fence",
    43: "oracle_comparison", 50: "cleanup", 51: "quarantine",
    255: "unknown",
}
PHYSICAL_LAYER_SUBSTAGES = {
    0: "unknown", 1: "plan_inventory", 2: "qsa_arena",
    3: "sidecar", 4: "rope", 5: "gdn_owner", 6: "qsa_owner",
    7: "inventory_assembly",
}
GDN_OWNER_SUBSTAGES = {
    0: "unknown", 1: "lease", 2: "plan_binder", 3: "globals",
    4: "moe_stage", 5: "moe_aot", 6: "storage", 7: "cutlass_graph",
    8: "hyperconnection", 9: "composite",
    10: "moe_aot_identity", 11: "moe_aot_module_data",
    12: "moe_aot_module_load", 13: "moe_participant_contract",
}
MOE_AOT_CUDA_FAILURES = {
    0: "success", 1: "invalid_value", 2: "invalid_image", 3: "invalid_ptx",
    4: "no_binary_for_gpu", 5: "out_of_memory", 6: "not_supported",
    7: "other",
}
EXECUTION_STAGES = {
    0: "validation", 1: "token_source_wait", 2: "layer_source_wait",
    3: "begin_sequence", 4: "begin_row", 5: "embedding_reduction",
    6: "embedding_comparison", 7: "layer_execution",
    8: "layer_comparison", 9: "final_norm", 10: "lm_head",
    11: "winner_exchange", 12: "terminal_fence",
    13: "final_norm_comparison", 14: "logits_comparison",
    15: "token_comparison", 16: "complete",
}
LAYER_EXECUTION_STAGES = {
    0: "none", 1: "state_preparation", 2: "attention_hyperconnection",
    3: "attention", 4: "attention_fence", 5: "attention_reduction",
    6: "mlp_hyperconnection", 7: "moe", 8: "moe_reduction",
    9: "final_hyperconnection",
}
GDN_GRAPH_STAGES = {
    0: "none", 1: "input_quantize", 2: "qkv_projection",
    3: "ba_projection", 4: "input_scale", 5: "core",
    6: "output_quantize", 7: "output_projection", 8: "output_scale",
    9: "launch_check", 10: "output_publication",
}
STARTUP_CONSTRUCTION_STAGES = {
    0: "none", 1: "dependency_validation",
    2: "token_io_pair_reduce_registration",
    3: "tokenizer_reauthentication",
    4: "executor_ownership_validation", 5: "final_publication",
}


class NativeRunStatusError(RuntimeError):
    def __init__(self, status: int, physical_substage: int = 0,
                 physical_layer: int = -1, gdn_owner_substage: int = 0,
                 moe_aot_cuda_failure: int = 0, execution_stage: int = 0,
                 execution_row: int = -1, execution_layer: int = -1,
                 execution_layer_stage: int = 0, gdn_graph_stage: int = 0,
                 startup_construction_stage: int = 0):
        self.stage = NATIVE_STATUS_STAGES.get(status, "unknown")
        self.physical_substage = PHYSICAL_LAYER_SUBSTAGES.get(
            physical_substage, "unknown")
        owner_stage = self.physical_substage in ("gdn_owner", "qsa_owner")
        self.physical_layer = (physical_layer if owner_stage
                               and 0 <= physical_layer < 48 else -1)
        self.gdn_owner_substage = GDN_OWNER_SUBSTAGES.get(
            gdn_owner_substage, "unknown") if self.physical_substage == (
                "gdn_owner") else "unknown"
        self.moe_aot_cuda_failure = MOE_AOT_CUDA_FAILURES.get(
            moe_aot_cuda_failure, "other")
        self.execution_stage = EXECUTION_STAGES.get(execution_stage, "unknown")
        self.execution_row = execution_row if 0 <= execution_row < 35 else -1
        self.execution_layer = (execution_layer if 0 <= execution_layer < 48
                                else -1)
        self.execution_layer_stage = LAYER_EXECUTION_STAGES.get(
            execution_layer_stage, "unknown")
        self.gdn_graph_stage = GDN_GRAPH_STAGES.get(gdn_graph_stage, "unknown")
        self.startup_construction_stage = STARTUP_CONSTRUCTION_STAGES.get(
            startup_construction_stage, "unknown")
        self.layer_boundary_diagnostics = ()
        super().__init__("native K0 run rejected")


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
        ("physical_layer_substage", ctypes.c_int32),
        ("physical_layer_index", ctypes.c_int32),
        ("gdn_owner_substage", ctypes.c_int32),
        ("moe_aot_cuda_failure", ctypes.c_int32),
        ("execution_stage", ctypes.c_int32),
        ("execution_row", ctypes.c_int32),
        ("execution_layer", ctypes.c_int32),
        ("execution_layer_stage", ctypes.c_int32),
        ("gdn_graph_stage", ctypes.c_int32),
        ("startup_construction_stage", ctypes.c_int32),
        ("layer_boundary_hashes", ctypes.c_uint64 * 6),
        ("layer_boundary_elements", ctypes.c_uint32 * 6),
        ("layer_boundary_zero_counts", ctypes.c_uint32 * 6),
        ("layer_boundary_nonfinite_counts", ctypes.c_uint32 * 6),
        ("layer_boundary_reference_mismatch_counts", ctypes.c_uint32 * 6),
        ("layer_boundary_reference_first_mismatches", ctypes.c_uint32 * 6),
        ("layer_boundary_reference_compared", ctypes.c_uint8 * 6),
        ("layer_boundary_reference_exact", ctypes.c_uint8 * 6),
        ("oracle_domain_skip_counts", ctypes.c_uint32 * 5),
    )


def _secret(path: Path) -> bytes:
    value = path.read_bytes()
    if len(value) != 32:
        raise ValueError("bootstrap secret extent changed")
    return value


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
        elif isinstance(current, SlabError):
            kind, stage = "slab_contract", "accepted_loader_contract"
        elif isinstance(current, NativeTargetSlabFinalizeError):
            kind, stage = "native_finalize", current.stage
        elif isinstance(current, Layer3FactoryError):
            kind, stage = "layer_factory", "native_finalize"
        elif isinstance(current, NativeRunStatusError):
            kind, stage = "native_run", current.stage
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


def _emit_failure(counter: object, rank: int, phase: str,
                  terminal: dict[str, str], error: BaseException) -> None:
    try:
        attributes = {"rank": rank, "phase": phase, "outcome": "failure",
                      "failure.class": terminal["class"],
                      "failure.stage": terminal["stage"]}
        if isinstance(error, NativeRunStatusError):
            attributes["failure.physical_substage"] = error.physical_substage
            if error.physical_layer >= 0:
                attributes["failure.layer"] = error.physical_layer
            attributes["failure.gdn_owner_substage"] = error.gdn_owner_substage
            attributes["failure.moe_aot_cuda"] = error.moe_aot_cuda_failure
            attributes["failure.execution_stage"] = error.execution_stage
            if error.execution_row >= 0:
                attributes["failure.execution_row"] = error.execution_row
            if error.execution_layer >= 0:
                attributes["failure.execution_layer"] = error.execution_layer
            attributes["failure.execution_layer_stage"] = (
                error.execution_layer_stage)
            attributes["failure.gdn_graph_stage"] = error.gdn_graph_stage
            attributes["failure.startup_construction_stage"] = (
                error.startup_construction_stage)
        counter.add(1, attributes)
    except BaseException:
        pass


def _native_symbols(library: Path) -> CtypesNativeTargetSlabLeaseFactory:
    native = ctypes.CDLL(str(library))
    getattr(native, "qwen38_target_k0_oracle35_run")
    return CtypesNativeTargetSlabLeaseFactory(
        library, retain_symbol="qwen38_target_k0_retain_accepted_loader",
    )


def _descriptors(output: Path) -> None:
    """Authenticate a complete pre-generated inventory without source-slab I/O."""

    expected = {f"rank{rank}-layer{layer}.json"
                for rank in (0, 1) for layer in range(48)}
    observed = {item.name for item in output.iterdir()}
    if observed != expected:
        raise ValueError("descriptor inventory changed")
    allowlist = load_descriptor_allowlist(
        ENGINE / "src/decode/target_layer_descriptor_identities.json")
    for rank in (0, 1):
        for layer in range(48):
            source = output / f"rank{rank}-layer{layer}.json"
            if source.is_symlink() or not source.is_file():
                raise ValueError("descriptor inventory file changed")
            payload = source.read_bytes()
            descriptor = json.loads(payload)
            if payload != canonical_bytes(descriptor) + b"\n":
                raise ValueError("descriptor canonical encoding changed")
            if descriptor.get("rank") != rank or descriptor.get("layer") != layer:
                raise ValueError("descriptor filename identity changed")
            claimed = descriptor.get("descriptor_sha256")
            unsigned = dict(descriptor)
            unsigned.pop("descriptor_sha256", None)
            if claimed != hashlib.sha256(canonical_bytes(unsigned)).hexdigest():
                raise ValueError("descriptor digest changed")
            extents = descriptor.get("extents")
            if descriptor.get("native_binding_inventory_sha256") != hashlib.sha256(
                    canonical_bytes(extents)).hexdigest():
                raise ValueError("descriptor extent inventory changed")
            if not authenticate_descriptor_identity(
                    descriptor_identity(descriptor), allowlist):
                raise ValueError("descriptor identity is not allowlisted")


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
        error = NativeRunStatusError(
            status, result.physical_layer_substage,
            result.physical_layer_index, result.gdn_owner_substage,
            result.moe_aot_cuda_failure, result.execution_stage,
            result.execution_row, result.execution_layer,
            result.execution_layer_stage, result.gdn_graph_stage,
            result.startup_construction_stage,
        )
        error.layer_boundary_diagnostics = _layer_boundary_diagnostics(result)
        raise error
    return result


LAYER_BOUNDARY_NAMES = (
    "attention_output", "attention_reduction", "hc_combine_mix",
    "moe_output", "moe_reduction", "final_hc",
)


def _layer_boundary_diagnostics(result: _NativeResult) -> tuple[dict[str, int | str], ...]:
    return tuple({"boundary": name,
                  "hash": int(result.layer_boundary_hashes[index]),
                  "elements": int(result.layer_boundary_elements[index]),
                  "zero_count": int(result.layer_boundary_zero_counts[index]),
                  "nonfinite_count": int(
                      result.layer_boundary_nonfinite_counts[index]),
                  "reference_compared": bool(
                      result.layer_boundary_reference_compared[index]),
                  "reference_exact": bool(
                      result.layer_boundary_reference_exact[index]),
                  "reference_mismatch_count": int(
                      result.layer_boundary_reference_mismatch_counts[index]),
                  "reference_first_mismatch": int(
                      result.layer_boundary_reference_first_mismatches[index])}
                 for index, name in enumerate(LAYER_BOUNDARY_NAMES)
                 if result.layer_boundary_elements[index])


def _snapshot(result: _NativeResult) -> dict[str, object]:
    return {
        "execution_domain": "packed_decode_rows",
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
        "physical_layer_substage": PHYSICAL_LAYER_SUBSTAGES.get(
            result.physical_layer_substage, "unknown"),
        "gdn_owner_substage": GDN_OWNER_SUBSTAGES.get(
            result.gdn_owner_substage, "unknown"),
        "moe_aot_cuda_failure": MOE_AOT_CUDA_FAILURES.get(
            result.moe_aot_cuda_failure, "other"),
        "execution_stage": EXECUTION_STAGES.get(result.execution_stage,
                                                 "unknown"),
        "execution_row": (result.execution_row
                          if 0 <= result.execution_row < 35 else -1),
        "execution_layer": (result.execution_layer
                            if 0 <= result.execution_layer < 48 else -1),
        "execution_layer_stage": LAYER_EXECUTION_STAGES.get(
            result.execution_layer_stage, "unknown"),
        "gdn_graph_stage": GDN_GRAPH_STAGES.get(result.gdn_graph_stage,
                                                 "unknown"),
        "startup_construction_stage": STARTUP_CONSTRUCTION_STAGES.get(
            result.startup_construction_stage, "unknown"),
        "layer_boundary_diagnostics": _layer_boundary_diagnostics(result),
        "oracle_domain_skip_counts": list(result.oracle_domain_skip_counts),
    }


def worker(args: argparse.Namespace) -> int:
    from opentelemetry import metrics, trace
    import torch

    tracer = trace.get_tracer("rocket.qwen38.k0_oracle35")
    meter = metrics.get_meter("rocket.qwen38.k0_oracle35")
    terminal = meter.create_counter("rocket.qwen38.k0_oracle35", unit="{run}")
    phase = "session"
    started = time.perf_counter_ns()
    try:
        sessions = tuple(_secret(path) for path in (
            args.layer_session_file, args.embedding_session_file,
            args.nccl_session_file, args.nccl_authentication_key_file,
        ))
        phase = "native_symbols"
        finalizer = _native_symbols(args.library)
        phase = "descriptor_inventory"
        _descriptors(args.descriptor_directory)
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
        cause_chain = _typed_cause_chain(error, phase)
        failure = cause_chain[-1]
        _emit_failure(terminal, args.rank, phase, failure, error)
        physical = {}
        if isinstance(error, NativeRunStatusError):
            physical["physical_layer_substage"] = error.physical_substage
            if error.physical_layer >= 0:
                physical["physical_layer_index"] = error.physical_layer
            physical["gdn_owner_substage"] = error.gdn_owner_substage
            physical["moe_aot_cuda_failure"] = error.moe_aot_cuda_failure
            physical["execution_stage"] = error.execution_stage
            if error.execution_row >= 0:
                physical["execution_row"] = error.execution_row
            if error.execution_layer >= 0:
                physical["execution_layer"] = error.execution_layer
            physical["execution_layer_stage"] = error.execution_layer_stage
            physical["gdn_graph_stage"] = error.gdn_graph_stage
            physical["startup_construction_stage"] = (
                error.startup_construction_stage)
            physical["layer_boundary_diagnostics"] = (
                error.layer_boundary_diagnostics)
        print(json.dumps({"schema": SCHEMA, "valid": False, "complete": False,
                          "rank": args.rank, "phase": phase,
                          "failure_class": failure["class"],
                          "failure_stage": failure["stage"],
                          "cause_chain": cause_chain,
                          "elapsed_ns": time.perf_counter_ns() - started,
                          **physical},
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

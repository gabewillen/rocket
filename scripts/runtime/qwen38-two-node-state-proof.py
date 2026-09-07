#!/usr/bin/env python3
"""Run Qwen full-extent CUDA capture and a physical TP2 owner fixture.

The coordinator starts one JSON-line worker per physical node.  Worker control
messages are synchronous, so both physical decoder owners remain held while
the existing ``TwoRankRestoreCoordinator`` orders the two concrete restores.
This fixture exercises state ownership and byte movement.  It does not execute
model kernels or test token continuation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

from qwen38_slab.device_decode import DevicePhase, DevicePublication
from qwen38_slab.runtime_state import CudaStateBinding, DeviceState, RuntimeBoundary
from qwen38_slab.state_owner import DecoderStateOwner, TwoRankRestoreCoordinator
from qwen38_slab.state_txn import AuthenticatedState, FamilyPayload, STATE_FAMILIES
from qwen38_slab.torch_cuda import TorchCudaRuntime

DEFAULT_IMAGE = "vllm/vllm-openai:qwen38-flash-next"
CONTAINER_REPO = "/rocket"
PLAN = "scripts/memory/qwen38-state-capacity-plan.json"
SCRIPT = "scripts/runtime/qwen38-two-node-state-proof.py"
PLAN_FAMILIES = (
    "target.full_attention.kv",
    "target.qsa.raw",
    "target.qsa.compressed",
    "target.linear.conv",
    "target.linear.recurrent",
    "target.ple.conv",
    "mtp.full_attention.kv",
    "mtp.qsa.raw",
    "mtp.qsa.compressed",
)


class _Span:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): del key, value
    def record_exception(self, exception): del exception


class _Tracer:
    def start_as_current_span(self, name):
        del name
        return _Span()


class _Decoder:
    def __init__(self):
        self.phase = DevicePhase.IDLE
        self.publication = None

    def upload_and_launch(self, generation):
        self.publication = DevicePublication(generation, 1, generation % 2)
        return self.publication


def _boundary() -> RuntimeBoundary:
    digest = hashlib.sha256(b"qwen38-c16-262144-two-node-fixture-v1").hexdigest()
    return RuntimeBoundary(262144, digest, 1)


def _extents(plan_path: Path) -> dict[str, int]:
    plan = json.loads(plan_path.read_text())
    families = plan["families"]
    if tuple(family["id"] for family in families) != PLAN_FAMILIES:
        raise RuntimeError("capacity plan family order changed")
    measured = {
        state_family: planned["logical_bytes_per_stream"] * 16
        for state_family, planned in zip(STATE_FAMILIES, families, strict=True)
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in measured.values()
    ):
        raise RuntimeError("capacity plan does not contain the canonical extents")
    return measured


def _fixture(extents: dict[str, int]) -> AuthenticatedState:
    payloads = {
        family: FamilyPayload(bytes(len(family) + 1)) for family in extents
    }
    boundary = _boundary()
    return AuthenticatedState._from_verified(
        token_count=boundary.token_count,
        token_hash=boundary.token_hash,
        rank_payloads={0: payloads, 1: payloads},
    )


def _runtime(rank: int):
    import torch

    tracer = _Tracer()
    decoder = _Decoder()
    owner = DecoderStateOwner(rank=rank, decoder=decoder, tracer=tracer)
    boundary = _boundary()
    owner.accept_boundary(owner.upload_and_launch(1), boundary)
    runtime = TorchCudaRuntime(
        owner=owner,
        compute_streams=(torch.cuda.Stream(device="cuda:0"),),
        torch_api=torch,
        device="cuda:0",
    )
    return torch, owner, CudaStateBinding(rank, runtime, tracer), boundary


def full_capture(rank: int, plan_path: Path) -> dict[str, object]:
    torch, owner, binding, boundary = _runtime(rank)
    extents = _extents(plan_path)
    start = time.perf_counter()
    tensors = {}
    for index, (family, size) in enumerate(extents.items()):
        tensor = torch.empty(size, dtype=torch.uint8, device="cuda:0")
        tensor.fill_((rank * len(STATE_FAMILIES) + index + 1) % 251)
        tensors[family] = tensor
    torch.cuda.synchronize()
    allocated_seconds = time.perf_counter() - start
    allocated_bytes = torch.cuda.memory_allocated()
    sources = {
        family: DeviceState(family, tensor, extents[family], tensor.numel())
        for family, tensor in tensors.items()
    }
    capture_start = time.perf_counter()
    accepted, captured = binding.capture(boundary, sources)
    capture_seconds = time.perf_counter() - capture_start
    for index, family in enumerate(STATE_FAMILIES):
        expected = (rank * len(STATE_FAMILIES) + index + 1) % 251
        payload = captured[family].accepted
        if len(payload) != extents[family] or payload[0] != expected or payload[-1] != expected:
            raise RuntimeError(f"captured extent or sentinel changed for {family}")
    if accepted.token_count != boundary.token_count or owner.faulted:
        raise RuntimeError("full capture did not resume the accepted owner")
    return {
        "node": socket.gethostname(),
        "device": torch.cuda.get_device_name(0),
        "rank": rank,
        "families": len(captured),
        "logical_bytes": sum(extents.values()),
        "cuda_memory_allocated_bytes": allocated_bytes,
        "allocate_seconds": round(allocated_seconds, 6),
        "capture_seconds": round(capture_seconds, 6),
    }


def worker(rank: int, plan_path: Path) -> None:
    torch, owner, binding, boundary = _runtime(rank)
    extents = _extents(plan_path)
    authenticated = _fixture(extents)
    print(json.dumps({
        "ok": True,
        "event": "ready",
        "rank": rank,
        "node": socket.gethostname(),
        "device": torch.cuda.get_device_name(0),
    }), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        command = request.get("command")
        try:
            if command == "hold":
                owner.hold_launch_gate(boundary)
                result = {"phase": owner.phase.value}
            elif command == "restore":
                if request.get("inject_failure") is True:
                    raise RuntimeError("injected physical-rank restore failure")
                start = time.perf_counter()
                binding.restore(authenticated, generation_epoch=1)
                torch.cuda.synchronize()
                result = {
                    "seconds": round(time.perf_counter() - start, 6),
                    "bytes": sum(len(value.accepted) for value in authenticated.rank_payload(rank).values()),
                    "phase": owner.phase.value,
                }
            elif command == "release":
                owner.release_launch_gate(boundary)
                result = {"phase": owner.phase.value}
            elif command == "fault":
                owner.fault_closed(boundary)
                result = {"phase": owner.phase.value}
            elif command == "status":
                launch_rejected = False
                if request.get("probe_launch") is True:
                    try:
                        owner.upload_and_launch(2)
                    except Exception:
                        launch_rejected = True
                result = {"phase": owner.phase.value, "launch_rejected": launch_rejected}
            elif command == "stop":
                print(json.dumps({"ok": True, "result": {"phase": owner.phase.value}}), flush=True)
                return
            else:
                raise RuntimeError("unknown worker command")
        except Exception as exc:
            print(json.dumps({"ok": False, "error": type(exc).__name__}), flush=True)
        else:
            print(json.dumps({"ok": True, "result": result}), flush=True)


class _Process:
    def __init__(self, command: list[str]):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        ready = self._read()
        if ready.get("event") != "ready":
            raise RuntimeError("physical rank worker did not become ready")
        self.ready = ready

    def request(self, command: str, **values):
        if self.process.stdin is None:
            raise RuntimeError("physical rank worker input is closed")
        self.process.stdin.write(json.dumps({"command": command, **values}) + "\n")
        self.process.stdin.flush()
        response = self._read()
        if response.get("ok") is not True:
            raise RuntimeError(f"physical worker rejected {command}: {response.get('error')}")
        return response["result"]

    def _read(self):
        if self.process.stdout is None:
            raise RuntimeError("physical rank worker output is closed")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"physical rank worker exited with {self.process.poll()}")
        return json.loads(line)

    def close(self):
        if self.process.poll() is None:
            try:
                self.request("stop")
            except Exception:
                self.process.terminate()
            self.process.wait(timeout=30)


class _OwnerProxy:
    def __init__(self, rank: int, process: _Process, boundary: RuntimeBoundary):
        self.rank = rank
        self.process = process
        self.accepted_boundary = boundary

    def hold_launch_gate(self, boundary):
        if boundary != self.accepted_boundary: raise RuntimeError("proxy boundary mismatch")
        self.process.request("hold")

    def release_launch_gate(self, boundary):
        if boundary != self.accepted_boundary: raise RuntimeError("proxy boundary mismatch")
        self.process.request("release")

    def fault_closed(self, boundary):
        if boundary != self.accepted_boundary: raise RuntimeError("proxy boundary mismatch")
        self.process.request("fault")


class _BindingProxy:
    def __init__(self, owner: _OwnerProxy, results: list[dict], fault_rank: int | None):
        self.rank = owner.rank
        self.state_owner = owner
        self.owner = owner
        self.results = results
        self.fault_rank = fault_rank

    def restore(self, authenticated, generation_epoch):
        if authenticated.boundary.token_hash != self.owner.accepted_boundary.token_hash or generation_epoch != 1:
            raise RuntimeError("proxy authenticated boundary mismatch")
        result = self.owner.process.request(
            "restore", inject_failure=self.rank == self.fault_rank
        )
        self.results.append({"rank": self.rank, **result})


def _worker_command(repo: Path, image: str, rank: int, remote: str | None) -> list[str]:
    docker = [
        "docker", "run", "--rm", "--gpus", "all", "--entrypoint", "python3",
        "-i", "-v", f"{repo}:{CONTAINER_REPO}:ro",
        "-e", f"PYTHONPATH={CONTAINER_REPO}/engines/qwen38-flash-next-nvfp4-2b/src",
        image, f"{CONTAINER_REPO}/{SCRIPT}", "--worker", "--rank", str(rank),
        "--plan", f"{CONTAINER_REPO}/{PLAN}",
    ]
    if remote is None:
        return docker
    return ["ssh", "-o", "BatchMode=yes", remote, shlex.join(docker)]


def coordinate(repo: Path, image: str, remote: str, fault_rank: int | None) -> dict[str, object]:
    boundary = _boundary()
    processes = (
        _Process(_worker_command(repo, image, 0, None)),
        _Process(_worker_command(repo, image, 1, remote)),
    )
    try:
        owners = (
            _OwnerProxy(0, processes[0], boundary),
            _OwnerProxy(1, processes[1], boundary),
        )
        results: list[dict] = []
        bindings = (
            _BindingProxy(owners[0], results, fault_rank),
            _BindingProxy(owners[1], results, fault_rank),
        )
        coordinator = TwoRankRestoreCoordinator(owners, bindings, _Tracer())
        extents = _extents(repo / PLAN)
        authenticated = _fixture(extents)
        failure = None
        start = time.perf_counter()
        try:
            coordinator.restore(authenticated, generation_epoch=1)
        except Exception as exc:
            failure = type(exc).__name__
        elapsed = time.perf_counter() - start
        statuses = tuple(
            process.request("status", probe_launch=failure is not None)
            for process in processes
        )
        restore_seconds = sum(result["seconds"] for result in results)
        control_seconds = max(0.0, elapsed - restore_seconds)
        floor_step_seconds = 16 / 320
        if fault_rank is None:
            if failure is not None or any(status["phase"] != "open" for status in statuses):
                raise RuntimeError("successful physical coordinator proof did not reopen both ranks")
        elif (
            failure != "DecoderStateOwnerError"
            or any(status["phase"] != "faulted" for status in statuses)
            or any(status["launch_rejected"] is not True for status in statuses)
        ):
            raise RuntimeError("physical coordinator failure did not fault both ranks closed")
        return {
            "coordinator_seconds": round(elapsed, 6),
            "control_seconds": round(control_seconds, 6),
            "control_fraction_of_320_tps_floor_step": round(
                control_seconds / floor_step_seconds, 6
            ),
            "failure": failure,
            "fault_rank": fault_rank,
            "rank_results": results,
            "rank_status": statuses,
            "workers": tuple(process.ready for process in processes),
        }
    finally:
        for process in processes:
            process.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--full-capture", action="store_true")
    parser.add_argument("--rank", type=int, choices=(0, 1))
    parser.add_argument("--plan", type=Path, default=Path(PLAN))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--remote")
    parser.add_argument("--fault-rank", type=int, choices=(0, 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.rank is None: raise SystemExit("--worker requires --rank")
        worker(args.rank, args.plan)
    elif args.full_capture:
        if args.rank is None: raise SystemExit("--full-capture requires --rank")
        print(json.dumps(full_capture(args.rank, args.plan), sort_keys=True))
    else:
        if not args.remote: raise SystemExit("coordinator requires --remote")
        print(json.dumps(coordinate(args.repo.resolve(), args.image, args.remote, args.fault_rank), sort_keys=True))


if __name__ == "__main__":
    main()

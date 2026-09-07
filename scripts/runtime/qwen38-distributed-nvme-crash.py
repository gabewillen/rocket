#!/usr/bin/env python3
"""Run the owner-local Qwen TP2 NVMe crash matrix on two physical nodes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from qwen38_slab.distributed_state_txn import (
    AuthenticationReceipt,
    CommitReceipt,
    DistributedStateCoordinator,
    GeneratedFamilySource,
    LocalRankEndpoint,
    PrepareReceipt,
    RankStateStore,
    RestoreInspection,
)
from qwen38_slab.decode import Depth
from qwen38_slab.device_decode import DevicePhase, DevicePublication
from qwen38_slab.local_cuda_restore import LocalCudaStateBinding
from qwen38_slab.mtp_policy import (
    AdaptiveMtpPolicy,
    ConcurrencyCeiling,
    PolicyConfig,
)
from qwen38_slab.runtime_state import RuntimeBoundary
from qwen38_slab.state_txn import (
    STATE_FAMILIES,
    AcceptedBoundary,
    StateIdentity,
    StateTransactionError,
    Transition,
)
from qwen38_slab.state_owner import DecoderStateOwner, OwnerPhase
from qwen38_slab.torch_cuda import TorchCudaRuntime

IMAGE = "vllm/vllm-openai:qwen38-flash-next"
CONTAINER_REPO = "/rocket"
SCRIPT = "scripts/runtime/qwen38-distributed-nvme-crash.py"
PLAN = "scripts/memory/qwen38-state-capacity-plan.json"
RESERVE_BYTES = 8 * 1024**3


class _Span:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): del key, value
    def record_exception(self, exception): del exception


class _Tracer:
    def start_as_current_span(self, name): del name; return _Span()


def _boundary() -> AcceptedBoundary:
    return AcceptedBoundary(
        262144,
        hashlib.sha256(b"qwen38-physical-nvme-crash-v1").hexdigest(),
        True,
    )


def _identity() -> StateIdentity:
    return StateIdentity(
        "fc694-physical-nvme",
        hashlib.sha256(b"qwen38-planner-c16-262144-v1").hexdigest(),
    )


def _policy_state() -> bytes:
    policy = _policy()
    return policy.dump_state(policy.initial_state())


def _policy() -> AdaptiveMtpPolicy:
    return AdaptiveMtpPolicy(
        _Tracer(),
        PolicyConfig((
            ConcurrencyCeiling(1, Depth.K7),
            ConcurrencyCeiling(16, Depth.K1),
        )),
    )


def _sources(plan_path: Path, rank: int):
    plan = json.loads(plan_path.read_text())
    families = plan["families"]
    if len(families) != len(STATE_FAMILIES):
        raise RuntimeError("capacity plan family count changed")
    sources = {
        family: GeneratedFamilySource(
            planned["logical_bytes_per_stream"] * 16,
            rank * len(STATE_FAMILIES) + index + 1,
            planned["nvme_padded_bytes_per_stream"] * 16,
        )
        for index, (family, planned) in enumerate(
            zip(STATE_FAMILIES, families, strict=True)
        )
    }
    padded = sum(
        source.stored_bytes
        for source in sources.values()
    )
    expected = plan["totals_per_rank"]["nvme_padded_bytes_c16"]
    if padded != expected:
        raise RuntimeError("capacity plan NVMe extent changed")
    return sources, expected


def _cuda_layout(plan_path: Path):
    plan = json.loads(plan_path.read_text())
    logical = {
        family: item["logical_bytes_per_stream"] * 16
        for family, item in zip(STATE_FAMILIES, plan["families"], strict=True)
    }
    allocated = {
        family: item["cuda_allocated_bytes_c16"]
        for family, item in zip(STATE_FAMILIES, plan["families"], strict=True)
    }
    identifiers = {
        item["id"]: family
        for family, item in zip(STATE_FAMILIES, plan["families"], strict=True)
    }
    owners = {
        family: identifiers.get(item["shares_cuda_allocation_with"], family)
        for family, item in zip(STATE_FAMILIES, plan["families"], strict=True)
    }
    counted = sum(
        allocated[family] for family in STATE_FAMILIES if owners[family] == family
    )
    if counted != plan["totals_per_rank"]["cuda_allocated_bytes_c16"]:
        raise RuntimeError("capacity plan CUDA allocation changed")
    return logical, allocated, owners, counted


class _Decoder:
    def __init__(self):
        self.phase = DevicePhase.IDLE
        self.publication = None

    def upload_and_launch(self, generation):
        self.publication = DevicePublication(generation, 16, generation % 2)
        return self.publication


def _encode(value):
    result = asdict(value)
    if "boundary" in result:
        result["boundary"] = asdict(value.boundary)
    return result


def _prepare(value):
    value = dict(value); value["boundary"] = AcceptedBoundary(**value["boundary"])
    return PrepareReceipt(**value)


def _inspection(value):
    value = dict(value); value["boundary"] = AcceptedBoundary(**value["boundary"])
    return RestoreInspection(**value)


def _authentication(value):
    value = dict(value); value["boundary"] = AcceptedBoundary(**value["boundary"])
    return AuthenticationReceipt(**value)


def _footprint(store: Path) -> int:
    if not store.exists(): return 0
    return sum(path.stat().st_size for path in store.rglob("*.state"))


def worker(rank: int, store_path: Path, plan_path: Path, cuda_restore: bool) -> None:
    store_path.mkdir(parents=True, exist_ok=True)
    sources, record_bytes = _sources(plan_path, rank)
    available = shutil.disk_usage(store_path).free
    reclaimable = _footprint(store_path)
    if available + reclaimable < record_bytes + RESERVE_BYTES:
        raise RuntimeError("owner-local NVMe preflight failed")
    publications = []
    owner = None
    cuda_bytes = 0
    publisher = publications.append
    if cuda_restore:
        import torch

        logical, allocated, owners, cuda_bytes = _cuda_layout(plan_path)
        owner = DecoderStateOwner(rank=rank, decoder=_Decoder(), tracer=_Tracer())
        runtime_boundary = RuntimeBoundary(
            _boundary().token_count, _boundary().token_hash, 1
        )
        owner.accept_boundary(owner.upload_and_launch(1), runtime_boundary)
        runtime = TorchCudaRuntime(
            owner=owner,
            compute_streams=(torch.cuda.Stream(device="cuda:0"),),
            torch_api=torch,
            device="cuda:0",
            allocation_bytes=allocated,
            allocation_owners=owners,
        )
        binding = LocalCudaStateBinding(
            rank, runtime, _policy(), logical, _Tracer()
        )

        def publish_cuda(state):
            binding.restore_local(state, 1)
            publications.append(state)

        publisher = publish_cuda
    endpoint = LocalRankEndpoint(
        RankStateStore(rank, store_path, _identity(), _Tracer(), _policy()),
        sources,
        publisher,
    )
    print(json.dumps({
        "ok": True, "event": "ready", "rank": rank,
        "free_bytes": available, "reclaimable_bytes": reclaimable,
        "record_bytes": record_bytes,
    }), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        command = request.get("command")
        try:
            start = time.perf_counter()
            if command == "prepare":
                encoded_policy = request.get("policy_state")
                result = _encode(endpoint.prepare(
                    request["session_id"], request["transaction_id"],
                    AcceptedBoundary(**request["boundary"]),
                    None if encoded_policy is None else base64.b64decode(
                        encoded_policy, validate=True
                    ),
                ))
            elif command == "commit":
                result = _encode(endpoint.commit(tuple(
                    _prepare(value) for value in request["receipts"]
                )))
            elif command == "index":
                endpoint.index(tuple(CommitReceipt(**value) for value in request["receipts"]))
                result = {}
            elif command == "inspect":
                result = _encode(endpoint.inspect_restore(request["session_id"]))
            elif command == "authenticate":
                result = _encode(endpoint.authenticate(_inspection(request["inspection"])))
            elif command == "publish":
                endpoint.publish(_authentication(request["receipt"]))
                result = {"publications": len(publications)}
            elif command == "hold":
                if owner is None: raise RuntimeError("CUDA owner is unavailable")
                owner.hold_launch_gate(RuntimeBoundary(
                    _boundary().token_count, _boundary().token_hash, 1
                ))
                result = {"phase": owner.phase.value}
            elif command == "release":
                if owner is None: raise RuntimeError("CUDA owner is unavailable")
                owner.release_launch_gate(RuntimeBoundary(
                    _boundary().token_count, _boundary().token_hash, 1
                ))
                result = {"phase": owner.phase.value}
            elif command == "fault":
                if owner is None: raise RuntimeError("CUDA owner is unavailable")
                owner.fault_closed(RuntimeBoundary(
                    _boundary().token_count, _boundary().token_hash, 1
                ))
                result = {"phase": owner.phase.value}
            elif command == "status":
                result = {
                    "publications": len(publications),
                    "payload_bytes": _footprint(store_path),
                    "free_bytes": shutil.disk_usage(store_path).free,
                }
                if owner is not None:
                    import torch

                    samples = []
                    if owner.active_state is not None:
                        for index, family in enumerate(STATE_FAMILIES):
                            tensor = owner.active_state[family]
                            expected = rank * len(STATE_FAMILIES) + index + 1
                            offsets = (0, tensor.numel() // 2, tensor.numel() - 1)
                            samples.append(all(
                                int(tensor[offset].item()) == expected
                                for offset in offsets
                            ))
                    result.update({
                        "owner_phase": owner.phase.value,
                        "cuda_bytes": cuda_bytes,
                        "cuda_storage_bytes": 0 if owner.active_state is None else sum({
                            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                            for tensor in owner.active_state.values()
                        }.values()),
                        "policy_digest": None if owner.active_policy_state is None else hashlib.sha256(
                            owner.active_policy_state
                        ).hexdigest(),
                        "sentinels": samples,
                        "device": torch.cuda.get_device_name(0),
                    })
            elif command == "cleanup":
                for child in (store_path / "transactions", store_path / "sessions"):
                    if child.exists(): shutil.rmtree(child)
                publications.clear()
                endpoint.pending = None
                result = {"payload_bytes": _footprint(store_path)}
            elif command == "stop":
                print(json.dumps({"ok": True, "result": {}}), flush=True)
                return
            else:
                raise RuntimeError("unknown worker command")
            result["seconds"] = round(time.perf_counter() - start, 6)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": type(exc).__name__}), flush=True)
        else:
            print(json.dumps({"ok": True, "result": result}), flush=True)


class _Process:
    def __init__(self, command):
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        self.ready = self._read()
        if self.ready.get("event") != "ready": raise RuntimeError("worker not ready")

    def request(self, command, **values):
        self.process.stdin.write(json.dumps({"command": command, **values}) + "\n")
        self.process.stdin.flush()
        response = self._read()
        if response.get("ok") is not True:
            raise StateTransactionError(
                f"rank worker rejected {command}: {response.get('error')}"
            )
        return response["result"]

    def _read(self):
        line = self.process.stdout.readline()
        if not line: raise RuntimeError(f"rank worker exited: {self.process.poll()}")
        return json.loads(line)

    def close(self):
        if self.process.poll() is None:
            try: self.request("stop")
            except Exception: self.process.terminate()
            self.process.wait(timeout=30)


class _EndpointProxy:
    def __init__(self, rank, process):
        self.rank = rank
        self.process = process
        self.timings = []

    def _timed(self, command, **values):
        result = self.process.request(command, **values)
        self.timings.append((command, result.pop("seconds")))
        return result

    def prepare(self, session_id, transaction_id, boundary, policy_state):
        return _prepare(self._timed(
            "prepare", session_id=session_id, transaction_id=transaction_id,
            boundary=asdict(boundary),
            policy_state=None if policy_state is None else base64.b64encode(
                policy_state
            ).decode("ascii"),
        ))
    def commit(self, receipts):
        return CommitReceipt(**self._timed(
            "commit", receipts=[_encode(value) for value in receipts]
        ))
    def index(self, receipts):
        self._timed("index", receipts=[_encode(value) for value in receipts])
    def inspect_restore(self, session_id):
        return _inspection(self._timed("inspect", session_id=session_id))
    def authenticate(self, inspection):
        return _authentication(self._timed("authenticate", inspection=_encode(inspection)))
    def publish(self, receipt):
        self._timed("publish", receipt=_encode(receipt))


def _command(
    repo: Path, rank: int, remote: str | None, container_name: str,
    cuda_restore: bool = False,
):
    owner_path = f"/var/lib/rocket/qwen38-state/rank{rank}"
    command = [
        "docker", "run", "--rm", "--name", container_name,
        "--entrypoint", "python3", "-i",
        "-v", f"{repo}:{CONTAINER_REPO}:ro", "-v", f"{owner_path}:/state",
        "-e", f"PYTHONPATH={CONTAINER_REPO}/engines/qwen38-flash-next-nvfp4-2b/src",
        IMAGE, f"{CONTAINER_REPO}/{SCRIPT}", "--worker", "--rank", str(rank),
        "--store", "/state", "--plan", f"{CONTAINER_REPO}/{PLAN}",
    ]
    if cuda_restore:
        command[2:2] = ["--gpus", "all"]
        command.append("--cuda-restore")
    return command if remote is None else ["ssh", "-o", "BatchMode=yes", remote, shlex.join(command)]


def _start_processes(repo: Path, remote: str, prefix: str):
    return (
        _Process(_command(repo, 0, None, f"{prefix}-rank0")),
        _Process(_command(repo, 1, remote, f"{prefix}-rank1")),
    )


def _start_cuda_processes(repo: Path, remote: str, prefix: str):
    return (
        _Process(_command(repo, 0, None, f"{prefix}-rank0", True)),
        _Process(_command(repo, 1, remote, f"{prefix}-rank1", True)),
    )


def crash_case(repo: Path, remote: str, transition: Transition, prefix: str):
    processes = _start_processes(repo, remote, prefix)
    endpoints = (
        _EndpointProxy(0, processes[0]), _EndpointProxy(1, processes[1])
    )
    coordinator = DistributedStateCoordinator(endpoints, _Tracer())
    policy_state = _policy_state()

    def kill_at_boundary(observed):
        if observed is transition:
            print(json.dumps({
                "event": "sigkill",
                "transition": transition.value,
                "policy_digest": hashlib.sha256(policy_state).hexdigest(),
                "rank_timings": tuple(endpoint.timings for endpoint in endpoints),
                "workers": tuple(process.ready for process in processes),
            }, sort_keys=True), flush=True)
            os.kill(os.getpid(), signal.SIGKILL)

    coordinator.commit(
        "physical", f"txn-{transition.value}", _boundary(),
        policy_state=policy_state, inject_fault=kill_at_boundary,
    )
    raise RuntimeError("coordinator passed configured SIGKILL transition")


def _container_exists(remote: str | None, name: str) -> bool:
    command = ["docker", "inspect", name]
    if remote is not None:
        command = ["ssh", "-o", "BatchMode=yes", remote, shlex.join(command)]
    return subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def _await_worker_exit(remote: str, prefix: str) -> None:
    owners = ((None, f"{prefix}-rank0"), (remote, f"{prefix}-rank1"))
    deadline = time.monotonic() + 30
    while any(_container_exists(owner, name) for owner, name in owners):
        if time.monotonic() >= deadline:
            raise RuntimeError("SIGKILL worker descendants did not exit on stdin EOF")
        time.sleep(0.25)


def _run_killed_case(repo: Path, remote: str, transition: Transition, prefix: str):
    command = [
        sys.executable, str(repo / SCRIPT), "--crash-case", transition.value,
        "--repo", str(repo), "--remote", remote, "--container-prefix", prefix,
    ]
    start = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True)
    seconds = time.perf_counter() - start
    if result.returncode != -signal.SIGKILL:
        raise RuntimeError(
            f"coordinator crash returned {result.returncode}: {result.stderr[-400:]}"
        )
    lines = tuple(line for line in result.stdout.splitlines() if line)
    if len(lines) != 1:
        raise RuntimeError("coordinator SIGKILL evidence is missing or ambiguous")
    evidence = json.loads(lines[0])
    if evidence.get("event") != "sigkill" or evidence.get("transition") != transition.value:
        raise RuntimeError("coordinator died outside the configured transition")
    _await_worker_exit(remote, prefix)
    evidence["coordinator_seconds"] = round(seconds, 6)
    evidence["returncode"] = result.returncode
    return evidence


def matrix(repo: Path, remote: str):
    cases = []
    for transition in Transition:
        prefix = f"qwen38-state-{transition.value.replace('_', '-')}"
        cleanup_processes = _start_processes(repo, remote, f"{prefix}-preflight")
        try:
            cleanup = tuple(process.request("cleanup") for process in cleanup_processes)
            if any(value["payload_bytes"] for value in cleanup):
                raise RuntimeError("preflight cleanup left payload bytes")
        finally:
            for process in cleanup_processes: process.close()
        evidence = _run_killed_case(repo, remote, transition, prefix)
        processes = _start_processes(repo, remote, prefix)
        try:
            endpoints = (
                _EndpointProxy(0, processes[0]), _EndpointProxy(1, processes[1])
            )
            coordinator = DistributedStateCoordinator(endpoints, _Tracer())
            before = tuple(process.request("status") for process in processes)
            restored = False
            restore_seconds = None
            try:
                restore_start = time.perf_counter()
                coordinator.restore("physical")
                restore_seconds = time.perf_counter() - restore_start
                restored = True
            except StateTransactionError:
                pass
            after = tuple(process.request("status") for process in processes)
            expected = transition is Transition.AFTER_INDEX_RANK1
            if restored is not expected:
                raise RuntimeError(f"restore eligibility changed at {transition.value}")
            if expected:
                if tuple(value["publications"] for value in after) != (1, 1):
                    raise RuntimeError("final restore did not publish both ranks")
            elif tuple(value["publications"] for value in after) != (0, 0):
                raise RuntimeError("pre-final crash exposed a publication")
            cases.append({
                "transition": transition.value,
                "crash": evidence,
                "restore_seconds": None if restore_seconds is None else round(restore_seconds, 6),
                "restored": restored,
                "payload_bytes": tuple(value["payload_bytes"] for value in before),
                "recovery_timings": tuple(endpoint.timings for endpoint in endpoints),
            })
            cleanup = tuple(process.request("cleanup") for process in processes)
            if any(value["payload_bytes"] for value in cleanup):
                raise RuntimeError("physical crash matrix cleanup left payload bytes")
            cases[-1]["cleanup"] = cleanup
        finally:
            for process in processes: process.close()
    return {"signal": "SIGKILL", "cases": cases}


def cuda_restore(repo: Path, remote: str):
    prefix = "qwen38-state-cuda-restore"
    cleanup_processes = _start_processes(repo, remote, f"{prefix}-preflight")
    try:
        for process in cleanup_processes: process.request("cleanup")
    finally:
        for process in cleanup_processes: process.close()
    processes = _start_cuda_processes(repo, remote, prefix)
    try:
        endpoints = (
            _EndpointProxy(0, processes[0]), _EndpointProxy(1, processes[1])
        )
        coordinator = DistributedStateCoordinator(endpoints, _Tracer())
        policy = _policy_state()
        coordinator.commit(
            "physical", "txn-cuda-restore", _boundary(), policy_state=policy
        )
        for process in processes: process.request("hold")
        start = time.perf_counter()
        try:
            coordinator.restore("physical")
            for process in processes: process.request("release")
        except BaseException:
            for process in processes:
                try: process.request("fault")
                except BaseException: pass
            raise
        seconds = time.perf_counter() - start
        status = tuple(process.request("status") for process in processes)
        if any(
            value["owner_phase"] != OwnerPhase.OPEN.value
            or value["cuda_storage_bytes"] != value["cuda_bytes"]
            or value["policy_digest"] != hashlib.sha256(policy).hexdigest()
            or value["sentinels"] != [True] * len(STATE_FAMILIES)
            for value in status
        ):
            raise RuntimeError("two-rank CUDA restore publication changed")
        cleanup = tuple(process.request("cleanup") for process in processes)
        return {
            "restore_seconds": round(seconds, 6),
            "policy_bytes": len(policy),
            "policy_digest": hashlib.sha256(policy).hexdigest(),
            "workers": tuple(process.ready for process in processes),
            "status": status,
            "cleanup": cleanup,
        }
    finally:
        for process in processes: process.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cuda-restore", action="store_true")
    parser.add_argument("--cuda-final", action="store_true")
    parser.add_argument("--crash-case", choices=tuple(value.value for value in Transition))
    parser.add_argument("--container-prefix")
    parser.add_argument("--rank", type=int, choices=(0, 1))
    parser.add_argument("--store", type=Path)
    parser.add_argument("--plan", type=Path, default=Path(PLAN))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--remote")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        if args.rank is None or args.store is None: raise SystemExit("worker requires rank/store")
        worker(args.rank, args.store, args.plan, args.cuda_restore)
    elif args.crash_case:
        if not args.remote or not args.container_prefix:
            raise SystemExit("crash case requires remote/container-prefix")
        crash_case(
            args.repo.resolve(), args.remote, Transition(args.crash_case),
            args.container_prefix,
        )
    elif args.cuda_final:
        if not args.remote: raise SystemExit("CUDA restore requires remote")
        print(json.dumps(cuda_restore(args.repo.resolve(), args.remote), sort_keys=True))
    else:
        if not args.remote: raise SystemExit("matrix requires remote")
        print(json.dumps(matrix(args.repo.resolve(), args.remote), sort_keys=True))


if __name__ == "__main__": main()

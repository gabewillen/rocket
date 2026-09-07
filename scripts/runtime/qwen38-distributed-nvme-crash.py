#!/usr/bin/env python3
"""Run the owner-local Qwen TP2 NVMe crash matrix on two physical nodes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import shutil
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
from qwen38_slab.mtp_policy import (
    AdaptiveMtpPolicy,
    ConcurrencyCeiling,
    PolicyConfig,
)
from qwen38_slab.state_txn import (
    STATE_FAMILIES,
    AcceptedBoundary,
    StateIdentity,
    StateTransactionError,
    Transition,
)

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
    policy = AdaptiveMtpPolicy(
        _Tracer(),
        PolicyConfig((
            ConcurrencyCeiling(1, Depth.K7),
            ConcurrencyCeiling(16, Depth.K1),
        )),
    )
    return policy.dump_state(policy.initial_state())


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


def worker(rank: int, store_path: Path, plan_path: Path) -> None:
    store_path.mkdir(parents=True, exist_ok=True)
    sources, record_bytes = _sources(plan_path, rank)
    available = shutil.disk_usage(store_path).free
    if available < record_bytes + RESERVE_BYTES:
        raise RuntimeError("owner-local NVMe preflight failed")
    publications = []
    endpoint = LocalRankEndpoint(
        RankStateStore(rank, store_path, _identity(), _Tracer()),
        sources,
        publications.append,
    )
    print(json.dumps({
        "ok": True, "event": "ready", "rank": rank,
        "free_bytes": available, "record_bytes": record_bytes,
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
            elif command == "status":
                result = {
                    "publications": len(publications),
                    "payload_bytes": _footprint(store_path),
                    "free_bytes": shutil.disk_usage(store_path).free,
                }
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


def _command(repo: Path, rank: int, remote: str | None):
    owner_path = f"/var/lib/rocket/qwen38-state/rank{rank}"
    command = [
        "docker", "run", "--rm", "--entrypoint", "python3", "-i",
        "-v", f"{repo}:{CONTAINER_REPO}:ro", "-v", f"{owner_path}:/state",
        "-e", f"PYTHONPATH={CONTAINER_REPO}/engines/qwen38-flash-next-nvfp4-2b/src",
        IMAGE, f"{CONTAINER_REPO}/{SCRIPT}", "--worker", "--rank", str(rank),
        "--store", "/state", "--plan", f"{CONTAINER_REPO}/{PLAN}",
    ]
    return command if remote is None else ["ssh", "-o", "BatchMode=yes", remote, shlex.join(command)]


class InjectedCrash(RuntimeError): pass


def matrix(repo: Path, remote: str):
    processes = (
        _Process(_command(repo, 0, None)),
        _Process(_command(repo, 1, remote)),
    )
    try:
        cases = []
        for transition in Transition:
            for process in processes: process.request("cleanup")
            endpoints = (
                _EndpointProxy(0, processes[0]), _EndpointProxy(1, processes[1])
            )
            coordinator = DistributedStateCoordinator(endpoints, _Tracer())

            def inject(observed):
                if observed is transition: raise InjectedCrash(transition.value)

            start = time.perf_counter()
            try:
                coordinator.commit(
                    "physical", f"txn-{transition.value}", _boundary(),
                    policy_state=_policy_state(),
                    inject_fault=inject,
                )
            except InjectedCrash:
                pass
            commit_seconds = time.perf_counter() - start
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
                "commit_seconds": round(commit_seconds, 6),
                "restore_seconds": None if restore_seconds is None else round(restore_seconds, 6),
                "restored": restored,
                "payload_bytes": tuple(value["payload_bytes"] for value in before),
                "rank_timings": tuple(endpoint.timings for endpoint in endpoints),
            })
        cleanup = tuple(process.request("cleanup") for process in processes)
        if any(value["payload_bytes"] for value in cleanup):
            raise RuntimeError("physical crash matrix cleanup left payload bytes")
        return {"workers": tuple(process.ready for process in processes), "cases": cases, "cleanup": cleanup}
    finally:
        for process in processes: process.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
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
        worker(args.rank, args.store, args.plan)
    else:
        if not args.remote: raise SystemExit("matrix requires remote")
        print(json.dumps(matrix(args.repo.resolve(), args.remote), sort_keys=True))


if __name__ == "__main__": main()

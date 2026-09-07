#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the controller-gated TP2 Torch state restore on two worker processes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import selectors
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from qwen38_slab.controller_restore import DecoderStateRestoreController
from qwen38_slab.controller_rpc import (
    AuthenticatedRpcChannel,
    AuthenticatedRpcWorker,
    ControllerRpcError,
    MAX_RPC_BYTES,
    PhysicalDecoderRestoreController,
    RpcCudaRuntime,
    RpcRankStore,
)
from qwen38_slab.decode import Depth
from qwen38_slab.device_decode import DevicePhase, DevicePublication
from qwen38_slab.distributed_state_txn import RankStateStore, RestoreInspection
from qwen38_slab.local_cuda_restore import (
    DecoderStateTransactionGate,
    LocalCudaStateBinding,
    TwoRankLocalCudaCoordinator,
)
from qwen38_slab.mtp_policy import AdaptiveMtpPolicy, ConcurrencyCeiling, PolicyConfig
from qwen38_slab.runtime_state import RuntimeBoundary
from qwen38_slab.state_owner import DecoderStateOwner, OwnerPhase
from qwen38_slab.state_txn import STATE_FAMILIES, StateIdentity
from qwen38_slab.torch_cuda import TorchCudaRuntime


class Span:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): del key, value
    def record_exception(self, exception): del exception


class Tracer:
    def start_as_current_span(self, name): del name; return Span()


class Decoder:
    def __init__(self):
        self.phase = DevicePhase.IDLE
        self.publication = None

    def upload_and_launch(self, value):
        generation = value if isinstance(value, int) else value.lease.generation
        self.publication = DevicePublication(generation, 16, generation % 2)
        return self.publication


def policy():
    return AdaptiveMtpPolicy(
        Tracer(),
        PolicyConfig((ConcurrencyCeiling(1, Depth.K7), ConcurrencyCeiling(16, Depth.K1))),
    )


def identity():
    return StateIdentity(
        "fc694-physical-nvme",
        hashlib.sha256(b"qwen38-planner-c16-262144-v1").hexdigest(),
    )


def layout(plan_path):
    plan = json.loads(plan_path.read_text())
    logical = {
        family: item["logical_bytes_per_stream"] * 16
        for family, item in zip(STATE_FAMILIES, plan["families"], strict=True)
    }
    allocated = {
        family: item["cuda_allocated_bytes_c16"]
        for family, item in zip(STATE_FAMILIES, plan["families"], strict=True)
    }
    by_id = {
        item["id"]: family
        for family, item in zip(STATE_FAMILIES, plan["families"], strict=True)
    }
    owners = {
        family: by_id.get(item["shares_cuda_allocation_with"], family)
        for family, item in zip(STATE_FAMILIES, plan["families"], strict=True)
    }
    return logical, allocated, owners


def inspection_dict(value):
    result = asdict(value)
    result["boundary"] = asdict(value.boundary)
    return result


class WorkerState:
    def __init__(
        self, rank, store_path, plan_path, session_id, device,
        fail_after_copy_family=None,
    ):
        import torch

        self.rank = rank
        self.session_id = session_id
        self.store = RankStateStore(rank, store_path, identity(), Tracer(), policy())
        inspection = self.store.inspect_restore(session_id)
        self.inspection = inspection
        self.authenticated = None
        boundary = RuntimeBoundary(
            inspection.boundary.token_count, inspection.boundary.token_hash, 7
        )
        decoder = Decoder()
        self.owner = DecoderStateOwner(rank=rank, decoder=decoder, tracer=Tracer())
        self.owner.accept_boundary(self.owner.upload_and_launch(7), boundary)
        logical, allocated, owners = layout(plan_path)
        self.runtime = TorchCudaRuntime(
            owner=self.owner,
            compute_streams=(torch.cuda.Stream(device=device),),
            torch_api=torch,
            device=device,
            allocation_bytes=allocated,
            allocation_owners=owners,
        )
        self.allocations = {}
        self.pending_boundary = None
        self.pending_commit = None
        self.gate_boundary = None
        self.accepted_continuation = None
        self.armed_continuation = None
        self.released_continuation = None
        self.fail_after_copy_family = fail_after_copy_family

    def handle(self, command, payload):
        if command == "status":
            active = self.owner.active_state
            return {
                "rank": self.rank,
                "phase": self.owner.phase.value,
                "active_commit_sha256": self.owner.active_commit_sha256,
                "active_family_count": 0 if active is None else len(active),
            }
        if command == "inspect":
            if payload.get("session_id") != self.session_id:
                raise RuntimeError("session changed")
            self.inspection = self.store.inspect_restore(self.session_id)
            return inspection_dict(self.inspection)
        if command == "authenticate":
            requested = payload.get("inspection")
            if requested != inspection_dict(self.inspection):
                raise RuntimeError("inspection changed")
            state = self.store.authenticate(self.inspection)
            self.authenticated = state
            return {
                "rank": self.rank,
                "boundary": asdict(state.boundary),
                "commit_sha256": state.commit_sha256,
                "policy_digest": state.policy_digest,
                "policy_state": base64.b64encode(state.policy_state).decode(),
                "families": {
                    family: {
                        "path": str(extent.path),
                        "logical_bytes": extent.logical_bytes,
                        "length_bytes": extent.length_bytes,
                        "logical_sha256": extent.logical_sha256,
                        "padded_sha256": extent.padded_sha256,
                    }
                    for family, extent in state.families.items()
                },
            }
        if command == "quiesce":
            boundary = RuntimeBoundary(**payload["boundary"])
            self.owner.hold_launch_gate(boundary)
            self.gate_boundary = boundary
            receipt = self.runtime.quiesce(boundary)
            return {
                "compute_fenced": receipt.compute_fenced,
                "pending_launches": receipt.pending_launches,
            }
        if command == "allocate":
            family, size = payload.get("family"), payload.get("logical_bytes")
            destination = self.runtime.allocate_staging(family, size)
            token = f"r{self.rank}-{len(self.allocations)}"
            self.allocations[token] = destination
            return {"token": token}
        if command == "copy":
            token, raw = payload.get("token"), payload.get("extent")
            if self.authenticated is None or token not in self.allocations:
                raise RuntimeError("copy has no authenticated allocation")
            family = raw.get("family")
            expected = self.authenticated.families.get(family)
            if expected is None or raw != {
                "family": expected.family, "path": str(expected.path),
                "logical_bytes": expected.logical_bytes,
                "length_bytes": expected.length_bytes,
                "logical_sha256": expected.logical_sha256,
                "padded_sha256": expected.padded_sha256,
            }:
                raise RuntimeError("copy extent changed")
            self.runtime.copy_extent_to_device(self.allocations[token], expected)
            if family == self.fail_after_copy_family:
                self.fail_after_copy_family = None
                raise RuntimeError("injected post-copy worker fault")
            return {}
        if command == "finish":
            self.runtime.finish_transfers(); return {}
        if command == "prepare":
            if self.authenticated is None:
                raise RuntimeError("prepare has no authenticated state")
            boundary = RuntimeBoundary(**payload["boundary"])
            policy_state = base64.b64decode(payload["policy_state"], validate=True)
            commit = payload.get("commit_sha256")
            if policy_state != self.authenticated.policy_state or commit != self.authenticated.commit_sha256:
                raise RuntimeError("prepare identity changed")
            tokens = payload.get("tokens")
            staged = {family: self.allocations[tokens[family]] for family in STATE_FAMILIES}
            self.runtime.prepare_local(staged, boundary, policy_state, commit)
            self.pending_boundary, self.pending_commit = boundary, commit
            return {}
        if command in {"commit", "rollback", "finalize"}:
            boundary = RuntimeBoundary(**payload["boundary"])
            commit = payload.get("commit_sha256")
            if boundary != self.pending_boundary or commit != self.pending_commit:
                raise RuntimeError("pending identity changed")
            getattr(self.runtime, f"{command}_local")(boundary, commit)
            return {}
        if command == "discard":
            tensors = tuple(self.allocations[token] for token in payload.get("tokens", ()))
            self.runtime.discard(tensors); return {}
        if command == "retain_gate":
            if self.owner.phase == OwnerPhase.OPEN:
                raise RuntimeError("physical gate opened before common receipt")
            return {"phase": self.owner.phase.value}
        if command in {"accept_common", "arm_open", "open_common"}:
            continuation = payload.get("continuation")
            if command == "open_common" and continuation == self.released_continuation:
                return {
                    "rank": self.rank, "phase": "open",
                    "token_hash": self.pending_boundary.token_hash,
                    "generation_epoch": self.pending_boundary.generation_epoch,
                }
            if (
                self.pending_boundary is None
                or continuation.get("token_hash") != self.pending_boundary.token_hash
                or continuation.get("generation_epoch") != self.pending_boundary.generation_epoch
                or continuation.get("commit_sha256") != self.pending_commit
                or self.owner.phase == OwnerPhase.OPEN
            ):
                raise RuntimeError("common continuation changed")
            if command == "accept_common":
                if self.accepted_continuation not in (None, continuation):
                    raise RuntimeError("different common continuation was already accepted")
                self.accepted_continuation = dict(continuation)
                phase = "accepted"
            elif command == "arm_open":
                if continuation != self.accepted_continuation:
                    raise RuntimeError("common continuation was not accepted while closed")
                # Drop rollback and the runtime hold only after both ranks accepted the
                # receipt.  The independent coordinator hold keeps this physical owner
                # CLOSED while the controller arms its peer.
                self.runtime.complete_local(self.pending_boundary, self.pending_commit)
                self.runtime.resume(self.pending_boundary)
                self.owner.validate_launch_gate_release(self.pending_boundary)
                self.armed_continuation = dict(continuation)
                phase = "armed"
            else:
                if continuation != self.armed_continuation:
                    raise RuntimeError("common continuation open was not armed while closed")
                # arm_open exhausted the validation/failure branches.  Admission stays
                # serialized by the controller latch while both local assignments run.
                self.owner._commit_launch_gate_release(self.pending_boundary)
                self.released_continuation = dict(continuation)
                phase = "open"
            return {
                "rank": self.rank, "phase": phase,
                "token_hash": self.pending_boundary.token_hash,
                "generation_epoch": self.pending_boundary.generation_epoch,
            }
        if command == "fault":
            try:
                if self.pending_boundary is not None:
                    self.runtime.rollback_local(self.pending_boundary, self.pending_commit)
            except Exception:
                pass
            self.owner.fault_closed(self.pending_boundary or self.gate_boundary)
            return {"phase": self.owner.phase.value}
        if command == "launch":
            continuation = payload.get("continuation")
            generation = payload.get("generation")
            if continuation != self.released_continuation:
                raise RuntimeError("launch lacks the published common continuation")
            publication = self.owner.upload_and_launch(generation)
            return {"generation": publication.generation}
        raise RuntimeError("unknown worker command")


class ProcessTransport:
    def __init__(self, command, key, timeout):
        self.process = subprocess.Popen(
            shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, bufsize=0,
        )
        self.timeout = timeout
        bootstrap = base64.b64encode(key) + b"\n"
        self.process.stdin.write(bootstrap); self.process.stdin.flush()
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def exchange(self, request, timeout_seconds):
        self.process.stdin.write(request + b"\n"); self.process.stdin.flush()
        if not self.selector.select(min(timeout_seconds, self.timeout)):
            raise TimeoutError("worker response timeout")
        response = self.process.stdout.readline(MAX_RPC_BYTES + 2)
        if not response or len(response) > MAX_RPC_BYTES + 1:
            raise ControllerRpcError("worker response is absent or oversized")
        return response.rstrip(b"\n")

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=30)


def run_worker(args):
    encoded = sys.stdin.buffer.readline(128)
    key = base64.b64decode(encoded.strip(), validate=True)
    state = WorkerState(
        args.rank, args.store, args.plan, args.session, args.device,
        args.fail_after_copy_family,
    )
    server = AuthenticatedRpcWorker(args.rank, key, state.handle)
    for line in sys.stdin.buffer:
        if len(line) > MAX_RPC_BYTES + 1:
            raise ControllerRpcError("worker request is oversized")
        response = server.handle(line.rstrip(b"\n"))
        sys.stdout.buffer.write(response + b"\n"); sys.stdout.buffer.flush()


def run_controller(args):
    key = os.urandom(32)
    transports = (
        ProcessTransport(args.rank0_command, key, args.timeout),
        ProcessTransport(args.rank1_command, key, args.timeout),
    )
    try:
        channels = tuple(
            AuthenticatedRpcChannel(rank, key, transport, args.timeout)
            for rank, transport in enumerate(transports)
        )
        for channel in channels:
            channel.request("status", {})
        stores = tuple(RpcRankStore(rank, channel) for rank, channel in enumerate(channels))
        inspection = stores[0].inspect_restore(args.session)
        boundary = RuntimeBoundary(
            inspection.boundary.token_count, inspection.boundary.token_hash, 7
        )
        tracer = Tracer()
        owners = []
        runtimes = []
        bindings = []
        logical, _allocated, _owners = layout(args.plan)
        for rank, channel in enumerate(channels):
            decoder = Decoder()
            owner = DecoderStateOwner(rank=rank, decoder=decoder, tracer=tracer)
            owner.accept_boundary(owner.upload_and_launch(7), boundary)
            runtime = RpcCudaRuntime(owner, channel)
            owners.append(owner); runtimes.append(runtime)
            bindings.append(LocalCudaStateBinding(rank, runtime, policy(), logical, tracer))
        coordinator = TwoRankLocalCudaCoordinator(tuple(owners), tuple(bindings), tracer)
        gate = DecoderStateTransactionGate(coordinator, 7, tracer)
        sources = tuple({family: object() for family in STATE_FAMILIES} for _ in range(2))
        logical_controller = DecoderStateRestoreController(stores, sources, gate, tracer)
        physical = PhysicalDecoderRestoreController(
            logical_controller, tuple(runtimes)
        )
        try:
            continuation = physical.restore(args.session)
        except ControllerRpcError as exc:
            if not args.expect_failure:
                raise
            statuses = tuple(channel.request("status", {}) for channel in channels)
            if any(
                value.get("phase") != "faulted"
                or value.get("active_family_count") != 0
                or value.get("active_commit_sha256") is not None
                for value in statuses
            ):
                raise ControllerRpcError(
                    "failed physical restore exposed a partial active generation"
                ) from exc
            print(json.dumps({
                "result": "qwen38-controller-state-proof",
                "expected_failure": True,
                "worker_status": statuses,
            }, sort_keys=True))
            return
        if args.expect_failure:
            raise ControllerRpcError("expected physical restore failure did not occur")
        launched = tuple(physical.launch(rank, 8) for rank in range(2))
        if tuple(value.get("generation") for value in launched) != (8, 8):
            raise ControllerRpcError("physical continuation launch generations changed")
        print(json.dumps({
            "result": "qwen38-controller-state-proof",
            **asdict(continuation),
            "continuation_launches": launched,
        }, sort_keys=True))
    finally:
        for transport in transports:
            transport.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--rank", type=int, choices=(0, 1), required=True)
    worker.add_argument("--store", type=Path, required=True)
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument("--session", required=True)
    worker.add_argument("--device", default="cuda:0")
    worker.add_argument("--fail-after-copy-family", choices=STATE_FAMILIES)
    controller = sub.add_parser("controller")
    controller.add_argument("--rank0-command", required=True)
    controller.add_argument("--rank1-command", required=True)
    controller.add_argument("--plan", type=Path, required=True)
    controller.add_argument("--session", required=True)
    controller.add_argument("--timeout", type=float, default=300.0)
    controller.add_argument("--expect-failure", action="store_true")
    args = parser.parse_args()
    run_worker(args) if args.mode == "worker" else run_controller(args)


if __name__ == "__main__":
    main()

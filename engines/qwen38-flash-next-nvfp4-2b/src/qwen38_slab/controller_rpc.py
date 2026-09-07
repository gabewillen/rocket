# SPDX-License-Identifier: Apache-2.0
"""Authenticated bounded RPC adapters for physical TP2 state publication."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .controller_restore import DecoderContinuation, DecoderStateRestoreController
from .distributed_state_txn import (
    AuthenticatedFamilyExtent,
    LocalAuthenticatedState,
    RestoreInspection,
)
from .local_cuda_restore import DecoderStateTransactionGate
from .runtime_state import QuiesceReceipt, RuntimeBoundary
from .state_txn import STATE_FAMILIES, AcceptedBoundary, _HEX_256

RPC_SCHEMA = "rocket.qwen38.controller-worker.v1"
MAX_RPC_BYTES = 65_536


class ControllerRpcError(RuntimeError):
    """Authenticated RPC, timeout, or physical publication failure."""


class RpcTransport:
    def exchange(self, request: bytes, timeout_seconds: float) -> bytes: ...


class AuthenticatedRpcChannel:
    """Strict request/response sequence with HMAC-bound rank and direction."""

    def __init__(self, rank: int, key: bytes, transport: RpcTransport, timeout_seconds=30.0):
        if (
            isinstance(rank, bool)
            or rank not in (0, 1)
            or not isinstance(key, bytes)
            or len(key) != 32
            or not callable(getattr(transport, "exchange", None))
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0 < timeout_seconds <= 300
        ):
            raise ControllerRpcError("rank, 256-bit key, transport, and bounded timeout are required")
        self.rank = rank
        self._key = key
        self._transport = transport
        self._timeout = float(timeout_seconds)
        self._sequence = 0
        self._exchange_lock = threading.RLock()

    def request(self, command: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        if not isinstance(command, str) or not command or len(command) > 64:
            raise ControllerRpcError("RPC command is invalid")
        with self._exchange_lock:
            self._sequence += 1
            request = self._encode("request", self._sequence, command, dict(payload))
            try:
                response = self._transport.exchange(request, self._timeout)
            except TimeoutError as exc:
                raise ControllerRpcError(f"rank {self.rank} RPC timed out") from exc
            decoded = self._decode(response, "response", self._sequence, command)
            if decoded.get("ok") is not True or not isinstance(decoded.get("result"), dict):
                raise ControllerRpcError(f"rank {self.rank} rejected {command}")
            return decoded["result"]

    def _encode(self, direction, sequence, command, payload):
        envelope = {
            "schema": RPC_SCHEMA,
            "direction": direction,
            "rank": self.rank,
            "sequence": sequence,
            "command": command,
            "payload": payload,
        }
        canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        envelope["hmac_sha256"] = hmac.new(self._key, canonical, hashlib.sha256).hexdigest()
        encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > MAX_RPC_BYTES:
            raise ControllerRpcError("RPC message exceeds 65536 bytes")
        return encoded

    def _decode(self, encoded, direction, sequence, command):
        if not isinstance(encoded, bytes) or not 0 < len(encoded) <= MAX_RPC_BYTES:
            raise ControllerRpcError("RPC response extent is invalid")
        try:
            envelope = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControllerRpcError("RPC response is not canonical JSON") from exc
        if not isinstance(envelope, dict):
            raise ControllerRpcError("RPC response root is invalid")
        signature = envelope.pop("hmac_sha256", None)
        canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        expected = hmac.new(self._key, canonical, hashlib.sha256).hexdigest()
        if (
            not isinstance(signature, str)
            or not hmac.compare_digest(signature, expected)
            or envelope.get("schema") != RPC_SCHEMA
            or envelope.get("direction") != direction
            or envelope.get("rank") != self.rank
            or envelope.get("sequence") != sequence
            or envelope.get("command") != command
            or not isinstance(envelope.get("payload"), dict)
        ):
            raise ControllerRpcError("RPC response authentication changed")
        return envelope["payload"]


class AuthenticatedRpcWorker:
    """Worker-side strict sequence verifier for one bounded command handler."""

    def __init__(self, rank: int, key: bytes, handler):
        if (
            isinstance(rank, bool)
            or rank not in (0, 1)
            or not isinstance(key, bytes)
            or len(key) != 32
            or not callable(handler)
        ):
            raise ControllerRpcError("worker rank, 256-bit key, and handler are required")
        self.rank = rank
        self._key = key
        self._handler = handler
        self._sequence = 0

    def handle(self, encoded: bytes) -> bytes:
        self._sequence += 1
        command, payload = self._decode(encoded)
        try:
            result = self._handler(command, payload)
            response = {"ok": True, "result": result}
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            response = {"ok": False, "result": {}}
        return self._encode(command, response)

    def _decode(self, encoded):
        if not isinstance(encoded, bytes) or not 0 < len(encoded) <= MAX_RPC_BYTES:
            raise ControllerRpcError("RPC request extent is invalid")
        try:
            envelope = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControllerRpcError("RPC request is not canonical JSON") from exc
        if not isinstance(envelope, dict):
            raise ControllerRpcError("RPC request root is invalid")
        signature = envelope.pop("hmac_sha256", None)
        canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        expected = hmac.new(self._key, canonical, hashlib.sha256).hexdigest()
        if (
            not isinstance(signature, str)
            or not hmac.compare_digest(signature, expected)
            or envelope.get("schema") != RPC_SCHEMA
            or envelope.get("direction") != "request"
            or envelope.get("rank") != self.rank
            or envelope.get("sequence") != self._sequence
            or not isinstance(envelope.get("command"), str)
            or not isinstance(envelope.get("payload"), dict)
        ):
            raise ControllerRpcError("RPC request authentication changed")
        return envelope["command"], envelope["payload"]

    def _encode(self, command, payload):
        envelope = {
            "schema": RPC_SCHEMA,
            "direction": "response",
            "rank": self.rank,
            "sequence": self._sequence,
            "command": command,
            "payload": payload,
        }
        canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        envelope["hmac_sha256"] = hmac.new(self._key, canonical, hashlib.sha256).hexdigest()
        encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > MAX_RPC_BYTES:
            raise ControllerRpcError("RPC response exceeds 65536 bytes")
        return encoded


@dataclass(frozen=True)
class RemoteAllocation:
    rank: int
    family: str
    logical_bytes: int
    token: str


class RpcRankStore:
    """Controller proxy for owner-local inspect and authentication only."""

    def __init__(self, rank: int, channel: AuthenticatedRpcChannel):
        if channel.rank != rank:
            raise ControllerRpcError("rank store channel changed")
        self.rank = rank
        self.channel = channel

    def prepare(self, *args):
        raise ControllerRpcError("physical proof consumes an existing durable commit")

    def commit(self, *args):
        raise ControllerRpcError("physical proof consumes an existing durable commit")

    def index(self, *args):
        raise ControllerRpcError("physical proof consumes an existing durable commit")

    def inspect_restore(self, session_id):
        result = self.channel.request("inspect", {"session_id": session_id})
        return _inspection(result)

    def authenticate(self, inspection):
        result = self.channel.request("authenticate", {"inspection": _inspection_dict(inspection)})
        try:
            boundary = AcceptedBoundary(**result["boundary"])
            policy_state = base64.b64decode(result["policy_state"], validate=True)
            families = {
                family: AuthenticatedFamilyExtent(
                    family,
                    Path(result["families"][family]["path"]),
                    result["families"][family]["logical_bytes"],
                    result["families"][family]["length_bytes"],
                    result["families"][family]["logical_sha256"],
                    result["families"][family]["padded_sha256"],
                )
                for family in STATE_FAMILIES
            }
            state = LocalAuthenticatedState._from_verified(
                self.rank,
                boundary,
                result["policy_digest"],
                policy_state,
                families,
                result["commit_sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ControllerRpcError("rank authenticated-state response is invalid") from exc
        if (
            boundary != inspection.boundary
            or state.commit_sha256 != inspection.commit_sha256
            or state.policy_digest != inspection.policy_digest
        ):
            raise ControllerRpcError("rank authenticated state changed after inspection")
        return state


class RpcCudaRuntime:
    """Local binding adapter whose storage and physical gate live on one worker."""

    def __init__(self, owner, channel: AuthenticatedRpcChannel):
        if getattr(owner, "rank", None) != channel.rank:
            raise ControllerRpcError("CUDA runtime owner/channel rank changed")
        self.owner = owner
        self.channel = channel
        self._allocations = {}

    def quiesce(self, boundary):
        result = self.channel.request("quiesce", {"boundary": asdict(boundary)})
        self.owner.close_launch_gate(boundary)
        return QuiesceReceipt(boundary, result.get("compute_fenced"), result.get("pending_launches"))

    def allocate_staging(self, family, logical_bytes):
        result = self.channel.request("allocate", {"family": family, "logical_bytes": logical_bytes})
        token = result.get("token")
        if not isinstance(token, str) or not token or len(token) > 64 or token in self._allocations:
            raise ControllerRpcError("worker allocation token is invalid")
        allocation = RemoteAllocation(self.owner.rank, family, logical_bytes, token)
        self._allocations[token] = allocation
        return allocation

    def copy_extent_to_device(self, destination, extent):
        self._require_allocation(destination, extent.family, extent.logical_bytes)
        self.channel.request("copy", {"token": destination.token, "extent": _extent_dict(extent)})

    def finish_transfers(self):
        self.channel.request("finish", {})

    def prepare_local(self, staged, boundary, policy_state, commit_sha256):
        tokens = self._tokens(staged)
        self.channel.request(
            "prepare",
            {"tokens": tokens, "boundary": asdict(boundary),
             "policy_state": base64.b64encode(policy_state).decode(),
             "commit_sha256": commit_sha256},
        )
        self.owner.prepare_state_with_policy(staged, policy_state, boundary, commit_sha256)

    def commit_local(self, boundary, commit_sha256):
        self.channel.request("commit", {"boundary": asdict(boundary), "commit_sha256": commit_sha256})
        self.owner.commit_prepared_state(boundary, commit_sha256)

    def rollback_local(self, boundary, commit_sha256):
        try:
            self.channel.request("rollback", {"boundary": asdict(boundary), "commit_sha256": commit_sha256})
        finally:
            self.owner.rollback_prepared_state(boundary, commit_sha256)

    def finalize_local(self, boundary, commit_sha256):
        self.channel.request("finalize", {"boundary": asdict(boundary), "commit_sha256": commit_sha256})
        self.owner.finalize_prepared_state(boundary, commit_sha256)

    def complete_local(self, boundary, commit_sha256):
        del boundary, commit_sha256
        # Remote undo remains live until both workers acknowledge the common receipt.
        self.owner._discard_prepared_rollback()

    def discard(self, staged):
        tokens = [value.token for value in staged if isinstance(value, RemoteAllocation)]
        self.channel.request("discard", {"tokens": tokens})

    def resume(self, boundary):
        self.channel.request("retain_gate", {"boundary": asdict(boundary)})
        self.owner.open_launch_gate(boundary)

    def accept_common(self, continuation: DecoderContinuation):
        return self.channel.request("accept_common", {"continuation": asdict(continuation)})

    def release_common(self, continuation: DecoderContinuation):
        return self.channel.request("arm_open", {"continuation": asdict(continuation)})

    def open_common(self, continuation: DecoderContinuation):
        return self.channel.request("open_common", {"continuation": asdict(continuation)})

    def launch(self, continuation: DecoderContinuation, generation: int):
        return self.channel.request(
            "launch", {"continuation": asdict(continuation), "generation": generation}
        )

    def fault_physical(self, continuation=None):
        payload = {} if continuation is None else {"continuation": asdict(continuation)}
        try:
            self.channel.request("fault", payload)
        except ControllerRpcError:
            pass

    def _require_allocation(self, value, family, logical_bytes):
        if (
            not isinstance(value, RemoteAllocation)
            or self._allocations.get(value.token) is not value
            or value.rank != self.owner.rank
            or value.family != family
            or value.logical_bytes != logical_bytes
        ):
            raise ControllerRpcError("remote allocation ownership changed")

    def _tokens(self, staged):
        if tuple(staged) != STATE_FAMILIES:
            raise ControllerRpcError("remote staged table is incomplete")
        for family in STATE_FAMILIES:
            self._require_allocation(staged[family], family, staged[family].logical_bytes)
        return {family: staged[family].token for family in STATE_FAMILIES}


class PhysicalDecoderRestoreController:
    """Keep physical gates closed until both workers accept one common receipt."""

    def __init__(self, controller: DecoderStateRestoreController, runtimes):
        if (
            not callable(getattr(controller, "restore", None))
            or not isinstance(runtimes, tuple)
            or len(runtimes) != 2
            or tuple(runtime.owner.rank for runtime in runtimes) != (0, 1)
        ):
            raise ControllerRpcError("ordered controller and physical runtimes are required")
        self.controller = controller
        self.runtimes = runtimes
        self._terminal = False
        self._admission = threading.RLock()
        self._published: DecoderContinuation | None = None

    def admit(self, continuation: DecoderContinuation) -> None:
        """Authorize a controller-owned decoder launch after both physical opens."""

        with self._admission:
            if self._published != continuation:
                raise ControllerRpcError("physical TP2 continuation is not published")

    def launch(self, rank: int, generation: int):
        """Route the only authenticated worker launch entry through publication."""

        with self._admission:
            if self._published is None:
                raise ControllerRpcError("physical TP2 continuation is not published")
            if isinstance(rank, bool) or rank not in (0, 1):
                raise ControllerRpcError("physical launch rank is invalid")
            return self.runtimes[rank].launch(self._published, generation)

    def restore(self, session_id):
        if self._terminal:
            raise ControllerRpcError("physical restore is one-shot")
        self._terminal = True
        continuation = None
        with self._admission:
            try:
                continuation = self.controller.restore(session_id)
                accepted = tuple(
                    runtime.accept_common(continuation) for runtime in self.runtimes
                )
                self._validate_acknowledgements(accepted, continuation, "accepted")
                released = tuple(
                    runtime.release_common(continuation) for runtime in self.runtimes
                )
                self._validate_acknowledgements(released, continuation, "armed")
                opened = tuple(
                    runtime.open_common(continuation) for runtime in self.runtimes
                )
                self._validate_acknowledgements(opened, continuation, "open")
                self._published = continuation
                return continuation
            except BaseException as exc:
                for runtime in self.runtimes:
                    runtime.fault_physical(continuation)
                if not isinstance(exc, Exception):
                    raise
                if isinstance(exc, ControllerRpcError):
                    raise
                raise ControllerRpcError("physical TP2 restore failed") from exc

    @staticmethod
    def _validate_acknowledgements(values, continuation, phase):
        if (
            len(values) != 2
            or tuple(value.get("rank") for value in values) != (0, 1)
            or any(value.get("phase") != phase for value in values)
            or any(value.get("token_hash") != continuation.token_hash for value in values)
            or any(value.get("generation_epoch") != continuation.generation_epoch for value in values)
        ):
            raise ControllerRpcError("physical continuation acknowledgement changed")


def _inspection(value):
    try:
        copied = dict(value)
        copied["boundary"] = AcceptedBoundary(**copied["boundary"])
        return RestoreInspection(**copied)
    except (KeyError, TypeError, ValueError) as exc:
        raise ControllerRpcError("restore inspection response is invalid") from exc


def _inspection_dict(value):
    result = asdict(value)
    result["boundary"] = asdict(value.boundary)
    return result


def _extent_dict(value):
    return {
        "family": value.family,
        "path": str(value.path),
        "logical_bytes": value.logical_bytes,
        "length_bytes": value.length_bytes,
        "logical_sha256": value.logical_sha256,
        "padded_sha256": value.padded_sha256,
    }


__all__ = [
    "AuthenticatedRpcChannel", "AuthenticatedRpcWorker", "ControllerRpcError", "DecoderContinuation",
    "MAX_RPC_BYTES", "PhysicalDecoderRestoreController", "RPC_SCHEMA",
    "RemoteAllocation", "RpcCudaRuntime", "RpcRankStore",
]

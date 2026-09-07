"""Owner-local two-rank state transactions exchanging metadata only.

Each :class:`RankStateStore` owns exactly one rank directory. Payload files
never leave that node. The coordinator exchanges bounded receipts, writes one
shared commit decision and session index on both ranks, then authenticates each
rank locally before allowing either local publish callback.

Payload I/O uses 65536-byte aligned ``posix_memalign`` buffers with
``O_DIRECT`` ``preadv``/``pwritev``. There is no buffered payload fallback.
OpenTelemetry attributes remain bounded to phase, rank, family, and outcome.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Protocol

from .mtp_policy import AdaptiveMtpPolicy, MtpPolicyError
from .state_txn import (
    IO_CHUNK_BYTES,
    PAGE_BYTES,
    SCHEMA,
    STATE_FAMILIES,
    AcceptedBoundary,
    FaultInjector,
    OtelTracer,
    StateIdentity,
    StateTransactionError,
    Transition,
    _HEX_256,
    _SAFE_ID,
    _align_up,
    _atomic_write,
    _canonical_bytes,
    _decode_record,
    _ensure_directory,
    _fsync_directory,
    _require_keys,
)

DISTRIBUTED_SCHEMA = "rocket.qwen38-distributed-state.v1"
MAX_POLICY_BYTES = 65_536
_FAMILY_FILES = {
    family: f"{index:02d}-{family}.state"
    for index, family in enumerate(STATE_FAMILIES)
}


@dataclass(frozen=True)
class GeneratedFamilySource:
    """Deterministic owner-local source with constant logical bytes."""

    logical_bytes: int
    fill_byte: int
    stored_bytes: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.logical_bytes, bool)
            or not isinstance(self.logical_bytes, int)
            or self.logical_bytes <= 0
            or isinstance(self.fill_byte, bool)
            or not isinstance(self.fill_byte, int)
            or not 0 <= self.fill_byte <= 255
            or (
                self.stored_bytes is not None
                and (
                    isinstance(self.stored_bytes, bool)
                    or not isinstance(self.stored_bytes, int)
                    or self.stored_bytes < self.logical_bytes
                    or self.stored_bytes % PAGE_BYTES
                )
            )
        ):
            raise StateTransactionError("generated family source is invalid")

    def fill(self, view: memoryview, offset: int, accepted_bytes: int) -> None:
        del offset
        view[:accepted_bytes] = bytes((self.fill_byte,)) * accepted_bytes


@dataclass(frozen=True)
class PrepareReceipt:
    rank: int
    session_id: str
    transaction_id: str
    boundary: AcceptedBoundary
    prepared_sha256: str
    rank_sha256: str
    policy_digest: str


@dataclass(frozen=True)
class CommitReceipt:
    rank: int
    session_id: str
    transaction_id: str
    commit_sha256: str


@dataclass(frozen=True)
class RestoreInspection:
    rank: int
    session_id: str
    transaction_id: str
    boundary: AcceptedBoundary
    commit_sha256: str
    prepared_sha256: str
    rank_sha256: str
    policy_digest: str


@dataclass(frozen=True)
class AuthenticatedFamilyExtent:
    family: str
    path: Path
    logical_bytes: int
    length_bytes: int
    logical_sha256: str
    padded_sha256: str


@dataclass(frozen=True)
class LocalAuthenticatedState:
    """Local immutable descriptor; payload paths never enter a receipt."""

    rank: int
    boundary: AcceptedBoundary
    policy_digest: str
    policy_state: bytes
    families: Mapping[str, AuthenticatedFamilyExtent]


@dataclass(frozen=True)
class AuthenticationReceipt:
    rank: int
    boundary: AcceptedBoundary
    commit_sha256: str
    rank_sha256: str
    policy_digest: str


class RankEndpoint(Protocol):
    rank: int

    def prepare(
        self, session_id: str, transaction_id: str,
        boundary: AcceptedBoundary, policy_state: bytes,
    ) -> PrepareReceipt: ...
    def commit(self, receipts: tuple[PrepareReceipt, PrepareReceipt]) -> CommitReceipt: ...
    def index(self, receipts: tuple[CommitReceipt, CommitReceipt]) -> None: ...
    def inspect_restore(self, session_id: str) -> RestoreInspection: ...
    def authenticate(self, inspection: RestoreInspection) -> AuthenticationReceipt: ...
    def publish(self, receipt: AuthenticationReceipt) -> None: ...


class _AlignedBuffer:
    """Owned heap buffer aligned without a file-backed or anonymous mapping."""

    _libc = ctypes.CDLL(None)
    _libc.posix_memalign.argtypes = (
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t
    )
    _libc.posix_memalign.restype = ctypes.c_int
    _libc.free.argtypes = (ctypes.c_void_p,)

    def __init__(self, length: int):
        if length <= 0 or length % PAGE_BYTES:
            raise StateTransactionError("direct buffer length is not 65536-byte aligned")
        pointer = ctypes.c_void_p()
        result = self._libc.posix_memalign(
            ctypes.byref(pointer), PAGE_BYTES, length
        )
        if result != 0 or pointer.value is None:
            raise StateTransactionError(f"posix_memalign failed with errno {result}")
        self._pointer = pointer
        self._array = (ctypes.c_ubyte * length).from_address(pointer.value)
        self.view = memoryview(self._array).cast("B")

    def close(self) -> None:
        self.view.release()
        del self.view
        del self._array
        self._libc.free(self._pointer)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): self.close()


class RankStateStore:
    """One rank's durable transaction participant and local authenticator."""

    def __init__(
        self,
        rank: int,
        store: Path,
        identity: StateIdentity,
        tracer: OtelTracer,
        policy: AdaptiveMtpPolicy,
    ):
        if isinstance(rank, bool) or rank not in (0, 1):
            raise StateTransactionError("rank store rank must be 0 or 1")
        if not _SAFE_ID.fullmatch(identity.revision) or not _HEX_256.fullmatch(
            identity.precision_digest
        ):
            raise StateTransactionError("rank store identity is invalid")
        if identity.topology != "qwen3.8-flash-next:tp2:rank-native:v1":
            raise StateTransactionError("rank store topology is invalid")
        if tracer is None:
            raise StateTransactionError("an OpenTelemetry tracer is required")
        if not isinstance(policy, AdaptiveMtpPolicy):
            raise StateTransactionError("an adaptive MTP policy is required")
        self.rank = rank
        self.store = Path(store)
        self.identity = identity
        self.tracer = tracer
        self.policy = policy

    def prepare(
        self,
        session_id: str,
        transaction_id: str,
        boundary: AcceptedBoundary,
        sources: Mapping[str, GeneratedFamilySource],
        policy_state: bytes,
    ) -> PrepareReceipt:
        policy_state = _canonical_policy_state(self.policy, policy_state)
        policy_digest = hashlib.sha256(policy_state).hexdigest()
        _validate_common(session_id, transaction_id, boundary, policy_digest)
        if tuple(sources) != STATE_FAMILIES or any(
            not isinstance(source, GeneratedFamilySource)
            for source in sources.values()
        ):
            raise StateTransactionError("rank source requires canonical nine-family order")
        directory = self._transaction_dir(transaction_id)
        try:
            _ensure_directory(directory.parent)
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "families").mkdir()
        except OSError as exc:
            raise StateTransactionError("cannot create rank transaction directory") from exc
        table = []
        rank_digest = hashlib.sha256()
        for family in STATE_FAMILIES:
            source = sources[family]
            length = source.stored_bytes or _align_up(source.logical_bytes, PAGE_BYTES)
            logical_digest, padded_digest = self._write_family(
                family, directory / "families" / _FAMILY_FILES[family], source, length
            )
            rank_digest.update(family.encode() + b"\0" + bytes.fromhex(padded_digest))
            table.append({
                "family": family,
                "file": _FAMILY_FILES[family],
                "logical_bytes": source.logical_bytes,
                "length_bytes": length,
                "logical_sha256": logical_digest,
                "padded_sha256": padded_digest,
            })
        _fsync_directory(directory / "families")
        _atomic_write(directory / "POLICY.json", policy_state)
        record = {
            "schema": DISTRIBUTED_SCHEMA,
            "record": "PREPARED",
            "rank": self.rank,
            "session_id": session_id,
            "transaction_id": transaction_id,
            "revision": self.identity.revision,
            "precision_digest": self.identity.precision_digest,
            "topology": self.identity.topology,
            "token_count": boundary.token_count,
            "token_hash": boundary.token_hash,
            "policy_digest": policy_digest,
            "family_table": table,
            "rank_sha256": rank_digest.hexdigest(),
        }
        raw = _distributed_bytes(record)
        with self._observed("prepare", "none"):
            _atomic_write(directory / "PREPARED.json", raw)
            _fsync_directory(directory.parent)
        return PrepareReceipt(
            self.rank, session_id, transaction_id, boundary,
            hashlib.sha256(raw).hexdigest(), rank_digest.hexdigest(), policy_digest,
        )

    def commit(
        self, receipts: tuple[PrepareReceipt, PrepareReceipt]
    ) -> CommitReceipt:
        _validate_prepare_receipts(receipts)
        local = receipts[self.rank]
        raw = _commit_bytes(receipts, self.identity)
        prepared = self._read_bounded(
            self._transaction_dir(local.transaction_id) / "PREPARED.json"
        )
        if hashlib.sha256(prepared).hexdigest() != local.prepared_sha256:
            raise StateTransactionError("local prepared receipt changed before commit")
        with self._observed("commit", "none"):
            _atomic_write(
                self._transaction_dir(local.transaction_id) / "COMMITTED.json", raw
            )
        return CommitReceipt(
            self.rank, local.session_id, local.transaction_id,
            hashlib.sha256(raw).hexdigest(),
        )

    def index(self, receipts: tuple[CommitReceipt, CommitReceipt]) -> None:
        if (
            tuple(receipt.rank for receipt in receipts) != (0, 1)
            or receipts[0].commit_sha256 != receipts[1].commit_sha256
            or receipts[0].session_id != receipts[1].session_id
            or receipts[0].transaction_id != receipts[1].transaction_id
        ):
            raise StateTransactionError("commit receipts do not agree")
        local = receipts[self.rank]
        commit = _decode_distributed(
            self._read_bounded(
                self._transaction_dir(local.transaction_id) / "COMMITTED.json"
            ),
            "COMMITTED",
        )
        index = _distributed_bytes({
            "schema": DISTRIBUTED_SCHEMA,
            "record": "SESSION_INDEX",
            "session_id": commit["session_id"],
            "transaction_id": commit["transaction_id"],
            "commit_sha256": receipts[self.rank].commit_sha256,
        })
        with self._observed("index", "none"):
            _atomic_write(
                self.store / "sessions" / local.session_id / "index.json", index
            )

    def inspect_restore(self, session_id: str) -> RestoreInspection:
        if not _SAFE_ID.fullmatch(session_id):
            raise StateTransactionError("session ID is invalid")
        index = _decode_distributed(
            self._read_bounded(self.store / "sessions" / session_id / "index.json"),
            "SESSION_INDEX",
        )
        transaction_id = index.get("transaction_id")
        if not isinstance(transaction_id, str) or not _SAFE_ID.fullmatch(transaction_id):
            raise StateTransactionError("session transaction ID is invalid")
        commit_raw = self._read_bounded(
            self._transaction_dir(transaction_id) / "COMMITTED.json"
        )
        commit_sha = hashlib.sha256(commit_raw).hexdigest()
        if commit_sha != index.get("commit_sha256"):
            raise StateTransactionError("local commit digest does not match index")
        commit = _decode_distributed(commit_raw, "COMMITTED")
        _validate_commit_identity(commit, session_id, transaction_id, self.identity)
        prepared_raw = self._read_bounded(
            self._transaction_dir(transaction_id) / "PREPARED.json"
        )
        prepared_sha = hashlib.sha256(prepared_raw).hexdigest()
        if prepared_sha != commit["prepared_sha256"][str(self.rank)]:
            raise StateTransactionError("local prepared digest does not match commit")
        prepared = _decode_distributed(prepared_raw, "PREPARED")
        if prepared.get("rank") != self.rank or prepared.get("rank_sha256") != commit["rank_sha256"][str(self.rank)]:
            raise StateTransactionError("local prepared rank identity changed")
        boundary = AcceptedBoundary(commit["token_count"], commit["token_hash"], True)
        return RestoreInspection(
            self.rank, session_id, transaction_id, boundary, commit_sha,
            prepared_sha, prepared["rank_sha256"], commit["policy_digest"],
        )

    def authenticate(self, inspection: RestoreInspection) -> LocalAuthenticatedState:
        if inspection.rank != self.rank:
            raise StateTransactionError("restore inspection belongs to another rank")
        prepared = _decode_distributed(
            self._read_bounded(
                self._transaction_dir(inspection.transaction_id) / "PREPARED.json"
            ),
            "PREPARED",
        )
        table = prepared.get("family_table")
        if not isinstance(table, list) or len(table) != len(STATE_FAMILIES):
            raise StateTransactionError("prepared family table is incomplete")
        extents = {}
        rank_digest = hashlib.sha256()
        for family, extent in zip(STATE_FAMILIES, table, strict=True):
            if not isinstance(extent, dict) or extent.get("family") != family:
                raise StateTransactionError("prepared family order changed")
            authenticated = self._read_family(
                family, self._transaction_dir(inspection.transaction_id) / "families",
                extent,
            )
            rank_digest.update(
                family.encode() + b"\0" + bytes.fromhex(authenticated.padded_sha256)
            )
            extents[family] = authenticated
        if rank_digest.hexdigest() != inspection.rank_sha256:
            raise StateTransactionError("local whole-rank checksum changed")
        return LocalAuthenticatedState(
            self.rank, inspection.boundary, inspection.policy_digest,
            self._read_policy(inspection),
            MappingProxyType(extents),
        )

    def _read_policy(self, inspection):
        policy = self._read_bounded(
            self._transaction_dir(inspection.transaction_id) / "POLICY.json",
            MAX_POLICY_BYTES,
        )
        if hashlib.sha256(policy).hexdigest() != inspection.policy_digest:
            raise StateTransactionError("local policy state digest changed")
        return _canonical_policy_state(self.policy, policy)

    def _write_family(self, family, path, source, length):
        flags = getattr(os, "O_DIRECT", None)
        if flags is None:
            raise StateTransactionError("O_DIRECT is unavailable")
        fd = -1
        logical_digest = hashlib.sha256()
        padded_digest = hashlib.sha256()
        with self._observed("write", family):
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | flags, 0o600)
                for offset in range(0, length, IO_CHUNK_BYTES):
                    size = min(IO_CHUNK_BYTES, length - offset)
                    accepted = min(size, max(0, source.logical_bytes - offset))
                    with _AlignedBuffer(size) as buffer:
                        source.fill(buffer.view, offset, accepted)
                        if accepted < size:
                            buffer.view[accepted:] = bytes(size - accepted)
                        count = os.pwritev(fd, [buffer.view], offset)
                        if count != size:
                            raise StateTransactionError("short O_DIRECT rank write")
                        logical_digest.update(buffer.view[:accepted])
                        padded_digest.update(buffer.view)
                os.fdatasync(fd)
            except OSError as exc:
                raise StateTransactionError("O_DIRECT rank write failed") from exc
            finally:
                if fd >= 0: os.close(fd)
        return logical_digest.hexdigest(), padded_digest.hexdigest()

    def _read_family(self, family, directory, extent):
        required = {
            "family", "file", "logical_bytes", "length_bytes",
            "logical_sha256", "padded_sha256",
        }
        if set(extent) != required or extent["file"] != _FAMILY_FILES[family]:
            raise StateTransactionError("family extent metadata changed")
        logical = extent["logical_bytes"]
        length = extent["length_bytes"]
        if (
            isinstance(logical, bool) or not isinstance(logical, int) or logical <= 0
            or isinstance(length, bool) or not isinstance(length, int)
            or length < logical or length % PAGE_BYTES
        ):
            raise StateTransactionError("family extent violates direct-I/O contract")
        path = directory / extent["file"]
        fd = -1
        logical_digest = hashlib.sha256()
        padded_digest = hashlib.sha256()
        with self._observed("read", family):
            try:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                if os.fstat(fd).st_size != length:
                    raise StateTransactionError("family file size changed")
                for offset in range(0, length, IO_CHUNK_BYTES):
                    size = min(IO_CHUNK_BYTES, length - offset)
                    accepted = min(size, max(0, logical - offset))
                    with _AlignedBuffer(size) as buffer:
                        if os.preadv(fd, [buffer.view], offset) != size:
                            raise StateTransactionError("short O_DIRECT rank read")
                        if any(buffer.view[accepted:]):
                            raise StateTransactionError("family padding is not zero")
                        logical_digest.update(buffer.view[:accepted])
                        padded_digest.update(buffer.view)
            except OSError as exc:
                raise StateTransactionError("O_DIRECT rank read failed") from exc
            finally:
                if fd >= 0: os.close(fd)
        if logical_digest.hexdigest() != extent["logical_sha256"] or padded_digest.hexdigest() != extent["padded_sha256"]:
            raise StateTransactionError("family checksum changed")
        return AuthenticatedFamilyExtent(
            family, path, logical, length,
            logical_digest.hexdigest(), padded_digest.hexdigest(),
        )

    def _transaction_dir(self, transaction_id):
        return self.store / "transactions" / transaction_id

    @staticmethod
    def _read_bounded(path: Path, maximum: int = 1_048_576) -> bytes:
        try:
            size = path.stat().st_size
            if size <= 0 or size > maximum:
                raise StateTransactionError("metadata record size is outside bounds")
            return path.read_bytes()
        except OSError as exc:
            raise StateTransactionError("required local metadata is unavailable") from exc

    @contextmanager
    def _observed(self, phase: str, family: str) -> Iterator[None]:
        with self.tracer.start_as_current_span("rocket.qwen38.state.distributed") as span:
            span.set_attribute("phase", phase)
            span.set_attribute("rank", self.rank)
            span.set_attribute("family", family)
            try:
                yield
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                raise
            else:
                span.set_attribute("outcome", "success")


class LocalRankEndpoint:
    """In-process endpoint; network adapters expose the same metadata surface."""

    def __init__(self, store, sources, publisher):
        if not callable(publisher):
            raise StateTransactionError("local rank publisher is required")
        self.rank = store.rank
        self.store = store
        self.sources = sources
        self.publisher = publisher
        self.pending: LocalAuthenticatedState | None = None

    def prepare(self, session_id, transaction_id, boundary, policy_state):
        return self.store.prepare(
            session_id, transaction_id, boundary, self.sources, policy_state
        )
    def commit(self, receipts): return self.store.commit(receipts)
    def index(self, receipts): self.store.index(receipts)
    def inspect_restore(self, session_id): return self.store.inspect_restore(session_id)
    def authenticate(self, inspection):
        self.pending = self.store.authenticate(inspection)
        return AuthenticationReceipt(
            self.rank, inspection.boundary, inspection.commit_sha256,
            inspection.rank_sha256, inspection.policy_digest,
        )
    def publish(self, receipt):
        if self.pending is None or receipt.rank != self.rank:
            raise StateTransactionError("rank has no matching authenticated state")
        pending, self.pending = self.pending, None
        self.publisher(pending)


class DistributedStateCoordinator:
    """Two-rank metadata coordinator; endpoint payloads are never parameters."""

    def __init__(self, endpoints: tuple[RankEndpoint, RankEndpoint], tracer: OtelTracer):
        if len(endpoints) != 2 or tuple(endpoint.rank for endpoint in endpoints) != (0, 1):
            raise StateTransactionError("ordered rank endpoints are required")
        if tracer is None:
            raise StateTransactionError("an OpenTelemetry tracer is required")
        self.endpoints = endpoints
        self.tracer = tracer

    def commit(
        self, session_id, transaction_id, boundary, *, policy_state,
        inject_fault: FaultInjector | None = None,
    ):
        policy_digest = _policy_digest(policy_state)
        _validate_common(session_id, transaction_id, boundary, policy_digest)
        prepared = []
        for endpoint in self.endpoints:
            with self._observed("prepare", endpoint.rank):
                prepared.append(endpoint.prepare(
                    session_id, transaction_id, boundary, policy_state
                ))
            _fault(inject_fault, Transition(f"after_prepare_rank{endpoint.rank}"))
        receipts = tuple(prepared)
        committed = []
        for endpoint in self.endpoints:
            with self._observed("commit", endpoint.rank):
                committed.append(endpoint.commit(receipts))
            _fault(inject_fault, Transition(f"after_commit_rank{endpoint.rank}"))
        commits = tuple(committed)
        for endpoint in self.endpoints:
            with self._observed("index", endpoint.rank):
                endpoint.index(commits)
            _fault(inject_fault, Transition(f"after_index_rank{endpoint.rank}"))
        return commits

    def restore(self, session_id):
        inspections = []
        for endpoint in self.endpoints:
            with self._observed("inspect", endpoint.rank):
                inspections.append(endpoint.inspect_restore(session_id))
        inspections = tuple(inspections)
        _validate_inspections(inspections)
        authenticated = []
        for endpoint, inspection in zip(self.endpoints, inspections, strict=True):
            with self._observed("authenticate", endpoint.rank):
                authenticated.append(endpoint.authenticate(inspection))
        receipts = tuple(authenticated)
        _validate_authentication_receipts(receipts)
        for endpoint, receipt in zip(self.endpoints, receipts, strict=True):
            with self._observed("publish", endpoint.rank):
                endpoint.publish(receipt)
        return receipts

    @contextmanager
    def _observed(self, phase, rank):
        with self.tracer.start_as_current_span(
            "rocket.qwen38.state.distributed.coordinator"
        ) as span:
            span.set_attribute("phase", phase)
            span.set_attribute("rank", rank)
            span.set_attribute("family", "none")
            try:
                yield
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                raise
            else:
                span.set_attribute("outcome", "success")


def _validate_common(session, transaction, boundary, policy_digest):
    if not _SAFE_ID.fullmatch(session) or not _SAFE_ID.fullmatch(transaction):
        raise StateTransactionError("session and transaction IDs are invalid")
    if (
        not isinstance(boundary, AcceptedBoundary) or boundary.quiesced is not True
        or isinstance(boundary.token_count, bool) or boundary.token_count < 0
        or not _HEX_256.fullmatch(boundary.token_hash)
    ):
        raise StateTransactionError("accepted boundary is invalid")
    if not isinstance(policy_digest, str) or not _HEX_256.fullmatch(policy_digest):
        raise StateTransactionError("canonical policy digest is invalid")


def _policy_digest(policy_state):
    if (
        not isinstance(policy_state, bytes)
        or not 0 < len(policy_state) <= MAX_POLICY_BYTES
    ):
        raise StateTransactionError("policy state must be 1 through 65536 bytes")
    return hashlib.sha256(policy_state).hexdigest()


def _canonical_policy_state(policy, encoded):
    _policy_digest(encoded)
    try:
        decoded = policy.load_state(encoded)
        canonical = policy.dump_state(decoded)
    except MtpPolicyError as exc:
        raise StateTransactionError("adaptive MTP policy state is invalid") from exc
    if canonical != encoded:
        raise StateTransactionError("adaptive MTP policy state is not canonical")
    return canonical


def _validate_prepare_receipts(receipts):
    if tuple(receipt.rank for receipt in receipts) != (0, 1):
        raise StateTransactionError("prepare receipts require both ranks")
    first, second = receipts
    if (
        first.session_id != second.session_id
        or first.transaction_id != second.transaction_id
        or first.boundary != second.boundary
        or first.policy_digest != second.policy_digest
        or any(not _HEX_256.fullmatch(receipt.prepared_sha256) for receipt in receipts)
        or any(not _HEX_256.fullmatch(receipt.rank_sha256) for receipt in receipts)
    ):
        raise StateTransactionError("prepare receipts do not agree")


def _commit_bytes(receipts, identity):
    _validate_prepare_receipts(receipts)
    first = receipts[0]
    return _distributed_bytes({
        "schema": DISTRIBUTED_SCHEMA,
        "record": "COMMITTED",
        "session_id": first.session_id,
        "transaction_id": first.transaction_id,
        "revision": identity.revision,
        "precision_digest": identity.precision_digest,
        "topology": identity.topology,
        "token_count": first.boundary.token_count,
        "token_hash": first.boundary.token_hash,
        "policy_digest": first.policy_digest,
        "families": list(STATE_FAMILIES),
        "prepared_sha256": {str(value.rank): value.prepared_sha256 for value in receipts},
        "rank_sha256": {str(value.rank): value.rank_sha256 for value in receipts},
    })


def _validate_commit_identity(commit, session, transaction, identity):
    expected = {
        "schema", "record", "session_id", "transaction_id", "revision",
        "precision_digest", "topology", "token_count", "token_hash",
        "policy_digest", "families", "prepared_sha256", "rank_sha256",
    }
    _require_keys(commit, expected)
    if (
        commit["session_id"] != session or commit["transaction_id"] != transaction
        or commit["revision"] != identity.revision
        or commit["precision_digest"] != identity.precision_digest
        or commit["topology"] != identity.topology
        or commit["families"] != list(STATE_FAMILIES)
    ):
        raise StateTransactionError("distributed commit identity changed")
    boundary = AcceptedBoundary(commit["token_count"], commit["token_hash"], True)
    _validate_common(session, transaction, boundary, commit["policy_digest"])


def _validate_inspections(values):
    if tuple(value.rank for value in values) != (0, 1):
        raise StateTransactionError("restore inspections require both ranks")
    first, second = values
    if (
        first.session_id != second.session_id
        or first.transaction_id != second.transaction_id
        or first.boundary != second.boundary
        or first.commit_sha256 != second.commit_sha256
        or first.policy_digest != second.policy_digest
    ):
        raise StateTransactionError("rank restore inspections do not agree")


def _validate_authentication_receipts(values):
    if tuple(value.rank for value in values) != (0, 1):
        raise StateTransactionError("authentication receipts require both ranks")
    first, second = values
    if (
        first.boundary != second.boundary
        or first.commit_sha256 != second.commit_sha256
        or first.policy_digest != second.policy_digest
    ):
        raise StateTransactionError("rank authentication receipts do not agree")


def _distributed_bytes(record):
    raw = _canonical_bytes(record)
    return raw.replace(SCHEMA.encode(), DISTRIBUTED_SCHEMA.encode(), 1)


def _decode_distributed(raw, record_type):
    translated = raw.replace(DISTRIBUTED_SCHEMA.encode(), SCHEMA.encode(), 1)
    value = _decode_record(translated, record_type)
    value["schema"] = DISTRIBUTED_SCHEMA
    return value


def _fault(injector, transition):
    if injector is not None:
        injector(transition)


__all__ = [
    "AuthenticationReceipt", "CommitReceipt", "DistributedStateCoordinator",
    "GeneratedFamilySource", "LocalAuthenticatedState", "LocalRankEndpoint",
    "PrepareReceipt", "RankStateStore", "RestoreInspection",
]

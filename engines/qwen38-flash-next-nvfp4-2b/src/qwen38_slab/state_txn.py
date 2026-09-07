"""Atomic two-rank NVMe transactions for Qwen3.8 generated-token state.

This module owns a synchronous, externally serialized host protocol.  Callers
must quiesce generation at an accepted token boundary before ``commit`` and
must keep both rank stores on independent failure domains.  Payload inputs are
borrowed for the call and copied to anonymous aligned staging buffers.  Restore
returns one opaque accepted boundary plus owned immutable family snapshots to a
publish callback only after both ranks have been authenticated in full.

Expected validation and I/O failures raise :class:`StateTransactionError`.
Failed commits may leave PREPARED data, but such data is never restore-eligible.

OpenTelemetry cardinality: the only attributes are ``phase`` (six values),
``rank`` (0, 1, or -1 for joint publish), ``family`` (nine values plus ``none``), and ``outcome``
(two values).  Session and transaction identifiers are never attributes.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Protocol

PAGE_BYTES = 65_536
IO_CHUNK_BYTES = 64 * 1024 * 1024
SCHEMA = "rocket.qwen38-state-transaction.v1"

STATE_FAMILIES = (
    "target_full_attention_main_kv",
    "target_qsa_raw",
    "target_qsa_compressed",
    "target_gdn_conv",
    "target_gdn_recurrent",
    "ple_conv",
    "mtp_main_kv",
    "mtp_qsa_raw",
    "mtp_qsa_compressed",
)

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
_FAMILY_FILES = {name: f"{index:02d}-{name}.state" for index, name in enumerate(STATE_FAMILIES)}


class StateTransactionError(RuntimeError):
    """A validation, persistence, or authentication failure with no publication."""


class Transition(str, Enum):
    """Durable transitions exposed to deterministic fault injection."""

    AFTER_PREPARE_RANK0 = "after_prepare_rank0"
    AFTER_PREPARE_RANK1 = "after_prepare_rank1"
    AFTER_COMMIT_RANK0 = "after_commit_rank0"
    AFTER_COMMIT_RANK1 = "after_commit_rank1"
    AFTER_INDEX_RANK0 = "after_index_rank0"
    AFTER_INDEX_RANK1 = "after_index_rank1"


@dataclass(frozen=True)
class AcceptedBoundary:
    """A caller-owned snapshot boundary; ``token_hash`` is lowercase SHA-256."""

    token_count: int
    token_hash: str
    quiesced: bool


@dataclass(frozen=True)
class FamilyPayload:
    """Borrowed state at accepted T plus an ignored speculative suffix.

    Only ``accepted`` is persisted.  ``speculative_tail`` is intentionally
    excluded, and every byte after accepted data in its final page is zeroed.
    """

    accepted: bytes
    speculative_tail: bytes = b""


_AUTHENTICATED_SEAL = object()


class AuthenticatedState:
    """Owned immutable state emitted only after complete restore authentication.

    Construction is owned by :class:`StateTransactionStore`.  Consumers borrow
    the read-only mappings and immutable family bytes.  The boundary and bytes
    cannot be supplied independently to the runtime restore API.
    """

    __slots__ = ("_boundary", "_rank_payloads", "_seal")

    def __init__(self) -> None:
        raise TypeError("authenticated state is created only by StateTransactionStore")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("authenticated state is immutable")

    @classmethod
    def _from_verified(
        cls,
        *,
        token_count: int,
        token_hash: str,
        rank_payloads: Mapping[int, Mapping[str, FamilyPayload]],
    ) -> "AuthenticatedState":
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
            or not isinstance(token_hash, str)
            or not _HEX_256.fullmatch(token_hash)
            or tuple(rank_payloads) != (0, 1)
        ):
            raise StateTransactionError("verified state boundary or rank inventory is invalid")
        frozen_ranks: dict[int, Mapping[str, FamilyPayload]] = {}
        for rank in range(2):
            payloads = rank_payloads[rank]
            if tuple(payloads) != STATE_FAMILIES:
                raise StateTransactionError("verified state family inventory is invalid")
            copied: dict[str, FamilyPayload] = {}
            for family in STATE_FAMILIES:
                payload = payloads[family]
                if (
                    not isinstance(payload, FamilyPayload)
                    or not isinstance(payload.accepted, bytes)
                    or not payload.accepted
                    or payload.speculative_tail != b""
                ):
                    raise StateTransactionError("verified family payload is invalid")
                copied[family] = payload
            frozen_ranks[rank] = MappingProxyType(copied)
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "_boundary",
            AcceptedBoundary(token_count, token_hash, quiesced=True),
        )
        object.__setattr__(instance, "_rank_payloads", MappingProxyType(frozen_ranks))
        object.__setattr__(instance, "_seal", _AUTHENTICATED_SEAL)
        return instance

    @property
    def boundary(self) -> AcceptedBoundary:
        """Return the authenticated accepted-token boundary."""

        return self._boundary

    @property
    def rank_payloads(self) -> Mapping[int, Mapping[str, FamilyPayload]]:
        """Return the read-only two-rank payload snapshot."""

        return self._rank_payloads

    def rank_payload(self, rank: int) -> Mapping[str, FamilyPayload]:
        """Return one borrowed canonical rank mapping, or fail validation."""

        if isinstance(rank, bool) or rank not in (0, 1):
            raise StateTransactionError("authenticated state rank must be 0 or 1")
        return self._rank_payloads[rank]

    def _is_store_authenticated(self) -> bool:
        return getattr(self, "_seal", None) is _AUTHENTICATED_SEAL


@dataclass(frozen=True)
class StateIdentity:
    """Immutable restore identity for one Qwen revision and TP2 state layout."""

    revision: str
    precision_digest: str
    topology: str = "qwen3.8-flash-next:tp2:rank-native:v1"


class _Span(Protocol):
    def __enter__(self) -> "_Span": ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
    def set_attribute(self, key: str, value: str | int) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...


class OtelTracer(Protocol):
    def start_as_current_span(self, name: str) -> _Span: ...


FaultInjector = Callable[[Transition], None]
Publisher = Callable[[AuthenticatedState], None]


class _AlignedBuffer:
    """Owned anonymous mmap with one PAGE_BYTES-aligned writable view."""

    def __init__(self, length: int):
        if length <= 0 or length % PAGE_BYTES:
            raise StateTransactionError("direct-I/O buffer length must be positive and 65536-byte aligned")
        self._mapping = mmap.mmap(-1, length + PAGE_BYTES)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        displacement = (-base) % PAGE_BYTES
        self.view = memoryview(self._mapping)[displacement:displacement + length]
        address = ctypes.addressof(ctypes.c_char.from_buffer(self.view))
        if address % PAGE_BYTES:
            self.close()
            raise StateTransactionError("anonymous direct-I/O buffer address is not 65536-byte aligned")

    def close(self) -> None:
        if hasattr(self, "view"):
            self.view.release()
            del self.view
        self._mapping.close()

    def __enter__(self) -> "_AlignedBuffer":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class StateTransactionStore:
    """Single-owner host transaction coordinator for exactly two rank stores.

    ``commit`` creates files only beneath the two supplied stores.  It is not
    reentrant or process-safe; the caller must serialize transactions for a
    session.  A returned commit has two byte-identical COMMITTED records and two
    byte-identical session indexes.  ``restore`` performs direct reads into
    private staging and calls ``publish`` exactly once after all checks pass.
    """

    def __init__(self, rank_stores: tuple[Path, Path], identity: StateIdentity, tracer: OtelTracer):
        if len(rank_stores) != 2 or rank_stores[0] == rank_stores[1]:
            raise StateTransactionError("two distinct rank stores are required")
        if not _SAFE_ID.fullmatch(identity.revision) or not _HEX_256.fullmatch(identity.precision_digest):
            raise StateTransactionError("revision and lowercase SHA-256 precision digest are required")
        if identity.topology != "qwen3.8-flash-next:tp2:rank-native:v1":
            raise StateTransactionError("unsupported state topology")
        if tracer is None:
            raise StateTransactionError("an OpenTelemetry tracer is required")
        self._stores = tuple(Path(path) for path in rank_stores)
        self._identity = identity
        self._tracer = tracer

    def commit(
        self,
        session_id: str,
        transaction_id: str,
        boundary: AcceptedBoundary,
        rank_payloads: Mapping[int, Mapping[str, FamilyPayload]],
        inject_fault: FaultInjector | None = None,
    ) -> bytes:
        """Persist and publish one accepted boundary, or raise explicitly.

        Side effects before both session indexes are durable are ineligible
        transaction garbage.  A fault after the second index occurs after the
        transaction has become eligible and therefore does not roll it back.
        """
        self._validate_ids(session_id, transaction_id)
        self._validate_boundary(boundary)
        self._validate_payloads(rank_payloads)
        prepared_digests: dict[str, str] = {}
        for rank in range(2):
            prepared = self._prepare_rank(
                rank, session_id, transaction_id, boundary, rank_payloads[rank]
            )
            prepared_digests[str(rank)] = hashlib.sha256(prepared).hexdigest()
            self._fault(inject_fault, Transition(f"after_prepare_rank{rank}"))

        commit_record = {
            "schema": SCHEMA,
            "record": "COMMITTED",
            "session_id": session_id,
            "transaction_id": transaction_id,
            "revision": self._identity.revision,
            "precision_digest": self._identity.precision_digest,
            "topology": self._identity.topology,
            "token_count": boundary.token_count,
            "token_hash": boundary.token_hash,
            "families": list(STATE_FAMILIES),
            "prepared_sha256": prepared_digests,
        }
        commit_bytes = _canonical_bytes(commit_record)
        for rank in range(2):
            path = self._transaction_dir(rank, transaction_id) / "COMMITTED.json"
            with self._observed("commit", rank, "none"):
                _atomic_write(path, commit_bytes)
            self._fault(inject_fault, Transition(f"after_commit_rank{rank}"))

        commit_digest = hashlib.sha256(commit_bytes).hexdigest()
        index_bytes = _canonical_bytes({
            "schema": SCHEMA,
            "record": "SESSION_INDEX",
            "session_id": session_id,
            "transaction_id": transaction_id,
            "commit_sha256": commit_digest,
        })
        for rank in range(2):
            index = self._stores[rank] / "sessions" / session_id / "index.json"
            with self._observed("index", rank, "none"):
                _atomic_write(index, index_bytes)
            self._fault(inject_fault, Transition(f"after_index_rank{rank}"))
        return commit_bytes

    def restore(self, session_id: str, publish: Publisher) -> AuthenticatedState:
        """Authenticate a complete transaction, then invoke one pointer swap.

        The callback borrows the returned opaque authenticated snapshot.  Its
        boundary and family bytes remain paired through one read-only object.
        """
        self._validate_ids(session_id, "validation-only")
        if not callable(publish):
            raise StateTransactionError("publish callback is required")
        index_bytes = [self._read_bounded(store / "sessions" / session_id / "index.json")
                       for store in self._stores]
        if index_bytes[0] != index_bytes[1]:
            raise StateTransactionError("rank session indexes are not identical")
        index = _decode_record(index_bytes[0], "SESSION_INDEX")
        _require_keys(index, {
            "schema", "record", "session_id", "transaction_id", "commit_sha256",
        })
        if index.get("session_id") != session_id:
            raise StateTransactionError("session index identity mismatch")
        transaction_id = index.get("transaction_id")
        if not isinstance(transaction_id, str) or not _SAFE_ID.fullmatch(transaction_id):
            raise StateTransactionError("session index transaction ID is invalid")
        commit_bytes = [self._read_bounded(self._transaction_dir(rank, transaction_id) / "COMMITTED.json")
                        for rank in range(2)]
        if commit_bytes[0] != commit_bytes[1]:
            raise StateTransactionError("rank commit records are not identical")
        if hashlib.sha256(commit_bytes[0]).hexdigest() != index.get("commit_sha256"):
            raise StateTransactionError("session index commit checksum mismatch")
        commit = _decode_record(commit_bytes[0], "COMMITTED")
        self._validate_commit(commit, session_id, transaction_id)

        staged: dict[int, dict[str, FamilyPayload]] = {}
        for rank in range(2):
            prepared_bytes = self._read_bounded(self._transaction_dir(rank, transaction_id) / "PREPARED.json")
            if hashlib.sha256(prepared_bytes).hexdigest() != commit["prepared_sha256"].get(str(rank)):
                raise StateTransactionError("prepared record checksum mismatch")
            prepared = _decode_record(prepared_bytes, "PREPARED")
            self._validate_prepared(prepared, rank, session_id, transaction_id, commit)
            staged[rank] = {}
            rank_digest = hashlib.sha256()
            for expected_family, extent in zip(STATE_FAMILIES, prepared["family_table"], strict=True):
                family = extent.get("family")
                if family != expected_family:
                    raise StateTransactionError("prepared family order mismatch")
                payload, padded_digest = self._read_family(rank, transaction_id, extent)
                if padded_digest != extent.get("padded_sha256"):
                    raise StateTransactionError("family padded checksum mismatch")
                if hashlib.sha256(payload).hexdigest() != extent.get("logical_sha256"):
                    raise StateTransactionError("family logical checksum mismatch")
                rank_digest.update(expected_family.encode() + b"\0" + bytes.fromhex(padded_digest))
                staged[rank][expected_family] = FamilyPayload(payload)
            if rank_digest.hexdigest() != prepared.get("rank_sha256"):
                raise StateTransactionError("whole-rank checksum mismatch")
        authenticated = AuthenticatedState._from_verified(
            token_count=commit["token_count"],
            token_hash=commit["token_hash"],
            rank_payloads=staged,
        )
        with self._observed("publish", -1, "none"):
            publish(authenticated)
        return authenticated

    def _prepare_rank(
        self,
        rank: int,
        session_id: str,
        transaction_id: str,
        boundary: AcceptedBoundary,
        payloads: Mapping[str, FamilyPayload],
    ) -> bytes:
        transaction_dir = self._transaction_dir(rank, transaction_id)
        try:
            _ensure_directory(transaction_dir.parent)
            transaction_dir.mkdir(parents=True, exist_ok=False)
            (transaction_dir / "families").mkdir()
        except OSError as exc:
            raise StateTransactionError("cannot create unique transaction directory") from exc
        table = []
        rank_digest = hashlib.sha256()
        for family in STATE_FAMILIES:
            accepted = payloads[family].accepted
            stored_bytes = _align_up(len(accepted), PAGE_BYTES)
            filename = _FAMILY_FILES[family]
            padded_digest = self._write_family(
                rank, family, transaction_dir / "families" / filename, accepted, stored_bytes
            )
            logical_digest = hashlib.sha256(accepted).hexdigest()
            rank_digest.update(family.encode() + b"\0" + bytes.fromhex(padded_digest))
            table.append({
                "family": family,
                "file": filename,
                "offset_bytes": 0,
                "logical_bytes": len(accepted),
                "length_bytes": stored_bytes,
                "logical_sha256": logical_digest,
                "padded_sha256": padded_digest,
            })
        _fsync_directory(transaction_dir / "families")
        record = {
            "schema": SCHEMA,
            "record": "PREPARED",
            "rank": rank,
            "session_id": session_id,
            "transaction_id": transaction_id,
            "revision": self._identity.revision,
            "precision_digest": self._identity.precision_digest,
            "topology": self._identity.topology,
            "token_count": boundary.token_count,
            "token_hash": boundary.token_hash,
            "family_table": table,
            "rank_sha256": rank_digest.hexdigest(),
        }
        prepared = _canonical_bytes(record)
        with self._observed("prepare", rank, "none"):
            _atomic_write(transaction_dir / "PREPARED.json", prepared)
            _fsync_directory(transaction_dir.parent)
        return prepared

    def _write_family(self, rank: int, family: str, path: Path, payload: bytes, length: int) -> str:
        flags = getattr(os, "O_DIRECT", None)
        if flags is None:
            raise StateTransactionError("O_DIRECT is unavailable")
        fd = -1
        with self._observed("write", rank, family):
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | flags, 0o600)
                digest = hashlib.sha256()
                for offset in range(0, length, IO_CHUNK_BYTES):
                    chunk_bytes = min(IO_CHUNK_BYTES, length - offset)
                    with _AlignedBuffer(chunk_bytes) as buffer:
                        buffer.view[:] = b"\0" * chunk_bytes
                        accepted_bytes = min(chunk_bytes, max(0, len(payload) - offset))
                        if accepted_bytes:
                            buffer.view[:accepted_bytes] = payload[offset:offset + accepted_bytes]
                        count = os.pwritev(fd, [buffer.view], offset)
                        if count != chunk_bytes:
                            raise StateTransactionError(f"short O_DIRECT write: {count}/{chunk_bytes}")
                        digest.update(buffer.view)
                os.fdatasync(fd)
                return digest.hexdigest()
            except OSError as exc:
                raise StateTransactionError("O_DIRECT pwritev failed; fallback is forbidden") from exc
            finally:
                if fd >= 0:
                    os.close(fd)

    def _read_family(self, rank: int, transaction_id: str, extent: object) -> tuple[bytes, str]:
        if not isinstance(extent, dict):
            raise StateTransactionError("family extent is not an object")
        family = extent.get("family")
        filename = extent.get("file")
        offset = extent.get("offset_bytes")
        logical = extent.get("logical_bytes")
        length = extent.get("length_bytes")
        if (
            family not in STATE_FAMILIES
            or filename != _FAMILY_FILES[family]
            or isinstance(offset, bool) or offset != 0
            or isinstance(logical, bool) or not isinstance(logical, int) or logical <= 0
            or isinstance(length, bool) or not isinstance(length, int) or length < logical
            or length % PAGE_BYTES
        ):
            raise StateTransactionError("family extent violates 65536-byte contract")
        path = self._transaction_dir(rank, transaction_id) / "families" / filename
        flags = getattr(os, "O_DIRECT", None)
        if flags is None:
            raise StateTransactionError("O_DIRECT is unavailable")
        fd = -1
        with self._observed("read", rank, family):
            try:
                fd = os.open(path, os.O_RDONLY | flags)
                if os.fstat(fd).st_size != length:
                    raise StateTransactionError("family file size does not match extent")
                payload = bytearray(logical)
                digest = hashlib.sha256()
                for chunk_offset in range(0, length, IO_CHUNK_BYTES):
                    chunk_bytes = min(IO_CHUNK_BYTES, length - chunk_offset)
                    with _AlignedBuffer(chunk_bytes) as buffer:
                        count = os.preadv(fd, [buffer.view], offset + chunk_offset)
                        if count != chunk_bytes:
                            raise StateTransactionError(f"short O_DIRECT read: {count}/{chunk_bytes}")
                        accepted_bytes = min(chunk_bytes, max(0, logical - chunk_offset))
                        if accepted_bytes:
                            payload[chunk_offset:chunk_offset + accepted_bytes] = buffer.view[:accepted_bytes]
                        if any(buffer.view[accepted_bytes:]):
                            raise StateTransactionError("speculative tail or page padding is not canonical zero")
                        digest.update(buffer.view)
                return bytes(payload), digest.hexdigest()
            except OSError as exc:
                raise StateTransactionError("O_DIRECT preadv failed; fallback is forbidden") from exc
            finally:
                if fd >= 0:
                    os.close(fd)

    def _validate_commit(self, record: dict[str, object], session: str, transaction: str) -> None:
        _require_keys(record, {
            "schema", "record", "session_id", "transaction_id", "revision",
            "precision_digest", "topology", "token_count", "token_hash",
            "families", "prepared_sha256",
        })
        expected = {
            "session_id": session,
            "transaction_id": transaction,
            "revision": self._identity.revision,
            "precision_digest": self._identity.precision_digest,
            "topology": self._identity.topology,
            "families": list(STATE_FAMILIES),
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise StateTransactionError("commit identity, topology, or family table mismatch")
        self._validate_token_fields(record)
        digests = record.get("prepared_sha256")
        if not isinstance(digests, dict) or set(digests) != {"0", "1"} or any(
            not isinstance(value, str) or not _HEX_256.fullmatch(value) for value in digests.values()
        ):
            raise StateTransactionError("commit prepared checksums are invalid")

    def _validate_prepared(
        self, record: dict[str, object], rank: int, session: str, transaction: str,
        commit: dict[str, object],
    ) -> None:
        _require_keys(record, {
            "schema", "record", "rank", "session_id", "transaction_id",
            "revision", "precision_digest", "topology", "token_count",
            "token_hash", "family_table", "rank_sha256",
        })
        expected = {
            "rank": rank,
            "session_id": session,
            "transaction_id": transaction,
            "revision": self._identity.revision,
            "precision_digest": self._identity.precision_digest,
            "topology": self._identity.topology,
            "token_count": commit["token_count"],
            "token_hash": commit["token_hash"],
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise StateTransactionError("prepared identity, boundary, or topology mismatch")
        table = record.get("family_table")
        if not isinstance(table, list) or len(table) != len(STATE_FAMILIES):
            raise StateTransactionError("prepared family table is incomplete")
        for extent in table:
            if not isinstance(extent, dict):
                raise StateTransactionError("prepared family extent is not an object")
            _require_keys(extent, {
                "family", "file", "offset_bytes", "logical_bytes", "length_bytes",
                "logical_sha256", "padded_sha256",
            })
        if not isinstance(record.get("rank_sha256"), str) or not _HEX_256.fullmatch(record["rank_sha256"]):
            raise StateTransactionError("whole-rank checksum is invalid")

    @staticmethod
    def _validate_ids(session_id: str, transaction_id: str) -> None:
        if not _SAFE_ID.fullmatch(session_id) or not _SAFE_ID.fullmatch(transaction_id):
            raise StateTransactionError("session and transaction IDs must be safe bounded path components")

    @staticmethod
    def _validate_boundary(boundary: AcceptedBoundary) -> None:
        if (
            not boundary.quiesced
            or isinstance(boundary.token_count, bool)
            or not isinstance(boundary.token_count, int)
            or boundary.token_count < 0
            or not _HEX_256.fullmatch(boundary.token_hash)
        ):
            raise StateTransactionError("boundary must be quiesced with nonnegative T and token SHA-256")

    @staticmethod
    def _validate_payloads(payloads: Mapping[int, Mapping[str, FamilyPayload]]) -> None:
        if set(payloads) != {0, 1}:
            raise StateTransactionError("rank payloads must contain exactly ranks 0 and 1")
        for rank in range(2):
            if tuple(payloads[rank].keys()) != STATE_FAMILIES:
                raise StateTransactionError("rank payloads must use the canonical nine-family order")
            for payload in payloads[rank].values():
                if not isinstance(payload, FamilyPayload) or not isinstance(payload.accepted, bytes) or not payload.accepted:
                    raise StateTransactionError("every accepted family payload must be nonempty immutable bytes")
                if not isinstance(payload.speculative_tail, bytes):
                    raise StateTransactionError("speculative tails must be immutable bytes")

    @staticmethod
    def _validate_token_fields(record: Mapping[str, object]) -> None:
        count = record.get("token_count")
        digest = record.get("token_hash")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise StateTransactionError("committed token count is invalid")
        if not isinstance(digest, str) or not _HEX_256.fullmatch(digest):
            raise StateTransactionError("committed token hash is invalid")

    def _transaction_dir(self, rank: int, transaction_id: str) -> Path:
        return self._stores[rank] / "transactions" / transaction_id

    @staticmethod
    def _read_bounded(path: Path, maximum: int = 1_048_576) -> bytes:
        fd = -1
        try:
            fd = os.open(path, os.O_RDONLY)
            size = os.fstat(fd).st_size
            if size <= 0 or size > maximum:
                raise StateTransactionError("transaction record size is outside bounds")
            chunks = []
            remaining = size
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    raise StateTransactionError("transaction record was truncated during read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise StateTransactionError("transaction record grew during bounded read")
            return b"".join(chunks)
        except OSError as exc:
            raise StateTransactionError("required transaction record is unavailable") from exc
        finally:
            if fd >= 0:
                os.close(fd)

    @staticmethod
    def _fault(injector: FaultInjector | None, transition: Transition) -> None:
        if injector is not None:
            injector(transition)

    @contextmanager
    def _observed(self, phase: str, rank: int, family: str) -> Iterator[None]:
        with self._tracer.start_as_current_span("rocket.qwen38.state.transaction") as span:
            span.set_attribute("phase", phase)
            span.set_attribute("rank", rank)
            span.set_attribute("family", family)
            try:
                yield
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                raise
            else:
                span.set_attribute("outcome", "success")


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _canonical_bytes(record: object) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _decode_record(raw: bytes, record_type: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise StateTransactionError("transaction record contains duplicate fields")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StateTransactionError("transaction record is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("record") != record_type:
        raise StateTransactionError(f"invalid {record_type} transaction record")
    return value


def _require_keys(record: Mapping[str, object], expected: set[str]) -> None:
    if set(record) != expected:
        raise StateTransactionError("transaction record field set is invalid")


def _atomic_write(path: Path, payload: bytes) -> None:
    """Persist one file with temp+fdatasync+rename+directory fsync."""
    try:
        _ensure_directory(path.parent)
        suffix = hashlib.sha256(payload).hexdigest()[:16]
        temporary = path.with_name(f"{path.name}.tmp-{suffix}")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view = memoryview(payload)
            try:
                written = 0
                while written < len(view):
                    count = os.write(fd, view[written:])
                    if count <= 0:
                        raise StateTransactionError("short atomic metadata write")
                    written += count
                os.fdatasync(fd)
            finally:
                view.release()
        finally:
            os.close(fd)
        os.rename(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise StateTransactionError("atomic metadata persistence failed") from exc


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise StateTransactionError("directory fsync failed") from exc


def _ensure_directory(path: Path) -> None:
    """Create a directory chain and make every new directory entry durable."""
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if not cursor.is_dir():
        raise StateTransactionError("transaction directory ancestor is not a directory")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except OSError as exc:
            raise StateTransactionError("cannot create transaction directory") from exc
        _fsync_directory(directory.parent)

"""Accepted-prefix transaction owner for native Qwen3.8 MTP execution.

The native executor computes every requested draft step in one coarse call and
keeps a causal snapshot for each prefix.  This ledger publishes only snapshots
selected by target verification.  It never executes model work and never loops
over draft steps on the Python hot path.

The object has one logical writer and is not thread-safe.  Inputs are borrowed
for each call.  Returned snapshots and publications are immutable owned values.
Validation failures before publication leave the current phase unchanged.  A
native causal-boundary mismatch faults the ledger because device state may have
been published and the executor must be reconstructed.

OpenTelemetry attributes are bounded: operation is prepare or commit, depth is
k0 through k7, batch is 1, 2, 4, 8, or 16, and outcome is success or failure.
Sequence slots, token IDs, prefix digests, and generations are excluded.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Protocol

from .controller_restore import DecoderContinuation
from .decode import Depth, GRAPH_BATCHES


class DraftTransactionError(RuntimeError):
    """Accepted-prefix identity or native causal publication failed."""


class DraftTransactionPhase(str, Enum):
    IDLE = "idle"
    PENDING = "pending"
    FAULTED = "faulted"


@dataclass(frozen=True)
class DraftSequenceState:
    """One sequence's accepted target count and published MTP cache count."""

    slot: int
    accepted_tokens: int
    mtp_cached_tokens: int
    native_generation: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.slot, bool)
            or not isinstance(self.slot, int)
            or not 0 <= self.slot < 16
            or isinstance(self.accepted_tokens, bool)
            or not isinstance(self.accepted_tokens, int)
            or not 0 <= self.accepted_tokens <= 262_144
            or isinstance(self.mtp_cached_tokens, bool)
            or not isinstance(self.mtp_cached_tokens, int)
            or self.mtp_cached_tokens not in (
                self.accepted_tokens,
                max(0, self.accepted_tokens - 1),
            )
            or isinstance(self.native_generation, bool)
            or not isinstance(self.native_generation, int)
            or self.native_generation < 0
        ):
            raise DraftTransactionError("draft sequence causal state is invalid")


@dataclass(frozen=True)
class DraftBatchSnapshot:
    """Private native result awaiting target verification."""

    transaction: int
    prefix: DecoderContinuation
    depth: Depth
    sequences: tuple[DraftSequenceState, ...]
    proposal_tokens: tuple[tuple[int, ...], ...]
    native_generation: int


@dataclass(frozen=True)
class AcceptedDraftPublication:
    """Owned accepted state after native snapshot publication."""

    prefix: DecoderContinuation
    sequences: tuple[DraftSequenceState, ...]
    accepted_proposals: tuple[tuple[int, ...], ...]
    native_generation: int


class AcceptedPrefixSource(Protocol):
    @property
    def continuation(self) -> DecoderContinuation | None: ...


class _Span(Protocol):
    def __enter__(self) -> "_Span": ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
    def set_attribute(self, key: str, value: str | int) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...


class OtelTracer(Protocol):
    def start_as_current_span(self, name: str) -> _Span: ...


class DraftStateLedger:
    """Single-writer owner of one pending speculative state transaction."""

    def __init__(self, prefix_source: AcceptedPrefixSource, tracer: OtelTracer):
        if (
            prefix_source is None
            or not hasattr(prefix_source, "continuation")
            or tracer is None
            or not callable(getattr(tracer, "start_as_current_span", None))
        ):
            raise DraftTransactionError("accepted-prefix source and OTEL tracer are required")
        self._prefix_source = prefix_source
        self._tracer = tracer
        self._phase = DraftTransactionPhase.IDLE
        self._pending: DraftBatchSnapshot | None = None
        self._next_transaction = 1

    @property
    def phase(self) -> DraftTransactionPhase:
        return self._phase

    def prepare(
        self,
        *,
        prefix: DecoderContinuation,
        sequences: tuple[DraftSequenceState, ...],
        depth: Depth,
        proposal_tokens: tuple[tuple[int, ...], ...],
        native_generation: int,
    ) -> DraftBatchSnapshot:
        """Retain a complete native result without publishing causal state."""

        safe_depth = _safe_depth(depth)
        safe_batch = len(sequences) if isinstance(sequences, tuple) else 0
        with self._observed("prepare", safe_depth, safe_batch):
            if self._phase is DraftTransactionPhase.FAULTED:
                raise DraftTransactionError("faulted draft ledger cannot be reused")
            if self._phase is not DraftTransactionPhase.IDLE or self._pending is not None:
                raise DraftTransactionError("a draft transaction is already pending")
            current = self._current_prefix()
            if not _valid_prefix(prefix) or prefix is not current:
                raise DraftTransactionError("exact current accepted prefix is required")
            try:
                if isinstance(depth, bool):
                    raise ValueError("boolean")
                selected = Depth(depth)
            except (TypeError, ValueError) as exc:
                raise DraftTransactionError("draft depth must be K1 through K7") from exc
            if selected is Depth.K0:
                raise DraftTransactionError("K0 is target-only and has no draft transaction")
            _validate_sequences(sequences, selected)
            if (
                not isinstance(proposal_tokens, tuple)
                or len(proposal_tokens) != len(sequences)
                or any(
                    not isinstance(row, tuple)
                    or len(row) != int(selected)
                    or any(
                        isinstance(token, bool)
                        or not isinstance(token, int)
                        or not 0 <= token < 248_320
                        for token in row
                    )
                    for row in proposal_tokens
                )
            ):
                raise DraftTransactionError("proposal width or token range changed")
            if (
                isinstance(native_generation, bool)
                or not isinstance(native_generation, int)
                or native_generation <= max(item.native_generation for item in sequences)
            ):
                raise DraftTransactionError("native draft generation did not advance")
            snapshot = DraftBatchSnapshot(
                self._next_transaction,
                prefix,
                selected,
                tuple(sequences),
                tuple(tuple(row) for row in proposal_tokens),
                native_generation,
            )
            self._next_transaction += 1
            self._pending = snapshot
            self._phase = DraftTransactionPhase.PENDING
            return snapshot

    def commit(
        self,
        snapshot: DraftBatchSnapshot,
        *,
        accepted_widths: tuple[int, ...],
        published_mtp_tokens: tuple[int, ...],
        prefix: DecoderContinuation,
    ) -> AcceptedDraftPublication:
        """Publish only target-accepted native snapshots and clear the transaction."""

        depth = _safe_depth(getattr(snapshot, "depth", Depth.K0))
        batch = len(snapshot.sequences) if isinstance(snapshot, DraftBatchSnapshot) else 0
        with self._observed("commit", depth, batch):
            if self._phase is DraftTransactionPhase.FAULTED:
                raise DraftTransactionError("faulted draft ledger cannot be reused")
            if self._phase is not DraftTransactionPhase.PENDING or self._pending is None:
                raise DraftTransactionError("no draft transaction is pending")
            if snapshot is not self._pending:
                raise DraftTransactionError("commit does not name the pending snapshot")
            if (
                not isinstance(accepted_widths, tuple)
                or len(accepted_widths) != len(snapshot.sequences)
                or any(
                    isinstance(width, bool)
                    or not isinstance(width, int)
                    or not 1 <= width <= int(snapshot.depth) + 1
                    for width in accepted_widths
                )
                or not isinstance(published_mtp_tokens, tuple)
                or len(published_mtp_tokens) != len(snapshot.sequences)
            ):
                raise DraftTransactionError("accepted-prefix widths are invalid")
            current = self._current_prefix()
            if (
                not _valid_prefix(prefix)
                or prefix is not current
                or prefix is snapshot.prefix
                or prefix.generation_epoch <= snapshot.prefix.generation_epoch
                or prefix.token_count <= snapshot.prefix.token_count
            ):
                raise DraftTransactionError("a new accepted prefix must be published first")
            expected_mtp = tuple(
                state.mtp_cached_tokens + min(width, int(snapshot.depth))
                for state, width in zip(snapshot.sequences, accepted_widths, strict=True)
            )
            if published_mtp_tokens != expected_mtp:
                self._phase = DraftTransactionPhase.FAULTED
                raise DraftTransactionError("native MTP causal boundary does not match acceptance")
            if any(
                state.accepted_tokens + width > 262_144
                for state, width in zip(snapshot.sequences, accepted_widths, strict=True)
            ):
                self._phase = DraftTransactionPhase.FAULTED
                raise DraftTransactionError("accepted draft publication exceeds context")
            next_states = tuple(
                DraftSequenceState(
                    state.slot,
                    state.accepted_tokens + width,
                    mtp_tokens,
                    snapshot.native_generation,
                )
                for state, width, mtp_tokens in zip(
                    snapshot.sequences,
                    accepted_widths,
                    published_mtp_tokens,
                    strict=True,
                )
            )
            accepted = tuple(
                row[: width - 1]
                for row, width in zip(
                    snapshot.proposal_tokens, accepted_widths, strict=True
                )
            )
            publication = AcceptedDraftPublication(
                prefix, next_states, accepted, snapshot.native_generation
            )
            self._pending = None
            self._phase = DraftTransactionPhase.IDLE
            return publication

    def _current_prefix(self) -> DecoderContinuation:
        try:
            current = self._prefix_source.continuation
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            raise DraftTransactionError("accepted-prefix source failed") from exc
        if not isinstance(current, DecoderContinuation):
            raise DraftTransactionError("accepted-prefix source has no publication")
        return current

    @contextmanager
    def _observed(self, operation: str, depth: Depth, batch: int) -> Iterator[None]:
        with self._tracer.start_as_current_span(
            "rocket.qwen38.mtp.transaction"
        ) as span:
            span.set_attribute("operation", operation)
            span.set_attribute("depth", f"k{int(depth)}")
            span.set_attribute("batch", batch if batch in GRAPH_BATCHES else 0)
            try:
                yield
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                raise
            else:
                span.set_attribute("outcome", "success")


def _validate_sequences(
    sequences: tuple[DraftSequenceState, ...], depth: Depth
) -> None:
    if (
        not isinstance(sequences, tuple)
        or len(sequences) not in GRAPH_BATCHES
        or any(not isinstance(value, DraftSequenceState) for value in sequences)
        or tuple(item.slot for item in sequences)
        != tuple(sorted({item.slot for item in sequences}))
        or any(item.accepted_tokens + int(depth) + 1 > 262_144 for item in sequences)
    ):
        raise DraftTransactionError("draft sequence batch contract changed")


def _safe_depth(value: object) -> Depth:
    try:
        if isinstance(value, bool):
            raise ValueError("boolean")
        return Depth(value)
    except (TypeError, ValueError):
        return Depth.K0


def _valid_prefix(value: object) -> bool:
    if not isinstance(value, DecoderContinuation):
        return False
    digests = (value.token_hash, value.commit_sha256, value.policy_digest)
    return (
        not isinstance(value.token_count, bool)
        and isinstance(value.token_count, int)
        and 0 <= value.token_count <= 262_144
        and not isinstance(value.generation_epoch, bool)
        and isinstance(value.generation_epoch, int)
        and value.generation_epoch > 0
        and all(
            isinstance(item, str)
            and len(item) == 64
            and set(item) <= set("0123456789abcdef")
            for item in digests
        )
    )


__all__ = [
    "AcceptedDraftPublication",
    "AcceptedPrefixSource",
    "DraftBatchSnapshot",
    "DraftSequenceState",
    "DraftStateLedger",
    "DraftTransactionError",
    "DraftTransactionPhase",
]

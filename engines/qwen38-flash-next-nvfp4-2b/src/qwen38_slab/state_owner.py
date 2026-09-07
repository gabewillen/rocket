"""Decoder-owned accepted-state gate and two-rank restore coordination.

All specialized decoder launches pass through :class:`DecoderStateOwner`.
The owner binds an accepted K0 publication to one scheduler boundary and owns
the complete nine-family CUDA pointer table used by ``TorchCudaRuntime``.
Publication is possible only while the matching launch gate is closed.

The coordinator validates both rank boundaries before either runtime binding
is called.  It holds both launch gates across the two synchronous restores.
Any failure after those holds faults both owners closed, preventing execution
through a mixed rank publication.  Cross-rank pointer replacement remains two
ordered local publications rather than a simultaneous CUDA operation.

OpenTelemetry attributes have finite cardinality: ``phase`` (fifteen values),
``rank`` (-1, 0, or 1), and ``outcome`` (two values).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Iterator, Protocol, runtime_checkable

from .decode import PreparedDecode
from .device_decode import DevicePhase, DevicePublication
from .runtime_state import RuntimeBoundary
from .state_txn import AuthenticatedState, OtelTracer, STATE_FAMILIES

_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")


class DecoderStateOwnerError(RuntimeError):
    """Validation or ownership failure at the decoder launch boundary."""


class OwnerPhase(str, Enum):
    """Externally observable launch-gate state."""

    OPEN = "open"
    CLOSED = "closed"
    FAULTED = "faulted"


class CoordinatorPhase(str, Enum):
    """Externally observable two-rank coordinator state."""

    IDLE = "idle"
    RESTORING = "restoring"
    FAULTED = "faulted"


class DecoderBinding(Protocol):
    """Specialized synchronous decoder surface owned by the gate."""

    @property
    def phase(self) -> DevicePhase: ...

    @property
    def publication(self) -> DevicePublication | None: ...

    def upload_and_launch(self, prepared: PreparedDecode) -> DevicePublication: ...


class RankRuntimeBinding(Protocol):
    """One authenticated rank restore binding."""

    @property
    def rank(self) -> int: ...

    @property
    def state_owner(self) -> object | None: ...

    def restore(
        self, authenticated: AuthenticatedState, generation_epoch: int
    ) -> None: ...


@runtime_checkable
class CoordinatedStateOwner(Protocol):
    """Rank owner surface, including a synchronous physical-node proxy."""

    @property
    def rank(self) -> int: ...

    @property
    def accepted_boundary(self) -> RuntimeBoundary | None: ...

    def hold_launch_gate(self, boundary: RuntimeBoundary) -> None: ...
    def release_launch_gate(self, boundary: RuntimeBoundary) -> None: ...
    def fault_closed(self, boundary: RuntimeBoundary) -> None: ...


class DecoderStateOwner:
    """Single-owner gate joining decoder launches and the state pointer table.

    Calls are synchronous, externally serialized, and non-reentrant.  A
    coordinator hold keeps the gate closed while each rank's runtime adapter
    independently closes and reopens its local restore hold.
    """

    _DECODER_METHODS = ("upload_and_launch",)

    def __init__(self, *, rank: int, decoder: DecoderBinding, tracer: OtelTracer):
        _validate_rank(rank)
        if decoder is None or any(
            not callable(getattr(decoder, name, None))
            for name in self._DECODER_METHODS
        ):
            raise DecoderStateOwnerError("decoder binding contract is incomplete")
        if not hasattr(decoder, "phase") or not hasattr(decoder, "publication"):
            raise DecoderStateOwnerError("decoder binding ownership is incomplete")
        if tracer is None or not callable(
            getattr(tracer, "start_as_current_span", None)
        ):
            raise DecoderStateOwnerError("an OpenTelemetry tracer is required")
        self._rank = rank
        self._decoder = decoder
        self._tracer = tracer
        self._phase = OwnerPhase.OPEN
        self._accepted_boundary: RuntimeBoundary | None = None
        self._closed_boundary: RuntimeBoundary | None = None
        self._runtime_hold = False
        self._coordinator_hold = False
        self._active_state: Mapping[str, object] | None = None
        self._active_policy_state: bytes | None = None
        self._active_commit_sha256: str | None = None
        self._prepared_state: Mapping[str, object] | None = None
        self._prepared_policy_state: bytes | None = None
        self._prepared_commit_sha256: str | None = None
        self._rollback_state: Mapping[str, object] | None = None
        self._rollback_policy_state: bytes | None = None
        self._rollback_commit_sha256: str | None = None
        self._prepared_committed = False
        self._publication_lock = RLock()
        self._publication_lock_bound = False

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def phase(self) -> OwnerPhase:
        return self._phase

    @property
    def faulted(self) -> bool:
        return self._phase is OwnerPhase.FAULTED

    @property
    def accepted_boundary(self) -> RuntimeBoundary | None:
        return self._accepted_boundary

    @property
    def active_state(self) -> Mapping[str, object] | None:
        return self._active_state

    @property
    def active_policy_state(self) -> bytes | None:
        return self._active_policy_state

    @property
    def active_commit_sha256(self) -> str | None:
        return self._active_commit_sha256

    def upload_and_launch(self, prepared: PreparedDecode) -> DevicePublication:
        """Launch only while open; success awaits explicit boundary acceptance."""

        with self._publication_lock:
            with self._observed("launch"):
                self._require_open()
                publication = self._decoder.upload_and_launch(prepared)
                if (
                    not isinstance(publication, DevicePublication)
                    or publication is not self._decoder.publication
                ):
                    self._fault()
                    raise DecoderStateOwnerError(
                        "decoder launch returned a non-current publication"
                    )
                self._accepted_boundary = None
                return publication

    def accept_boundary(
        self, publication: DevicePublication, boundary: RuntimeBoundary
    ) -> None:
        """Authenticate the current decoder publication as scheduler-accepted."""

        with self._publication_lock:
            with self._observed("accept"):
                self._require_open()
                _validate_boundary(boundary)
                if (
                    not isinstance(publication, DevicePublication)
                    or publication is not self._decoder.publication
                    or publication.generation != boundary.generation_epoch
                    or self._decoder.phase is not DevicePhase.IDLE
                ):
                    raise DecoderStateOwnerError(
                        "accepted boundary does not describe the current decoder publication"
                    )
                if (
                    self._accepted_boundary is not None
                    and self._accepted_boundary != boundary
                ):
                    raise DecoderStateOwnerError(
                        "current decoder publication already has another boundary"
                    )
                self._accepted_boundary = boundary

    def _bind_publication_lock(self, lock) -> None:
        """Bind both TP ranks to the coordinator's decoder-admission lock."""

        if not hasattr(lock, "acquire") or not hasattr(lock, "release"):
            raise DecoderStateOwnerError("publication lock contract is invalid")
        if self._publication_lock_bound:
            raise DecoderStateOwnerError("publication lock is already bound")
        self._publication_lock = lock
        self._publication_lock_bound = True

    def hold_launch_gate(self, boundary: RuntimeBoundary) -> None:
        """Hold the matching rank gate for an entire coordinator restore."""

        with self._observed("hold"):
            self._require_matching_accepted(boundary)
            if self._runtime_hold or self._coordinator_hold:
                raise DecoderStateOwnerError("decoder launch gate is already closed")
            self._coordinator_hold = True
            self._closed_boundary = boundary
            self._phase = OwnerPhase.CLOSED

    def close_launch_gate(self, boundary: RuntimeBoundary) -> RuntimeBoundary:
        """Close the local runtime hold after matching accepted publication."""

        with self._observed("close"):
            self._require_matching_accepted(boundary)
            if self._runtime_hold:
                raise DecoderStateOwnerError("runtime launch gate is already closed")
            if self._coordinator_hold:
                if self._closed_boundary != boundary:
                    raise DecoderStateOwnerError("coordinator boundary does not match")
            else:
                self._closed_boundary = boundary
                self._phase = OwnerPhase.CLOSED
            self._runtime_hold = True
            return boundary

    def publish_state(
        self, staged: Mapping[str, object], boundary: RuntimeBoundary
    ) -> None:
        """Replace the complete nine-family table in one owner assignment."""

        with self._observed("publish"):
            if self._phase is OwnerPhase.FAULTED:
                raise DecoderStateOwnerError("decoder launch gate is faulted")
            if not self._runtime_hold or self._closed_boundary != boundary:
                raise DecoderStateOwnerError(
                    "state publication requires the matching closed runtime gate"
                )
            if not isinstance(staged, Mapping) or tuple(staged) != STATE_FAMILIES:
                raise DecoderStateOwnerError(
                    "state publication requires the canonical nine-family table"
                )
            copied = {family: staged[family] for family in STATE_FAMILIES}
            if any(value is None for value in copied.values()):
                raise DecoderStateOwnerError(
                    "state publication contains an invalid family pointer"
                )
            self._active_state = MappingProxyType(copied)

    def prepare_state_with_policy(
        self, staged: Mapping[str, object], policy_state: bytes,
        boundary: RuntimeBoundary, commit_sha256: str,
    ) -> None:
        """Validate one inactive generation without changing live pointers."""

        with self._observed("prepare_state"):
            if self._phase is OwnerPhase.FAULTED:
                raise DecoderStateOwnerError("decoder launch gate is faulted")
            if not self._runtime_hold or self._closed_boundary != boundary:
                raise DecoderStateOwnerError(
                    "state publication requires the matching closed runtime gate"
                )
            if not isinstance(staged, Mapping) or tuple(staged) != STATE_FAMILIES:
                raise DecoderStateOwnerError(
                    "state publication requires the canonical nine-family table"
                )
            copied = {family: staged[family] for family in STATE_FAMILIES}
            if any(value is None for value in copied.values()):
                raise DecoderStateOwnerError(
                    "state publication contains an invalid family pointer"
                )
            if not isinstance(policy_state, bytes) or not 0 < len(policy_state) <= 65_536:
                raise DecoderStateOwnerError("canonical adaptive policy state is required")
            if not isinstance(commit_sha256, str) or not _HEX_256.fullmatch(commit_sha256):
                raise DecoderStateOwnerError("durable commit digest is required")
            if self._prepared_state is not None:
                raise DecoderStateOwnerError("inactive state generation already exists")
            self._prepared_state = MappingProxyType(copied)
            self._prepared_policy_state = policy_state
            self._prepared_commit_sha256 = commit_sha256
            self._prepared_committed = False

    def commit_prepared_state(
        self, boundary: RuntimeBoundary, commit_sha256: str
    ) -> None:
        """Swap the validated inactive generation while retaining rollback state."""

        with self._observed("commit_state"):
            self._require_prepared(boundary, commit_sha256)
            if self._prepared_committed:
                raise DecoderStateOwnerError("inactive state generation already committed")
            self._rollback_state = self._active_state
            self._rollback_policy_state = self._active_policy_state
            self._rollback_commit_sha256 = self._active_commit_sha256
            self._active_state = self._prepared_state
            self._active_policy_state = self._prepared_policy_state
            self._active_commit_sha256 = self._prepared_commit_sha256
            self._prepared_committed = True

    def rollback_prepared_state(
        self, boundary: RuntimeBoundary, commit_sha256: str
    ) -> None:
        """Discard or undo one inactive generation before the gate can reopen."""

        with self._observed("rollback_state"):
            self._require_prepared(boundary, commit_sha256)
            if self._prepared_committed:
                self._active_state = self._rollback_state
                self._active_policy_state = self._rollback_policy_state
                self._active_commit_sha256 = self._rollback_commit_sha256
            self._clear_prepared()

    def finalize_prepared_state(
        self, boundary: RuntimeBoundary, commit_sha256: str
    ) -> None:
        """Acknowledge a committed generation while retaining rollback state."""

        with self._observed("finalize_state"):
            self._require_prepared(boundary, commit_sha256)
            if not self._prepared_committed:
                raise DecoderStateOwnerError("inactive state generation is not committed")

    def validate_launch_gate_release(self, boundary: RuntimeBoundary) -> None:
        """Preflight a coordinator release before the global commit point."""

        with self._observed("validate_release"):
            self._require_launch_gate_release(boundary)

    def _commit_launch_gate_release(self, boundary: RuntimeBoundary) -> None:
        """Apply a prevalidated release without a remaining failure branch."""

        self._coordinator_hold = False
        self._closed_boundary = None
        self._phase = OwnerPhase.OPEN

    def _discard_prepared_rollback(self) -> None:
        """Drop undo state after the coordinator's global commit point."""

        self._clear_prepared()

    def open_launch_gate(self, boundary: RuntimeBoundary) -> None:
        """Release the runtime hold; a coordinator hold may remain active."""

        with self._observed("open"):
            if self._phase is OwnerPhase.FAULTED:
                raise DecoderStateOwnerError("decoder launch gate is faulted")
            if not self._runtime_hold or self._closed_boundary != boundary:
                raise DecoderStateOwnerError("runtime launch gate boundary does not match")
            self._runtime_hold = False
            if not self._coordinator_hold:
                self._closed_boundary = None
                self._phase = OwnerPhase.OPEN

    def release_launch_gate(self, boundary: RuntimeBoundary) -> None:
        """Release a successful coordinator hold after both ranks restore."""

        with self._observed("release"):
            self._require_launch_gate_release(boundary)
            self._commit_launch_gate_release(boundary)

    def fault_closed(self, boundary: RuntimeBoundary) -> None:
        """Permanently reject decoder launches after uncertain rank publication."""

        with self._observed("fault"):
            _validate_boundary(boundary)
            self._closed_boundary = boundary
            self._runtime_hold = False
            self._coordinator_hold = True
            self._phase = OwnerPhase.FAULTED

    def _require_open(self) -> None:
        if self._phase is not OwnerPhase.OPEN:
            raise DecoderStateOwnerError(
                f"decoder launch gate is {self._phase.value}"
            )

    def _require_matching_accepted(self, boundary: RuntimeBoundary) -> None:
        if self._phase is OwnerPhase.FAULTED:
            raise DecoderStateOwnerError("decoder launch gate is faulted")
        _validate_boundary(boundary)
        if (
            self._accepted_boundary != boundary
            or self._decoder.publication is None
            or self._decoder.publication.generation != boundary.generation_epoch
            or self._decoder.phase is not DevicePhase.IDLE
        ):
            raise DecoderStateOwnerError(
                "decoder accepted boundary does not match the current publication"
            )

    def _require_prepared(
        self, boundary: RuntimeBoundary, commit_sha256: str
    ) -> None:
        if (
            self._phase is OwnerPhase.FAULTED
            or self._closed_boundary != boundary
            or not (self._runtime_hold or self._coordinator_hold)
            or self._prepared_state is None
            or self._prepared_policy_state is None
            or self._prepared_commit_sha256 != commit_sha256
        ):
            raise DecoderStateOwnerError("inactive state generation does not match")

    def _require_launch_gate_release(self, boundary: RuntimeBoundary) -> None:
        if self._phase is OwnerPhase.FAULTED:
            raise DecoderStateOwnerError("decoder launch gate is faulted")
        if (
            not self._coordinator_hold
            or self._runtime_hold
            or self._closed_boundary != boundary
        ):
            raise DecoderStateOwnerError(
                "coordinator launch gate boundary does not match"
            )

    def _clear_prepared(self) -> None:
        self._prepared_state = None
        self._prepared_policy_state = None
        self._prepared_commit_sha256 = None
        self._rollback_state = None
        self._rollback_policy_state = None
        self._rollback_commit_sha256 = None
        self._prepared_committed = False

    def _fault(self) -> None:
        self._runtime_hold = False
        self._coordinator_hold = True
        self._phase = OwnerPhase.FAULTED

    @contextmanager
    def _observed(self, phase: str) -> Iterator[None]:
        with self._tracer.start_as_current_span("rocket.qwen38.state.owner") as span:
            span.set_attribute("phase", phase)
            span.set_attribute("rank", self._rank)
            try:
                yield
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                raise
            else:
                span.set_attribute("outcome", "success")


class TwoRankRestoreCoordinator:
    """Fail-closed coordinator for one authenticated TP2 restore."""

    def __init__(
        self,
        owners: tuple[CoordinatedStateOwner, CoordinatedStateOwner],
        bindings: tuple[RankRuntimeBinding, RankRuntimeBinding],
        tracer: OtelTracer,
    ):
        if (
            not isinstance(owners, tuple)
            or len(owners) != 2
            or any(not isinstance(owner, CoordinatedStateOwner) for owner in owners)
            or tuple(getattr(owner, "rank", None) for owner in owners) != (0, 1)
            or not isinstance(bindings, tuple)
            or len(bindings) != 2
            or tuple(getattr(binding, "rank", None) for binding in bindings) != (0, 1)
            or any(
                not callable(getattr(binding, "restore", None))
                for binding in bindings
            )
            or any(
                getattr(binding, "state_owner", None) is not owner
                for owner, binding in zip(owners, bindings, strict=True)
            )
        ):
            raise DecoderStateOwnerError(
                "coordinator requires ordered rank 0 and rank 1 owners and bindings"
            )
        if tracer is None or not callable(
            getattr(tracer, "start_as_current_span", None)
        ):
            raise DecoderStateOwnerError("an OpenTelemetry tracer is required")
        self._owners = owners
        self._bindings = bindings
        self._tracer = tracer
        self._phase = CoordinatorPhase.IDLE

    @property
    def phase(self) -> CoordinatorPhase:
        return self._phase

    def restore(
        self, authenticated: AuthenticatedState, *, generation_epoch: int
    ) -> None:
        """Restore both ranks only from one exactly matching accepted boundary."""

        with self._observed("validate", -1):
            if self._phase is not CoordinatorPhase.IDLE:
                raise DecoderStateOwnerError(
                    f"two-rank coordinator is not idle: {self._phase.value}"
                )
            if (
                not isinstance(authenticated, AuthenticatedState)
                or not authenticated._is_store_authenticated()
            ):
                raise DecoderStateOwnerError(
                    "coordinator requires host-authenticated state"
                )
            boundary = authenticated.boundary
            expected = RuntimeBoundary(
                boundary.token_count, boundary.token_hash, generation_epoch
            )
            _validate_boundary(expected)
            if any(owner.accepted_boundary != expected for owner in self._owners):
                raise DecoderStateOwnerError(
                    "accepted boundary for both ranks must match authenticated state"
                )

        self._phase = CoordinatorPhase.RESTORING
        failed_rank = -1
        try:
            for owner in self._owners:
                owner.hold_launch_gate(expected)
            for binding in self._bindings:
                failed_rank = binding.rank
                with self._observed("restore", binding.rank):
                    binding.restore(authenticated, generation_epoch)
            for owner in self._owners:
                owner.release_launch_gate(expected)
        except BaseException as exc:
            self._phase = CoordinatorPhase.FAULTED
            for owner in self._owners:
                try:
                    owner.fault_closed(expected)
                except BaseException:
                    pass
            if not isinstance(exc, Exception):
                raise
            raise DecoderStateOwnerError(
                f"two-rank restore failed at rank {failed_rank}"
            ) from exc
        self._phase = CoordinatorPhase.IDLE

    @contextmanager
    def _observed(self, phase: str, rank: int) -> Iterator[None]:
        with self._tracer.start_as_current_span(
            "rocket.qwen38.state.coordinator"
        ) as span:
            span.set_attribute("phase", phase)
            span.set_attribute("rank", rank)
            try:
                yield
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                raise
            else:
                span.set_attribute("outcome", "success")


def _validate_rank(rank: int) -> None:
    if isinstance(rank, bool) or rank not in (0, 1):
        raise DecoderStateOwnerError("decoder state owner rank must be 0 or 1")


def _validate_boundary(boundary: RuntimeBoundary) -> None:
    if (
        not isinstance(boundary, RuntimeBoundary)
        or isinstance(boundary.token_count, bool)
        or not isinstance(boundary.token_count, int)
        or boundary.token_count < 0
        or not isinstance(boundary.token_hash, str)
        or not _HEX_256.fullmatch(boundary.token_hash)
        or isinstance(boundary.generation_epoch, bool)
        or not isinstance(boundary.generation_epoch, int)
        or boundary.generation_epoch <= 0
    ):
        raise DecoderStateOwnerError("decoder boundary is invalid")


__all__ = [
    "CoordinatedStateOwner",
    "CoordinatorPhase",
    "DecoderStateOwner",
    "DecoderStateOwnerError",
    "OwnerPhase",
    "TwoRankRestoreCoordinator",
]

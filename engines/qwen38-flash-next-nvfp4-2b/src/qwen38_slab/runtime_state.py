"""CUDA/runtime binding for Qwen3.8 accepted-token state.

The engine embedder owns one :class:`CudaStateBinding` per rank and serializes
all calls.  The injected runtime adapter is the CUDA boundary: ``quiesce`` must
stop new generation launches, fence every stream that can mutate state, and
return a matching receipt.  A quiesce failure faults the binding unless it is a
typed :class:`CudaQuiesceError` proving that the launch gate reopened.  A
successful receipt transfers responsibility for resuming generation to this
binding.

``capture`` copies exactly the accepted extent of every canonical family into
owned immutable host bytes.  ``restore`` borrows authenticated host payloads,
copies all of them into private device allocations, fences those transfers,
then gives one complete pointer table to an all-or-none ``publish`` operation.
Before publication, failures discard private staging.  After publication, a
resume failure leaves the binding faulted and the new pointer table owned by
the runtime.  The binding never publishes a partial family inventory.

OpenTelemetry cardinality is finite.  Span attributes are ``phase`` (eight
values), ``rank`` (two values), ``family`` (nine values plus ``none``), and
``outcome`` (two values).  Tokens, epochs, handles, sessions, requests, and
transaction identifiers are forbidden as attributes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterator, Protocol

from .state_txn import (
    STATE_FAMILIES,
    AcceptedBoundary,
    AuthenticatedState,
    FamilyPayload,
    OtelTracer,
)

MAX_FAMILY_BYTES = 24 * 1024**3
MAX_RANK_BYTES = 64 * 1024**3
_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")


class RuntimeStateError(RuntimeError):
    """Validation or CUDA-adapter failure at a stable runtime boundary."""


class CudaQuiesceError(RuntimeError):
    """Quiesce failure carrying launch-gate recovery evidence.

    ``safe_to_retry`` may be true only when the adapter proved that its launch
    gate reopened after the failed fence.  Untyped and unsafe failures
    permanently fault the binding.
    """

    def __init__(self, message: str, *, safe_to_retry: bool):
        super().__init__(message)
        self.safe_to_retry = safe_to_retry


class CudaRuntimeFatalError(RuntimeStateError):
    """CUDA failure after which runtime ownership cannot be proven safe."""


class BindingPhase(str, Enum):
    """Externally observable lifecycle snapshot for one single-owner binding."""

    IDLE = "idle"
    QUIESCING = "quiescing"
    QUIESCED = "quiesced"
    CAPTURING = "capturing"
    STAGING = "staging"
    SYNCHRONIZING = "synchronizing"
    PUBLISHING = "publishing"
    DISCARDING = "discarding"
    RESUMING = "resuming"
    FAULTED = "faulted"


@dataclass(frozen=True)
class RuntimeBoundary:
    """Accepted scheduler boundary requested from the CUDA runtime.

    ``generation_epoch`` is a rank-local monotonically increasing scheduler
    generation.  It is compared in the quiesce receipt but never emitted to
    metrics or spans.
    """

    token_count: int
    token_hash: str
    generation_epoch: int


@dataclass(frozen=True)
class QuiesceReceipt:
    """CUDA adapter attestation returned after its launch gate and fences hold."""

    boundary: RuntimeBoundary
    compute_fenced: bool
    pending_launches: int


@dataclass(frozen=True)
class DeviceState:
    """Borrowed device state extent; ``handle`` remains runtime-owned."""

    family: str
    handle: object
    accepted_bytes: int
    allocated_bytes: int


class CudaRuntime(Protocol):
    """Engine-owned CUDA adapter used only by :class:`CudaStateBinding`.

    Methods are synchronous.  Device and stream handles are adapter-private.
    ``publish`` must atomically replace the complete pointer table or raise
    before replacing it.  It consumes every staged allocation only on success.
    ``discard`` consumes unpublished allocations and ``resume`` reopens the
    generation launch gate.  All operations have bounded work proportional to
    the exact nine-family inventory and its validated byte extents.
    """

    def quiesce(self, boundary: RuntimeBoundary) -> QuiesceReceipt: ...
    def copy_device_to_host(
        self, source: DeviceState, logical_bytes: int
    ) -> bytes: ...
    def allocate_staging(self, family: str, logical_bytes: int) -> object: ...
    def copy_host_to_device(self, destination: object, payload: bytes) -> None: ...
    def finish_transfers(self) -> None: ...
    def publish(self, staged: Mapping[str, object], boundary: RuntimeBoundary) -> None: ...
    def discard(self, staged: tuple[object, ...]) -> None: ...
    def resume(self, boundary: RuntimeBoundary) -> None: ...


class CudaStateBinding:
    """Single-rank, single-owner accepted-state capture and restore boundary.

    Public calls are non-reentrant and not thread-safe.  Inputs are borrowed;
    capture returns owned immutable bytes.  Restore transfers ownership of
    private staging to the runtime only after one successful atomic publish.
    Expected validation and adapter failures raise :class:`RuntimeStateError`.
    A ``FAULTED`` binding cannot be reused because launch-gate or pointer-table
    state can no longer be proven from this process.
    """

    _REQUIRED_METHODS = (
        "quiesce",
        "copy_device_to_host",
        "allocate_staging",
        "copy_host_to_device",
        "finish_transfers",
        "publish",
        "discard",
        "resume",
    )

    def __init__(self, rank: int, runtime: CudaRuntime, tracer: OtelTracer):
        if isinstance(rank, bool) or rank not in (0, 1):
            raise RuntimeStateError("runtime binding rank must be 0 or 1")
        if runtime is None or any(
            not callable(getattr(runtime, method, None)) for method in self._REQUIRED_METHODS
        ):
            raise RuntimeStateError("CUDA runtime adapter contract is incomplete")
        if tracer is None or not callable(getattr(tracer, "start_as_current_span", None)):
            raise RuntimeStateError("an OpenTelemetry tracer is required")
        self._rank = rank
        self._runtime = runtime
        self._tracer = tracer
        self._phase = BindingPhase.IDLE

    @property
    def phase(self) -> BindingPhase:
        """Return a point-in-time lifecycle snapshot; callers must not drive it."""

        return self._phase

    @property
    def rank(self) -> int:
        """Return the fixed TP rank owned by this binding."""

        return self._rank

    @property
    def state_owner(self) -> object | None:
        """Return the runtime's fixed gate owner when the adapter exposes it."""

        return getattr(self._runtime, "owner", None)

    def capture(
        self,
        boundary: RuntimeBoundary,
        sources: Mapping[str, DeviceState],
    ) -> tuple[AcceptedBoundary, dict[str, FamilyPayload]]:
        """Fence CUDA mutation and copy nine accepted extents to owned bytes.

        Reads the borrowed source handles and mutates only the injected runtime
        launch gate.  Success always resumes generation.  Copy or resume
        failures publish no host checkpoint and raise ``RuntimeStateError``.
        """

        with self._observed("validate", "none"):
            self._require_idle()
            self._validate_boundary(boundary)
            self._validate_sources(sources)
        self._quiesce(boundary)
        captured: dict[str, FamilyPayload] = {}
        operation_error: BaseException | None = None
        self._transition(BindingPhase.QUIESCED, BindingPhase.CAPTURING)
        try:
            for family in STATE_FAMILIES:
                source = sources[family]
                with self._observed("capture", family):
                    try:
                        payload = self._runtime.copy_device_to_host(
                            source, source.accepted_bytes
                        )
                    except CudaRuntimeFatalError:
                        raise
                    except Exception as exc:
                        raise RuntimeStateError(
                            f"CUDA capture failed for family {family}"
                        ) from exc
                    if not isinstance(payload, bytes) or len(payload) != source.accepted_bytes:
                        raise RuntimeStateError(
                            f"CUDA capture returned an invalid extent for family {family}"
                        )
                    captured[family] = FamilyPayload(payload)
        except BaseException as exc:
            operation_error = exc
        resume_error = self._resume(boundary)
        if resume_error is not None:
            raise resume_error
        if operation_error is not None:
            if isinstance(operation_error, CudaRuntimeFatalError):
                self._phase = BindingPhase.FAULTED
            raise operation_error
        return (
            AcceptedBoundary(boundary.token_count, boundary.token_hash, quiesced=True),
            captured,
        )

    def restore(
        self,
        authenticated: AuthenticatedState,
        generation_epoch: int,
    ) -> None:
        """Stage and atomically publish one authenticated nine-family snapshot.

        ``authenticated`` is the opaque object returned by the host transaction
        layer after both ranks pass authentication.  Its payloads are borrowed
        until return.  Validation fails before CUDA quiesce.  Before a
        successful publish, failures discard all allocated staging and resume.
        A resume failure after publish raises and permanently faults the
        binding while leaving the newly published table runtime-owned.
        """

        with self._observed("validate", "none"):
            self._require_idle()
            if (
                not isinstance(authenticated, AuthenticatedState)
                or not authenticated._is_store_authenticated()
            ):
                raise RuntimeStateError("restore requires host-authenticated state")
            boundary = authenticated.boundary
            runtime_boundary = self._restore_boundary(boundary, generation_epoch)
            payloads = authenticated.rank_payload(self._rank)
            self._validate_payloads(payloads)
        self._quiesce(runtime_boundary)
        staged: dict[str, object] = {}
        operation_error: BaseException | None = None
        published = False
        self._transition(BindingPhase.QUIESCED, BindingPhase.STAGING)
        try:
            for family in STATE_FAMILIES:
                payload = payloads[family].accepted
                with self._observed("stage", family):
                    try:
                        destination = self._runtime.allocate_staging(family, len(payload))
                        if destination is None:
                            raise RuntimeStateError(
                                f"CUDA staging returned no allocation for family {family}"
                            )
                        staged[family] = destination
                        self._runtime.copy_host_to_device(destination, payload)
                    except RuntimeStateError:
                        raise
                    except Exception as exc:
                        raise RuntimeStateError(
                            f"CUDA stage failed for family {family}"
                        ) from exc
            self._transition(BindingPhase.STAGING, BindingPhase.SYNCHRONIZING)
            with self._observed("sync", "none"):
                try:
                    self._runtime.finish_transfers()
                except CudaRuntimeFatalError:
                    raise
                except Exception as exc:
                    raise RuntimeStateError("CUDA staging fence failed") from exc
            self._transition(BindingPhase.SYNCHRONIZING, BindingPhase.PUBLISHING)
            with self._observed("publish", "none"):
                try:
                    self._runtime.publish(MappingProxyType(staged), runtime_boundary)
                except Exception as exc:
                    raise RuntimeStateError("CUDA pointer-table publish failed") from exc
            published = True
        except BaseException as exc:
            operation_error = exc

        cleanup_error: RuntimeStateError | None = None
        if not published:
            cleanup_error = self._discard_staging(tuple(staged.values()))
        resume_error = self._resume(runtime_boundary)
        if cleanup_error is not None or resume_error is not None:
            details = []
            if operation_error is not None:
                details.append(str(operation_error))
            if cleanup_error is not None:
                details.append(str(cleanup_error))
            if resume_error is not None:
                details.append(str(resume_error))
            raise RuntimeStateError("; ".join(details)) from (
                resume_error or cleanup_error or operation_error
            )
        if operation_error is not None:
            if isinstance(operation_error, CudaRuntimeFatalError):
                self._phase = BindingPhase.FAULTED
            raise operation_error

    def _quiesce(self, boundary: RuntimeBoundary) -> None:
        self._transition(BindingPhase.IDLE, BindingPhase.QUIESCING)
        with self._observed("quiesce", "none"):
            try:
                receipt = self._runtime.quiesce(boundary)
            except BaseException as exc:
                safe_to_retry = (
                    isinstance(exc, CudaQuiesceError) and exc.safe_to_retry is True
                )
                self._transition(
                    BindingPhase.QUIESCING,
                    BindingPhase.IDLE if safe_to_retry else BindingPhase.FAULTED,
                )
                if not isinstance(exc, Exception):
                    raise
                raise RuntimeStateError("CUDA quiesce failed before receipt") from exc
            if (
                not isinstance(receipt, QuiesceReceipt)
                or receipt.boundary != boundary
                or receipt.compute_fenced is not True
                or isinstance(receipt.pending_launches, bool)
                or receipt.pending_launches != 0
            ):
                self._transition(BindingPhase.QUIESCING, BindingPhase.FAULTED)
                raise RuntimeStateError("CUDA quiesce receipt is invalid")
        self._transition(BindingPhase.QUIESCING, BindingPhase.QUIESCED)

    def _discard_staging(self, staged: tuple[object, ...]) -> RuntimeStateError | None:
        prior = self._phase
        self._transition(prior, BindingPhase.DISCARDING)
        try:
            with self._observed("discard", "none"):
                self._runtime.discard(staged)
        except BaseException as exc:
            self._phase = BindingPhase.FAULTED
            error = RuntimeStateError("CUDA private staging discard failed")
            error.__cause__ = exc
            return error
        return None

    def _resume(self, boundary: RuntimeBoundary) -> RuntimeStateError | None:
        prior = self._phase
        if prior is not BindingPhase.FAULTED:
            self._transition(prior, BindingPhase.RESUMING)
        try:
            with self._observed("resume", "none"):
                self._runtime.resume(boundary)
        except BaseException as exc:
            self._phase = BindingPhase.FAULTED
            error = RuntimeStateError("CUDA generation resume failed")
            error.__cause__ = exc
            return error
        if prior is BindingPhase.FAULTED:
            return RuntimeStateError("CUDA staging cleanup failed; binding is faulted")
        self._transition(BindingPhase.RESUMING, BindingPhase.IDLE)
        return None

    def _transition(self, expected: BindingPhase, target: BindingPhase) -> None:
        if self._phase is not expected:
            observed = self._phase
            self._phase = BindingPhase.FAULTED
            raise RuntimeStateError(
                f"runtime lifecycle transition expected {expected.value}, got {observed.value}"
            )
        self._phase = target

    def _require_idle(self) -> None:
        if self._phase is not BindingPhase.IDLE:
            raise RuntimeStateError(
                f"runtime state binding is not idle: {self._phase.value}"
            )

    @staticmethod
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
            or boundary.generation_epoch < 0
        ):
            raise RuntimeStateError("runtime boundary is not an accepted token boundary")

    @classmethod
    def _restore_boundary(
        cls, boundary: AcceptedBoundary, generation_epoch: int
    ) -> RuntimeBoundary:
        if not isinstance(boundary, AcceptedBoundary) or boundary.quiesced is not True:
            raise RuntimeStateError("restore requires an authenticated quiesced boundary")
        runtime_boundary = RuntimeBoundary(
            boundary.token_count, boundary.token_hash, generation_epoch
        )
        cls._validate_boundary(runtime_boundary)
        return runtime_boundary

    @staticmethod
    def _validate_sources(sources: Mapping[str, DeviceState]) -> None:
        if not isinstance(sources, Mapping) or tuple(sources) != STATE_FAMILIES:
            raise RuntimeStateError("device sources must use the canonical nine-family order")
        total = 0
        for family in STATE_FAMILIES:
            source = sources[family]
            if (
                not isinstance(source, DeviceState)
                or source.family != family
                or source.handle is None
                or isinstance(source.accepted_bytes, bool)
                or not isinstance(source.accepted_bytes, int)
                or source.accepted_bytes <= 0
                or source.accepted_bytes > MAX_FAMILY_BYTES
                or isinstance(source.allocated_bytes, bool)
                or not isinstance(source.allocated_bytes, int)
                or source.allocated_bytes < source.accepted_bytes
                or source.allocated_bytes > MAX_FAMILY_BYTES
            ):
                raise RuntimeStateError(f"invalid device extent for family {family}")
            total += source.accepted_bytes
        if total > MAX_RANK_BYTES:
            raise RuntimeStateError("accepted device state exceeds the per-rank byte bound")

    @staticmethod
    def _validate_payloads(payloads: Mapping[str, FamilyPayload]) -> None:
        if not isinstance(payloads, Mapping) or tuple(payloads) != STATE_FAMILIES:
            raise RuntimeStateError("restore payloads must use the canonical nine-family order")
        total = 0
        for family in STATE_FAMILIES:
            payload = payloads[family]
            if (
                not isinstance(payload, FamilyPayload)
                or not isinstance(payload.accepted, bytes)
                or not payload.accepted
                or len(payload.accepted) > MAX_FAMILY_BYTES
                or payload.speculative_tail != b""
            ):
                raise RuntimeStateError(f"invalid accepted payload for family {family}")
            total += len(payload.accepted)
        if total > MAX_RANK_BYTES:
            raise RuntimeStateError("restore state exceeds the per-rank byte bound")

    @contextmanager
    def _observed(self, phase: str, family: str) -> Iterator[None]:
        with self._tracer.start_as_current_span("rocket.qwen38.state.runtime") as span:
            span.set_attribute("phase", phase)
            span.set_attribute("rank", self._rank)
            span.set_attribute("family", family)
            try:
                yield
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                raise
            else:
                span.set_attribute("outcome", "success")


__all__ = [
    "BindingPhase",
    "CudaQuiesceError",
    "CudaRuntimeFatalError",
    "CudaRuntime",
    "CudaStateBinding",
    "DeviceState",
    "MAX_FAMILY_BYTES",
    "MAX_RANK_BYTES",
    "QuiesceReceipt",
    "RuntimeBoundary",
    "RuntimeStateError",
]

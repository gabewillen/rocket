"""Publish rank-local authenticated NVMe state through the decoder CUDA gate.

The adapter follows the pinned vLLM checkpoint fence ordering: close the
launch gate, fence every mutating compute stream, restore into private device
storage, fence the transfer stream, then publish.  The TP2 coordinator holds
both rank gates until both local publications complete.  Any uncertainty faults
both owners closed.

OpenTelemetry attributes are bounded to phase, rank, family, and outcome.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

from .distributed_state_txn import LocalAuthenticatedState
from .mtp_policy import AdaptiveMtpPolicy, MtpPolicyError
from .runtime_state import QuiesceReceipt, RuntimeBoundary
from .state_owner import CoordinatedStateOwner, DecoderStateOwner
from .state_txn import STATE_FAMILIES, OtelTracer, _HEX_256

EXACT_C16_LOGICAL_BYTES_PER_RANK = 32_352_705_536


class LocalCudaRestoreError(RuntimeError):
    """Rank-local CUDA restore validation or ownership failure."""


@dataclass(frozen=True)
class StagedLocalCudaState:
    """Binding-owned inactive generation; it is never a decoder publication."""

    rank: int
    boundary: RuntimeBoundary
    commit_sha256: str
    policy_digest: str


@dataclass(frozen=True)
class DecoderStatePublication:
    """Common durable generation exposed only after both rank gates reopen."""

    boundary: RuntimeBoundary
    commit_sha256: str
    policy_digest: str
    family_count: int


class LocalCudaStateBinding:
    """One rank's sealed NVMe descriptor to private CUDA staging transaction."""

    _METHODS = (
        "quiesce", "allocate_staging", "copy_extent_to_device",
        "finish_transfers", "prepare_local", "commit_local",
        "rollback_local", "finalize_local", "complete_local", "discard", "resume",
    )

    def __init__(self, rank, runtime, policy, expected_extents, tracer):
        if isinstance(rank, bool) or rank not in (0, 1):
            raise LocalCudaRestoreError("local CUDA rank must be 0 or 1")
        if runtime is None or any(
            not callable(getattr(runtime, method, None)) for method in self._METHODS
        ):
            raise LocalCudaRestoreError("local CUDA runtime contract is incomplete")
        if (
            not isinstance(getattr(runtime, "owner", None), CoordinatedStateOwner)
            or runtime.owner.rank != rank
        ):
            raise LocalCudaRestoreError("runtime must expose the matching decoder owner")
        if not isinstance(policy, AdaptiveMtpPolicy):
            raise LocalCudaRestoreError("adaptive MTP policy is required")
        if (
            not isinstance(expected_extents, Mapping)
            or tuple(expected_extents) != STATE_FAMILIES
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in expected_extents.values()
            )
            or sum(expected_extents.values()) != EXACT_C16_LOGICAL_BYTES_PER_RANK
        ):
            raise LocalCudaRestoreError("exact c16 rank extents are required")
        if tracer is None or not callable(getattr(tracer, "start_as_current_span", None)):
            raise LocalCudaRestoreError("an OpenTelemetry tracer is required")
        self.rank = rank
        self.runtime = runtime
        self.policy = policy
        self.expected_extents = MappingProxyType(dict(expected_extents))
        self.tracer = tracer
        self._pending: StagedLocalCudaState | None = None

    @property
    def state_owner(self):
        return getattr(self.runtime, "owner", None)

    def restore_local(self, authenticated, generation_epoch):
        staged = self.stage_local(authenticated, generation_epoch)
        try:
            self.commit_staged(staged)
            self.finalize_staged(staged)
            self.resume_staged(staged)
            self.complete_staged(staged)
        except BaseException:
            try: self.rollback_staged(staged)
            except BaseException: self.state_owner.fault_closed(staged.boundary)
            raise

    def stage_local(self, authenticated, generation_epoch):
        boundary = self._validate(authenticated, generation_epoch)
        if self._pending is not None:
            raise LocalCudaRestoreError("inactive CUDA generation already exists")
        staged = {}
        quiesced = False
        try:
            with self._observed("quiesce", "none"):
                receipt = self.runtime.quiesce(boundary)
                if (
                    not isinstance(receipt, QuiesceReceipt)
                    or receipt.boundary != boundary
                    or receipt.compute_fenced is not True
                    or isinstance(receipt.pending_launches, bool)
                    or not isinstance(receipt.pending_launches, int)
                    or receipt.pending_launches != 0
                ):
                    raise LocalCudaRestoreError("CUDA quiesce receipt is invalid")
                quiesced = True
            for family in STATE_FAMILIES:
                with self._observed("stage", family):
                    extent = authenticated.families[family]
                    destination = self.runtime.allocate_staging(
                        family, extent.logical_bytes
                    )
                    if destination is None:
                        raise LocalCudaRestoreError("CUDA staging returned no allocation")
                    staged[family] = destination
                    self.runtime.copy_extent_to_device(destination, extent)
            with self._observed("sync", "none"):
                self.runtime.finish_transfers()
            with self._observed("prepare", "none"):
                self.runtime.prepare_local(
                    MappingProxyType(staged), boundary, authenticated.policy_state,
                    authenticated.commit_sha256,
                )
        except BaseException as exc:
            if quiesced:
                try:
                    with self._observed("discard", "none"):
                        self.runtime.discard(tuple(staged.values()))
                except BaseException as discard_error:
                    self.state_owner.fault_closed(boundary)
                    raise LocalCudaRestoreError("CUDA staging discard failed") from discard_error
            if quiesced:
                try:
                    self.runtime.resume(boundary)
                except BaseException as resume_error:
                    self.state_owner.fault_closed(boundary)
                    raise LocalCudaRestoreError("CUDA restore gate could not reopen") from resume_error
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, LocalCudaRestoreError):
                raise
            raise LocalCudaRestoreError("rank-local CUDA restore failed") from exc
        self._pending = StagedLocalCudaState(
            self.rank, boundary, authenticated.commit_sha256,
            authenticated.policy_digest,
        )
        return self._pending

    def commit_staged(self, staged):
        self._require_pending(staged)
        with self._observed("commit", "none"):
            self.runtime.commit_local(staged.boundary, staged.commit_sha256)

    def resume_staged(self, staged):
        self._require_pending(staged)
        with self._observed("resume", "none"):
            self.runtime.resume(staged.boundary)

    def finalize_staged(self, staged):
        self._require_pending(staged)
        with self._observed("finalize", "none"):
            self.runtime.finalize_local(staged.boundary, staged.commit_sha256)

    def complete_staged(self, staged):
        self._require_pending(staged)
        self.runtime.complete_local(staged.boundary, staged.commit_sha256)
        self._pending = None

    def rollback_staged(self, staged):
        self._require_pending(staged)
        with self._observed("rollback", "none"):
            self.runtime.rollback_local(staged.boundary, staged.commit_sha256)
        self._pending = None

    def _require_pending(self, staged):
        if not isinstance(staged, StagedLocalCudaState) or staged is not self._pending:
            raise LocalCudaRestoreError("inactive CUDA generation ownership changed")

    def _validate(self, authenticated, generation_epoch):
        with self._observed("validate", "none"):
            if (
                not isinstance(authenticated, LocalAuthenticatedState)
                or not authenticated._is_store_authenticated()
                or authenticated.rank != self.rank
                or not _HEX_256.fullmatch(authenticated.commit_sha256)
                or tuple(authenticated.families) != STATE_FAMILIES
                or any(
                    authenticated.families[family].logical_bytes
                    != self.expected_extents[family]
                    for family in STATE_FAMILIES
                )
            ):
                raise LocalCudaRestoreError("sealed exact rank state is required")
            if (
                isinstance(generation_epoch, bool)
                or not isinstance(generation_epoch, int)
                or generation_epoch <= 0
            ):
                raise LocalCudaRestoreError("positive generation epoch is required")
            try:
                decoded = self.policy.load_state(authenticated.policy_state)
                canonical = self.policy.dump_state(decoded)
            except MtpPolicyError as exc:
                raise LocalCudaRestoreError("adaptive policy state is invalid") from exc
            if (
                canonical != authenticated.policy_state
                or hashlib.sha256(canonical).hexdigest() != authenticated.policy_digest
            ):
                raise LocalCudaRestoreError("adaptive policy authentication changed")
            accepted = authenticated.boundary
            return RuntimeBoundary(
                accepted.token_count, accepted.token_hash, generation_epoch
            )

    @contextmanager
    def _observed(self, phase, family):
        with self.tracer.start_as_current_span("rocket.qwen38.state.local_cuda") as span:
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


class TwoRankLocalCudaCoordinator:
    """Hold both decoder gates across two owner-local CUDA publications."""

    def __init__(self, owners, bindings, tracer):
        if (
            not isinstance(owners, tuple) or len(owners) != 2
            or tuple(getattr(owner, "rank", None) for owner in owners) != (0, 1)
            or any(not isinstance(owner, DecoderStateOwner) for owner in owners)
            or not isinstance(bindings, tuple) or len(bindings) != 2
            or tuple(getattr(binding, "rank", None) for binding in bindings) != (0, 1)
            or any(
                binding.state_owner is not owner
                for owner, binding in zip(owners, bindings, strict=True)
            )
        ):
            raise LocalCudaRestoreError("ordered TP2 owners and bindings are required")
        if tracer is None or not callable(getattr(tracer, "start_as_current_span", None)):
            raise LocalCudaRestoreError("an OpenTelemetry tracer is required")
        self.owners = owners
        self.bindings = bindings
        self.tracer = tracer
        self._publication_lock = RLock()
        self._publication: DecoderStatePublication | None = None
        if any(owner._publication_lock_bound for owner in self.owners):
            raise LocalCudaRestoreError("decoder publication lock is already bound")
        for owner in self.owners:
            owner._bind_publication_lock(self._publication_lock)

    @property
    def publication(self):
        return self._publication

    def restore(self, states, generation_epoch):
        if (
            not isinstance(states, tuple) or len(states) != 2
            or any(not isinstance(state, LocalAuthenticatedState) for state in states)
            or tuple(getattr(state, "rank", None) for state in states) != (0, 1)
            or any(not state._is_store_authenticated() for state in states)
            or states[0].boundary != states[1].boundary
            or states[0].policy_digest != states[1].policy_digest
            or states[0].policy_state != states[1].policy_state
            or not _HEX_256.fullmatch(states[0].commit_sha256)
            or states[0].commit_sha256 != states[1].commit_sha256
            or any(tuple(state.families) != STATE_FAMILIES for state in states)
        ):
            raise LocalCudaRestoreError(
                "TP2 authenticated state, durable commit, or family table does not agree"
            )
        accepted = states[0].boundary
        boundary = RuntimeBoundary(
            accepted.token_count, accepted.token_hash, generation_epoch
        )
        with self._publication_lock:
            return self._restore_locked(states, generation_epoch, boundary)

    def _restore_locked(self, states, generation_epoch, boundary):
        if any(owner.accepted_boundary != boundary for owner in self.owners):
            raise LocalCudaRestoreError("TP2 decoder accepted boundary does not agree")
        failed_rank = -1
        staged = []
        try:
            for owner in self.owners:
                owner.hold_launch_gate(boundary)
            for binding, state in zip(self.bindings, states, strict=True):
                failed_rank = binding.rank
                staged.append(binding.stage_local(state, generation_epoch))
            for binding, inactive in zip(self.bindings, staged, strict=True):
                failed_rank = binding.rank
                binding.commit_staged(inactive)
            for binding, inactive in zip(self.bindings, staged, strict=True):
                failed_rank = binding.rank
                binding.resume_staged(inactive)
            for binding, inactive in zip(self.bindings, staged, strict=True):
                failed_rank = binding.rank
                binding.finalize_staged(inactive)
            for owner in self.owners:
                failed_rank = owner.rank
                owner.validate_launch_gate_release(boundary)
            # The externally serialized coordinator reaches one global commit
            # point only after every fallible rank acknowledgement succeeds.
            for owner in self.owners:
                owner._commit_launch_gate_release(boundary)
            for binding, inactive in zip(self.bindings, staged, strict=True):
                binding.complete_staged(inactive)
        except BaseException as exc:
            for binding, inactive in reversed(tuple(zip(
                self.bindings[:len(staged)], staged, strict=True
            ))):
                try: binding.rollback_staged(inactive)
                except BaseException: pass
            for owner in self.owners:
                try: owner.fault_closed(boundary)
                except BaseException: pass
            if not isinstance(exc, Exception):
                raise
            raise LocalCudaRestoreError(
                f"TP2 local CUDA restore failed at rank {failed_rank}"
            ) from exc
        publication = DecoderStatePublication(
            boundary, states[0].commit_sha256, states[0].policy_digest,
            len(STATE_FAMILIES),
        )
        self._publication = publication
        return publication


class DecoderStateTransactionGate:
    """One-shot bridge from durable rank authentication to decoder publication."""

    def __init__(self, coordinator, generation_epoch, tracer):
        if not isinstance(coordinator, TwoRankLocalCudaCoordinator):
            raise LocalCudaRestoreError("exact two-rank CUDA coordinator is required")
        if (
            isinstance(generation_epoch, bool)
            or not isinstance(generation_epoch, int)
            or generation_epoch <= 0
        ):
            raise LocalCudaRestoreError("positive decoder generation is required")
        if tracer is None or not callable(getattr(tracer, "start_as_current_span", None)):
            raise LocalCudaRestoreError("an OpenTelemetry tracer is required")
        self.coordinator = coordinator
        self.generation_epoch = generation_epoch
        self.tracer = tracer
        self._states = {}
        self._publication = None
        self._faulted = False
        self._lock = RLock()

    @property
    def publication(self):
        return self._publication

    def publisher(self, rank):
        if isinstance(rank, bool) or rank not in (0, 1):
            raise LocalCudaRestoreError("decoder publisher rank must be 0 or 1")

        def accept(authenticated):
            with self._lock:
                with self._observed("accept", rank):
                    if self._faulted or self._publication is not None:
                        raise LocalCudaRestoreError("decoder transaction gate is terminal")
                    if (
                        not isinstance(authenticated, LocalAuthenticatedState)
                        or not authenticated._is_store_authenticated()
                        or authenticated.rank != rank
                        or rank in self._states
                        or not _HEX_256.fullmatch(authenticated.commit_sha256)
                    ):
                        self._faulted = True
                        self._states.clear()
                        raise LocalCudaRestoreError("durable rank publication is invalid")
                    self._states[rank] = authenticated
                if len(self._states) != 2:
                    return
                try:
                    with self._observed("publish", -1):
                        self._publication = self.coordinator.restore(
                            (self._states[0], self._states[1]), self.generation_epoch
                        )
                except BaseException:
                    self._faulted = True
                    self._states.clear()
                    raise
                self._states.clear()

        return accept

    @contextmanager
    def _observed(self, phase, rank):
        with self.tracer.start_as_current_span(
            "rocket.qwen38.state.decoder_transaction_gate"
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


__all__ = [
    "DecoderStatePublication", "DecoderStateTransactionGate",
    "EXACT_C16_LOGICAL_BYTES_PER_RANK", "LocalCudaRestoreError",
    "LocalCudaStateBinding", "StagedLocalCudaState",
    "TwoRankLocalCudaCoordinator",
]

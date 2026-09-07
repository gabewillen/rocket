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
from types import MappingProxyType

from .distributed_state_txn import LocalAuthenticatedState
from .mtp_policy import AdaptiveMtpPolicy, MtpPolicyError
from .runtime_state import QuiesceReceipt, RuntimeBoundary
from .state_owner import CoordinatedStateOwner
from .state_txn import STATE_FAMILIES, OtelTracer

EXACT_C16_LOGICAL_BYTES_PER_RANK = 32_352_705_536


class LocalCudaRestoreError(RuntimeError):
    """Rank-local CUDA restore validation or ownership failure."""


class LocalCudaStateBinding:
    """One rank's sealed NVMe descriptor to private CUDA staging transaction."""

    _METHODS = (
        "quiesce", "allocate_staging", "copy_extent_to_device",
        "finish_transfers", "publish_local", "discard", "resume",
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

    @property
    def state_owner(self):
        return getattr(self.runtime, "owner", None)

    def restore_local(self, authenticated, generation_epoch):
        boundary = self._validate(authenticated, generation_epoch)
        staged = {}
        quiesced = False
        published = False
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
            with self._observed("publish", "none"):
                self.runtime.publish_local(
                    MappingProxyType(staged), boundary, authenticated.policy_state
                )
                published = True
        except BaseException as exc:
            if quiesced and not published:
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
        try:
            with self._observed("resume", "none"):
                self.runtime.resume(boundary)
        except BaseException as exc:
            self.state_owner.fault_closed(boundary)
            raise LocalCudaRestoreError("published CUDA gate could not reopen") from exc

    def _validate(self, authenticated, generation_epoch):
        with self._observed("validate", "none"):
            if (
                not isinstance(authenticated, LocalAuthenticatedState)
                or not authenticated._is_store_authenticated()
                or authenticated.rank != self.rank
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
            or any(not isinstance(owner, CoordinatedStateOwner) for owner in owners)
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

    def restore(self, states, generation_epoch):
        if (
            not isinstance(states, tuple) or len(states) != 2
            or any(not isinstance(state, LocalAuthenticatedState) for state in states)
            or tuple(getattr(state, "rank", None) for state in states) != (0, 1)
            or any(not state._is_store_authenticated() for state in states)
            or states[0].boundary != states[1].boundary
            or states[0].policy_digest != states[1].policy_digest
            or states[0].policy_state != states[1].policy_state
        ):
            raise LocalCudaRestoreError("TP2 authenticated state does not agree")
        accepted = states[0].boundary
        boundary = RuntimeBoundary(
            accepted.token_count, accepted.token_hash, generation_epoch
        )
        if any(owner.accepted_boundary != boundary for owner in self.owners):
            raise LocalCudaRestoreError("TP2 decoder accepted boundary does not agree")
        failed_rank = -1
        try:
            for owner in self.owners:
                owner.hold_launch_gate(boundary)
            for binding, state in zip(self.bindings, states, strict=True):
                failed_rank = binding.rank
                binding.restore_local(state, generation_epoch)
            for owner in self.owners:
                owner.release_launch_gate(boundary)
        except BaseException as exc:
            for owner in self.owners:
                try: owner.fault_closed(boundary)
                except BaseException: pass
            if not isinstance(exc, Exception):
                raise
            raise LocalCudaRestoreError(
                f"TP2 local CUDA restore failed at rank {failed_rank}"
            ) from exc


__all__ = [
    "EXACT_C16_LOGICAL_BYTES_PER_RANK", "LocalCudaRestoreError",
    "LocalCudaStateBinding", "TwoRankLocalCudaCoordinator",
]

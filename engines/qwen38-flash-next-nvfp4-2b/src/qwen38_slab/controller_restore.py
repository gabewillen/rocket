# SPDX-License-Identifier: Apache-2.0
"""Controller-owned bridge from rank-local authentication to decoder publish."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from .distributed_state_txn import (
    AuthenticationReceipt,
    DistributedStateCoordinator,
    LocalAuthenticatedState,
)
from .local_cuda_restore import (
    DecoderStatePublication,
    DecoderStateTransactionGate,
    LocalCudaRestoreError,
)
from .state_txn import STATE_FAMILIES


class ControllerRestoreError(RuntimeError):
    """Controller endpoint, receipt, or one-shot publication failure."""


@dataclass(frozen=True)
class DecoderContinuation:
    """Common accepted token boundary authorized for both TP2 decoders."""

    token_count: int
    token_hash: str
    generation_epoch: int
    commit_sha256: str
    policy_digest: str


class DecoderGateRankEndpoint:
    """Rank-local durable endpoint whose publisher is fixed to one decoder gate."""

    _STORE_METHODS = (
        "prepare", "commit", "index", "inspect_restore", "authenticate"
    )

    def __init__(self, store, sources, gate: DecoderStateTransactionGate):
        rank = getattr(store, "rank", None)
        if (
            isinstance(rank, bool)
            or rank not in (0, 1)
            or any(not callable(getattr(store, method, None)) for method in self._STORE_METHODS)
            or not isinstance(sources, Mapping)
            or tuple(sources) != STATE_FAMILIES
            or not isinstance(gate, DecoderStateTransactionGate)
        ):
            raise ControllerRestoreError(
                "rank store, canonical sources, and decoder gate are required"
            )
        self.rank = rank
        self.store = store
        self.sources = sources
        self._publish = gate.publisher(rank)
        self._pending_state: LocalAuthenticatedState | None = None
        self._pending_receipt: AuthenticationReceipt | None = None

    def prepare(self, session_id, transaction_id, boundary, policy_state):
        return self.store.prepare(
            session_id, transaction_id, boundary, self.sources, policy_state
        )

    def commit(self, receipts):
        return self.store.commit(receipts)

    def index(self, receipts):
        self.store.index(receipts)

    def inspect_restore(self, session_id):
        return self.store.inspect_restore(session_id)

    def authenticate(self, inspection):
        if self._pending_state is not None:
            raise ControllerRestoreError("rank authentication is already pending")
        state = self.store.authenticate(inspection)
        if (
            not isinstance(state, LocalAuthenticatedState)
            or not state._is_store_authenticated()
            or state.rank != self.rank
            or state.boundary != inspection.boundary
            or state.commit_sha256 != inspection.commit_sha256
            or state.policy_digest != inspection.policy_digest
            or tuple(state.families) != STATE_FAMILIES
        ):
            raise ControllerRestoreError("rank store returned an invalid sealed state")
        receipt = AuthenticationReceipt(
            self.rank,
            inspection.boundary,
            inspection.commit_sha256,
            inspection.rank_sha256,
            inspection.policy_digest,
        )
        self._pending_state = state
        self._pending_receipt = receipt
        return receipt

    def publish(self, receipt):
        if (
            not isinstance(receipt, AuthenticationReceipt)
            or receipt is not self._pending_receipt
            or self._pending_state is None
        ):
            raise ControllerRestoreError("rank has no matching authenticated receipt")
        state = self._pending_state
        self._pending_state = None
        self._pending_receipt = None
        self._publish(state)


class DecoderStateRestoreController:
    """One-shot controller composition for durable restore and decoder publish."""

    def __init__(self, stores, sources, gate, tracer):
        if (
            not isinstance(stores, tuple)
            or len(stores) != 2
            or tuple(getattr(store, "rank", None) for store in stores) != (0, 1)
            or not isinstance(sources, tuple)
            or len(sources) != 2
            or not isinstance(gate, DecoderStateTransactionGate)
            or tracer is None
            or not callable(getattr(tracer, "start_as_current_span", None))
        ):
            raise ControllerRestoreError(
                "ordered TP2 stores, sources, gate, and tracer are required"
            )
        self._gate = gate
        self._tracer = tracer
        self._endpoints = tuple(
            DecoderGateRankEndpoint(store, source, gate)
            for store, source in zip(stores, sources, strict=True)
        )
        self._distributed = DistributedStateCoordinator(self._endpoints, tracer)
        self._continuation: DecoderContinuation | None = None
        self._terminal = False

    @property
    def continuation(self):
        return self._continuation

    def restore(self, session_id: str) -> DecoderContinuation:
        if self._terminal:
            raise ControllerRestoreError("controller restore is one-shot")
        self._terminal = True
        try:
            with self._observed():
                receipts = self._distributed.restore(session_id)
                publication = self._gate.publication
                self._continuation = self._validated_continuation(
                    receipts, publication
                )
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, ControllerRestoreError):
                raise
            raise ControllerRestoreError("TP2 decoder restore failed") from exc
        return self._continuation

    @staticmethod
    def _validated_continuation(receipts, publication):
        if (
            not isinstance(receipts, tuple)
            or len(receipts) != 2
            or tuple(receipt.rank for receipt in receipts) != (0, 1)
            or not isinstance(publication, DecoderStatePublication)
            or publication.family_count != len(STATE_FAMILIES)
            or any(
                receipt.boundary.token_count != publication.boundary.token_count
                or receipt.boundary.token_hash != publication.boundary.token_hash
                for receipt in receipts
            )
            or any(receipt.commit_sha256 != publication.commit_sha256 for receipt in receipts)
            or any(receipt.policy_digest != publication.policy_digest for receipt in receipts)
        ):
            raise ControllerRestoreError(
                "decoder publication does not match both authentication receipts"
            )
        return DecoderContinuation(
            publication.boundary.token_count,
            publication.boundary.token_hash,
            publication.boundary.generation_epoch,
            publication.commit_sha256,
            publication.policy_digest,
        )

    @contextmanager
    def _observed(self):
        with self._tracer.start_as_current_span(
            "rocket.qwen38.state.controller_restore"
        ) as span:
            span.set_attribute("phase", "restore")
            span.set_attribute("rank", -1)
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
    "ControllerRestoreError",
    "DecoderContinuation",
    "DecoderGateRankEndpoint",
    "DecoderStateRestoreController",
]

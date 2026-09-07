# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from qwen38_slab.controller_restore import (
    ControllerRestoreError,
    DecoderGateRankEndpoint,
    DecoderStateRestoreController,
)
from qwen38_slab.decode import Depth
from qwen38_slab.device_decode import DevicePhase, DevicePublication
from qwen38_slab.distributed_state_txn import (
    AuthenticatedFamilyExtent,
    LocalAuthenticatedState,
    RestoreInspection,
)
from qwen38_slab.local_cuda_restore import (
    DecoderStateTransactionGate,
    EXACT_C16_LOGICAL_BYTES_PER_RANK,
    LocalCudaStateBinding,
    TwoRankLocalCudaCoordinator,
)
from qwen38_slab.mtp_policy import (
    AdaptiveMtpPolicy,
    ConcurrencyCeiling,
    PolicyConfig,
)
from qwen38_slab.runtime_state import QuiesceReceipt, RuntimeBoundary
from qwen38_slab.state_owner import DecoderStateOwner, OwnerPhase
from qwen38_slab.state_txn import STATE_FAMILIES, AcceptedBoundary


class Span:
    def __init__(self, tracer, name):
        self.tracer = tracer
        self.name = name
        self.attributes = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.tracer.spans.append(self)

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exception):
        del exception


class Tracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name):
        return Span(self, name)


class Decoder:
    def __init__(self):
        self.phase = DevicePhase.IDLE
        self.publication = None

    def upload_and_launch(self, prepared):
        generation = prepared if isinstance(prepared, int) else prepared.lease.generation
        self.publication = DevicePublication(generation, 1, generation % 2)
        return self.publication


class Runtime:
    def __init__(self, owner, fail_family=None):
        self.owner = owner
        self.fail_family = fail_family
        self.staged = {}

    def quiesce(self, boundary):
        return QuiesceReceipt(self.owner.close_launch_gate(boundary), True, 0)

    def allocate_staging(self, family, logical_bytes):
        value = (family, logical_bytes, object())
        self.staged[family] = value
        return value

    def copy_extent_to_device(self, destination, extent):
        if extent.family == self.fail_family:
            raise RuntimeError("injected rank-local copy failure")
        if destination[1] != extent.logical_bytes:
            raise RuntimeError("extent changed")

    def finish_transfers(self):
        pass

    def prepare_local(self, staged, boundary, policy_state, commit_sha256):
        self.owner.prepare_state_with_policy(
            staged, policy_state, boundary, commit_sha256
        )

    def commit_local(self, boundary, commit_sha256):
        self.owner.commit_prepared_state(boundary, commit_sha256)

    def rollback_local(self, boundary, commit_sha256):
        self.owner.rollback_prepared_state(boundary, commit_sha256)

    def finalize_local(self, boundary, commit_sha256):
        self.owner.finalize_prepared_state(boundary, commit_sha256)

    def complete_local(self, boundary, commit_sha256):
        del boundary, commit_sha256
        self.owner._discard_prepared_rollback()

    def discard(self, staged):
        del staged
        self.staged.clear()

    def resume(self, boundary):
        self.owner.open_launch_gate(boundary)


class Store:
    def __init__(self, state, inspection):
        self.rank = state.rank
        self.state = state
        self.inspection = inspection

    def prepare(self, *args):
        raise AssertionError("restore must not prepare")

    def commit(self, *args):
        raise AssertionError("restore must not commit")

    def index(self, *args):
        raise AssertionError("restore must not index")

    def inspect_restore(self, session_id):
        if session_id != self.inspection.session_id:
            raise RuntimeError("session changed")
        return self.inspection

    def authenticate(self, inspection):
        if inspection is not self.inspection:
            raise RuntimeError("inspection ownership changed")
        return self.state


class ControllerRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan = json.loads(
            Path("scripts/memory/qwen38-state-capacity-plan.json").read_text()
        )
        cls.extents = {
            family: item["logical_bytes_per_stream"] * 16
            for family, item in zip(STATE_FAMILIES, plan["families"], strict=True)
        }
        assert sum(cls.extents.values()) == EXACT_C16_LOGICAL_BYTES_PER_RANK

    def setUp(self):
        self.tracer = Tracer()
        self.policy = AdaptiveMtpPolicy(
            self.tracer,
            PolicyConfig(
                (
                    ConcurrencyCeiling(1, Depth.K7),
                    ConcurrencyCeiling(16, Depth.K1),
                )
            ),
        )
        self.policy_state = self.policy.dump_state(self.policy.initial_state())
        self.policy_digest = hashlib.sha256(self.policy_state).hexdigest()
        self.accepted = AcceptedBoundary(
            262_144,
            hashlib.sha256(b"controller-shared-token-boundary").hexdigest(),
            True,
        )
        self.boundary = RuntimeBoundary(
            self.accepted.token_count, self.accepted.token_hash, 7
        )
        self.commit = hashlib.sha256(b"controller-durable-commit").hexdigest()

    def _state(self, rank):
        families = {
            family: AuthenticatedFamilyExtent(
                family,
                Path(f"/rank{rank}/{index}"),
                logical,
                logical,
                hashlib.sha256(f"logical-{rank}-{index}".encode()).hexdigest(),
                hashlib.sha256(f"padded-{rank}-{index}".encode()).hexdigest(),
            )
            for index, (family, logical) in enumerate(self.extents.items())
        }
        return LocalAuthenticatedState._from_verified(
            rank,
            self.accepted,
            self.policy_digest,
            self.policy_state,
            families,
            self.commit,
        )

    def _rank(self, rank, fail_family=None):
        decoder = Decoder()
        owner = DecoderStateOwner(rank=rank, decoder=decoder, tracer=self.tracer)
        owner.accept_boundary(owner.upload_and_launch(7), self.boundary)
        runtime = Runtime(owner, fail_family)
        binding = LocalCudaStateBinding(
            rank, runtime, self.policy, self.extents, self.tracer
        )
        return decoder, owner, binding

    def _stores(self):
        states = (self._state(0), self._state(1))
        stores = []
        for rank, state in enumerate(states):
            inspection = RestoreInspection(
                rank,
                "controller-session",
                "controller-transaction",
                self.accepted,
                self.commit,
                hashlib.sha256(f"prepared-{rank}".encode()).hexdigest(),
                hashlib.sha256(f"rank-{rank}".encode()).hexdigest(),
                self.policy_digest,
            )
            stores.append(Store(state, inspection))
        return tuple(stores)

    def _controller(self, ranks):
        coordinator = TwoRankLocalCudaCoordinator(
            tuple(item[1] for item in ranks),
            tuple(item[2] for item in ranks),
            self.tracer,
        )
        gate = DecoderStateTransactionGate(coordinator, 7, self.tracer)
        sources = tuple(
            {family: object() for family in STATE_FAMILIES} for _ in range(2)
        )
        return DecoderStateRestoreController(
            self._stores(), sources, gate, self.tracer
        ), gate

    @staticmethod
    def _seed_previous(ranks, boundary):
        previous = []
        for _decoder, owner, _binding in ranks:
            table = {family: object() for family in STATE_FAMILIES}
            owner.close_launch_gate(boundary)
            owner.publish_state(table, boundary)
            owner.open_launch_gate(boundary)
            previous.append(dict(owner.active_state))
        return tuple(previous)

    def test_rank0_receipt_alone_cannot_publish_active_generation(self):
        ranks = (self._rank(0), self._rank(1))
        previous = self._seed_previous(ranks, self.boundary)
        controller, gate = self._controller(ranks)
        endpoint = controller._endpoints[0]
        inspection = endpoint.inspect_restore("controller-session")
        receipt = endpoint.authenticate(inspection)
        endpoint.publish(receipt)
        self.assertIsNone(gate.publication)
        self.assertEqual(
            tuple(dict(item[1].active_state) for item in ranks), previous
        )

    def test_rank1_prepublication_failure_preserves_both_active_tables(self):
        ranks = (
            self._rank(0),
            self._rank(1, fail_family=STATE_FAMILIES[-1]),
        )
        previous = self._seed_previous(ranks, self.boundary)
        controller, gate = self._controller(ranks)
        with self.assertRaisesRegex(ControllerRestoreError, "restore failed"):
            controller.restore("controller-session")
        self.assertIsNone(controller.continuation)
        self.assertIsNone(gate.publication)
        for expected, (_decoder, owner, _binding) in zip(
            previous, ranks, strict=True
        ):
            self.assertEqual(dict(owner.active_state), expected)
            self.assertEqual(owner.phase, OwnerPhase.FAULTED)

    def test_control_restore_returns_matching_continuation_token_and_hash(self):
        ranks = (self._rank(0), self._rank(1))
        controller, gate = self._controller(ranks)
        continuation = controller.restore("controller-session")
        self.assertIs(continuation, controller.continuation)
        self.assertEqual(continuation.token_count, self.accepted.token_count)
        self.assertEqual(continuation.token_hash, self.accepted.token_hash)
        self.assertEqual(continuation.generation_epoch, 7)
        self.assertEqual(continuation.commit_sha256, self.commit)
        self.assertEqual(continuation.policy_digest, self.policy_digest)
        self.assertIsNotNone(gate.publication)
        for _decoder, owner, _binding in ranks:
            self.assertEqual(tuple(owner.active_state), STATE_FAMILIES)
            self.assertEqual(owner.active_policy_state, self.policy_state)
            self.assertEqual(owner.phase, OwnerPhase.OPEN)
        controller_spans = [
            span
            for span in self.tracer.spans
            if span.name == "rocket.qwen38.state.controller_restore"
        ]
        self.assertEqual(len(controller_spans), 1)
        self.assertEqual(
            controller_spans[0].attributes,
            {"phase": "restore", "rank": -1, "family": "none", "outcome": "success"},
        )


if __name__ == "__main__":
    unittest.main()

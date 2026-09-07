from __future__ import annotations

import hashlib
import unittest

from qwen38_slab.device_decode import DevicePhase, DevicePublication
from qwen38_slab.runtime_state import RuntimeBoundary, RuntimeStateError
from qwen38_slab.state_owner import (
    CoordinatorPhase,
    DecoderStateOwner,
    DecoderStateOwnerError,
    TwoRankRestoreCoordinator,
)
from qwen38_slab.state_txn import AuthenticatedState, FamilyPayload, STATE_FAMILIES


class Span:
    def __init__(self): self.attributes = {}; self.exceptions = []
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, exception): self.exceptions.append(type(exception).__name__)


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, name):
        span = Span(); span.name = name; self.spans.append(span); return span


class DecoderBinding:
    def __init__(self, rank):
        self.rank = rank
        self.phase = DevicePhase.IDLE
        self.publication = None
        self.launches = []

    def upload_and_launch(self, prepared):
        self.launches.append(prepared)
        self.publication = DevicePublication(prepared, 1, prepared % 2)
        return self.publication


class RuntimeBinding:
    def __init__(self, rank, owner, events):
        self.rank = rank
        self.owner = owner
        self.state_owner = owner
        self.events = events
        self.fail = False

    def restore(self, authenticated, generation_epoch):
        self.events.append(("restore", self.rank, authenticated.boundary, generation_epoch))
        boundary = self.owner.accepted_boundary
        self.owner.close_launch_gate(boundary)
        if self.fail:
            raise RuntimeStateError(f"rank {self.rank} restore")
        self.owner.publish_state(
            {family: object() for family in STATE_FAMILIES}, boundary
        )
        self.owner.open_launch_gate(boundary)


class OwnerProxy:
    def __init__(self, owner):
        self.owner = owner

    @property
    def rank(self): return self.owner.rank
    @property
    def accepted_boundary(self): return self.owner.accepted_boundary
    def hold_launch_gate(self, boundary): self.owner.hold_launch_gate(boundary)
    def release_launch_gate(self, boundary): self.owner.release_launch_gate(boundary)
    def fault_closed(self, boundary): self.owner.fault_closed(boundary)


class ProxyBinding:
    def __init__(self, rank, owner, events):
        self.rank = rank
        self.state_owner = owner
        self.events = events

    def restore(self, authenticated, generation_epoch):
        self.events.append((self.rank, authenticated.boundary, generation_epoch))


class StateOwnerTests(unittest.TestCase):
    def setUp(self):
        self.digest = hashlib.sha256(b"accepted-41").hexdigest()
        self.boundary = RuntimeBoundary(41, self.digest, 7)
        payloads = {
            family: FamilyPayload(f"state:{family}".encode())
            for family in STATE_FAMILIES
        }
        self.authenticated = AuthenticatedState._from_verified(
            token_count=41,
            token_hash=self.digest,
            rank_payloads={0: payloads, 1: payloads},
        )

    def _owner(self, rank):
        tracer = Tracer()
        decoder = DecoderBinding(rank)
        owner = DecoderStateOwner(rank=rank, decoder=decoder, tracer=tracer)
        publication = owner.upload_and_launch(7)
        owner.accept_boundary(publication, self.boundary)
        return owner, decoder, tracer

    def test_owner_gates_decoder_and_atomically_replaces_complete_pointer_table(self):
        owner, decoder, _tracer = self._owner(0)
        self.assertEqual(owner.close_launch_gate(self.boundary), self.boundary)
        with self.assertRaisesRegex(DecoderStateOwnerError, "closed"):
            owner.upload_and_launch(8)
        table = {family: object() for family in STATE_FAMILIES}
        owner.publish_state(table, self.boundary)
        self.assertEqual(tuple(owner.active_state), STATE_FAMILIES)
        owner.open_launch_gate(self.boundary)
        self.assertEqual(owner.upload_and_launch(8).generation, 8)
        self.assertEqual(decoder.launches, [7, 8])

    def test_owner_rejects_stale_publication_boundary_and_partial_table(self):
        owner, decoder, _tracer = self._owner(0)
        with self.assertRaisesRegex(DecoderStateOwnerError, "current"):
            owner.accept_boundary(DevicePublication(6, 1, 0), self.boundary)
        owner.close_launch_gate(self.boundary)
        partial = {family: object() for family in STATE_FAMILIES[:-1]}
        with self.assertRaisesRegex(DecoderStateOwnerError, "nine-family"):
            owner.publish_state(partial, self.boundary)
        self.assertIsNone(owner.active_state)
        owner.open_launch_gate(self.boundary)
        self.assertEqual(decoder.phase, DevicePhase.IDLE)

    def test_two_rank_match_restores_both_and_mismatch_calls_neither(self):
        owners = (self._owner(0)[0], self._owner(1)[0])
        events = []
        bindings = (
            RuntimeBinding(0, owners[0], events),
            RuntimeBinding(1, owners[1], events),
        )
        coordinator = TwoRankRestoreCoordinator(owners, bindings, Tracer())
        coordinator.restore(self.authenticated, generation_epoch=7)
        self.assertEqual([event[:2] for event in events], [("restore", 0), ("restore", 1)])
        self.assertEqual(coordinator.phase, CoordinatorPhase.IDLE)

        other_boundary = RuntimeBoundary(42, hashlib.sha256(b"other").hexdigest(), 7)
        other_tracer = Tracer()
        other_decoder = DecoderBinding(1)
        other = DecoderStateOwner(rank=1, decoder=other_decoder, tracer=other_tracer)
        other.accept_boundary(other.upload_and_launch(7), other_boundary)
        rejected_events = []
        rejected = TwoRankRestoreCoordinator(
            (owners[0], other),
            (
                RuntimeBinding(0, owners[0], rejected_events),
                RuntimeBinding(1, other, rejected_events),
            ),
            Tracer(),
        )
        with self.assertRaisesRegex(DecoderStateOwnerError, "boundary"):
            rejected.restore(self.authenticated, generation_epoch=7)
        self.assertEqual(rejected_events, [])
        self.assertEqual(rejected.phase, CoordinatorPhase.IDLE)

    def test_rank_failure_faults_both_decoder_owners_closed(self):
        owners = (self._owner(0)[0], self._owner(1)[0])
        events = []
        bindings = (
            RuntimeBinding(0, owners[0], events),
            RuntimeBinding(1, owners[1], events),
        )
        bindings[1].fail = True
        coordinator = TwoRankRestoreCoordinator(owners, bindings, Tracer())
        with self.assertRaisesRegex(DecoderStateOwnerError, "rank 1"):
            coordinator.restore(self.authenticated, generation_epoch=7)
        self.assertEqual(coordinator.phase, CoordinatorPhase.FAULTED)
        self.assertTrue(all(owner.faulted for owner in owners))
        for owner in owners:
            with self.assertRaisesRegex(DecoderStateOwnerError, "faulted"):
                owner.upload_and_launch(8)

    def test_coordinator_rejects_binding_with_another_pointer_table_owner(self):
        owners = (self._owner(0)[0], self._owner(1)[0])
        impostor = self._owner(0)[0]
        with self.assertRaisesRegex(DecoderStateOwnerError, "ordered rank"):
            TwoRankRestoreCoordinator(
                owners,
                (
                    RuntimeBinding(0, impostor, []),
                    RuntimeBinding(1, owners[1], []),
                ),
                Tracer(),
            )

    def test_coordinator_accepts_strict_physical_owner_proxy_contract(self):
        concrete = (self._owner(0)[0], self._owner(1)[0])
        proxies = (OwnerProxy(concrete[0]), OwnerProxy(concrete[1]))
        events = []
        coordinator = TwoRankRestoreCoordinator(
            proxies,
            (
                ProxyBinding(0, proxies[0], events),
                ProxyBinding(1, proxies[1], events),
            ),
            Tracer(),
        )
        coordinator.restore(self.authenticated, generation_epoch=7)
        self.assertEqual([event[0] for event in events], [0, 1])
        self.assertTrue(all(owner.phase.value == "open" for owner in concrete))

    def test_owner_and_coordinator_otel_dimensions_are_bounded(self):
        owner0, _decoder0, tracer0 = self._owner(0)
        owner1, _decoder1, tracer1 = self._owner(1)
        events = []
        tracer = Tracer()
        coordinator = TwoRankRestoreCoordinator(
            (owner0, owner1),
            (
                RuntimeBinding(0, owner0, events),
                RuntimeBinding(1, owner1, events),
            ),
            tracer,
        )
        coordinator.restore(self.authenticated, generation_epoch=7)
        spans = tracer0.spans + tracer1.spans + tracer.spans
        allowed = {"phase", "rank", "outcome"}
        self.assertTrue(spans)
        self.assertTrue(all(set(span.attributes) == allowed for span in spans))
        self.assertTrue(all(span.attributes["phase"] in {
            "accept", "launch", "hold", "close", "publish", "open", "release",
            "validate", "restore"
        } for span in spans))
        self.assertTrue(all(span.attributes["rank"] in {-1, 0, 1} for span in spans))
        self.assertTrue(all(span.attributes["outcome"] in {"success", "failure"} for span in spans))


if __name__ == "__main__":
    unittest.main()

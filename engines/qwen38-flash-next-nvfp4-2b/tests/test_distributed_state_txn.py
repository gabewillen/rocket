from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from qwen38_slab.distributed_state_txn import (
    DistributedStateCoordinator,
    GeneratedFamilySource,
    LocalRankEndpoint,
    RankStateStore,
)
from qwen38_slab.decode import Depth
from qwen38_slab.mtp_policy import (
    AdaptiveMtpPolicy,
    ConcurrencyCeiling,
    PolicyConfig,
)
from qwen38_slab.state_txn import (
    STATE_FAMILIES,
    AcceptedBoundary,
    StateIdentity,
    StateTransactionError,
    Transition,
)


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


class DistributedStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temporary.name)
        self.identity = StateIdentity(
            "fc694-distributed", hashlib.sha256(b"precision").hexdigest()
        )
        self.boundary = AcceptedBoundary(
            262144, hashlib.sha256(b"accepted-boundary").hexdigest(), True
        )
        self.policy_digest = hashlib.sha256(b"typed-k0-k7-policy-slot").hexdigest()
        policy = AdaptiveMtpPolicy(
            Tracer(),
            PolicyConfig((ConcurrencyCeiling(1, Depth.K7), ConcurrencyCeiling(16, Depth.K1))),
        )
        self.policy_state = policy.dump_state(policy.initial_state())
        self.policy_digest = hashlib.sha256(self.policy_state).hexdigest()
        self.policy = policy

    def tearDown(self):
        self.temporary.cleanup()

    def _coordinator(self, root):
        publications = ([], [])
        endpoints = []
        for rank in range(2):
            sources = {
                family: GeneratedFamilySource(17 + index, rank * 16 + index + 1)
                for index, family in enumerate(STATE_FAMILIES)
            }
            store = RankStateStore(
                rank, root / f"rank{rank}", self.identity, Tracer(), self.policy
            )
            endpoints.append(
                LocalRankEndpoint(store, sources, publications[rank].append)
            )
        return DistributedStateCoordinator(tuple(endpoints), Tracer()), publications

    def test_receipts_coordinate_owner_local_payloads_and_publish_complete(self):
        coordinator, publications = self._coordinator(self.root / "complete")
        coordinator.commit(
            "session", "txn", self.boundary, policy_state=self.policy_state
        )
        receipts = coordinator.restore("session")
        self.assertEqual(tuple(receipt.rank for receipt in receipts), (0, 1))
        self.assertTrue(all(receipt.policy_digest == self.policy_digest for receipt in receipts))
        self.assertEqual(tuple(len(items) for items in publications), (1, 1))
        for rank, items in enumerate(publications):
            descriptor = items[0]
            self.assertEqual(descriptor.rank, rank)
            self.assertEqual(tuple(descriptor.families), STATE_FAMILIES)
            self.assertEqual(descriptor.boundary, self.boundary)
            self.assertEqual(descriptor.policy_state, self.policy_state)
            self.assertEqual(descriptor.commit_sha256, receipts[rank].commit_sha256)
        spans = coordinator.tracer.spans
        self.assertTrue(spans)
        self.assertTrue(all(
            set(span.attributes) == {"phase", "rank", "family", "outcome"}
            for span in spans
        ))
        self.assertTrue(all(span.attributes["phase"] in {
            "prepare", "commit", "index", "inspect", "authenticate", "publish"
        } for span in spans))
        self.assertTrue(all(span.attributes["rank"] in {0, 1} for span in spans))
        self.assertTrue(all(span.attributes["family"] == "none" for span in spans))

    def test_six_crash_points_publish_none_until_second_index(self):
        for transition in Transition:
            with self.subTest(transition=transition):
                coordinator, publications = self._coordinator(
                    self.root / transition.value
                )

                def inject(observed):
                    if observed is transition:
                        raise InjectedCrash(transition.value)

                with self.assertRaises(InjectedCrash):
                    coordinator.commit(
                        "session", "txn", self.boundary,
                        policy_state=self.policy_state, inject_fault=inject,
                    )
                if transition is Transition.AFTER_INDEX_RANK1:
                    receipts = coordinator.restore("session")
                    self.assertEqual(len(receipts), 2)
                    self.assertEqual(tuple(len(items) for items in publications), (1, 1))
                else:
                    with self.assertRaises(StateTransactionError):
                        coordinator.restore("session")
                    self.assertEqual(publications, ([], []))

    def test_invalid_policy_digest_and_cross_rank_receipt_drift_fail_closed(self):
        coordinator, publications = self._coordinator(self.root / "invalid")
        with self.assertRaisesRegex(StateTransactionError, "policy"):
            coordinator.commit("session", "txn", self.boundary, policy_state=b"")
        self.assertEqual(publications, ([], []))

    def test_rank_policy_configuration_mismatch_fails_before_commit(self):
        coordinator, publications = self._coordinator(self.root / "mismatch")
        mismatched = AdaptiveMtpPolicy(
            Tracer(),
            PolicyConfig((
                ConcurrencyCeiling(1, Depth.K6),
                ConcurrencyCeiling(16, Depth.K1),
            )),
        )
        coordinator.endpoints[1].store.policy = mismatched
        with self.assertRaisesRegex(StateTransactionError, "policy"):
            coordinator.commit(
                "session", "txn", self.boundary, policy_state=self.policy_state
            )
        self.assertEqual(publications, ([], []))

    def test_rank_policy_payload_tamper_blocks_both_publications(self):
        coordinator, publications = self._coordinator(self.root / "tamper")
        coordinator.commit(
            "session", "txn", self.boundary, policy_state=self.policy_state
        )
        policy_path = (
            self.root / "tamper" / "rank1" / "transactions" / "txn" / "POLICY.json"
        )
        policy_path.write_bytes(self.policy_state + b" ")
        with self.assertRaisesRegex(StateTransactionError, "policy"):
            coordinator.restore("session")
        self.assertEqual(publications, ([], []))


class InjectedCrash(RuntimeError):
    pass


if __name__ == "__main__":
    unittest.main()

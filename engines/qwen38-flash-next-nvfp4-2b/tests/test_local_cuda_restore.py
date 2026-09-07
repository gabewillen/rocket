from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from threading import Barrier, Event, Thread

from qwen38_slab.decode import Depth
from qwen38_slab.device_decode import DevicePhase, DevicePublication
from qwen38_slab.distributed_state_txn import (
    AuthenticatedFamilyExtent,
    LocalAuthenticatedState,
)
from qwen38_slab.local_cuda_restore import (
    DecoderStateTransactionGate,
    EXACT_C16_LOGICAL_BYTES_PER_RANK,
    LocalCudaRestoreError,
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
    def __init__(self): self.attributes = {}
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, exception): del exception


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, name):
        span = Span(); span.name = name; self.spans.append(span); return span


class Decoder:
    def __init__(self): self.phase = DevicePhase.IDLE; self.publication = None
    def upload_and_launch(self, prepared):
        generation = prepared if isinstance(prepared, int) else prepared.lease.generation
        self.publication = DevicePublication(generation, 1, generation % 2)
        return self.publication


class Runtime:
    def __init__(
        self, owner, fail_family=None, fail_after_commit=False,
        fail_finalize=False,
    ):
        self.owner = owner; self.fail_family = fail_family
        self.fail_after_commit = fail_after_commit
        self.fail_finalize = fail_finalize; self.staged = {}
    def quiesce(self, boundary):
        return QuiesceReceipt(self.owner.close_launch_gate(boundary), True, 0)
    def allocate_staging(self, family, logical_bytes):
        value = (family, logical_bytes); self.staged[family] = value; return value
    def copy_extent_to_device(self, destination, extent):
        if extent.family == self.fail_family: raise RuntimeError("copy")
        if destination[1] != extent.logical_bytes: raise RuntimeError("extent")
    def finish_transfers(self): pass
    def prepare_local(self, staged, boundary, policy_state, commit_sha256):
        self.owner.prepare_state_with_policy(
            staged, policy_state, boundary, commit_sha256
        )
    def commit_local(self, boundary, commit_sha256):
        self.owner.commit_prepared_state(boundary, commit_sha256)
        if self.fail_after_commit: raise RuntimeError("commit publication")
    def rollback_local(self, boundary, commit_sha256):
        self.owner.rollback_prepared_state(boundary, commit_sha256)
    def finalize_local(self, boundary, commit_sha256):
        if self.fail_finalize: raise RuntimeError("finalize acknowledgement")
        self.owner.finalize_prepared_state(boundary, commit_sha256)
    def complete_local(self, boundary, commit_sha256):
        self.owner._discard_prepared_rollback()
    def discard(self, staged): self.staged.clear()
    def resume(self, boundary): self.owner.open_launch_gate(boundary)


class LocalCudaRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan = json.loads(Path("scripts/memory/qwen38-state-capacity-plan.json").read_text())
        cls.extents = {
            family: item["logical_bytes_per_stream"] * 16
            for family, item in zip(STATE_FAMILIES, plan["families"], strict=True)
        }
        assert sum(cls.extents.values()) == EXACT_C16_LOGICAL_BYTES_PER_RANK

    def setUp(self):
        self.policy = AdaptiveMtpPolicy(
            Tracer(), PolicyConfig((
                ConcurrencyCeiling(1, Depth.K7),
                ConcurrencyCeiling(16, Depth.K1),
            )),
        )
        self.policy_state = self.policy.dump_state(self.policy.initial_state())
        self.policy_digest = hashlib.sha256(self.policy_state).hexdigest()
        self.accepted = AcceptedBoundary(
            262144, hashlib.sha256(b"shared-token-boundary").hexdigest(), True
        )
        self.runtime_boundary = RuntimeBoundary(
            self.accepted.token_count, self.accepted.token_hash, 7
        )

    def _state(self, rank, *, policy_state=None, commit_sha256=None):
        families = {
            family: AuthenticatedFamilyExtent(
                family, Path(f"/rank{rank}/{index}"), logical, logical,
                hashlib.sha256(f"logical-{rank}-{index}".encode()).hexdigest(),
                hashlib.sha256(f"padded-{rank}-{index}".encode()).hexdigest(),
            )
            for index, (family, logical) in enumerate(self.extents.items())
        }
        state = self.policy_state if policy_state is None else policy_state
        return LocalAuthenticatedState._from_verified(
            rank, self.accepted, hashlib.sha256(state).hexdigest(), state, families,
            commit_sha256 or hashlib.sha256(b"shared-durable-commit").hexdigest(),
        )

    def _rank(
        self, rank, fail_family=None, fail_after_commit=False,
        fail_finalize=False,
    ):
        decoder = Decoder(); owner = DecoderStateOwner(rank=rank, decoder=decoder, tracer=Tracer())
        owner.accept_boundary(owner.upload_and_launch(7), self.runtime_boundary)
        runtime = Runtime(owner, fail_family, fail_after_commit, fail_finalize)
        binding = LocalCudaStateBinding(
            rank, runtime, self.policy, self.extents, Tracer()
        )
        return decoder, owner, binding

    def _coordinator(self, ranks):
        return TwoRankLocalCudaCoordinator(
            tuple(value[1] for value in ranks),
            tuple(value[2] for value in ranks), Tracer(),
        )

    def _seed_previous(self, ranks):
        previous = []
        for _decoder, owner, _binding in ranks:
            table = {family: object() for family in STATE_FAMILIES}
            owner.close_launch_gate(self.runtime_boundary)
            owner.publish_state(table, self.runtime_boundary)
            owner.open_launch_gate(self.runtime_boundary)
            previous.append(dict(owner.active_state))
        return previous

    def _assert_previous_faulted(self, ranks, previous):
        for expected, (_decoder, owner, _binding) in zip(
            previous, ranks, strict=True
        ):
            self.assertEqual(dict(owner.active_state), expected)
            self.assertIsNone(owner.active_policy_state)
            self.assertIsNone(owner.active_commit_sha256)
            self.assertEqual(owner.phase, OwnerPhase.FAULTED)

    def test_exact_tp2_restore_publishes_policy_and_reopens_both_gates(self):
        ranks = (self._rank(0), self._rank(1))
        coordinator = TwoRankLocalCudaCoordinator(
            tuple(value[1] for value in ranks),
            tuple(value[2] for value in ranks), Tracer(),
        )
        publication = coordinator.restore((self._state(0), self._state(1)), 7)
        self.assertEqual(publication.boundary, self.runtime_boundary)
        self.assertEqual(publication.family_count, len(STATE_FAMILIES))
        for decoder, owner, _binding in ranks:
            self.assertEqual(owner.phase, OwnerPhase.OPEN)
            self.assertEqual(tuple(owner.active_state), STATE_FAMILIES)
            self.assertEqual(owner.active_policy_state, self.policy_state)
            self.assertEqual(owner.upload_and_launch(8).generation, 8)

    def test_rank_failure_faults_both_launch_gates(self):
        ranks = (self._rank(0), self._rank(1, STATE_FAMILIES[-1]))
        coordinator = TwoRankLocalCudaCoordinator(
            tuple(value[1] for value in ranks),
            tuple(value[2] for value in ranks), Tracer(),
        )
        with self.assertRaisesRegex(LocalCudaRestoreError, "rank 1"):
            coordinator.restore((self._state(0), self._state(1)), 7)
        self.assertTrue(all(value[1].phase is OwnerPhase.FAULTED for value in ranks))
        self.assertTrue(all(value[1].active_state is None for value in ranks))
        for _decoder, owner, _binding in ranks:
            with self.assertRaisesRegex(Exception, "faulted"):
                owner.upload_and_launch(8)

    def test_one_rank_durable_commit_never_reaches_decoder_publication(self):
        ranks = (self._rank(0), self._rank(1))
        coordinator = TwoRankLocalCudaCoordinator(
            tuple(value[1] for value in ranks),
            tuple(value[2] for value in ranks), Tracer(),
        )
        states = (
            self._state(0),
            self._state(
                1, commit_sha256=hashlib.sha256(b"rank1-only-commit").hexdigest()
            ),
        )
        with self.assertRaisesRegex(LocalCudaRestoreError, "commit"):
            coordinator.restore(states, 7)
        self.assertTrue(all(value[1].active_state is None for value in ranks))

    def test_durable_rank_callbacks_share_the_exact_decoder_gate(self):
        ranks = (self._rank(0), self._rank(1))
        tracer = Tracer()
        coordinator = TwoRankLocalCudaCoordinator(
            tuple(value[1] for value in ranks),
            tuple(value[2] for value in ranks), tracer,
        )
        gate = DecoderStateTransactionGate(coordinator, 7, tracer)
        gate.publisher(0)(self._state(0))
        self.assertIsNone(gate.publication)
        self.assertTrue(all(value[1].active_state is None for value in ranks))
        gate.publisher(1)(self._state(1))
        self.assertEqual(gate.publication.boundary, self.runtime_boundary)
        self.assertTrue(all(tuple(value[1].active_state) == STATE_FAMILIES for value in ranks))
        gate_spans = [
            span for span in tracer.spans
            if span.name == "rocket.qwen38.state.decoder_transaction_gate"
        ]
        self.assertEqual(
            {(span.attributes["phase"], span.attributes["rank"]) for span in gate_spans},
            {("accept", 0), ("accept", 1), ("publish", -1)},
        )
        self.assertTrue(all(
            set(span.attributes) == {"phase", "rank", "family", "outcome"}
            and span.attributes["family"] == "none"
            for span in gate_spans
        ))

    def test_concurrent_rank_callbacks_publish_exactly_once(self):
        for attempt in range(8):
            with self.subTest(attempt=attempt):
                ranks = (self._rank(0), self._rank(1))
                gate = DecoderStateTransactionGate(
                    self._coordinator(ranks), 7, Tracer()
                )
                restores = []
                original_restore = gate.coordinator.restore

                def counted_restore(states, generation_epoch):
                    restores.append((states, generation_epoch))
                    return original_restore(states, generation_epoch)

                gate.coordinator.restore = counted_restore
                barrier = Barrier(2)
                failures = []

                def accept(rank):
                    try:
                        barrier.wait()
                        gate.publisher(rank)(self._state(rank))
                    except BaseException as exc:
                        failures.append(exc)

                threads = tuple(Thread(target=accept, args=(rank,)) for rank in (0, 1))
                for thread in threads: thread.start()
                for thread in threads: thread.join(2)
                self.assertFalse(failures)
                self.assertIsNotNone(gate.publication)
                self.assertEqual(len(restores), 1)
                self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_concurrent_duplicate_rank_callbacks_fault_once(self):
        ranks = (self._rank(0), self._rank(1))
        gate = DecoderStateTransactionGate(self._coordinator(ranks), 7, Tracer())
        restores = []
        original_restore = gate.coordinator.restore

        def counted_restore(states, generation_epoch):
            restores.append((states, generation_epoch))
            return original_restore(states, generation_epoch)

        gate.coordinator.restore = counted_restore
        barrier = Barrier(2)
        failures = []

        def accept():
            try:
                barrier.wait()
                gate.publisher(0)(self._state(0))
            except BaseException as exc:
                failures.append(exc)

        threads = (Thread(target=accept), Thread(target=accept))
        for thread in threads: thread.start()
        for thread in threads: thread.join(2)
        self.assertEqual(len(failures), 1)
        self.assertRegex(str(failures[0]), "invalid")
        with self.assertRaisesRegex(LocalCudaRestoreError, "terminal"):
            gate.publisher(1)(self._state(1))
        self.assertIsNone(gate.publication)
        self.assertFalse(restores)

    def test_owner_rejects_a_second_publication_coordinator(self):
        ranks = (self._rank(0), self._rank(1))
        self._coordinator(ranks)
        with self.assertRaisesRegex(Exception, "already bound"):
            self._coordinator(ranks)

    def test_post_swap_rank_failure_rolls_both_active_generations_back(self):
        ranks = (self._rank(0), self._rank(1, fail_after_commit=True))
        previous = self._seed_previous(ranks)
        coordinator = self._coordinator(ranks)
        with self.assertRaisesRegex(LocalCudaRestoreError, "rank 1"):
            coordinator.restore((self._state(0), self._state(1)), 7)
        self._assert_previous_faulted(ranks, previous)

    def test_finalize_ack_failure_retains_undo_for_both_ranks(self):
        for failed_rank in (0, 1):
            with self.subTest(failed_rank=failed_rank):
                ranks = tuple(
                    self._rank(rank, fail_finalize=rank == failed_rank)
                    for rank in (0, 1)
                )
                previous = self._seed_previous(ranks)
                with self.assertRaisesRegex(
                    LocalCudaRestoreError, f"rank {failed_rank}"
                ):
                    self._coordinator(ranks).restore(
                        (self._state(0), self._state(1)), 7
                    )
                self._assert_previous_faulted(ranks, previous)

    def test_release_ack_failure_precedes_global_publication(self):
        for failed_rank in (0, 1):
            with self.subTest(failed_rank=failed_rank):
                ranks = (self._rank(0), self._rank(1))
                previous = self._seed_previous(ranks)
                owner = ranks[failed_rank][1]

                def fail_release(_boundary):
                    raise RuntimeError("release acknowledgement")

                owner.validate_launch_gate_release = fail_release
                with self.assertRaisesRegex(
                    LocalCudaRestoreError, f"rank {failed_rank}"
                ):
                    self._coordinator(ranks).restore(
                        (self._state(0), self._state(1)), 7
                    )
                self._assert_previous_faulted(ranks, previous)

    def test_global_publication_lock_blocks_launch_between_rank_releases(self):
        for attempt in range(8):
            with self.subTest(attempt=attempt):
                ranks = (self._rank(0), self._rank(1))
                coordinator = self._coordinator(ranks)
                first_released = Event()
                continue_release = Event()
                launch_done = Event()
                results = []
                failures = []
                owner = ranks[0][1]
                original = owner._commit_launch_gate_release

                def pause_after_first(boundary):
                    original(boundary)
                    first_released.set()
                    if not continue_release.wait(2):
                        raise RuntimeError("test release barrier timed out")

                owner._commit_launch_gate_release = pause_after_first

                def restore():
                    try:
                        results.append(coordinator.restore(
                            (self._state(0), self._state(1)), 7
                        ))
                    except BaseException as exc:
                        failures.append(exc)

                def launch():
                    try:
                        ranks[0][1].upload_and_launch(8)
                    except BaseException as exc:
                        failures.append(exc)
                    finally:
                        launch_done.set()

                restore_thread = Thread(target=restore)
                restore_thread.start()
                self.assertTrue(first_released.wait(2))
                launch_thread = Thread(target=launch)
                launch_thread.start()
                self.assertFalse(launch_done.wait(0.01))
                self.assertIsNone(coordinator.publication)
                continue_release.set()
                restore_thread.join(2)
                launch_thread.join(2)
                self.assertFalse(restore_thread.is_alive())
                self.assertFalse(launch_thread.is_alive())
                self.assertFalse(failures)
                self.assertIs(coordinator.publication, results[0])

    def test_blocked_launch_wakes_faulted_when_release_ack_fails(self):
        ranks = (self._rank(0), self._rank(1))
        previous = self._seed_previous(ranks)
        coordinator = self._coordinator(ranks)
        release_entered = Event()
        fail_release = Event()
        launch_done = Event()
        restore_failures = []
        launch_failures = []

        def pause_then_fail(_boundary):
            release_entered.set()
            if not fail_release.wait(2):
                raise RuntimeError("test release barrier timed out")
            raise RuntimeError("release acknowledgement")

        ranks[1][1].validate_launch_gate_release = pause_then_fail

        def restore():
            try:
                coordinator.restore((self._state(0), self._state(1)), 7)
            except BaseException as exc:
                restore_failures.append(exc)

        def launch():
            try:
                ranks[0][1].upload_and_launch(8)
            except BaseException as exc:
                launch_failures.append(exc)
            finally:
                launch_done.set()

        restore_thread = Thread(target=restore)
        restore_thread.start()
        self.assertTrue(release_entered.wait(2))
        launch_thread = Thread(target=launch)
        launch_thread.start()
        self.assertFalse(launch_done.wait(0.01))
        fail_release.set()
        restore_thread.join(2)
        launch_thread.join(2)
        self.assertEqual(len(restore_failures), 1)
        self.assertEqual(len(launch_failures), 1)
        self.assertRegex(str(launch_failures[0]), "faulted")
        self.assertIsNone(coordinator.publication)
        self._assert_previous_faulted(ranks, previous)

    def test_cross_rank_policy_mismatch_fails_before_holding_gates(self):
        ranks = (self._rank(0), self._rank(1))
        coordinator = TwoRankLocalCudaCoordinator(
            tuple(value[1] for value in ranks),
            tuple(value[2] for value in ranks), Tracer(),
        )
        other = self.policy_state.replace(b'"decisions":0', b'"decisions":1')
        with self.assertRaisesRegex(LocalCudaRestoreError, "does not agree"):
            coordinator.restore((self._state(0), self._state(1, policy_state=other)), 7)
        self.assertTrue(all(value[1].phase is OwnerPhase.OPEN for value in ranks))

    def test_unsealed_state_and_nonexact_extent_plan_are_rejected(self):
        with self.assertRaises(TypeError): LocalAuthenticatedState()
        decoder, owner, _binding = self._rank(0)
        runtime = Runtime(owner)
        changed = dict(self.extents); changed[STATE_FAMILIES[0]] -= 1
        with self.assertRaisesRegex(LocalCudaRestoreError, "exact c16"):
            LocalCudaStateBinding(0, runtime, self.policy, changed, Tracer())


if __name__ == "__main__": unittest.main()

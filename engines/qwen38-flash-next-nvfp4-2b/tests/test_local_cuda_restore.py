from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from qwen38_slab.decode import Depth
from qwen38_slab.device_decode import DevicePhase, DevicePublication
from qwen38_slab.distributed_state_txn import (
    AuthenticatedFamilyExtent,
    LocalAuthenticatedState,
)
from qwen38_slab.local_cuda_restore import (
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
    def __init__(self, owner, fail_family=None):
        self.owner = owner; self.fail_family = fail_family; self.staged = {}
    def quiesce(self, boundary):
        return QuiesceReceipt(self.owner.close_launch_gate(boundary), True, 0)
    def allocate_staging(self, family, logical_bytes):
        value = (family, logical_bytes); self.staged[family] = value; return value
    def copy_extent_to_device(self, destination, extent):
        if extent.family == self.fail_family: raise RuntimeError("copy")
        if destination[1] != extent.logical_bytes: raise RuntimeError("extent")
    def finish_transfers(self): pass
    def publish_local(self, staged, boundary, policy_state):
        self.owner.publish_state_with_policy(staged, policy_state, boundary)
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

    def _state(self, rank, *, policy_state=None):
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
            rank, self.accepted, hashlib.sha256(state).hexdigest(), state, families
        )

    def _rank(self, rank, fail_family=None):
        decoder = Decoder(); owner = DecoderStateOwner(rank=rank, decoder=decoder, tracer=Tracer())
        owner.accept_boundary(owner.upload_and_launch(7), self.runtime_boundary)
        runtime = Runtime(owner, fail_family)
        binding = LocalCudaStateBinding(
            rank, runtime, self.policy, self.extents, Tracer()
        )
        return decoder, owner, binding

    def test_exact_tp2_restore_publishes_policy_and_reopens_both_gates(self):
        ranks = (self._rank(0), self._rank(1))
        coordinator = TwoRankLocalCudaCoordinator(
            tuple(value[1] for value in ranks),
            tuple(value[2] for value in ranks), Tracer(),
        )
        coordinator.restore((self._state(0), self._state(1)), 7)
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
        for _decoder, owner, _binding in ranks:
            with self.assertRaisesRegex(Exception, "faulted"):
                owner.upload_and_launch(8)

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

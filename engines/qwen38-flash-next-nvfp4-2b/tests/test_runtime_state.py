from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from qwen38_slab.runtime_state import (
    BindingPhase,
    CudaQuiesceError,
    CudaStateBinding,
    DeviceState,
    MAX_FAMILY_BYTES,
    QuiesceReceipt,
    RuntimeBoundary,
    RuntimeStateError,
)
from qwen38_slab.state_txn import (
    AcceptedBoundary,
    AuthenticatedState,
    FamilyPayload,
    STATE_FAMILIES,
    StateIdentity,
    StateTransactionStore,
)


class Span:
    def __init__(self):
        self.attributes = {}
        self.exceptions = []

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, exception): self.exceptions.append(type(exception).__name__)


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, name):
        span = Span(); span.name = name; self.spans.append(span); return span


class SyntheticCudaRuntime:
    """Deterministic contract fake; each operation is one CUDA adapter boundary."""

    def __init__(self):
        self.events = []
        self.device = {
            family: (f"accepted:{family}".encode() + b":speculative")
            for family in STATE_FAMILIES
        }
        self.published = None
        self.fail_at = None
        self.on_copy = None

    def _event(self, name, *values):
        self.events.append((name, *values))
        if self.fail_at == name or (
            isinstance(self.fail_at, set) and name in self.fail_at
        ):
            raise SyntheticCudaFailure(name)

    def quiesce(self, boundary):
        self.events.append(("quiesce", boundary.generation_epoch))
        if self.fail_at == "quiesce":
            raise CudaQuiesceError("synthetic safe quiesce failure", safe_to_retry=True)
        return QuiesceReceipt(boundary, compute_fenced=True, pending_launches=0)

    def copy_device_to_host(self, source, logical_bytes):
        self._event("copy_d2h", source.family, logical_bytes)
        if self.on_copy is not None:
            callback, self.on_copy = self.on_copy, None
            callback()
        return bytes(self.device[source.family][:logical_bytes])

    def allocate_staging(self, family, logical_bytes):
        self._event("allocate", family, logical_bytes)
        return {"family": family, "payload": None, "bytes": logical_bytes}

    def copy_host_to_device(self, destination, payload):
        self._event("copy_h2d", destination["family"], len(payload))
        destination["payload"] = bytes(payload)

    def finish_transfers(self):
        self._event("finish_transfers")

    def publish(self, staged, boundary):
        self._event("publish", boundary.token_count)
        self.published = {
            family: staged[family]["payload"] for family in STATE_FAMILIES
        }

    def discard(self, staged):
        self._event("discard", len(staged))

    def resume(self, boundary):
        self._event("resume", boundary.generation_epoch)


class RuntimeStateTests(unittest.TestCase):
    def setUp(self):
        self.boundary = RuntimeBoundary(
            token_count=37,
            token_hash=hashlib.sha256(b"accepted tokens").hexdigest(),
            generation_epoch=11,
        )
        self.runtime = SyntheticCudaRuntime()
        self.tracer = Tracer()
        self.binding = CudaStateBinding(rank=0, runtime=self.runtime, tracer=self.tracer)
        self.sources = {
            family: DeviceState(
                family=family,
                handle=f"device:{family}",
                accepted_bytes=len(f"accepted:{family}".encode()),
                allocated_bytes=len(self.runtime.device[family]),
            )
            for family in STATE_FAMILIES
        }
        self.payloads = {
            family: FamilyPayload(f"restored:{family}".encode())
            for family in STATE_FAMILIES
        }
        self.authenticated = AuthenticatedState._from_verified(
            token_count=37,
            token_hash=self.boundary.token_hash,
            rank_payloads={0: self.payloads, 1: self.payloads},
        )

    def test_capture_fences_exact_nine_accepted_slices_then_resumes(self):
        accepted, payloads = self.binding.capture(self.boundary, self.sources)
        self.assertEqual(accepted, self.authenticated.boundary)
        self.assertEqual(tuple(payloads), STATE_FAMILIES)
        self.assertTrue(all(payloads[name].speculative_tail == b"" for name in STATE_FAMILIES))
        self.assertTrue(all(
            payloads[name].accepted == f"accepted:{name}".encode()
            for name in STATE_FAMILIES
        ))
        self.assertEqual(self.runtime.events[0], ("quiesce", 11))
        self.assertEqual(self.runtime.events[-1], ("resume", 11))
        self.assertEqual(
            [event[1] for event in self.runtime.events if event[0] == "copy_d2h"],
            list(STATE_FAMILIES),
        )
        self.assertEqual(self.binding.phase, BindingPhase.IDLE)

    def test_largest_measured_c16_family_reaches_adapter_copy_boundary(self):
        sources = dict(self.sources)
        first = STATE_FAMILIES[0]
        sources[first] = DeviceState(
            family=first,
            handle="measured-c16-device",
            accepted_bytes=24 * 1024**3,
            allocated_bytes=24 * 1024**3,
        )
        self.assertEqual(MAX_FAMILY_BYTES, sources[first].accepted_bytes)
        with self.assertRaisesRegex(RuntimeStateError, "invalid extent"):
            self.binding.capture(self.boundary, sources)
        self.assertEqual(self.runtime.events[0], ("quiesce", 11))

    def test_restore_stages_all_families_and_publishes_once_after_transfer_fence(self):
        self.binding.restore(self.authenticated, generation_epoch=12)
        names = [event[0] for event in self.runtime.events]
        self.assertEqual(names.count("publish"), 1)
        self.assertLess(names.index("finish_transfers"), names.index("publish"))
        self.assertLess(names.index("publish"), names.index("resume"))
        self.assertEqual(
            self.runtime.published,
            {family: payload.accepted for family, payload in self.payloads.items()},
        )
        self.assertNotIn("discard", names)
        self.assertEqual(self.binding.phase, BindingPhase.IDLE)

    def test_authenticated_host_restore_flows_directly_into_rank_runtime(self):
        payloads = {
            rank: {
                family: FamilyPayload(f"disk:{rank}:{family}".encode())
                for family in STATE_FAMILIES
            }
            for rank in range(2)
        }
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            store = StateTransactionStore(
                (root / "rank0", root / "rank1"),
                StateIdentity("fc694-integration", hashlib.sha256(b"map").hexdigest()),
                Tracer(),
            )
            boundary = AcceptedBoundary(37, self.boundary.token_hash, quiesced=True)
            store.commit("session", "transaction", boundary, payloads)
            authenticated = store.restore("session", lambda _state: None)
            self.binding.restore(authenticated, generation_epoch=12)
        self.assertEqual(
            self.runtime.published,
            {family: payload.accepted for family, payload in payloads[0].items()},
        )

    def test_validation_fails_before_quiesce_for_inventory_boundary_and_extent_drift(self):
        cases = []
        missing = dict(self.sources); missing.pop(STATE_FAMILIES[-1]); cases.append((self.boundary, missing))
        reordered = dict(reversed(tuple(self.sources.items()))); cases.append((self.boundary, reordered))
        bad_extent = dict(self.sources)
        first = STATE_FAMILIES[0]
        bad_extent[first] = DeviceState(first, "device", 100, 99)
        cases.append((self.boundary, bad_extent))
        bad_hash = RuntimeBoundary(37, "x" * 64, 11); cases.append((bad_hash, self.sources))
        for boundary, sources in cases:
            with self.subTest(boundary=boundary, keys=tuple(sources)):
                runtime = SyntheticCudaRuntime()
                binding = CudaStateBinding(0, runtime, Tracer())
                with self.assertRaises(RuntimeStateError):
                    binding.capture(boundary, sources)
                self.assertEqual(runtime.events, [])

        forged = object.__new__(AuthenticatedState)
        with self.assertRaisesRegex(RuntimeStateError, "host-authenticated"):
            self.binding.restore(forged, generation_epoch=12)
        self.assertEqual(self.runtime.events, [])

    def test_stage_failure_discards_private_allocations_never_publishes_and_resumes(self):
        self.runtime.fail_at = "copy_h2d"
        with self.assertRaisesRegex(RuntimeStateError, "stage"):
            self.binding.restore(self.authenticated, generation_epoch=12)
        names = [event[0] for event in self.runtime.events]
        self.assertNotIn("publish", names)
        self.assertIn("discard", names)
        self.assertEqual(names[-1], "resume")
        self.assertEqual(self.binding.phase, BindingPhase.IDLE)

    def test_each_prepublication_cuda_failure_discards_and_resumes(self):
        for failure in ("allocate", "copy_h2d", "finish_transfers", "publish"):
            with self.subTest(failure=failure):
                runtime = SyntheticCudaRuntime(); runtime.fail_at = failure
                binding = CudaStateBinding(0, runtime, Tracer())
                with self.assertRaises(RuntimeStateError):
                    binding.restore(self.authenticated, generation_epoch=12)
                names = [event[0] for event in runtime.events]
                self.assertIn("discard", names)
                self.assertEqual(names[-1], "resume")
                self.assertIsNone(runtime.published)
                self.assertEqual(binding.phase, BindingPhase.IDLE)

    def test_resume_failure_after_publish_faults_without_discarding_owned_table(self):
        self.runtime.fail_at = "resume"
        with self.assertRaisesRegex(RuntimeStateError, "resume"):
            self.binding.restore(self.authenticated, generation_epoch=12)
        names = [event[0] for event in self.runtime.events]
        self.assertEqual(names.count("publish"), 1)
        self.assertNotIn("discard", names)
        self.assertIsNotNone(self.runtime.published)
        self.assertEqual(self.binding.phase, BindingPhase.FAULTED)

    def test_discard_failure_still_attempts_resume_and_faults_binding(self):
        self.runtime.fail_at = {"copy_h2d", "discard"}
        with self.assertRaisesRegex(RuntimeStateError, "discard"):
            self.binding.restore(self.authenticated, generation_epoch=12)
        names = [event[0] for event in self.runtime.events]
        self.assertNotIn("publish", names)
        self.assertEqual(names[-2:], ["discard", "resume"])
        self.assertEqual(self.binding.phase, BindingPhase.FAULTED)

    def test_typed_safe_quiesce_failure_before_receipt_leaves_binding_idle(self):
        self.runtime.fail_at = "quiesce"
        with self.assertRaisesRegex(RuntimeStateError, "before receipt"):
            self.binding.capture(self.boundary, self.sources)
        self.assertEqual(self.binding.phase, BindingPhase.IDLE)
        self.assertEqual([event[0] for event in self.runtime.events], ["quiesce"])

    def test_bad_quiesce_receipt_stops_before_any_copy_or_publish(self):
        original = self.runtime.quiesce

        def bad_receipt(boundary):
            receipt = original(boundary)
            return QuiesceReceipt(receipt.boundary, compute_fenced=False, pending_launches=1)

        self.runtime.quiesce = bad_receipt
        with self.assertRaisesRegex(RuntimeStateError, "quiesce receipt"):
            self.binding.capture(self.boundary, self.sources)
        self.assertEqual([event[0] for event in self.runtime.events], ["quiesce"])
        self.assertEqual(self.binding.phase, BindingPhase.FAULTED)

    def test_single_owner_binding_rejects_reentry_while_quiesced(self):
        def reenter():
            with self.assertRaisesRegex(RuntimeStateError, "not idle"):
                self.binding.capture(self.boundary, self.sources)

        self.runtime.on_copy = reenter
        self.binding.capture(self.boundary, self.sources)
        self.assertEqual(self.binding.phase, BindingPhase.IDLE)

    def test_capture_cancellation_resumes_before_propagating(self):
        def cancel():
            raise SyntheticCancellation("cancel")

        self.runtime.on_copy = cancel
        with self.assertRaises(SyntheticCancellation):
            self.binding.capture(self.boundary, self.sources)
        self.assertEqual([event[0] for event in self.runtime.events][-1], "resume")
        self.assertEqual(self.binding.phase, BindingPhase.IDLE)

    def test_otel_dimensions_are_finite_and_identifier_free_on_success_and_failure(self):
        self.binding.capture(self.boundary, self.sources)
        failed_runtime = SyntheticCudaRuntime(); failed_runtime.fail_at = "copy_d2h"
        failed_tracer = Tracer(); failed = CudaStateBinding(1, failed_runtime, failed_tracer)
        with self.assertRaises(RuntimeStateError):
            failed.capture(self.boundary, self.sources)
        spans = self.tracer.spans + failed_tracer.spans
        allowed = {"phase", "rank", "family", "outcome"}
        self.assertTrue(spans)
        self.assertTrue(all(set(span.attributes) <= allowed for span in spans))
        self.assertTrue(all(span.attributes["phase"] in {
            "validate", "quiesce", "capture", "stage", "sync", "publish", "discard", "resume"
        } for span in spans))
        self.assertTrue(all(span.attributes["rank"] in {0, 1} for span in spans))
        self.assertTrue(all(span.attributes["family"] in {*STATE_FAMILIES, "none"} for span in spans))
        self.assertTrue(all(span.attributes["outcome"] in {"success", "failure"} for span in spans))
        self.assertTrue(any(span.attributes["outcome"] == "failure" for span in spans))


class SyntheticCudaFailure(RuntimeError):
    pass


class SyntheticCancellation(BaseException):
    pass


if __name__ == "__main__":
    unittest.main()

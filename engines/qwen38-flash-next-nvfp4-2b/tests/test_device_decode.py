from __future__ import annotations

import unittest
from dataclasses import replace

from qwen38_slab.decode import Depth, DepthZeroDecodeExecutor, StreamStep
from qwen38_slab.device_decode import (
    BUFFER_LAYOUT,
    DeviceDecodeError,
    DevicePhase,
    DevicePublication,
    K0DeviceBinding,
)


class Span:
    def __init__(self):
        self.attributes = {}
        self.exceptions = []

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, exception):
        self.exceptions.append(type(exception).__name__)


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, name):
        span = Span(); span.name = name; self.spans.append(span); return span


class SyntheticGraphRuntime:
    """Deterministic fault adapter honoring all-or-none publish."""

    def __init__(self):
        self.events = []
        self.fail_at = None
        self.active = None
        self.next_bank = 0

    def _event(self, name, *values):
        self.events.append((name, *values))
        if self.fail_at == name or (
            isinstance(self.fail_at, set) and name in self.fail_at
        ):
            raise SyntheticCudaFailure(name)

    def stage(self, buffers):
        self._event("stage")
        return {
            "bank": self.next_bank,
            "buffers": {
                name: bytes(getattr(buffers, name)) for name, _length in BUFFER_LAYOUT
            },
        }

    def launch_k0(self, staged, graph_batch):
        self._event("launch", staged["bank"], graph_batch)

    def finish(self, staged):
        self._event("finish", staged["bank"])

    def publish(self, staged, generation, graph_batch):
        self._event("publish", staged["bank"], generation, graph_batch)
        publication = DevicePublication(generation, graph_batch, staged["bank"])
        self.active = (publication, staged["buffers"])
        self.next_bank = 1 - staged["bank"]
        return publication

    def discard(self, staged):
        self._event("discard", staged["bank"])


class DeviceDecodeTests(unittest.TestCase):
    def setUp(self):
        self.tracer = Tracer()
        self.executor = DepthZeroDecodeExecutor(self.tracer)
        self.runtime = SyntheticGraphRuntime()
        self.binding = K0DeviceBinding(self.runtime, self.tracer)

    def _prepare(self, count=2, accepted=10):
        return self.executor.prepare(
            [StreamStep(slot, accepted + slot, Depth.K0) for slot in range(count)]
        )

    def test_success_publishes_complete_alternating_banks_after_fence(self):
        first = self._prepare()
        publication = self.binding.upload_and_launch(first)
        self.assertEqual(publication, DevicePublication(1, 2, 0))
        self.assertEqual(
            [event[0] for event in self.runtime.events],
            ["stage", "launch", "finish", "publish"],
        )
        self.assertEqual(
            self.runtime.active[1],
            {
                name: bytes(getattr(first.buffers, name))
                for name, _length in BUFFER_LAYOUT
            },
        )
        second = self._prepare(count=5, accepted=20)
        self.assertEqual(
            self.binding.upload_and_launch(second), DevicePublication(2, 8, 1)
        )
        self.assertEqual(self.binding.phase, DevicePhase.IDLE)

    def test_each_prepublication_failure_discards_and_preserves_active_bank(self):
        initial = self._prepare()
        self.binding.upload_and_launch(initial)
        prior = self.runtime.active
        for failure in ("stage", "launch", "finish", "publish"):
            with self.subTest(failure=failure):
                runtime = SyntheticGraphRuntime()
                runtime.active = prior
                runtime.next_bank = 1
                runtime.fail_at = failure
                binding = K0DeviceBinding(runtime, Tracer())
                prepared = self._prepare(accepted=30)
                with self.assertRaises(DeviceDecodeError):
                    binding.upload_and_launch(prepared)
                self.assertIs(runtime.active, prior)
                if failure == "stage":
                    self.assertNotIn("discard", [event[0] for event in runtime.events])
                else:
                    self.assertEqual(runtime.events[-1][0], "discard")
                self.assertEqual(binding.phase, DevicePhase.IDLE)

    def test_stale_duplicate_and_non_k0_input_stop_before_runtime(self):
        stale = self._prepare(accepted=10)
        current = self._prepare(accepted=20)
        with self.assertRaisesRegex(DeviceDecodeError, "stale"):
            self.binding.upload_and_launch(stale)
        self.assertEqual(self.runtime.events, [])
        forged = replace(
            current,
            lease=replace(current.lease, actual_batch=current.lease.actual_batch + 1),
        )
        with self.assertRaisesRegex(DeviceDecodeError, "stale"):
            self.binding.upload_and_launch(forged)
        self.assertEqual(self.runtime.events, [])
        self.binding.upload_and_launch(current)
        before = list(self.runtime.events)
        with self.assertRaisesRegex(DeviceDecodeError, "increase"):
            self.binding.upload_and_launch(current)
        self.assertEqual(self.runtime.events, before)

    def test_discard_failure_faults_binding_without_replacing_publication(self):
        runtime = SyntheticGraphRuntime()
        runtime.fail_at = {"launch", "discard"}
        binding = K0DeviceBinding(runtime, self.tracer)
        with self.assertRaisesRegex(DeviceDecodeError, "discard"):
            binding.upload_and_launch(self._prepare())
        self.assertIsNone(binding.publication)
        self.assertIsNone(runtime.active)
        self.assertEqual(binding.phase, DevicePhase.FAULTED)

    def test_invalid_publish_result_faults_without_discarding_unknown_bank(self):
        class InvalidPublishRuntime(SyntheticGraphRuntime):
            def publish(self, staged, generation, graph_batch):
                self._event("publish", staged["bank"], generation, graph_batch)
                self.active = "unknown"
                return DevicePublication(generation + 1, graph_batch, staged["bank"])

        runtime = InvalidPublishRuntime()
        binding = K0DeviceBinding(runtime, self.tracer)
        with self.assertRaisesRegex(DeviceDecodeError, "result contract"):
            binding.upload_and_launch(self._prepare())
        self.assertEqual(binding.phase, DevicePhase.FAULTED)
        self.assertNotIn("discard", [event[0] for event in runtime.events])

    def test_cancellation_discards_then_propagates(self):
        class CancellingRuntime(SyntheticGraphRuntime):
            def launch_k0(self, staged, graph_batch):
                self._event("launch", staged["bank"], graph_batch)
                raise SyntheticCancellation("cancel")

        runtime = CancellingRuntime()
        binding = K0DeviceBinding(runtime, self.tracer)
        with self.assertRaises(SyntheticCancellation):
            binding.upload_and_launch(self._prepare())
        self.assertEqual(runtime.events[-1][0], "discard")
        self.assertEqual(binding.phase, DevicePhase.IDLE)

    def test_otel_cardinality_is_bounded_on_success_and_failure(self):
        self.binding.upload_and_launch(self._prepare())
        failed_runtime = SyntheticGraphRuntime()
        failed_runtime.fail_at = "launch"
        failed_tracer = Tracer()
        with self.assertRaises(DeviceDecodeError):
            K0DeviceBinding(failed_runtime, failed_tracer).upload_and_launch(
                self._prepare(accepted=40)
            )
        spans = [
            span
            for span in self.tracer.spans + failed_tracer.spans
            if span.name == "rocket.qwen38.decode.device"
        ]
        allowed = {"phase", "depth", "graph_batch", "outcome"}
        self.assertTrue(all(set(span.attributes) == allowed for span in spans))
        self.assertTrue(
            all(
                span.attributes["phase"]
                in {"validate", "stage", "launch", "sync", "publish", "discard"}
                for span in spans
            )
        )
        self.assertTrue(all(span.attributes["depth"] == "k0" for span in spans))
        self.assertTrue(
            all(span.attributes["graph_batch"] in {1, 2, 4, 8, 16} for span in spans)
        )
        self.assertTrue(
            all(span.attributes["outcome"] in {"success", "failure"} for span in spans)
        )
        self.assertTrue(any(span.attributes["outcome"] == "failure" for span in spans))


class SyntheticCudaFailure(RuntimeError):
    pass


class SyntheticCancellation(BaseException):
    pass


if __name__ == "__main__":
    unittest.main()

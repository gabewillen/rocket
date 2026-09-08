from __future__ import annotations

import ctypes
import gc
import os
import threading
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path

from qwen38_slab.cuda_slab_loader import (
    CudaRankSlabLoader, CudaSlabLoadError, CudaSlabPublicationError,
    CudaSlabCleanupIncompleteError,
    accepted_native_handoff,
)
from test_stage_a import Fixture, Tracer


class FakeTensor:
    copies = 0

    def __init__(self, backing: bytearray, offset: int, length: int):
        self.backing = backing
        self.offset = offset
        self.length = length

    def data_ptr(self):
        return ctypes.addressof(ctypes.c_char.from_buffer(self.backing)) + self.offset

    def narrow(self, dimension, offset, length):
        if dimension != 0 or offset < 0 or length < 0 or offset + length > self.length:
            raise ValueError("invalid fake tensor view")
        return FakeTensor(self.backing, self.offset + offset, length)

    def copy_(self, source, *, non_blocking=False):
        if not non_blocking or self.length != source.length:
            raise ValueError("copy contract changed")
        self.backing[self.offset:self.offset + self.length] = (
            source.backing[source.offset:source.offset + source.length]
        )
        type(self).copies += 1
        return self

    def numpy(self):
        return memoryview(self.backing)[self.offset:self.offset + self.length]

    def bytes(self):
        return bytes(self.backing[self.offset:self.offset + self.length])


class FakeStream:
    def __init__(self, api, index):
        self.api = api
        self.index = index
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True
        self.api.synchronizations += 1
        if self.index in self.api.fail_stream_indices:
            raise ValueError("injected stream fence failure")


class FakeEvent:
    next_handle = 1
    def __init__(self, api=None):
        self.api = api
        self.recorded = False
        self._handle = type(self).next_handle
        type(self).next_handle += 1
    @property
    def cuda_event(self): return self._handle
    def record(self, stream): self.recorded = True
    def synchronize(self):
        if not self.recorded: raise ValueError("unrecorded event")
        if getattr(self.api, "fail_event", False):
            raise ValueError("injected event failure")


class FakeCuda:
    def __init__(self):
        self.streams = []
        self.synchronizations = 0
        self.fail_event = False
        self.fail_stream_indices = set()
        self.fail_device_sync = False

    def is_available(self): return True
    def Stream(self, device):
        if device != "cuda:0": raise ValueError("unexpected device")
        stream = FakeStream(self, len(self.streams))
        self.streams.append(stream)
        return stream
    def Event(self): return FakeEvent(self)
    def stream(self, stream): return nullcontext()
    def synchronize(self, device):
        if device != "cuda:0": raise ValueError("unexpected device")
        self.synchronizations += 1
        if self.fail_device_sync:
            raise ValueError("injected device fence failure")


class FakeTorch:
    uint8 = "uint8"

    def __init__(self): self.cuda = FakeCuda()

    def empty(self, length, *, dtype, device, pin_memory=False):
        if dtype != self.uint8 or length <= 0:
            raise ValueError("invalid allocation")
        if device == "cpu" and not pin_memory:
            raise ValueError("CPU staging must be pinned")
        return FakeTensor(bytearray(length), 0, length)


class Owner:
    def __init__(self, torch):
        self.torch = torch
        self.calls = []

    def publish_rank_slabs(self, rank, slabs):
        if not all(stream.synchronized for stream in self.torch.cuda.streams):
            raise ValueError("published before streams fenced")
        self.calls.append((rank, dict(slabs)))


class RejectingOwner(Owner):
    def publish_rank_slabs(self, rank, slabs):
        raise ValueError("injected Python publication rejection")


class Metric:
    def __init__(self, name, records): self.name = name; self.records = records
    def add(self, amount, attributes): self.records.append((self.name, amount, dict(attributes)))
    def record(self, amount, attributes): self.records.append((self.name, amount, dict(attributes)))


class InterruptingMetric:
    def add(self, amount, attributes): raise KeyboardInterrupt("injected telemetry failure")


class Meter:
    def __init__(self): self.records = []
    def create_counter(self, name, *, unit):
        self.records.append(("created", name, unit))
        return Metric(name, self.records)
    def create_histogram(self, name, *, unit):
        self.records.append(("created", name, unit))
        return Metric(name, self.records)


class ThrowingExitTracer(Tracer):
    def start_as_current_span(self, name):
        inner = super().start_as_current_span(name)
        class ThrowingSpan:
            def __enter__(self): return inner.__enter__()
            def __exit__(self, exc_type, exc, traceback):
                inner.__exit__(exc_type, exc, traceback)
                raise ValueError("injected tracer exit failure")
        return ThrowingSpan()


class BarrierLoader(CudaRankSlabLoader):
    def __init__(self, *args, **kwargs):
        self.barrier = threading.Barrier(2)
        super().__init__(*args, **kwargs)

    def _load_one(self, descriptor, destination, pipeline):
        self.barrier.wait(timeout=5)
        return super()._load_one(descriptor, destination, pipeline)


class MetricsFailureLoader(BarrierLoader):
    def _record_metrics(self, receipt):
        raise ValueError("injected metrics boundary failure")


class ResultFailureLoader(BarrierLoader):
    def _loaded_result(self, slabs, receipt, ready_event, capability):
        raise ValueError("injected result boundary failure")


class AdmissionBarrierLoader(BarrierLoader):
    def __init__(self, *args, **kwargs):
        self.admitted = threading.Event()
        self.continue_load = threading.Event()
        super().__init__(*args, **kwargs)
    def _load_locked(self):
        self.admitted.set()
        if not self.continue_load.wait(timeout=5):
            raise ValueError("admission barrier timed out")
        return super()._load_locked()


class NativeFinalizer:
    def __init__(self): self.calls = []
    def retain_accepted_loader(self, **publication):
        self.calls.append(publication)
        return object()


class RejectingNativeFinalizer:
    def retain_accepted_loader(self, **publication):
        raise ValueError("injected native publication rejection")


class BlockingNativeFinalizer(NativeFinalizer):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
    def retain_accepted_loader(self, **publication):
        self.calls.append(publication)
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise ValueError("blocking finalizer timed out")
        return object()


class CudaRankSlabLoaderTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temp.name))
        self.artifact = self.fixture.build()
        self.torch = FakeTorch()
        FakeTensor.copies = 0
        self.owner = Owner(self.torch)
        self.tracer = Tracer()
        self.meter = Meter()
        import qwen38_slab.cuda_slab_loader as loader_module
        loader_module._PROCESS_LIFETIME_NATIVE_SLAB_OWNERS[:] = [None, None]
        loader_module._FAILED_LOAD_FLIGHTS[:] = [[None, None], [None, None]]
        loader_module._RANK_LOAD_STATES[:] = ["idle", "idle"]

    def tearDown(self): self.temp.cleanup()

    def loader(self):
        return BarrierLoader(
            self.artifact,
            rank=0,
            owner=self.owner,
            tracer=self.tracer,
            meter=self.meter,
            torch_api=self.torch,
            device="cuda:0",
            contract=self.fixture.contract,
        )

    def test_two_owner_local_slabs_load_once_overlap_fence_and_publish(self):
        opened = []
        real_open = os.open

        def recording_open(path, flags, *args):
            opened.append(Path(path).name)
            return real_open(path, flags, *args)

        from unittest import mock
        with mock.patch("qwen38_slab.cuda_slab_loader.os.open", side_effect=recording_open):
            loaded = self.loader().load()
        self.assertCountEqual(opened, ["rank0-target.slab", "rank0-mtp.slab"])
        self.assertEqual(len(self.owner.calls), 1)
        self.assertEqual(loaded.receipt.bytes_read, 2 * 65_536)
        self.assertEqual(loaded.receipt.h2d_bytes, loaded.receipt.bytes_read)
        self.assertEqual(loaded.receipt.target.direct_reads, 1)
        self.assertEqual(loaded.receipt.mtp.direct_reads, 1)
        self.assertEqual(loaded.receipt.target.h2d_copies, 1)
        self.assertEqual(loaded.receipt.mtp.h2d_copies, 1)
        self.assertEqual(FakeTensor.copies, 2)
        self.assertIsNotNone(loaded.ready_event)
        self.assertTrue(loaded.ready_event.recorded)
        self.assertGreater(loaded.ready_event.cuda_event, 0)
        self.assertEqual(loaded.receipt.target.chunks[0].bytes, 65_536)
        self.assertGreater(loaded.receipt.target.chunks[0].direct_read_ns, 0)
        self.assertGreater(loaded.receipt.target.chunks[0].sha256_ns, 0)
        self.assertGreater(loaded.receipt.target.chunks[0].h2d_fence_ns, 0)
        self.assertGreater(loaded.receipt.reader_overlap_ns, 0)
        for key, tensor in loaded.slabs.items():
            self.assertEqual(tensor.bytes(), (self.artifact / f"{key}.slab").read_bytes())
        span = self.tracer.spans[-1]
        self.assertEqual(span.name, "rocket.qwen38.rank_slab.cuda_load")
        self.assertEqual(
            set(span.attributes),
            {"rank", "io.direct", "outcome"},
        )
        self.assertEqual(span.attributes["outcome"], "success")
        measurements = [record for record in self.meter.records if record[0] != "created"]
        self.assertEqual(len(measurements), 14)
        self.assertEqual(
            {frozenset(record[2]) for record in measurements},
            {
                frozenset(("rank", "slab.kind", "direction")),
                frozenset(("rank", "slab.kind", "stage")),
            },
        )

    def test_native_capability_is_minted_only_after_authenticated_publication(self):
        finalizer = NativeFinalizer()
        loader = BarrierLoader(
            self.artifact, rank=0, owner=self.owner, tracer=self.tracer,
            meter=self.meter, torch_api=self.torch, device="cuda:0",
            contract=self.fixture.contract, native_target_finalizer=finalizer,
            target_layout_sha256="1" * 64,
        )
        loaded = loader.load()
        capability = accepted_native_handoff(loaded)
        self.assertIsNotNone(capability)
        self.assertEqual(len(finalizer.calls), 1)
        publication = finalizer.calls[0]
        self.assertEqual(publication["device_base"], loaded.slabs["rank0-target"].data_ptr())
        self.assertEqual(publication["ready_event"], loaded.ready_event.cuda_event)
        self.assertIs(publication["receipt"], loaded.receipt.target)
        self.assertEqual(publication["receipt_sha256"], capability[1])
        self.assertEqual(publication["layout_sha256"], capability[2])
        self.assertEqual(publication["chunks_authenticated"], 1)
        self.assertTrue(self.owner.calls)
        target_owner = weakref.ref(loaded.slabs["rank0-target"])
        self.owner.calls.clear()
        del loaded
        gc.collect()
        self.assertIsNotNone(target_owner())

    def test_native_finalizer_rejection_never_publishes_python_owner(self):
        import qwen38_slab.cuda_slab_loader as loader_module
        loader = BarrierLoader(
            self.artifact, rank=0, owner=self.owner, tracer=self.tracer,
            meter=self.meter, torch_api=self.torch, device="cuda:0",
            contract=self.fixture.contract,
            native_target_finalizer=RejectingNativeFinalizer(),
            target_layout_sha256="1" * 64,
        )
        with self.assertRaisesRegex(CudaSlabLoadError, "before publication"):
            loader.load()
        self.assertEqual(self.owner.calls, [])
        self.assertIsNone(loader_module._PROCESS_LIFETIME_NATIVE_SLAB_OWNERS[0])
        retry = BarrierLoader(
            self.artifact, rank=0, owner=self.owner, tracer=self.tracer,
            meter=self.meter, torch_api=self.torch, device="cuda:0",
            contract=self.fixture.contract,
            native_target_finalizer=NativeFinalizer(),
            target_layout_sha256="1" * 64,
        ).load()
        self.assertIsNotNone(accepted_native_handoff(retry))
        self.assertEqual(len(self.owner.calls), 1)

    def test_post_native_python_publication_failure_quarantines_all_owners(self):
        import qwen38_slab.cuda_slab_loader as loader_module
        finalizer = NativeFinalizer()
        loader = BarrierLoader(
            self.artifact, rank=0, owner=RejectingOwner(self.torch),
            tracer=self.tracer, meter=self.meter, torch_api=self.torch,
            device="cuda:0", contract=self.fixture.contract,
            native_target_finalizer=finalizer, target_layout_sha256="1" * 64,
        )
        with self.assertRaisesRegex(CudaSlabPublicationError, "rejected publication"):
            loader.load()
        self.assertEqual(len(finalizer.calls), 1)
        self.assertIsNotNone(loader_module._PROCESS_LIFETIME_NATIVE_SLAB_OWNERS[0])

    def test_every_post_registration_exception_preserves_fixed_owner_slot(self):
        import qwen38_slab.cuda_slab_loader as loader_module
        for loader_type, message in (
            (MetricsFailureLoader, "metrics boundary"),
            (ResultFailureLoader, "result boundary"),
        ):
            loader_module._PROCESS_LIFETIME_NATIVE_SLAB_OWNERS[:] = [None, None]
            finalizer = NativeFinalizer()
            loader = loader_type(
                self.artifact, rank=0, owner=self.owner, tracer=self.tracer,
                meter=self.meter, torch_api=self.torch, device="cuda:0",
                contract=self.fixture.contract,
                native_target_finalizer=finalizer, target_layout_sha256="1" * 64,
            )
            with self.assertRaisesRegex(CudaSlabLoadError, "before publication"):
                loader.load()
            retained = loader_module._PROCESS_LIFETIME_NATIVE_SLAB_OWNERS[0]
            self.assertIsNotNone(retained, message)
            self.assertIsNotNone(retained.capability.handle, message)
            self.assertEqual(self.owner.calls, [], message)

    def test_same_rank_loader_instances_serialize_one_native_reservation(self):
        first_finalizer = BlockingNativeFinalizer()
        second_finalizer = NativeFinalizer()
        second_torch = FakeTorch()
        outcomes = []
        first = BarrierLoader(
            self.artifact, rank=0, owner=self.owner, tracer=self.tracer,
            meter=self.meter, torch_api=self.torch, device="cuda:0",
            contract=self.fixture.contract,
            native_target_finalizer=first_finalizer,
            target_layout_sha256="1" * 64,
        )
        second = BarrierLoader(
            self.artifact, rank=0, owner=Owner(second_torch), tracer=Tracer(),
            meter=Meter(), torch_api=second_torch, device="cuda:0",
            contract=self.fixture.contract,
            native_target_finalizer=second_finalizer,
            target_layout_sha256="1" * 64,
        )
        def run(loader):
            try:
                loader.load()
                outcomes.append("success")
            except CudaSlabLoadError as exc:
                outcomes.append(str(exc))
        first_thread = threading.Thread(target=run, args=(first,))
        second_thread = threading.Thread(target=run, args=(second,))
        first_thread.start()
        self.assertTrue(first_finalizer.entered.wait(timeout=5))
        second_thread.start()
        self.assertEqual(second_finalizer.calls, [])
        first_finalizer.release.set()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)
        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertEqual(outcomes.count("success"), 1)
        self.assertEqual(len(second_finalizer.calls), 0)
        self.assertEqual(second_torch.cuda.streams, [])
        self.assertTrue(any("already active" in outcome for outcome in outcomes))

    def test_tracer_exit_is_settled_before_terminal_owner_publication(self):
        loader = BarrierLoader(
            self.artifact, rank=0, owner=self.owner,
            tracer=ThrowingExitTracer(), meter=self.meter,
            torch_api=self.torch, device="cuda:0", contract=self.fixture.contract,
        )
        with self.assertRaisesRegex(ValueError, "tracer exit failure"):
            loader.load()
        self.assertEqual(self.owner.calls, [])

    def test_event_failure_attempts_all_streams_and_device_fallback(self):
        self.torch.cuda.fail_event = True
        self.torch.cuda.fail_stream_indices = {0, 2, 4}
        with self.assertRaisesRegex(CudaSlabLoadError, "rank slab load failed"):
            self.loader().load()
        self.assertEqual(len(self.torch.cuda.streams), 6)
        self.assertTrue(all(stream.synchronized for stream in self.torch.cuda.streams))
        self.assertGreaterEqual(self.torch.cuda.synchronizations, 8)
        self.assertEqual(self.owner.calls, [])

    def test_double_fence_failure_quarantines_full_flight_and_primary(self):
        import qwen38_slab.cuda_slab_loader as loader_module
        self.torch.cuda.fail_event = True
        self.torch.cuda.fail_stream_indices = {0, 1, 2, 3, 4, 5}
        self.torch.cuda.fail_device_sync = True
        with self.assertRaises(CudaSlabCleanupIncompleteError) as raised:
            self.loader().load()
        self.assertIsNotNone(raised.exception.primary)
        self.assertTrue(all(stream.synchronized for stream in self.torch.cuda.streams))
        retained = loader_module._FAILED_LOAD_FLIGHTS[0]
        self.assertTrue(any(owner is not None for owner in retained))
        self.assertEqual(self.owner.calls, [])
        retained_identity = tuple(id(owner) if owner is not None else None
                                  for owner in retained)
        second_torch = FakeTorch()
        with self.assertRaises(CudaSlabCleanupIncompleteError):
            BarrierLoader(
                self.artifact, rank=0, owner=Owner(second_torch),
                tracer=Tracer(), meter=Meter(), torch_api=second_torch,
                device="cuda:0", contract=self.fixture.contract,
            ).load()
        self.assertEqual(second_torch.streams if hasattr(second_torch, "streams") else
                         second_torch.cuda.streams, [])
        self.assertEqual(
            tuple(id(owner) if owner is not None else None for owner in retained),
            retained_identity,
        )

    def test_close_failure_still_attempts_every_stream_fence(self):
        from unittest import mock
        real_close = os.close
        def close_then_fail(fd):
            real_close(fd)
            raise OSError("injected close failure")
        with mock.patch("qwen38_slab.cuda_slab_loader.os.close",
                        side_effect=close_then_fail):
            with self.assertRaisesRegex(CudaSlabLoadError, "descriptor close"):
                self.loader().load()
        self.assertEqual(len(self.torch.cuda.streams), 6)
        self.assertTrue(all(stream.synchronized for stream in self.torch.cuda.streams))
        self.assertEqual(self.owner.calls, [])

    def test_cleanup_telemetry_cannot_interrupt_double_failure_quarantine(self):
        import qwen38_slab.cuda_slab_loader as loader_module
        self.torch.cuda.fail_event = True
        self.torch.cuda.fail_stream_indices = {0, 1, 2, 3, 4, 5}
        self.torch.cuda.fail_device_sync = True
        loader = self.loader()
        loader._cleanup_counter = InterruptingMetric()
        with self.assertRaises(CudaSlabCleanupIncompleteError):
            loader.load()
        self.assertTrue(all(stream.synchronized for stream in self.torch.cuda.streams))
        self.assertEqual(loader_module._RANK_LOAD_STATES[0], "poisoned")
        self.assertTrue(any(owner is not None
                            for owner in loader_module._FAILED_LOAD_FLIGHTS[0]))

    def test_rank_admission_rejects_concurrent_loader_before_double_failure(self):
        import qwen38_slab.cuda_slab_loader as loader_module
        self.torch.cuda.fail_event = True
        self.torch.cuda.fail_stream_indices = {0, 1, 2, 3, 4, 5}
        self.torch.cuda.fail_device_sync = True
        first = AdmissionBarrierLoader(
            self.artifact, rank=0, owner=self.owner, tracer=self.tracer,
            meter=self.meter, torch_api=self.torch, device="cuda:0",
            contract=self.fixture.contract,
        )
        outcome = []
        thread = threading.Thread(
            target=lambda: outcome.append(self._capture_load_error(first))
        )
        thread.start()
        self.assertTrue(first.admitted.wait(timeout=5))
        second_torch = FakeTorch()
        with self.assertRaisesRegex(CudaSlabLoadError, "already active"):
            BarrierLoader(
                self.artifact, rank=0, owner=Owner(second_torch),
                tracer=Tracer(), meter=Meter(), torch_api=second_torch,
                device="cuda:0", contract=self.fixture.contract,
            ).load()
        self.assertEqual(second_torch.cuda.streams, [])
        first.continue_load.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0], CudaSlabCleanupIncompleteError)
        self.assertEqual(loader_module._RANK_LOAD_STATES[0], "poisoned")

    @staticmethod
    def _capture_load_error(loader):
        try:
            loader.load()
        except BaseException as exc:
            return exc
        return None

    def test_digest_failure_drains_both_streams_and_never_publishes(self):
        path = self.artifact / "rank0-mtp.slab"
        os.chmod(path, 0o644)
        payload = bytearray(path.read_bytes())
        payload[0] ^= 1
        path.write_bytes(payload)
        with self.assertRaisesRegex(CudaSlabLoadError, "digest"):
            self.loader().load()
        self.assertEqual(self.owner.calls, [])
        self.assertEqual(len(self.torch.cuda.streams), 6)
        self.assertTrue(all(stream.synchronized for stream in self.torch.cuda.streams))
        self.assertEqual(FakeTensor.copies, 1)
        self.assertEqual(self.tracer.spans[-1].attributes["outcome"], "failure")

    def test_rank_and_owner_contracts_fail_closed(self):
        with self.assertRaisesRegex(CudaSlabLoadError, "rank"):
            CudaRankSlabLoader(
                self.artifact, rank=2, owner=self.owner, tracer=self.tracer,
                meter=self.meter,
                torch_api=self.torch, device="cuda:0", contract=self.fixture.contract,
            )
        with self.assertRaisesRegex(CudaSlabLoadError, "owner"):
            CudaRankSlabLoader(
                self.artifact, rank=0, owner=None, tracer=self.tracer,
                meter=self.meter,
                torch_api=self.torch, device="cuda:0", contract=self.fixture.contract,
            )


if __name__ == "__main__":
    unittest.main()

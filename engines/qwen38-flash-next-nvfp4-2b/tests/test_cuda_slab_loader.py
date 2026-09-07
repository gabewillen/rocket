from __future__ import annotations

import ctypes
import os
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path

from qwen38_slab.cuda_slab_loader import CudaRankSlabLoader, CudaSlabLoadError
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
    def __init__(self, api):
        self.api = api
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True
        self.api.synchronizations += 1


class FakeEvent:
    def __init__(self): self.recorded = False
    def record(self, stream): self.recorded = True
    def synchronize(self):
        if not self.recorded: raise ValueError("unrecorded event")


class FakeCuda:
    def __init__(self):
        self.streams = []
        self.synchronizations = 0

    def is_available(self): return True
    def Stream(self, device):
        if device != "cuda:0": raise ValueError("unexpected device")
        stream = FakeStream(self)
        self.streams.append(stream)
        return stream
    def Event(self): return FakeEvent()
    def stream(self, stream): return nullcontext()


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


class Metric:
    def __init__(self, name, records): self.name = name; self.records = records
    def add(self, amount, attributes): self.records.append((self.name, amount, dict(attributes)))
    def record(self, amount, attributes): self.records.append((self.name, amount, dict(attributes)))


class Meter:
    def __init__(self): self.records = []
    def create_counter(self, name, *, unit):
        self.records.append(("created", name, unit))
        return Metric(name, self.records)
    def create_histogram(self, name, *, unit):
        self.records.append(("created", name, unit))
        return Metric(name, self.records)


class BarrierLoader(CudaRankSlabLoader):
    def __init__(self, *args, **kwargs):
        self.barrier = threading.Barrier(2)
        super().__init__(*args, **kwargs)

    def _load_one(self, descriptor, destination, pipeline):
        self.barrier.wait(timeout=5)
        return super()._load_one(descriptor, destination, pipeline)


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

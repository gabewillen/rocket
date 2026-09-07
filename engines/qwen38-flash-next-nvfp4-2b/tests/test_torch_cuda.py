from __future__ import annotations

import hashlib
import unittest

from qwen38_slab.device_decode import DevicePhase, DevicePublication
from qwen38_slab.runtime_state import (
    BindingPhase,
    CudaStateBinding,
    DeviceState,
    RuntimeBoundary,
    RuntimeStateError,
)
from qwen38_slab.state_owner import DecoderStateOwner, OwnerPhase
from qwen38_slab.state_txn import AuthenticatedState, FamilyPayload, STATE_FAMILIES
from qwen38_slab.torch_cuda import TorchCudaRuntime, TorchCudaRuntimeError


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


class FakeNumpy:
    def __init__(self, payload): self.payload = payload
    def tobytes(self): return bytes(self.payload)


class FakeTensor:
    def __init__(self, api, payload, *, device, dtype, pinned=False):
        self.api = api
        self.payload = bytearray(payload)
        self.device = device
        self.dtype = dtype
        self.is_cuda = str(device).startswith("cuda")
        self.pinned = pinned

    def numel(self): return len(self.payload)
    def is_contiguous(self): return True
    def narrow(self, dimension, offset, length):
        if dimension != 0: raise AssertionError("only byte-vector views are supported")
        return FakeTensor(
            self.api,
            self.payload[offset:offset + length],
            device=self.device,
            dtype=self.dtype,
            pinned=self.pinned,
        )
    def copy_(self, source, non_blocking=False):
        self.api.events.append(("tensor_copy", self.device, non_blocking, len(source.payload)))
        if self.api.fail_copy:
            raise FakeCudaError("copy")
        if len(self.payload) != len(source.payload):
            raise FakeCudaError("extent")
        self.payload[:] = source.payload
        return self
    def numpy(self): return FakeNumpy(self.payload)


class FakeEvent:
    def __init__(self, api): self.api = api
    def record(self, stream):
        self.api.events.append(("record", stream.name))
        if self.api.fail_record: raise FakeCudaError("record")


class FakeStream:
    def __init__(self, api, name): self.api = api; self.name = name
    def wait_event(self, event): self.api.events.append(("wait", self.name))
    def synchronize(self):
        self.api.sync_count += 1
        self.api.events.append(("synchronize", self.name))
        if self.api.sync_count in self.api.fail_sync_at:
            raise FakeCudaError("synchronize")


class StreamContext:
    def __init__(self, api, stream): self.api = api; self.stream = stream
    def __enter__(self): self.api.events.append(("stream_enter", self.stream.name))
    def __exit__(self, exc_type, exc, traceback):
        self.api.events.append(("stream_exit", self.stream.name))


class FakeCuda:
    def __init__(self, api): self.api = api; self.created = 0
    def is_available(self): return True
    def Stream(self, device=None):
        self.created += 1
        return FakeStream(self.api, f"copy:{device}:{self.created}")
    def Event(self): return FakeEvent(self.api)
    def stream(self, stream): return StreamContext(self.api, stream)


class FakeTorch:
    uint8 = "uint8"

    def __init__(self):
        self.events = []
        self.fail_copy = False
        self.fail_record = False
        self.sync_count = 0
        self.fail_sync_at = set()
        self.cuda = FakeCuda(self)

    def empty(self, length, *, dtype, device, pin_memory=False):
        self.events.append(("empty", device, length, pin_memory))
        return FakeTensor(
            self, b"\0" * length, device=device, dtype=dtype, pinned=pin_memory
        )

    def frombuffer(self, buffer, *, dtype):
        self.events.append(("frombuffer", type(buffer).__name__))
        return FakeTensor(self, bytes(buffer), device="cpu", dtype=dtype)


class Owner:
    def __init__(self, events):
        self.events = events
        self.published = None
        self.fail_close = False
        self.fail_open = False

    def close_launch_gate(self, boundary):
        self.events.append(("gate_close", boundary.generation_epoch))
        if self.fail_close: raise FakeCudaError("close")
        return boundary

    def publish_state(self, staged, boundary):
        self.events.append(("owner_publish", boundary.token_count))
        self.published = dict(staged)

    def open_launch_gate(self, boundary):
        self.events.append(("gate_open", boundary.generation_epoch))
        if self.fail_open: raise FakeCudaError("open")


class SpecializedDecoder:
    def __init__(self):
        self.phase = DevicePhase.IDLE
        self.publication = None

    def upload_and_launch(self, generation):
        self.publication = DevicePublication(generation, 1, generation % 2)
        return self.publication


class TorchCudaTests(unittest.TestCase):
    def setUp(self):
        self.torch = FakeTorch()
        self.owner = Owner(self.torch.events)
        self.compute = (
            FakeStream(self.torch, "compute-0"),
            FakeStream(self.torch, "compute-1"),
        )
        self.runtime = TorchCudaRuntime(
            owner=self.owner,
            compute_streams=self.compute,
            torch_api=self.torch,
            device="cuda:0",
        )
        self.binding = CudaStateBinding(0, self.runtime, Tracer())
        self.boundary = RuntimeBoundary(
            41, hashlib.sha256(b"accepted-41").hexdigest(), 7
        )
        self.device = {
            family: FakeTensor(
                self.torch,
                f"{family}:accepted:spec".encode(),
                device="cuda:0",
                dtype=self.torch.uint8,
            )
            for family in STATE_FAMILIES
        }
        self.sources = {
            family: DeviceState(
                family,
                tensor,
                len(f"{family}:accepted".encode()),
                tensor.numel(),
            )
            for family, tensor in self.device.items()
        }
        payloads = {
            family: FamilyPayload(f"restore:{family}".encode())
            for family in STATE_FAMILIES
        }
        self.authenticated = AuthenticatedState._from_verified(
            token_count=41,
            token_hash=self.boundary.token_hash,
            rank_payloads={0: payloads, 1: payloads},
        )

    def test_concrete_adapter_fences_all_streams_copies_and_publishes_nine(self):
        accepted, captured = self.binding.capture(self.boundary, self.sources)
        self.assertEqual(accepted, self.authenticated.boundary)
        self.assertEqual(tuple(captured), STATE_FAMILIES)
        names = [event[0] for event in self.torch.events]
        self.assertLess(names.index("gate_close"), names.index("record"))
        self.assertEqual(names.count("record"), len(self.compute))
        self.assertEqual(names[-1], "gate_open")

        self.torch.events.clear()
        self.binding.restore(self.authenticated, generation_epoch=8)
        names = [event[0] for event in self.torch.events]
        self.assertEqual(tuple(self.owner.published), STATE_FAMILIES)
        self.assertLess(names.index("synchronize"), names.index("owner_publish"))
        self.assertLess(names.index("owner_publish"), names.index("gate_open"))
        self.assertTrue(all(
            bytes(self.owner.published[family].payload)
            == self.authenticated.rank_payload(0)[family].accepted
            for family in STATE_FAMILIES
        ))

    def test_concrete_adapter_uses_specialized_decoder_owner_gate_and_table(self):
        decoder = SpecializedDecoder()
        owner = DecoderStateOwner(rank=0, decoder=decoder, tracer=Tracer())
        owner.accept_boundary(owner.upload_and_launch(7), self.boundary)
        runtime = TorchCudaRuntime(
            owner=owner,
            compute_streams=self.compute,
            torch_api=self.torch,
            device="cuda:0",
        )
        binding = CudaStateBinding(0, runtime, Tracer())

        binding.restore(self.authenticated, generation_epoch=7)

        self.assertEqual(binding.rank, 0)
        self.assertIs(binding.state_owner, owner)
        self.assertEqual(owner.phase, OwnerPhase.OPEN)
        self.assertEqual(tuple(owner.active_state), STATE_FAMILIES)
        self.assertEqual(owner.upload_and_launch(8).generation, 8)

    def test_padded_plan_uses_eight_backings_for_nine_logical_views(self):
        logical = {
            family: len(self.authenticated.rank_payload(0)[family].accepted)
            for family in STATE_FAMILIES
        }
        allocated = {family: size + 64 for family, size in logical.items()}
        owners = {family: family for family in STATE_FAMILIES}
        conv, recurrent = STATE_FAMILIES[3:5]
        allocated[conv] = logical[conv] + logical[recurrent] + 64
        allocated[recurrent] = allocated[conv]
        owners[recurrent] = conv
        runtime = TorchCudaRuntime(
            owner=self.owner,
            compute_streams=self.compute,
            torch_api=self.torch,
            device="cuda:0",
            allocation_bytes=allocated,
            allocation_owners=owners,
        )

        CudaStateBinding(0, runtime, Tracer()).restore(
            self.authenticated, generation_epoch=8
        )

        cuda_allocations = [
            event[2]
            for event in self.torch.events
            if event[0] == "empty" and event[1] == "cuda:0"
        ]
        self.assertEqual(len(cuda_allocations), 8)
        self.assertEqual(
            sum(cuda_allocations),
            sum(allocated[family] for family in STATE_FAMILIES if owners[family] == family),
        )
        self.assertEqual(
            [event[1] for event in self.torch.events if event[0] == "frombuffer"],
            ["bytes"] * len(STATE_FAMILIES),
        )
        self.assertTrue(all(
            self.owner.published[family].numel() == logical[family]
            for family in STATE_FAMILIES
        ))

    def test_padded_plan_rejects_partial_inventory_and_undersized_backing(self):
        partial = {family: 64 for family in STATE_FAMILIES[:-1]}
        owners = {family: family for family in STATE_FAMILIES}
        with self.assertRaisesRegex(TorchCudaRuntimeError, "nine-family"):
            TorchCudaRuntime(
                owner=self.owner,
                compute_streams=self.compute,
                torch_api=self.torch,
                device="cuda:0",
                allocation_bytes=partial,
                allocation_owners=owners,
            )

        allocated = {family: 1 for family in STATE_FAMILIES}
        runtime = TorchCudaRuntime(
            owner=self.owner,
            compute_streams=self.compute,
            torch_api=self.torch,
            device="cuda:0",
            allocation_bytes=allocated,
            allocation_owners=owners,
        )
        binding = CudaStateBinding(0, runtime, Tracer())
        with self.assertRaisesRegex(RuntimeStateError, "stage failed"):
            binding.restore(self.authenticated, generation_epoch=8)
        self.assertEqual(binding.phase, BindingPhase.IDLE)
        self.assertIsNone(self.owner.published)

    def test_tensor_contract_drift_fails_without_publication_and_reopens_gate(self):
        first = STATE_FAMILIES[0]
        bad = dict(self.sources)
        bad[first] = DeviceState(first, self.device[first], 1, 1)
        with self.assertRaisesRegex(RuntimeStateError, "capture"):
            self.binding.capture(self.boundary, bad)
        self.assertIsNone(self.owner.published)
        self.assertEqual(self.binding.phase, BindingPhase.IDLE)
        self.assertEqual(self.torch.events[-1][0], "gate_open")

    def test_copy_failure_discards_private_tensors_and_never_publishes(self):
        self.torch.fail_copy = True
        with self.assertRaisesRegex(RuntimeStateError, "stage"):
            self.binding.restore(self.authenticated, generation_epoch=8)
        self.assertIsNone(self.owner.published)
        self.assertEqual(self.binding.phase, BindingPhase.IDLE)
        self.assertEqual(self.torch.events[-1][0], "gate_open")

    def test_fence_failure_reopens_gate_and_is_safe_to_retry(self):
        self.torch.fail_record = True
        with self.assertRaisesRegex(RuntimeStateError, "quiesce"):
            self.binding.capture(self.boundary, self.sources)
        self.assertEqual(self.binding.phase, BindingPhase.IDLE)
        self.assertEqual(self.torch.events[-1][0], "gate_open")

    def test_fence_and_reopen_failure_faults_binding(self):
        self.torch.fail_record = True
        self.owner.fail_open = True
        with self.assertRaisesRegex(RuntimeStateError, "quiesce"):
            self.binding.capture(self.boundary, self.sources)
        self.assertEqual(self.binding.phase, BindingPhase.FAULTED)

    def test_unrecoverable_capture_stream_failure_reopens_gate_but_faults_binding(self):
        self.torch.fail_sync_at = {2, 3}
        with self.assertRaisesRegex(RuntimeStateError, "fenced"):
            self.binding.capture(self.boundary, self.sources)
        self.assertEqual(self.torch.events[-1][0], "gate_open")
        self.assertEqual(self.binding.phase, BindingPhase.FAULTED)

    def test_constructor_rejects_missing_cuda_streams_and_device(self):
        cases = (
            {"compute_streams": ()},
            {"device": "cpu"},
        )
        for override in cases:
            with self.subTest(override=override):
                arguments = {
                    "owner": self.owner,
                    "compute_streams": self.compute,
                    "torch_api": self.torch,
                    "device": "cuda:0",
                }
                arguments.update(override)
                with self.assertRaises(TorchCudaRuntimeError):
                    TorchCudaRuntime(**arguments)


class FakeCudaError(RuntimeError):
    pass


if __name__ == "__main__":
    unittest.main()

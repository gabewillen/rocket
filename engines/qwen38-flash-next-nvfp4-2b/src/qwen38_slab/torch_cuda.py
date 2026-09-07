"""Concrete Torch CUDA adapter for the Qwen3.8 runtime state contract.

The adapter owns its copy stream and temporary pinned host tensors.  The engine
owner supplies every compute stream that can mutate one of the nine state
families plus launch-gate and pointer-table callbacks.  Closing the launch gate
precedes CUDA event recording.  Publication is one owner callback after every
host-to-device transfer has completed.

The caller injects the imported ``torch`` module explicitly.  This keeps CUDA
initialization at the engine startup boundary and makes the adapter contract
testable without importing or initializing Torch at package import time.
OpenTelemetry is emitted by :class:`qwen38_slab.runtime_state.CudaStateBinding`,
which wraps every method on this adapter with bounded dimensions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol

from .runtime_state import (
    CudaQuiesceError,
    CudaRuntimeFatalError,
    DeviceState,
    QuiesceReceipt,
    RuntimeBoundary,
)
from .state_txn import STATE_FAMILIES

MAX_COMPUTE_STREAMS = 16
_CUDA_DEVICE = re.compile(r"cuda:[0-9]+\Z")


class TorchCudaRuntimeError(RuntimeError):
    """Torch adapter validation or ownership failure."""


class TorchTensor(Protocol):
    device: object
    dtype: object
    is_cuda: bool

    def numel(self) -> int: ...
    def is_contiguous(self) -> bool: ...
    def narrow(self, dimension: int, offset: int, length: int) -> "TorchTensor": ...
    def copy_(self, source: "TorchTensor", *, non_blocking: bool = False) -> "TorchTensor": ...
    def numpy(self) -> object: ...


class TorchEvent(Protocol):
    def record(self, stream: "TorchStream") -> None: ...


class TorchStream(Protocol):
    def wait_event(self, event: TorchEvent) -> None: ...
    def synchronize(self) -> None: ...


class TorchCudaApi(Protocol):
    def is_available(self) -> bool: ...
    def Stream(self, device: str) -> TorchStream: ...
    def Event(self) -> TorchEvent: ...
    def stream(self, stream: TorchStream) -> object: ...


class TorchApi(Protocol):
    uint8: object
    cuda: TorchCudaApi

    def empty(
        self,
        length: int,
        *,
        dtype: object,
        device: str,
        pin_memory: bool = False,
    ) -> TorchTensor: ...

    def frombuffer(self, buffer: bytearray, *, dtype: object) -> TorchTensor: ...


class TorchStateOwner(Protocol):
    """Single-writer engine owner for scheduling and the live pointer table.

    ``close_launch_gate`` returns the actual accepted boundary held by the
    scheduler.  ``publish_state`` must consume the complete staged table with
    one atomic pointer swap or raise before changing the live table.
    """

    def close_launch_gate(self, boundary: RuntimeBoundary) -> RuntimeBoundary: ...
    def publish_state(
        self, staged: Mapping[str, TorchTensor], boundary: RuntimeBoundary
    ) -> None: ...
    def open_launch_gate(self, boundary: RuntimeBoundary) -> None: ...


class TorchCudaRuntime:
    """Torch implementation of the synchronous ``CudaRuntime`` protocol.

    One binding owns this adapter.  Calls are non-reentrant and not thread-safe.
    Device tensors are one-dimensional contiguous ``torch.uint8`` byte views.
    The adapter borrows capture tensors, owns unpublished restore tensors, and
    transfers those tensors to ``TorchStateOwner`` only on publish success.
    """

    def __init__(
        self,
        *,
        owner: TorchStateOwner,
        compute_streams: tuple[TorchStream, ...],
        torch_api: TorchApi,
        device: str,
    ) -> None:
        if owner is None or any(
            not callable(getattr(owner, method, None))
            for method in ("close_launch_gate", "publish_state", "open_launch_gate")
        ):
            raise TorchCudaRuntimeError("Torch state owner contract is incomplete")
        if (
            torch_api is None
            or not callable(getattr(torch_api, "empty", None))
            or not callable(getattr(torch_api, "frombuffer", None))
            or not callable(getattr(getattr(torch_api, "cuda", None), "stream", None))
        ):
            raise TorchCudaRuntimeError("Torch CUDA API contract is incomplete")
        if not torch_api.cuda.is_available():
            raise TorchCudaRuntimeError("Torch CUDA is unavailable")
        if not isinstance(device, str) or not _CUDA_DEVICE.fullmatch(device):
            raise TorchCudaRuntimeError("device must be an explicit cuda:N device")
        if (
            not isinstance(compute_streams, tuple)
            or not 1 <= len(compute_streams) <= MAX_COMPUTE_STREAMS
            or len({id(stream) for stream in compute_streams}) != len(compute_streams)
            or any(
                not callable(getattr(stream, method, None))
                for stream in compute_streams
                for method in ("wait_event", "synchronize")
            )
        ):
            raise TorchCudaRuntimeError(
                "one to sixteen distinct mutating compute streams are required"
            )
        self._owner = owner
        self._streams = compute_streams
        self._torch = torch_api
        self._device = device
        self._copy_stream = torch_api.cuda.Stream(device=device)
        self._gate_boundary: RuntimeBoundary | None = None
        self._pinned_staging: list[TorchTensor] = []

    @property
    def owner(self) -> TorchStateOwner:
        """Return the fixed launch-gate and pointer-table owner."""

        return self._owner

    def quiesce(self, boundary: RuntimeBoundary) -> QuiesceReceipt:
        """Close the launch gate and fence all enumerated mutating streams."""

        if self._gate_boundary is not None:
            raise CudaQuiesceError("Torch launch gate is already closed", safe_to_retry=False)
        attested: RuntimeBoundary | None = None
        gate_closed = False
        try:
            attested = self._owner.close_launch_gate(boundary)
            gate_closed = True
            if not isinstance(attested, RuntimeBoundary) or attested != boundary:
                raise TorchCudaRuntimeError("scheduler accepted boundary does not match request")
            self._gate_boundary = attested
            for stream in self._streams:
                event = self._torch.cuda.Event()
                event.record(stream)
                self._copy_stream.wait_event(event)
            self._copy_stream.synchronize()
            return QuiesceReceipt(attested, compute_fenced=True, pending_launches=0)
        except BaseException as exc:
            safe_to_retry = False
            if gate_closed:
                try:
                    self._owner.open_launch_gate(attested or boundary)
                except BaseException:
                    self._gate_boundary = attested or boundary
                else:
                    self._gate_boundary = None
                    safe_to_retry = True
            error = CudaQuiesceError(
                "Torch CUDA stream quiesce failed", safe_to_retry=safe_to_retry
            )
            raise error from exc

    def copy_device_to_host(self, source: DeviceState, logical_bytes: int) -> bytes:
        """Copy one validated accepted byte prefix into owned host bytes."""

        self._require_quiesced()
        tensor = self._validated_tensor(source.handle, source.allocated_bytes)
        if logical_bytes != source.accepted_bytes:
            raise TorchCudaRuntimeError("capture logical extent does not match source")
        host = self._torch.empty(
            logical_bytes,
            dtype=self._torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        with self._torch.cuda.stream(self._copy_stream):
            try:
                host.copy_(tensor.narrow(0, 0, logical_bytes), non_blocking=True)
            except BaseException as exc:
                self._drain_after_copy_failure(exc)
                raise
        try:
            self._copy_stream.synchronize()
        except BaseException as exc:
            self._drain_after_copy_failure(exc)
            raise CudaRuntimeFatalError("CUDA capture stream could not be fenced") from exc
        numpy_view = host.numpy()
        tobytes = getattr(numpy_view, "tobytes", None)
        if not callable(tobytes):
            raise TorchCudaRuntimeError("pinned host tensor has no byte export")
        payload = tobytes()
        if not isinstance(payload, bytes) or len(payload) != logical_bytes:
            raise TorchCudaRuntimeError("pinned host byte export has the wrong extent")
        return payload

    def allocate_staging(self, family: str, logical_bytes: int) -> TorchTensor:
        """Allocate one private CUDA byte tensor while the launch gate is closed."""

        self._require_quiesced()
        if family not in STATE_FAMILIES or logical_bytes <= 0:
            raise TorchCudaRuntimeError("invalid staging family or extent")
        with self._torch.cuda.stream(self._copy_stream):
            tensor = self._torch.empty(
                logical_bytes, dtype=self._torch.uint8, device=self._device
            )
        return self._validated_tensor(tensor, logical_bytes)

    def copy_host_to_device(self, destination: object, payload: bytes) -> None:
        """Queue one borrowed host payload into a private staging tensor."""

        self._require_quiesced()
        if not isinstance(payload, bytes) or not payload:
            raise TorchCudaRuntimeError("restore payload must be nonempty immutable bytes")
        tensor = self._validated_tensor(destination, len(payload))
        pinned = self._torch.empty(
            len(payload), dtype=self._torch.uint8, device="cpu", pin_memory=True
        )
        source = self._torch.frombuffer(bytearray(payload), dtype=self._torch.uint8)
        pinned.copy_(source)
        self._pinned_staging.append(pinned)
        with self._torch.cuda.stream(self._copy_stream):
            tensor.copy_(pinned, non_blocking=True)

    def finish_transfers(self) -> None:
        """Fence all queued host-to-device copies and release pinned sources."""

        self._require_quiesced()
        try:
            self._copy_stream.synchronize()
        except BaseException as exc:
            raise CudaRuntimeFatalError("CUDA restore stream could not be fenced") from exc
        self._pinned_staging.clear()

    def publish(
        self, staged: Mapping[str, object], boundary: RuntimeBoundary
    ) -> None:
        """Transfer one complete validated pointer table to the engine owner."""

        self._require_boundary(boundary)
        if tuple(staged) != STATE_FAMILIES or self._pinned_staging:
            raise TorchCudaRuntimeError("staged pointer table is incomplete or unfenced")
        validated: dict[str, TorchTensor] = {}
        for family in STATE_FAMILIES:
            tensor = staged[family]
            if not isinstance(getattr(tensor, "is_cuda", None), bool):
                raise TorchCudaRuntimeError(f"invalid staged tensor for family {family}")
            validated[family] = self._validated_tensor(
                tensor, int(getattr(tensor, "numel")())
            )
        self._owner.publish_state(MappingProxyType(validated), boundary)

    def discard(self, staged: tuple[object, ...]) -> None:
        """Release adapter-owned host references; caller releases CUDA tensors."""

        self._require_quiesced()
        if not isinstance(staged, tuple) or len(staged) > len(STATE_FAMILIES):
            raise TorchCudaRuntimeError("invalid unpublished staging inventory")
        try:
            self._copy_stream.synchronize()
        except BaseException as exc:
            raise CudaRuntimeFatalError("CUDA discard stream could not be fenced") from exc
        self._pinned_staging.clear()

    def resume(self, boundary: RuntimeBoundary) -> None:
        """Reopen the launch gate after capture, discard, or publication."""

        self._require_boundary(boundary)
        self._owner.open_launch_gate(boundary)
        self._gate_boundary = None

    def _validated_tensor(self, value: object, expected_bytes: int) -> TorchTensor:
        numel = getattr(value, "numel", None)
        contiguous = getattr(value, "is_contiguous", None)
        narrow = getattr(value, "narrow", None)
        copy = getattr(value, "copy_", None)
        if (
            not callable(numel)
            or not callable(contiguous)
            or not callable(narrow)
            or not callable(copy)
            or getattr(value, "is_cuda", None) is not True
            or getattr(value, "dtype", None) != self._torch.uint8
            or str(getattr(value, "device", "")) != self._device
            or numel() != expected_bytes
            or contiguous() is not True
        ):
            raise TorchCudaRuntimeError(
                "state tensor must be an exact contiguous CUDA uint8 byte vector"
            )
        return value

    def _require_quiesced(self) -> None:
        if self._gate_boundary is None:
            raise TorchCudaRuntimeError("Torch launch gate is not quiesced")

    def _require_boundary(self, boundary: RuntimeBoundary) -> None:
        if self._gate_boundary is None or self._gate_boundary != boundary:
            raise TorchCudaRuntimeError("Torch launch gate boundary mismatch")

    def _drain_after_copy_failure(self, failure: BaseException) -> None:
        try:
            self._copy_stream.synchronize()
        except BaseException as drain_error:
            raise CudaRuntimeFatalError(
                "CUDA copy failure could not be fenced before cleanup"
            ) from BaseExceptionGroup(
                "copy and drain failures", [failure, drain_error]
            )


__all__ = [
    "MAX_COMPUTE_STREAMS",
    "TorchApi",
    "TorchCudaRuntime",
    "TorchCudaRuntimeError",
    "TorchStateOwner",
    "TorchStream",
    "TorchTensor",
]

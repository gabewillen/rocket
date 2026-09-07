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

import hashlib
import os
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
from .distributed_state_txn import AuthenticatedFamilyExtent, _AlignedBuffer
from .state_txn import IO_CHUNK_BYTES
from .state_txn import STATE_FAMILIES

MAX_COMPUTE_STREAMS = 16
MAX_ALLOCATION_BYTES = 32 * 1024**3
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
        allocation_bytes: Mapping[str, int] | None = None,
        allocation_owners: Mapping[str, str] | None = None,
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
        self._allocation_bytes, self._allocation_owners = _allocation_plan(
            allocation_bytes, allocation_owners
        )
        self._copy_stream = torch_api.cuda.Stream(device=device)
        self._gate_boundary: RuntimeBoundary | None = None
        self._pinned_staging: list[TorchTensor] = []
        self._stage_views: dict[str, TorchTensor] = {}
        self._stage_backings: dict[str, TorchTensor] = {}
        self._stage_offsets: dict[str, int] = {}

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
        if family in self._stage_views:
            raise TorchCudaRuntimeError("staging family was allocated twice")
        owner = self._allocation_owners.get(family, family)
        allocated_bytes = self._allocation_bytes.get(family, logical_bytes)
        if allocated_bytes < logical_bytes:
            raise TorchCudaRuntimeError("planned CUDA allocation is below logical extent")
        with self._torch.cuda.stream(self._copy_stream):
            if owner == family:
                backing = self._torch.empty(
                    allocated_bytes, dtype=self._torch.uint8, device=self._device
                )
                self._stage_backings[owner] = self._validated_tensor(
                    backing, allocated_bytes
                )
                self._stage_offsets[owner] = 0
            elif owner not in self._stage_backings:
                raise TorchCudaRuntimeError(
                    "shared CUDA allocation owner must be staged first"
                )
            backing = self._stage_backings[owner]
            offset = self._stage_offsets[owner]
            if offset + logical_bytes > backing.numel():
                raise TorchCudaRuntimeError("shared CUDA allocation is too small")
            tensor = backing.narrow(0, offset, logical_bytes)
            self._stage_offsets[owner] = offset + logical_bytes
        validated = self._validated_tensor(tensor, logical_bytes)
        self._stage_views[family] = validated
        return validated

    def copy_host_to_device(self, destination: object, payload: bytes) -> None:
        """Queue one borrowed host payload into a private staging tensor."""

        self._require_quiesced()
        if not isinstance(payload, bytes) or not payload:
            raise TorchCudaRuntimeError("restore payload must be nonempty immutable bytes")
        tensor = self._validated_tensor(destination, len(payload))
        pinned = self._torch.empty(
            len(payload), dtype=self._torch.uint8, device="cpu", pin_memory=True
        )
        if not any(value is tensor for value in self._stage_views.values()):
            raise TorchCudaRuntimeError("destination is not private staging")
        source = self._torch.frombuffer(payload, dtype=self._torch.uint8)
        pinned.copy_(source)
        self._pinned_staging.append(pinned)
        with self._torch.cuda.stream(self._copy_stream):
            tensor.copy_(pinned, non_blocking=True)

    def copy_extent_to_device(
        self, destination: object, extent: AuthenticatedFamilyExtent
    ) -> None:
        """Stream one authenticated O_DIRECT extent into private CUDA staging."""

        self._require_quiesced()
        if not isinstance(extent, AuthenticatedFamilyExtent):
            raise TorchCudaRuntimeError("authenticated family extent is required")
        tensor = self._validated_tensor(destination, extent.logical_bytes)
        if not any(value is tensor for value in self._stage_views.values()):
            raise TorchCudaRuntimeError("destination is not private staging")
        logical_digest = hashlib.sha256()
        padded_digest = hashlib.sha256()
        fd = -1
        try:
            fd = os.open(extent.path, os.O_RDONLY | os.O_DIRECT)
            if os.fstat(fd).st_size != extent.length_bytes:
                raise TorchCudaRuntimeError("authenticated family extent size changed")
            for offset in range(0, extent.length_bytes, IO_CHUNK_BYTES):
                size = min(IO_CHUNK_BYTES, extent.length_bytes - offset)
                accepted = min(size, max(0, extent.logical_bytes - offset))
                with _AlignedBuffer(size) as direct:
                    if os.preadv(fd, [direct.view], offset) != size:
                        raise TorchCudaRuntimeError("short O_DIRECT CUDA restore read")
                    if any(direct.view[accepted:]):
                        raise TorchCudaRuntimeError("CUDA restore padding changed")
                    logical_digest.update(direct.view[:accepted])
                    padded_digest.update(direct.view)
                    if accepted:
                        pinned = self._torch.empty(
                            accepted, dtype=self._torch.uint8,
                            device="cpu", pin_memory=True,
                        )
                        source = self._torch.frombuffer(
                            direct.view[:accepted], dtype=self._torch.uint8
                        )
                        pinned.copy_(source)
                        with self._torch.cuda.stream(self._copy_stream):
                            try:
                                tensor.narrow(0, offset, accepted).copy_(
                                    pinned, non_blocking=True
                                )
                            except BaseException as exc:
                                self._drain_after_copy_failure(exc)
                                raise
                        try:
                            self._copy_stream.synchronize()
                        except BaseException as exc:
                            raise CudaRuntimeFatalError(
                                "CUDA extent restore stream could not be fenced"
                            ) from exc
            if (
                logical_digest.hexdigest() != extent.logical_sha256
                or padded_digest.hexdigest() != extent.padded_sha256
            ):
                raise TorchCudaRuntimeError("CUDA restore extent digest changed")
        except OSError as exc:
            raise TorchCudaRuntimeError("O_DIRECT CUDA restore read failed") from exc
        finally:
            if fd >= 0:
                os.close(fd)

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
            if self._stage_views.get(family) is not tensor:
                raise TorchCudaRuntimeError(
                    f"staged tensor ownership changed for family {family}"
                )
            if not isinstance(getattr(tensor, "is_cuda", None), bool):
                raise TorchCudaRuntimeError(f"invalid staged tensor for family {family}")
            validated[family] = self._validated_tensor(
                tensor, int(getattr(tensor, "numel")())
            )
        self._owner.publish_state(MappingProxyType(validated), boundary)
        self._stage_views.clear()
        self._stage_backings.clear()
        self._stage_offsets.clear()

    def publish_local(
        self, staged: Mapping[str, object], boundary: RuntimeBoundary,
        policy_state: bytes,
    ) -> None:
        """Publish a complete pointer table and authenticated adaptive policy."""

        self._require_boundary(boundary)
        if tuple(staged) != STATE_FAMILIES or self._pinned_staging:
            raise TorchCudaRuntimeError("staged pointer table is incomplete or unfenced")
        validated = {}
        for family in STATE_FAMILIES:
            tensor = staged[family]
            if self._stage_views.get(family) is not tensor:
                raise TorchCudaRuntimeError(
                    f"staged tensor ownership changed for family {family}"
                )
            validated[family] = self._validated_tensor(tensor, tensor.numel())
        publish = getattr(self._owner, "publish_state_with_policy", None)
        if not callable(publish):
            raise TorchCudaRuntimeError("owner lacks adaptive policy publication")
        publish(MappingProxyType(validated), policy_state, boundary)
        self._stage_views.clear()
        self._stage_backings.clear()
        self._stage_offsets.clear()

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
        self._stage_views.clear()
        self._stage_backings.clear()
        self._stage_offsets.clear()

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
    "MAX_ALLOCATION_BYTES",
    "MAX_COMPUTE_STREAMS",
    "TorchApi",
    "TorchCudaRuntime",
    "TorchCudaRuntimeError",
    "TorchStateOwner",
    "TorchStream",
    "TorchTensor",
]


def _allocation_plan(
    allocation_bytes: Mapping[str, int] | None,
    allocation_owners: Mapping[str, str] | None,
) -> tuple[Mapping[str, int], Mapping[str, str]]:
    if allocation_bytes is None and allocation_owners is None:
        return MappingProxyType({}), MappingProxyType({})
    if (
        not isinstance(allocation_bytes, Mapping)
        or tuple(allocation_bytes) != STATE_FAMILIES
        or not isinstance(allocation_owners, Mapping)
        or tuple(allocation_owners) != STATE_FAMILIES
    ):
        raise TorchCudaRuntimeError(
            "allocation plan requires canonical nine-family mappings"
        )
    copied_bytes: dict[str, int] = {}
    copied_owners: dict[str, str] = {}
    seen: set[str] = set()
    for family in STATE_FAMILIES:
        size = allocation_bytes[family]
        owner = allocation_owners[family]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= MAX_ALLOCATION_BYTES
            or owner not in STATE_FAMILIES
            or owner not in seen | {family}
        ):
            raise TorchCudaRuntimeError("CUDA allocation plan entry is invalid")
        if owner != family and allocation_owners[owner] != owner:
            raise TorchCudaRuntimeError("CUDA allocation owner must own its backing")
        if owner != family and size != copied_bytes[owner]:
            raise TorchCudaRuntimeError(
                "shared family must name its owner's complete allocation"
            )
        copied_bytes[family] = size
        copied_owners[family] = owner
        seen.add(family)
    return MappingProxyType(copied_bytes), MappingProxyType(copied_owners)

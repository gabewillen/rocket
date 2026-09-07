"""Atomic K0 metadata publication through captured CUDA graphs.

``K0DeviceBinding`` has one logical writer and accepts only a current
``PreparedDecode`` lease. It stages the seven fixed QSA metadata buffers into
an inactive device bank, launches the captured K0 graph, fences the stream, and
then publishes the complete bank. Failures before publication discard staging
and leave the prior device generation active. A provider contract violation
after ``publish`` faults the binding because device ownership is then unknown.

``Cuda13GraphRuntime`` is the GB10 adapter. It loads one explicit CUDA 13
Runtime library, owns a nonblocking stream, one pinned host staging set, two
device banks, and one graph executable for each bank and captured batch size.
Graph nodes currently upload metadata only. Target and QSA kernel nodes remain
the next integration boundary.

OpenTelemetry span attributes are ``phase`` (six values), ``depth`` (k0),
``graph_batch`` (1, 2, 4, 8, or 16), and ``outcome`` (success or failure).
Generations, stream slots, CUDA pointers, and error strings are excluded.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, Protocol

from .decode import (
    GRAPH_BATCHES,
    MAX_QUERY_ROWS,
    MAX_STREAMS,
    Depth,
    OtelTracer,
    PreparedDecode,
    QsaBuffers,
)

DEFAULT_CUDART = Path("/usr/local/cuda/lib64/libcudart.so")
BUFFER_LAYOUT = (
    ("query_start_loc", (MAX_STREAMS + 1) * 4),
    ("seq_lens", MAX_STREAMS * 4),
    ("stream_slots", MAX_STREAMS * 4),
    ("token_to_req", MAX_QUERY_ROWS * 4),
    ("logical_positions", MAX_QUERY_ROWS * 8),
    ("raw_ring_offsets", MAX_QUERY_ROWS * 4),
    ("compressed_positions", MAX_QUERY_ROWS * 8),
)
BUFFER_FORMATS = {
    "query_start_loc": ("i", 4),
    "seq_lens": ("i", 4),
    "stream_slots": ("i", 4),
    "token_to_req": ("i", 4),
    "logical_positions": ("q", 8),
    "raw_ring_offsets": ("i", 4),
    "compressed_positions": ("q", 8),
}


class DeviceDecodeError(RuntimeError):
    """Validation, CUDA, or provider failure at the K0 device boundary."""


class DevicePhase(str, Enum):
    """Externally observable lifecycle for one single-owner binding."""

    IDLE = "idle"
    STAGING = "staging"
    LAUNCHING = "launching"
    SYNCHRONIZING = "synchronizing"
    PUBLISHING = "publishing"
    DISCARDING = "discarding"
    FAULTED = "faulted"


class _DeviceEvent(str, Enum):
    BEGIN_STAGE = "begin_stage"
    STAGED = "staged"
    STAGE_REJECTED = "stage_rejected"
    LAUNCHED = "launched"
    SYNCHRONIZED = "synchronized"
    PUBLISHED = "published"
    PROVIDER_CONTRACT_BROKEN = "provider_contract_broken"
    BEGIN_DISCARD = "begin_discard"
    DISCARDED = "discarded"
    DISCARD_FAILED = "discard_failed"


_TRANSITIONS = MappingProxyType(
    {
        (DevicePhase.IDLE, _DeviceEvent.BEGIN_STAGE): DevicePhase.STAGING,
        (DevicePhase.STAGING, _DeviceEvent.STAGED): DevicePhase.LAUNCHING,
        (DevicePhase.STAGING, _DeviceEvent.STAGE_REJECTED): DevicePhase.IDLE,
        (DevicePhase.LAUNCHING, _DeviceEvent.LAUNCHED): DevicePhase.SYNCHRONIZING,
        (
            DevicePhase.SYNCHRONIZING,
            _DeviceEvent.SYNCHRONIZED,
        ): DevicePhase.PUBLISHING,
        (DevicePhase.PUBLISHING, _DeviceEvent.PUBLISHED): DevicePhase.IDLE,
        (
            DevicePhase.PUBLISHING,
            _DeviceEvent.PROVIDER_CONTRACT_BROKEN,
        ): DevicePhase.FAULTED,
        (DevicePhase.STAGING, _DeviceEvent.BEGIN_DISCARD): DevicePhase.DISCARDING,
        (DevicePhase.LAUNCHING, _DeviceEvent.BEGIN_DISCARD): DevicePhase.DISCARDING,
        (
            DevicePhase.SYNCHRONIZING,
            _DeviceEvent.BEGIN_DISCARD,
        ): DevicePhase.DISCARDING,
        (
            DevicePhase.PUBLISHING,
            _DeviceEvent.BEGIN_DISCARD,
        ): DevicePhase.DISCARDING,
        (DevicePhase.DISCARDING, _DeviceEvent.DISCARDED): DevicePhase.IDLE,
        (DevicePhase.DISCARDING, _DeviceEvent.DISCARD_FAILED): DevicePhase.FAULTED,
    }
)


@dataclass(frozen=True)
class DevicePublication:
    """Immutable snapshot of the active device metadata generation."""

    generation: int
    graph_batch: int
    bank: int


class DeviceGraphRuntime(Protocol):
    """Synchronous adapter with all-or-none device bank publication.

    ``stage`` borrows host views for the call and returns private staging. It
    must release any partial internal staging before raising without a token.
    ``publish`` must either return a complete publication matching its inputs or
    raise before changing the active bank. ``discard`` consumes unpublished
    staging. The binding externally serializes all methods.
    """

    def stage(self, buffers: QsaBuffers) -> object: ...
    def launch_k0(self, staged: object, graph_batch: int) -> None: ...
    def finish(self, staged: object) -> None: ...
    def publish(
        self, staged: object, generation: int, graph_batch: int
    ) -> DevicePublication: ...
    def discard(self, staged: object) -> None: ...


class K0DeviceBinding:
    """Single-owner transaction from current host metadata to one device bank."""

    _REQUIRED_METHODS = ("stage", "launch_k0", "finish", "publish", "discard")

    def __init__(self, runtime: DeviceGraphRuntime, tracer: OtelTracer):
        if runtime is None or any(
            not callable(getattr(runtime, name, None))
            for name in self._REQUIRED_METHODS
        ):
            raise DeviceDecodeError("device graph runtime contract is incomplete")
        if tracer is None or not callable(
            getattr(tracer, "start_as_current_span", None)
        ):
            raise DeviceDecodeError("an OpenTelemetry tracer is required")
        self._runtime = runtime
        self._tracer = tracer
        self._phase = DevicePhase.IDLE
        self._publication: DevicePublication | None = None

    @property
    def phase(self) -> DevicePhase:
        """Return a point-in-time lifecycle snapshot."""

        return self._phase

    @property
    def publication(self) -> DevicePublication | None:
        """Return the last complete device publication snapshot."""

        return self._publication

    def upload_and_launch(self, prepared: PreparedDecode) -> DevicePublication:
        """Publish one current K0 lease after its graph completes.

        Inputs are borrowed for the synchronous call. Success invalidates no
        host data and returns an owned immutable publication. Expected adapter
        failures leave the previous publication active and return the binding
        to ``IDLE`` after successful discard.
        """

        graph_batch = _safe_graph_batch(prepared)
        with self._observed("validate", graph_batch):
            self._require_idle()
            self._validate(prepared)

        staged: object | None = None
        try:
            self._dispatch(_DeviceEvent.BEGIN_STAGE)
            with self._observed("stage", graph_batch):
                staged = self._runtime.stage(prepared.buffers)
                if staged is None:
                    raise DeviceDecodeError("device graph staging returned no token")

            self._dispatch(_DeviceEvent.STAGED)
            with self._observed("launch", graph_batch):
                self._runtime.launch_k0(staged, graph_batch)

            self._dispatch(_DeviceEvent.LAUNCHED)
            with self._observed("sync", graph_batch):
                self._runtime.finish(staged)

            self._dispatch(_DeviceEvent.SYNCHRONIZED)
            with self._observed("publish", graph_batch):
                publication = self._runtime.publish(
                    staged, prepared.lease.generation, graph_batch
                )
                if (
                    not isinstance(publication, DevicePublication)
                    or publication.generation != prepared.lease.generation
                    or publication.graph_batch != graph_batch
                    or isinstance(publication.bank, bool)
                    or publication.bank not in (0, 1)
                ):
                    self._dispatch(_DeviceEvent.PROVIDER_CONTRACT_BROKEN)
                    raise DeviceDecodeError(
                        "device publish violated its result contract"
                    )
        except BaseException as exc:
            if self._phase is DevicePhase.FAULTED:
                raise
            failure_phase = self._phase.value
            if staged is not None:
                self._dispatch(_DeviceEvent.BEGIN_DISCARD)
                try:
                    with self._observed("discard", graph_batch):
                        self._runtime.discard(staged)
                except BaseException as discard_exc:
                    self._dispatch(_DeviceEvent.DISCARD_FAILED)
                    raise DeviceDecodeError(
                        "device graph discard failed"
                    ) from discard_exc
                self._dispatch(_DeviceEvent.DISCARDED)
            else:
                self._dispatch(_DeviceEvent.STAGE_REJECTED)
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, DeviceDecodeError):
                raise
            raise DeviceDecodeError(
                f"K0 device operation failed during {failure_phase}"
            ) from exc

        self._publication = publication
        self._dispatch(_DeviceEvent.PUBLISHED)
        return publication

    def _validate(self, prepared: PreparedDecode) -> None:
        if not isinstance(prepared, PreparedDecode):
            raise DeviceDecodeError("device binding requires PreparedDecode")
        if not prepared.is_current():
            raise DeviceDecodeError("prepared metadata lease is stale")
        if (
            prepared.lease.depth is not Depth.K0
            or prepared.bucket.depth is not Depth.K0
            or prepared.lease.graph_batch not in GRAPH_BATCHES
            or prepared.bucket.graph_batch != prepared.lease.graph_batch
            or prepared.bucket.actual_batch != prepared.lease.actual_batch
            or prepared.lease.actual_rows != prepared.lease.actual_batch
            or prepared.lease.graph_rows != prepared.lease.graph_batch
            or prepared.lease.generation <= 0
        ):
            raise DeviceDecodeError("prepared K0 shape or generation is invalid")
        if (
            self._publication is not None
            and prepared.lease.generation <= self._publication.generation
        ):
            raise DeviceDecodeError("device generation must increase monotonically")
        _validate_buffers(prepared.buffers)

    def _require_idle(self) -> None:
        if self._phase is not DevicePhase.IDLE:
            raise DeviceDecodeError(
                f"device graph binding is not idle: {self._phase.value}"
            )

    def _dispatch(self, event: _DeviceEvent) -> None:
        transition = (self._phase, event)
        if transition not in _TRANSITIONS:
            observed = self._phase.value
            self._phase = DevicePhase.FAULTED
            raise DeviceDecodeError(
                f"device event {event.value} is invalid from {observed}"
            )
        self._phase = _TRANSITIONS[transition]

    @contextmanager
    def _observed(self, phase: str, graph_batch: int | str) -> Iterator[None]:
        with self._tracer.start_as_current_span("rocket.qwen38.decode.device") as span:
            span.set_attribute("phase", phase)
            span.set_attribute("depth", "k0")
            span.set_attribute("graph_batch", graph_batch)
            try:
                yield
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                raise
            else:
                span.set_attribute("outcome", "success")


@dataclass(frozen=True)
class _CudaStage:
    nonce: int
    bank: int


class _Cuda13Api:
    """Typed CUDA 13 Runtime calls isolated from scheduling logic."""

    def __init__(self, library: Path):
        try:
            self.lib = ctypes.CDLL(str(library))
        except OSError as exc:
            raise DeviceDecodeError(
                f"cannot load CUDA Runtime library: {library}"
            ) from exc
        pointer = ctypes.c_void_p
        size = ctypes.c_size_t
        integer = ctypes.c_int
        unsigned = ctypes.c_uint
        ulonglong = ctypes.c_ulonglong
        self._bind("cudaGetDeviceCount", [ctypes.POINTER(integer)])
        self._bind("cudaSetDevice", [integer])
        self._bind("cudaHostAlloc", [ctypes.POINTER(pointer), size, unsigned])
        self._bind("cudaFreeHost", [pointer])
        self._bind("cudaMalloc", [ctypes.POINTER(pointer), size])
        self._bind("cudaFree", [pointer])
        self._bind("cudaStreamCreateWithFlags", [ctypes.POINTER(pointer), unsigned])
        self._bind("cudaStreamDestroy", [pointer])
        self._bind("cudaStreamBeginCapture", [pointer, integer])
        self._bind("cudaStreamEndCapture", [pointer, ctypes.POINTER(pointer)])
        self._bind("cudaMemcpyAsync", [pointer, pointer, size, integer, pointer])
        self._bind("cudaMemcpy", [pointer, pointer, size, integer])
        self._bind(
            "cudaGraphInstantiate", [ctypes.POINTER(pointer), pointer, ulonglong]
        )
        self._bind("cudaGraphDestroy", [pointer])
        self._bind("cudaGraphExecDestroy", [pointer])
        self._bind("cudaGraphLaunch", [pointer, pointer])
        self._bind("cudaStreamSynchronize", [pointer])
        self.lib.cudaGetErrorString.argtypes = [integer]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def _bind(self, name: str, arguments: list[object]) -> None:
        try:
            function = getattr(self.lib, name)
        except AttributeError as exc:
            raise DeviceDecodeError(f"CUDA Runtime symbol is missing: {name}") from exc
        function.argtypes = arguments
        function.restype = ctypes.c_int

    def call(self, name: str, *arguments: object) -> None:
        code = getattr(self.lib, name)(*arguments)
        if code:
            raw = self.lib.cudaGetErrorString(code)
            detail = raw.decode("utf-8", "replace") if raw else "unknown"
            raise DeviceDecodeError(f"{name} failed with CUDA {code}: {detail}")


class Cuda13GraphRuntime:
    """CUDA 13 double-buffered K0 metadata graph runtime for one GB10 device.

    Construction allocates every resource and captures ten upload graphs. Public
    operations are synchronous, single-owner, and not thread-safe. ``close``
    releases all resources and must run after the binding stops using the
    adapter. ``active_device_pointers`` returns borrowed integer addresses whose
    lifetime ends at ``close``.
    """

    def __init__(self, device: int = 0, library: Path = DEFAULT_CUDART):
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise DeviceDecodeError("CUDA device must be a nonnegative integer")
        if not isinstance(library, Path):
            raise DeviceDecodeError("CUDA Runtime library must be an explicit Path")
        self._api = _Cuda13Api(library)
        self._stream = ctypes.c_void_p()
        self._host: dict[str, ctypes.c_void_p] = {}
        self._device: list[dict[str, ctypes.c_void_p]] = [{}, {}]
        self._graphs: dict[tuple[int, int], ctypes.c_void_p] = {}
        self._execs: dict[tuple[int, int], ctypes.c_void_p] = {}
        self._active_bank = -1
        self._staged: _CudaStage | None = None
        self._launched_batch: int | None = None
        self._finished = False
        self._nonce = 0
        self._closed = False
        try:
            count = ctypes.c_int()
            self._api.call("cudaGetDeviceCount", ctypes.byref(count))
            if device >= count.value:
                raise DeviceDecodeError("CUDA device index is outside the visible set")
            self._api.call("cudaSetDevice", device)
            self._api.call(
                "cudaStreamCreateWithFlags", ctypes.byref(self._stream), 1
            )
            for name, length in BUFFER_LAYOUT:
                host = ctypes.c_void_p()
                self._api.call("cudaHostAlloc", ctypes.byref(host), length, 0)
                ctypes.memset(host, 0, length)
                self._host[name] = host
                for bank in range(2):
                    destination = ctypes.c_void_p()
                    self._api.call("cudaMalloc", ctypes.byref(destination), length)
                    self._device[bank][name] = destination
            for bank in range(2):
                for graph_batch in GRAPH_BATCHES:
                    self._capture(bank, graph_batch)
        except BaseException:
            self._close_noexcept()
            raise

    def stage(self, buffers: QsaBuffers) -> object:
        self._require_open()
        if self._staged is not None:
            raise DeviceDecodeError("CUDA runtime already has private staging")
        _validate_buffers(buffers)
        bank = 0 if self._active_bank != 0 else 1
        for name, length in BUFFER_LAYOUT:
            ctypes.memmove(self._host[name], bytes(getattr(buffers, name)), length)
        self._nonce += 1
        self._staged = _CudaStage(self._nonce, bank)
        self._launched_batch = None
        self._finished = False
        return self._staged

    def launch_k0(self, staged: object, graph_batch: int) -> None:
        token = self._require_stage(staged)
        if graph_batch not in GRAPH_BATCHES or self._launched_batch is not None:
            raise DeviceDecodeError("CUDA graph batch or launch state is invalid")
        self._api.call(
            "cudaGraphLaunch", self._execs[(token.bank, graph_batch)], self._stream
        )
        self._launched_batch = graph_batch

    def finish(self, staged: object) -> None:
        self._require_stage(staged)
        if self._launched_batch is None or self._finished:
            raise DeviceDecodeError("CUDA graph is not awaiting synchronization")
        self._api.call("cudaStreamSynchronize", self._stream)
        self._finished = True

    def publish(
        self, staged: object, generation: int, graph_batch: int
    ) -> DevicePublication:
        token = self._require_stage(staged)
        if (
            not self._finished
            or self._launched_batch != graph_batch
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation <= 0
        ):
            raise DeviceDecodeError("CUDA publication preconditions are not satisfied")
        publication = DevicePublication(generation, graph_batch, token.bank)
        self._active_bank = token.bank
        self._staged = None
        self._launched_batch = None
        self._finished = False
        return publication

    def discard(self, staged: object) -> None:
        self._require_stage(staged)
        self._api.call("cudaStreamSynchronize", self._stream)
        self._staged = None
        self._launched_batch = None
        self._finished = False

    @property
    def active_device_pointers(self) -> Mapping[str, int]:
        """Return borrowed pointers for the active complete metadata bank."""

        self._require_open()
        if self._active_bank not in (0, 1):
            raise DeviceDecodeError("no CUDA metadata bank has been published")
        return MappingProxyType(
            {
                name: int(pointer.value)
                for name, pointer in self._device[self._active_bank].items()
            }
        )

    def read_active(self, name: str) -> bytes:
        """Copy one active field to owned host bytes for smoke verification."""

        self._require_open()
        lengths = dict(BUFFER_LAYOUT)
        if name not in lengths or self._active_bank not in (0, 1):
            raise DeviceDecodeError("active CUDA metadata field is unavailable")
        output = ctypes.create_string_buffer(lengths[name])
        self._api.call(
            "cudaMemcpy",
            ctypes.addressof(output),
            self._device[self._active_bank][name],
            lengths[name],
            2,
        )
        return output.raw

    def close(self) -> None:
        """Synchronize and release all owned CUDA resources exactly once."""

        if self._closed:
            return
        errors = self._close_noexcept()
        if errors:
            raise DeviceDecodeError(
                "CUDA resource cleanup failed: " + "; ".join(errors)
            )

    def __enter__(self) -> "Cuda13GraphRuntime":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _capture(self, bank: int, graph_batch: int) -> None:
        graph = ctypes.c_void_p()
        executable = ctypes.c_void_p()
        self._api.call("cudaStreamBeginCapture", self._stream, 1)
        for name, length in BUFFER_LAYOUT:
            self._api.call(
                "cudaMemcpyAsync",
                self._device[bank][name],
                self._host[name],
                length,
                1,
                self._stream,
            )
        self._api.call("cudaStreamEndCapture", self._stream, ctypes.byref(graph))
        self._graphs[(bank, graph_batch)] = graph
        self._api.call(
            "cudaGraphInstantiate", ctypes.byref(executable), graph, 0
        )
        self._execs[(bank, graph_batch)] = executable

    def _require_stage(self, staged: object) -> _CudaStage:
        self._require_open()
        if not isinstance(staged, _CudaStage) or staged != self._staged:
            raise DeviceDecodeError("CUDA staging token is stale or foreign")
        return staged

    def _require_open(self) -> None:
        if self._closed:
            raise DeviceDecodeError("CUDA graph runtime is closed")

    def _close_noexcept(self) -> list[str]:
        errors = []

        def release(name: str, handle: ctypes.c_void_p) -> None:
            if not handle or not handle.value:
                return
            try:
                self._api.call(name, handle)
            except Exception as exc:
                errors.append(str(exc))

        if self._stream.value:
            try:
                self._api.call("cudaStreamSynchronize", self._stream)
            except Exception as exc:
                errors.append(str(exc))
        for handle in self._execs.values():
            release("cudaGraphExecDestroy", handle)
        for handle in self._graphs.values():
            release("cudaGraphDestroy", handle)
        for bank in self._device:
            for handle in bank.values():
                release("cudaFree", handle)
        for handle in self._host.values():
            release("cudaFreeHost", handle)
        release("cudaStreamDestroy", self._stream)
        self._execs.clear()
        self._graphs.clear()
        self._device = [{}, {}]
        self._host.clear()
        self._closed = True
        return errors


def _validate_buffers(buffers: QsaBuffers) -> None:
    if not isinstance(buffers, QsaBuffers):
        raise DeviceDecodeError("QSA device input must be QsaBuffers")
    for name, length in BUFFER_LAYOUT:
        value = getattr(buffers, name)
        expected_format, expected_itemsize = BUFFER_FORMATS[name]
        if (
            not isinstance(value, memoryview)
            or not value.readonly
            or value.format != expected_format
            or value.itemsize != expected_itemsize
            or value.nbytes != length
        ):
            raise DeviceDecodeError(f"QSA buffer ABI mismatch for {name}")


def _safe_graph_batch(prepared: object) -> int | str:
    try:
        value = prepared.lease.graph_batch
    except AttributeError:
        return "mixed"
    return (
        value
        if isinstance(value, int)
        and not isinstance(value, bool)
        and value in GRAPH_BATCHES
        else "mixed"
    )


__all__ = [
    "BUFFER_LAYOUT",
    "BUFFER_FORMATS",
    "Cuda13GraphRuntime",
    "DEFAULT_CUDART",
    "DeviceDecodeError",
    "DeviceGraphRuntime",
    "DevicePhase",
    "DevicePublication",
    "K0DeviceBinding",
]

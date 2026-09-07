# SPDX-License-Identifier: Apache-2.0
"""Owner-local one-pass rank-slab loading into final CUDA byte layouts.

The two slab readers reuse the interleaved disk/DMA ring structure from vLLM's
Apache-2.0 ``simple_kv_offload.disk_backend``. This model-specific boundary
removes block indirection and tensor dispatch: one worker owns target, one owns
MTP, and each performs exactly one authenticated read and one H2D copy per
manifest chunk before a single publication callback.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from .contract import PINNED_CONTRACT, SlabContract
from .loader import DirectSlabLoader, OtelTracer, SlabDescriptor

_PIPELINE_SLOTS = 2
_RANKS = (0, 1)


class CudaSlabLoadError(RuntimeError):
    """Validation, direct-I/O, CUDA-transfer, or publication failure."""


class _Tensor(Protocol):
    def data_ptr(self) -> int: ...
    def narrow(self, dimension: int, offset: int, length: int) -> "_Tensor": ...
    def copy_(self, source: "_Tensor", *, non_blocking: bool = False) -> "_Tensor": ...
    def numpy(self) -> object: ...


class _Event(Protocol):
    def record(self, stream: "_Stream") -> None: ...
    def synchronize(self) -> None: ...


class _Stream(Protocol):
    def synchronize(self) -> None: ...


class _Cuda(Protocol):
    def is_available(self) -> bool: ...
    def Stream(self, device: str) -> _Stream: ...
    def Event(self) -> _Event: ...
    def stream(self, stream: _Stream) -> object: ...


class _Torch(Protocol):
    uint8: object
    cuda: _Cuda

    def empty(
        self, length: int, *, dtype: object, device: str,
        pin_memory: bool = False,
    ) -> _Tensor: ...


class RankSlabOwner(Protocol):
    """Single-writer owner of the cold model pointer table.

    The callback consumes both private CUDA tensors or raises without changing
    the live table. It is invoked exactly once and only after both streams fence.
    """

    def publish_rank_slabs(self, rank: int, slabs: Mapping[str, _Tensor]) -> None: ...


@dataclass(frozen=True)
class SlabTransferReceipt:
    key: str
    bytes_read: int
    h2d_bytes: int
    direct_reads: int
    h2d_copies: int
    started_ns: int
    completed_ns: int


@dataclass(frozen=True)
class RankLoadReceipt:
    rank: int
    target: SlabTransferReceipt
    mtp: SlabTransferReceipt
    allocation_ns: int
    publish_ns: int
    load_to_publish_ns: int
    reader_overlap_ns: int
    clock: str = "time.perf_counter_ns"

    @property
    def bytes_read(self) -> int:
        return self.target.bytes_read + self.mtp.bytes_read

    @property
    def h2d_bytes(self) -> int:
        return self.target.h2d_bytes + self.mtp.h2d_bytes


@dataclass(frozen=True)
class LoadedRankSlabs:
    """Published final-layout CUDA tensors and their immutable load receipt."""

    slabs: Mapping[str, _Tensor]
    receipt: RankLoadReceipt


@dataclass
class _Pipeline:
    stream: _Stream
    slots: tuple[_Tensor, _Tensor]
    views: tuple[memoryview, memoryview]
    events: tuple[_Event, _Event]


class CudaRankSlabLoader:
    """Non-reentrant owner-local loader for one fixed TP rank.

    The artifact and destination tensors are borrowed until ``load`` returns.
    The loader owns two 2-slot pinned rings and two CUDA streams for that call.
    Failure drains both streams and publishes nothing. Calls are not thread-safe.

    OTEL cardinality is bounded: ``rank`` has 2 values, ``outcome`` has 2,
    ``io.direct`` is boolean, and the span name is fixed, for at most 8 series.
    Exact byte/copy counters and monotonic clocks are returned in the receipt,
    not attached as attributes. No path, digest, tensor name, request id, byte
    count, or timestamp is attached as an attribute.
    """

    def __init__(
        self,
        artifact: Path,
        *,
        rank: int,
        owner: RankSlabOwner,
        tracer: OtelTracer,
        torch_api: _Torch,
        device: str,
        contract: SlabContract = PINNED_CONTRACT,
    ) -> None:
        if rank not in _RANKS:
            raise CudaSlabLoadError("rank must be 0 or 1")
        if owner is None or not callable(getattr(owner, "publish_rank_slabs", None)):
            raise CudaSlabLoadError("rank slab owner contract is incomplete")
        if tracer is None:
            raise CudaSlabLoadError("an OpenTelemetry tracer is required")
        if torch_api is None or not torch_api.cuda.is_available():
            raise CudaSlabLoadError("Torch CUDA is unavailable")
        if not isinstance(device, str) or not device.startswith("cuda:"):
            raise CudaSlabLoadError("device must be an explicit cuda:N")
        self._rank = rank
        self._owner = owner
        self._tracer = tracer
        self._torch = torch_api
        self._device = device
        self._contract = contract
        self._manifest = DirectSlabLoader(artifact, tracer, contract)
        self._load_lock = threading.Lock()

    def load(self) -> LoadedRankSlabs:
        """Load target and MTP once, fence both, then publish one pointer table."""

        if not self._load_lock.acquire(blocking=False):
            raise CudaSlabLoadError("rank slab loader is already active")
        try:
            return self._load_locked()
        finally:
            self._load_lock.release()

    def _load_locked(self) -> LoadedRankSlabs:
        """Execute one load while the single-owner guard is held."""

        started_ns = time.perf_counter_ns()
        target = self._manifest.descriptor(f"rank{self._rank}-target")
        mtp = self._manifest.descriptor(f"rank{self._rank}-mtp")
        with self._tracer.start_as_current_span("rocket.qwen38.rank_slab.cuda_load") as span:
            span.set_attribute("rank", self._rank)
            span.set_attribute("io.direct", True)
            try:
                allocation_started = time.perf_counter_ns()
                destinations = {
                    target.key: self._torch.empty(
                        target.bytes, dtype=self._torch.uint8, device=self._device
                    ),
                    mtp.key: self._torch.empty(
                        mtp.bytes, dtype=self._torch.uint8, device=self._device
                    ),
                }
                pipelines = {
                    target.key: self._pipeline(target),
                    mtp.key: self._pipeline(mtp),
                }
                allocation_ns = time.perf_counter_ns() - allocation_started
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="qwen38-slab") as pool:
                    futures = (
                        pool.submit(
                            self._load_one, target, destinations[target.key],
                            pipelines[target.key],
                        ),
                        pool.submit(
                            self._load_one, mtp, destinations[mtp.key], pipelines[mtp.key]
                        ),
                    )
                    receipts = tuple(future.result() for future in futures)
                publish_started = time.perf_counter_ns()
                published = MappingProxyType(destinations)
                self._owner.publish_rank_slabs(self._rank, published)
                completed_ns = time.perf_counter_ns()
                receipt = RankLoadReceipt(
                    rank=self._rank,
                    target=receipts[0],
                    mtp=receipts[1],
                    allocation_ns=allocation_ns,
                    publish_ns=completed_ns - publish_started,
                    load_to_publish_ns=completed_ns - started_ns,
                    reader_overlap_ns=max(
                        0,
                        min(receipts[0].completed_ns, receipts[1].completed_ns)
                        - max(receipts[0].started_ns, receipts[1].started_ns),
                    ),
                )
                span.set_attribute("outcome", "success")
                return LoadedRankSlabs(published, receipt)
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                if isinstance(exc, CudaSlabLoadError):
                    raise
                raise CudaSlabLoadError("rank slab load failed before publication") from exc

    def _pipeline(self, descriptor: SlabDescriptor) -> _Pipeline:
        slot_bytes = max(chunk.length_bytes for chunk in descriptor.chunks)
        slots: list[_Tensor] = []
        views: list[memoryview] = []
        for _ in range(_PIPELINE_SLOTS):
            raw = self._torch.empty(
                slot_bytes + self._contract.page_bytes,
                dtype=self._torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            offset = -raw.data_ptr() % self._contract.page_bytes
            slot = raw.narrow(0, offset, slot_bytes)
            if slot.data_ptr() % self._contract.page_bytes:
                raise CudaSlabLoadError("pinned staging is not 64 KiB aligned")
            slots.append(slot)
            views.append(memoryview(slot.numpy()))
        return _Pipeline(
            stream=self._torch.cuda.Stream(device=self._device),
            slots=(slots[0], slots[1]),
            views=(views[0], views[1]),
            events=(self._torch.cuda.Event(), self._torch.cuda.Event()),
        )

    def _load_one(
        self, descriptor: SlabDescriptor, destination: _Tensor, pipeline: _Pipeline
    ) -> SlabTransferReceipt:
        started_ns = time.perf_counter_ns()
        fd = -1
        reads = 0
        copies = 0
        try:
            flags = getattr(os, "O_DIRECT", None)
            if flags is None:
                raise CudaSlabLoadError("O_DIRECT is unavailable")
            fd = os.open(descriptor.path, os.O_RDONLY | os.O_NOFOLLOW | flags)
            if os.fstat(fd).st_size != descriptor.bytes:
                raise CudaSlabLoadError("rank slab byte count changed after validation")
            for index, chunk in enumerate(descriptor.chunks):
                slot_index = index % _PIPELINE_SLOTS
                if index >= _PIPELINE_SLOTS:
                    pipeline.events[slot_index].synchronize()
                view = pipeline.views[slot_index][:chunk.length_bytes]
                count = os.preadv(fd, [view], chunk.offset_bytes)
                reads += 1
                if count != chunk.length_bytes:
                    raise CudaSlabLoadError("short O_DIRECT rank slab read")
                if hashlib.sha256(view).hexdigest() != chunk.sha256:
                    raise CudaSlabLoadError("rank slab chunk digest changed")
                with self._torch.cuda.stream(pipeline.stream):
                    destination.narrow(0, chunk.offset_bytes, chunk.length_bytes).copy_(
                        pipeline.slots[slot_index].narrow(0, 0, chunk.length_bytes),
                        non_blocking=True,
                    )
                    pipeline.events[slot_index].record(pipeline.stream)
                copies += 1
        except OSError as exc:
            raise CudaSlabLoadError("O_DIRECT rank slab read failed") from exc
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                pipeline.stream.synchronize()
            except BaseException as exc:
                raise CudaSlabLoadError("CUDA slab copy stream could not be fenced") from exc
        completed_ns = time.perf_counter_ns()
        return SlabTransferReceipt(
            key=descriptor.key,
            bytes_read=descriptor.bytes,
            h2d_bytes=descriptor.bytes,
            direct_reads=reads,
            h2d_copies=copies,
            started_ns=started_ns,
            completed_ns=completed_ns,
        )

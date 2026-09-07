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

_TARGET_PIPELINE_SLOTS = 4
_MTP_PIPELINE_SLOTS = 2
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


class _Metric(Protocol):
    def add(self, amount: int, attributes: Mapping[str, str | int]) -> None: ...
    def record(self, amount: int, attributes: Mapping[str, str | int]) -> None: ...


class OtelMeter(Protocol):
    def create_counter(self, name: str, *, unit: str) -> _Metric: ...
    def create_histogram(self, name: str, *, unit: str) -> _Metric: ...


@dataclass(frozen=True)
class ChunkTransferReceipt:
    index: int
    bytes: int
    direct_read_ns: int
    sha256_ns: int
    h2d_fence_ns: int


@dataclass(frozen=True)
class SlabTransferReceipt:
    key: str
    bytes_read: int
    h2d_bytes: int
    direct_reads: int
    h2d_copies: int
    started_ns: int
    completed_ns: int
    chunks: tuple[ChunkTransferReceipt, ...]


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
    slots: tuple[_Tensor, ...]
    views: tuple[memoryview, ...]
    streams: tuple[_Stream, ...]
    events: tuple[_Event, ...]


class CudaRankSlabLoader:
    """Non-reentrant owner-local loader for one fixed TP rank.

    The artifact and destination tensors are borrowed until ``load`` returns.
    The loader owns a 4-slot target ring and 2-slot MTP ring for that call.
    Failure drains both streams and publishes nothing. Calls are not thread-safe.

    OTEL cardinality is bounded: span dimensions produce at most 8 series.
    Post-publication stage histograms use rank(2) x slab kind(2) x stage(3) =
    12 series; byte/copy counters use rank(2) x kind(2) x direction(2) = 8.
    Measurements do not create series. No path, digest, tensor name, request id,
    byte count, duration, or timestamp is used as an attribute.
    """

    def __init__(
        self,
        artifact: Path,
        *,
        rank: int,
        owner: RankSlabOwner,
        tracer: OtelTracer,
        meter: OtelMeter,
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
        if meter is None:
            raise CudaSlabLoadError("an OpenTelemetry meter is required")
        if torch_api is None or not torch_api.cuda.is_available():
            raise CudaSlabLoadError("Torch CUDA is unavailable")
        if not isinstance(device, str) or not device.startswith("cuda:"):
            raise CudaSlabLoadError("device must be an explicit cuda:N")
        self._rank = rank
        self._owner = owner
        self._tracer = tracer
        self._stage_duration = meter.create_histogram(
            "rocket.qwen38.rank_slab.stage.duration", unit="ns"
        )
        self._byte_counter = meter.create_counter(
            "rocket.qwen38.rank_slab.transfer", unit="By"
        )
        self._copy_counter = meter.create_counter(
            "rocket.qwen38.rank_slab.copy", unit="{copy}"
        )
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
                self._record_metrics(receipt)
                span.set_attribute("outcome", "success")
                return LoadedRankSlabs(published, receipt)
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                if isinstance(exc, CudaSlabLoadError):
                    raise
                raise CudaSlabLoadError("rank slab load failed before publication") from exc

    def _record_metrics(self, receipt: RankLoadReceipt) -> None:
        """Record bounded metrics after publication, outside load-to-publish."""

        try:
            for slab in (receipt.target, receipt.mtp):
                kind = "target" if slab.key.endswith("-target") else "mtp"
                base = {"rank": self._rank, "slab.kind": kind}
                self._byte_counter.add(
                    slab.bytes_read, {**base, "direction": "direct_read"}
                )
                self._byte_counter.add(
                    slab.h2d_bytes, {**base, "direction": "h2d"}
                )
                self._copy_counter.add(
                    slab.direct_reads, {**base, "direction": "direct_read"}
                )
                self._copy_counter.add(
                    slab.h2d_copies, {**base, "direction": "h2d"}
                )
                for chunk in slab.chunks:
                    self._stage_duration.record(
                        chunk.direct_read_ns, {**base, "stage": "direct_read"}
                    )
                    self._stage_duration.record(
                        chunk.sha256_ns, {**base, "stage": "sha256"}
                    )
                    self._stage_duration.record(
                        chunk.h2d_fence_ns, {**base, "stage": "h2d_fence"}
                    )
        except Exception:
            # Telemetry is outside the publication boundary and cannot revoke it.
            return

    def _pipeline(self, descriptor: SlabDescriptor) -> _Pipeline:
        slot_bytes = max(chunk.length_bytes for chunk in descriptor.chunks)
        slot_count = (
            _TARGET_PIPELINE_SLOTS
            if descriptor.key.endswith("-target")
            else _MTP_PIPELINE_SLOTS
        )
        slots: list[_Tensor] = []
        views: list[memoryview] = []
        for _ in range(slot_count):
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
            slots=tuple(slots),
            views=tuple(views),
            streams=tuple(
                self._torch.cuda.Stream(device=self._device)
                for _ in range(slot_count)
            ),
            events=tuple(self._torch.cuda.Event() for _ in range(slot_count)),
        )

    def _authenticate_and_copy(
        self,
        *,
        chunk_index: int,
        chunk_bytes: int,
        expected_sha256: str,
        view: memoryview,
        destination: _Tensor,
        destination_offset: int,
        slot: _Tensor,
        stream: _Stream,
        event: _Event,
        direct_read_ns: int,
    ) -> ChunkTransferReceipt:
        sha_started = time.perf_counter_ns()
        observed_sha256 = hashlib.sha256(view).hexdigest()
        sha256_ns = time.perf_counter_ns() - sha_started
        if observed_sha256 != expected_sha256:
            raise CudaSlabLoadError("rank slab chunk digest changed")
        h2d_started = time.perf_counter_ns()
        with self._torch.cuda.stream(stream):
            destination.narrow(0, destination_offset, chunk_bytes).copy_(
                slot.narrow(0, 0, chunk_bytes), non_blocking=True
            )
            event.record(stream)
        event.synchronize()
        return ChunkTransferReceipt(
            index=chunk_index,
            bytes=chunk_bytes,
            direct_read_ns=direct_read_ns,
            sha256_ns=sha256_ns,
            h2d_fence_ns=time.perf_counter_ns() - h2d_started,
        )

    def _load_one(
        self, descriptor: SlabDescriptor, destination: _Tensor, pipeline: _Pipeline
    ) -> SlabTransferReceipt:
        started_ns = time.perf_counter_ns()
        fd = -1
        reads = 0
        chunk_receipts: list[ChunkTransferReceipt | None] = [
            None for _ in descriptor.chunks
        ]
        try:
            flags = getattr(os, "O_DIRECT", None)
            if flags is None:
                raise CudaSlabLoadError("O_DIRECT is unavailable")
            fd = os.open(descriptor.path, os.O_RDONLY | os.O_NOFOLLOW | flags)
            if os.fstat(fd).st_size != descriptor.bytes:
                raise CudaSlabLoadError("rank slab byte count changed after validation")
            pending = [None for _ in pipeline.slots]
            with ThreadPoolExecutor(
                max_workers=len(pipeline.slots),
                thread_name_prefix=f"{descriptor.key}-auth",
            ) as pool:
                for index, chunk in enumerate(descriptor.chunks):
                    slot_index = index % len(pipeline.slots)
                    previous = pending[slot_index]
                    if previous is not None:
                        previous_index, future = previous
                        chunk_receipts[previous_index] = future.result()
                    view = pipeline.views[slot_index][:chunk.length_bytes]
                    read_started = time.perf_counter_ns()
                    count = os.preadv(fd, [view], chunk.offset_bytes)
                    direct_read_ns = time.perf_counter_ns() - read_started
                    reads += 1
                    if count != chunk.length_bytes:
                        raise CudaSlabLoadError("short O_DIRECT rank slab read")
                    pending[slot_index] = (
                        index,
                        pool.submit(
                            self._authenticate_and_copy,
                            chunk_index=index,
                            chunk_bytes=chunk.length_bytes,
                            expected_sha256=chunk.sha256,
                            view=view,
                            destination=destination,
                            destination_offset=chunk.offset_bytes,
                            slot=pipeline.slots[slot_index],
                            stream=pipeline.streams[slot_index],
                            event=pipeline.events[slot_index],
                            direct_read_ns=direct_read_ns,
                        ),
                    )
                for previous in pending:
                    if previous is not None:
                        previous_index, future = previous
                        chunk_receipts[previous_index] = future.result()
        except OSError as exc:
            raise CudaSlabLoadError("O_DIRECT rank slab read failed") from exc
        finally:
            if fd >= 0:
                os.close(fd)
            for stream in pipeline.streams:
                try:
                    stream.synchronize()
                except BaseException as exc:
                    raise CudaSlabLoadError(
                        "CUDA slab copy stream could not be fenced"
                    ) from exc
        completed_ns = time.perf_counter_ns()
        if any(receipt is None for receipt in chunk_receipts):
            raise CudaSlabLoadError("rank slab chunk pipeline did not drain")
        completed_chunks = tuple(
            receipt for receipt in chunk_receipts if receipt is not None
        )
        return SlabTransferReceipt(
            key=descriptor.key,
            bytes_read=descriptor.bytes,
            h2d_bytes=descriptor.bytes,
            direct_reads=reads,
            h2d_copies=len(completed_chunks),
            started_ns=started_ns,
            completed_ns=completed_ns,
            chunks=completed_chunks,
        )

"""Production rank-slab loader using Linux O_DIRECT and anonymous aligned buffers."""

from __future__ import annotations

import hashlib
import mmap
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .contract import (
    PINNED_CONTRACT,
    SCHEMA,
    SLAB_KEYS,
    SlabContract,
    SlabError,
    canonical_bytes,
    load_json,
)


class _Span(Protocol):
    def __enter__(self) -> "_Span": ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
    def set_attribute(self, key: str, value: str | int | bool) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...


class OtelTracer(Protocol):
    def start_as_current_span(self, name: str) -> _Span: ...


@dataclass(frozen=True)
class SlabChunk:
    offset_bytes: int
    length_bytes: int
    sha256: str


@dataclass(frozen=True)
class SlabDescriptor:
    """Validated immutable extent borrowed from one rank-slab artifact."""

    key: str
    path: Path
    bytes: int
    chunks: tuple[SlabChunk, ...]


class DirectSlabLoader:
    """Single-owner, non-reentrant loader for one immutable slab.

    The caller supplies an OpenTelemetry tracer and a synchronous consumer. Each
    borrowed chunk view is valid only during its callback. Validation or I/O failure
    exposes no later chunk and closes all resources. Concurrent use is unsupported.

    Telemetry cardinality: span names are fixed; ``slab.kind`` has four values,
    ``io.direct`` is boolean, and ``chunk.index`` is bounded by slab size/chunk size.
    Paths, tensor names, digests, errors, and request identifiers are never attributes.
    """

    def __init__(self, artifact: Path, tracer: OtelTracer,
                 contract: SlabContract = PINNED_CONTRACT):
        if tracer is None:
            raise SlabError("an OpenTelemetry tracer is required")
        self._artifact = artifact
        self._tracer = tracer
        self._contract = contract
        self._manifest = load_json(artifact / "manifest.json", 512 * 1024 * 1024)
        claimed_key = self._manifest.get("artifact_key")
        digest_input = dict(self._manifest)
        digest_input.pop("artifact_key", None)
        observed_key = hashlib.sha256(canonical_bytes(digest_input)).hexdigest()
        if claimed_key != observed_key or artifact.name != observed_key:
            raise SlabError("manifest content-address digest mismatch")
        expected = {"schema": SCHEMA, "overlay_artifact_key": contract.artifact_key,
                    "overlay_sha256": contract.overlay_sha256, "tp_size": contract.tp_size,
                    "page_bytes": contract.page_bytes, "tensor_alignment_bytes": contract.tensor_alignment_bytes,
                    "chunk_bytes": contract.chunk_bytes}
        if contract.plan_sha256 is not None:
            expected["plan_sha256"] = contract.plan_sha256
        drift = {key: (self._manifest.get(key), value) for key, value in expected.items()
                 if self._manifest.get(key) != value}
        if drift or self._manifest.get("direct_io") != {
            "required": True, "flag": "O_DIRECT", "regular_file_mmap": False,
            "gds": False, "cufile": False, "nvidia_fs": False,
        }:
            raise SlabError(f"slab manifest contract drift: {drift}")
        if set(self._manifest.get("slabs", {})) != set(SLAB_KEYS):
            raise SlabError("slab manifest inventory is incomplete")

    def descriptor(self, slab_key: str) -> SlabDescriptor:
        """Return one validated contiguous chunk inventory without opening payload."""

        if slab_key not in SLAB_KEYS:
            raise SlabError(f"unknown slab key: {slab_key}")
        slab = self._manifest["slabs"].get(slab_key)
        if not isinstance(slab, dict):
            raise SlabError("slab descriptor is missing")
        byte_count = slab.get("bytes")
        file_name = slab.get("file")
        raw_chunks = slab.get("chunks")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or byte_count % self._contract.page_bytes
            or not isinstance(file_name, str)
            or not file_name
            or Path(file_name).name != file_name
            or not isinstance(raw_chunks, list)
            or not raw_chunks
        ):
            raise SlabError("slab descriptor shape changed")
        chunks: list[SlabChunk] = []
        total = 0
        for raw in raw_chunks:
            if not isinstance(raw, dict):
                raise SlabError("slab chunk descriptor changed")
            offset = raw.get("offset_bytes")
            length = raw.get("length_bytes")
            digest = raw.get("sha256")
            if (
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset != total
                or isinstance(length, bool)
                or not isinstance(length, int)
                or length <= 0
                or length > self._contract.chunk_bytes
                or length % self._contract.page_bytes
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise SlabError("slab chunk extent or digest changed")
            chunks.append(SlabChunk(offset, length, digest))
            total += length
        if total != byte_count:
            raise SlabError("slab chunks do not cover the declared byte count")
        return SlabDescriptor(
            key=slab_key,
            path=self._artifact / file_name,
            bytes=byte_count,
            chunks=tuple(chunks),
        )

    def read(self, slab_key: str, consume: Callable[[int, memoryview], None]) -> int:
        """Read and authenticate every chunk exactly once through O_DIRECT."""
        descriptor = self.descriptor(slab_key)
        flags = getattr(os, "O_DIRECT", None)
        if flags is None:
            raise SlabError("O_DIRECT is unavailable on this platform")
        with self._tracer.start_as_current_span("rocket.qwen38.slab.read") as span:
            span.set_attribute("slab.kind", slab_key)
            span.set_attribute("io.direct", True)
            fd = -1
            try:
                fd = os.open(descriptor.path, os.O_RDONLY | flags)
                identity = os.fstat(fd)
                if identity.st_size != descriptor.bytes:
                    raise SlabError("slab byte count or page alignment drift")
                total = 0
                for index, chunk in enumerate(descriptor.chunks):
                    offset, length = chunk.offset_bytes, chunk.length_bytes
                    buffer = mmap.mmap(-1, length)
                    view = memoryview(buffer)
                    try:
                        count = os.preadv(fd, [view], offset)
                        if count != length:
                            raise SlabError(f"short direct read: {count}/{length}")
                        if hashlib.sha256(view).hexdigest() != chunk.sha256:
                            raise SlabError(f"chunk digest mismatch at bounded index {index}")
                        span.set_attribute("chunk.index", index)
                        consume(offset, view)
                    finally:
                        view.release()
                        buffer.close()
                    total += length
                if total != identity.st_size:
                    raise SlabError("chunk inventory does not cover slab")
                return total
            except OSError as exc:
                failure = SlabError("O_DIRECT slab open or read failed")
                span.record_exception(failure)
                raise failure from exc
            except BaseException as exc:
                span.record_exception(exc)
                raise
            finally:
                if fd >= 0:
                    os.close(fd)

"""Production rank-slab loader using Linux O_DIRECT and anonymous aligned buffers."""

from __future__ import annotations

import hashlib
import mmap
import os
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
        drift = {key: (self._manifest.get(key), value) for key, value in expected.items()
                 if self._manifest.get(key) != value}
        if drift or self._manifest.get("direct_io") != {
            "required": True, "flag": "O_DIRECT", "regular_file_mmap": False,
            "gds": False, "cufile": False, "nvidia_fs": False,
        }:
            raise SlabError(f"slab manifest contract drift: {drift}")
        if set(self._manifest.get("slabs", {})) != set(SLAB_KEYS):
            raise SlabError("slab manifest inventory is incomplete")

    def read(self, slab_key: str, consume: Callable[[int, memoryview], None]) -> int:
        """Read and authenticate every chunk exactly once through O_DIRECT."""
        if slab_key not in SLAB_KEYS:
            raise SlabError(f"unknown slab key: {slab_key}")
        slab = self._manifest["slabs"][slab_key]
        path = self._artifact / slab["file"]
        flags = getattr(os, "O_DIRECT", None)
        if flags is None:
            raise SlabError("O_DIRECT is unavailable on this platform")
        with self._tracer.start_as_current_span("rocket.qwen38.slab.read") as span:
            span.set_attribute("slab.kind", slab_key)
            span.set_attribute("io.direct", True)
            fd = -1
            try:
                fd = os.open(path, os.O_RDONLY | flags)
                identity = os.fstat(fd)
                if identity.st_size != slab["bytes"] or identity.st_size % self._contract.page_bytes:
                    raise SlabError("slab byte count or page alignment drift")
                total = 0
                for index, chunk in enumerate(slab["chunks"]):
                    offset, length = chunk["offset_bytes"], chunk["length_bytes"]
                    if offset != total or length <= 0 or length % self._contract.page_bytes:
                        raise SlabError("chunk extent is not contiguous and 64 KiB aligned")
                    buffer = mmap.mmap(-1, length)
                    view = memoryview(buffer)
                    try:
                        count = os.preadv(fd, [view], offset)
                        if count != length:
                            raise SlabError(f"short direct read: {count}/{length}")
                        if hashlib.sha256(view).hexdigest() != chunk["sha256"]:
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

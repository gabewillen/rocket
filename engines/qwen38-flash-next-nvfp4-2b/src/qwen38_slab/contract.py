"""Canonical, fail-closed contract for Qwen3.8 immutable rank slabs."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "rocket.qwen38-rank-slab.v1"
PLAN_SCHEMA = "rocket.qwen38-rank-slab-plan.v2"
OVERLAY_SCHEMA = "rocket.qwen38.nvfp4-overlay.v2"
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
ARTIFACT_KEY = "23d2c39e9c2cf36a832cb1750f6542fa46f11f1befe195d80c082051a5a772b4"
OVERLAY_SHA256 = "57c5f07d518d302349c1f0f6f8ae75276ae9de602a8b6faadce4faf98a198d6f"
OVERLAY_TENSORS = 278
TP_SIZE = 2
PAGE_BYTES = 65_536
TENSOR_ALIGNMENT_BYTES = 256
CHUNK_BYTES = 256 * 1024 * 1024
SLAB_KEYS = ("rank0-target", "rank0-mtp", "rank1-target", "rank1-mtp")
MODEL_NVFP4_ABI = "modelopt_nvfp4_group16_cutlass_sm121_sfb"
MTP_FP8_ABI = "fp8_e4m3_block_128x128"


class SlabError(ValueError):
    """Validated boundary failure. Materialization never publishes partial output."""


@dataclass(frozen=True)
class SlabContract:
    """Inputs are borrowed; returned manifests are immutable snapshots.

    A caller must provide the exact revision, topology, page size, overlay identity,
    and overlay inventory. Validation failures have no published side effects.
    """

    revision: str
    artifact_key: str
    overlay_sha256: str
    overlay_tensors: int
    tp_size: int = TP_SIZE
    page_bytes: int = PAGE_BYTES
    tensor_alignment_bytes: int = TENSOR_ALIGNMENT_BYTES
    chunk_bytes: int = CHUNK_BYTES


PINNED_CONTRACT = SlabContract(
    revision=REVISION,
    artifact_key=ARTIFACT_KEY,
    overlay_sha256=OVERLAY_SHA256,
    overlay_tensors=OVERLAY_TENSORS,
)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk_bytes):
            digest.update(data)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_json(path: Path, maximum_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if not 0 < size <= maximum_bytes:
            raise SlabError(f"JSON size outside 1..{maximum_bytes}: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SlabError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SlabError(f"JSON root is not an object: {path}")
    return value


def validate_plan(plan: dict[str, Any], contract: SlabContract) -> None:
    expected = {
        "schema": PLAN_SCHEMA,
        "checkpoint_revision": contract.revision,
        "tensor_parallel_size": contract.tp_size,
        "tensor_alignment_bytes": contract.tensor_alignment_bytes,
        "io_alignment_bytes": contract.page_bytes,
        "io_chunk_bytes": contract.chunk_bytes,
        "payload_io_performed": False,
        "assignment_phase": "complete-before-payload-io",
    }
    drift = {key: (plan.get(key), value) for key, value in expected.items() if plan.get(key) != value}
    if drift:
        raise SlabError(f"rank plan contract drift: {drift}")
    slabs = plan.get("slabs")
    if not isinstance(slabs, dict) or tuple(sorted(slabs)) != tuple(sorted(SLAB_KEYS)):
        raise SlabError("rank plan must contain exactly four TP2 target/MTP slabs")
    for key in SLAB_KEYS:
        slab = slabs[key]
        if not isinstance(slab, dict) or slab.get("key") != key:
            raise SlabError(f"invalid slab record: {key}")
        entries = slab.get("entries")
        if not isinstance(entries, list):
            raise SlabError(f"slab entries are missing: {key}")
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("entry_type") not in {"payload", "reference"}:
                raise SlabError(f"invalid entry in {key}")
            if entry.get("entry_type") == "payload":
                offset = entry.get("slab_offset_bytes")
                if not isinstance(offset, int) or offset % contract.tensor_alignment_bytes:
                    raise SlabError(f"unaligned plan tensor in {key}")
            elif entry.get("payload_io") is not False or entry.get("physical_owner") not in SLAB_KEYS:
                raise SlabError(f"invalid shared reference in {key}")


def validate_overlay(root: Path, contract: SlabContract) -> tuple[dict[str, Any], Path]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    overlay = manifest.get("overlay")
    source = manifest.get("source")
    expected_families = ["base_ple", "base_routers", "full_attention", "linear_attention"]
    if (
        manifest.get("schema") != OVERLAY_SCHEMA
        or manifest.get("artifact_key") != contract.artifact_key
        or not isinstance(source, dict)
        or source.get("revision") != contract.revision
        or source.get("families") != expected_families
        or not isinstance(source.get("tensors"), list)
        or len(source["tensors"]) != contract.overlay_tensors
        or not isinstance(overlay, dict)
        or overlay.get("sha256") != contract.overlay_sha256
    ):
        raise SlabError("frozen overlay manifest identity, family, or inventory drift")
    names = [item.get("name") for item in source["tensors"] if isinstance(item, dict)]
    if len(names) != len(set(names)) or any(not isinstance(name, str) for name in names):
        raise SlabError("overlay source inventory contains duplicate or invalid names")
    overlay_path = root / str(overlay.get("file", ""))
    if not overlay_path.is_file() or sha256_file(overlay_path) != contract.overlay_sha256:
        raise SlabError("frozen overlay payload digest mismatch")
    return manifest, overlay_path


def read_safetensors_index(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    """Read a safetensors header offline; production slab loading never calls this."""
    try:
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise SlabError(f"short safetensors header length: {path}")
            length = struct.unpack("<Q", raw_length)[0]
            if not 0 < length <= 64 * 1024 * 1024 or 8 + length > os.fstat(stream.fileno()).st_size:
                raise SlabError(f"invalid safetensors header length: {path}")
            raw = stream.read(length)
        header = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SlabError(f"cannot read safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise SlabError(f"safetensors header is not an object: {path}")
    result: dict[str, dict[str, Any]] = {}
    payload_start = 8 + length
    payload_size = path.stat().st_size - payload_start
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise SlabError(f"invalid safetensors entry: {path}")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > payload_size
        ):
            raise SlabError(f"invalid safetensors extent for {name}")
        result[name] = metadata
    return payload_start, result


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def sf_swizzle(linear: bytes, rows: int, k: int) -> bytes:
    """Return CUTLASS SM121 SFB bytes for row-major ModelOpt group-16 scales."""
    if rows <= 0 or k <= 0 or k % 16:
        raise SlabError("NVFP4 scale shape requires positive rows and K divisible by 16")
    k_sf = k // 16
    if len(linear) != rows * k_sf:
        raise SlabError("NVFP4 scale byte count does not match matrix shape")
    mn_tiles = align_up(rows, 128) // 128
    k_tiles = align_up(k_sf, 4) // 4
    output = bytearray(mn_tiles * k_tiles * 512)
    for row in range(rows):
        row_tile, row_in_tile = divmod(row, 128)
        for sf_index in range(k_sf):
            k_tile, sf_in_tile = divmod(sf_index, 4)
            atom = (row_tile * k_tiles + k_tile) * 512
            offset = atom + (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4 + sf_in_tile
            output[offset] = linear[row * k_sf + sf_index]
    return bytes(output)

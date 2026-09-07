"""Authenticated device bindings for fixed Qwen full-attention graphs.

The published ``a9fcca...`` slab predates the QSA indexer classifier fix and
contains one 320-row half of the replicated BF16 ``index_qk_proj`` on each TP
rank.  This loader authenticates both owner chunks and reconstructs the exact
``[640, 2560]`` tensor once at cold load.  It never relabels or mutates the
published artifact and adds no decode-step communication.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .contract import MODEL_NVFP4_ABI, PINNED_CONTRACT, SCHEMA, canonical_bytes

FULL_ATTENTION_LAYERS = tuple(range(3, 48, 4))
INDEX_QK_ROWS = 640
INDEX_QK_HALF_ROWS = 320
HIDDEN = 2560
CURRENT_ARTIFACT_KEY = (
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
)


class FullAttentionBindingError(RuntimeError):
    """Slab identity, digest, layer, or extent failure."""


@dataclass(frozen=True)
class TensorExtent:
    name: str
    rank: int
    slab_path: Path
    offset_bytes: int
    length_bytes: int
    shape: tuple[int, ...]
    dtype: str
    layout: str


@dataclass(frozen=True)
class AuthenticatedFullAttentionLayer:
    revision: str
    artifact_key: str
    rank: int
    layer: int
    local_chunk_sha256: str
    peer_index_chunk_sha256: str
    local_chunk_offset: int
    peer_chunk_offset: int
    local_extents: tuple[TensorExtent, ...]
    index_qk_halves: tuple[TensorExtent, TensorExtent]


def load_full_attention_layer_binding(
    artifact: Path, *, rank: int, layer: int
) -> AuthenticatedFullAttentionLayer:
    """Authenticate one rank/layer plus both legacy index-QK source halves."""

    if not isinstance(artifact, Path):
        raise FullAttentionBindingError("rank slab artifact must be a Path")
    if isinstance(rank, bool) or rank not in (0, 1):
        raise FullAttentionBindingError("full-attention rank must be 0 or 1")
    if isinstance(layer, bool) or layer not in FULL_ATTENTION_LAYERS:
        raise FullAttentionBindingError("layer is not a fixed full-attention layer")
    try:
        manifest = json.loads((artifact / "manifest.json").read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FullAttentionBindingError("cannot load rank slab manifest") from exc
    if not isinstance(manifest, dict):
        raise FullAttentionBindingError("rank slab manifest root is invalid")
    authenticated = dict(manifest)
    artifact_key = authenticated.pop("artifact_key", None)
    if (
        artifact_key != hashlib.sha256(canonical_bytes(authenticated)).hexdigest()
        or artifact.name != artifact_key
        or artifact_key != CURRENT_ARTIFACT_KEY
        or manifest.get("schema") != SCHEMA
        or manifest.get("revision") != PINNED_CONTRACT.revision
        or manifest.get("overlay_artifact_key") != PINNED_CONTRACT.artifact_key
        or manifest.get("overlay_sha256") != PINNED_CONTRACT.overlay_sha256
        or manifest.get("tp_size") != 2
        or manifest.get("tensor_alignment_bytes")
        != PINNED_CONTRACT.tensor_alignment_bytes
    ):
        raise FullAttentionBindingError("rank slab identity changed")

    local_slab = _slab(manifest, rank)
    peer_slab = _slab(manifest, 1 - rank)
    local_by_name = {entry.get("name"): entry for entry in local_slab["entries"]}
    peer_by_name = {entry.get("name"): entry for entry in peer_slab["entries"]}
    prefix = f"model.language_model.layers.{layer}.self_attn"
    contracts = (
        ("q_proj.weight", (6144, 1280), "U8", "packed_e2m1_row_major", 7864320, MODEL_NVFP4_ABI),
        ("q_proj.weight_scale", (983040,), "F8_E4M3", "cutlass_sm121_sfb", 983040, MODEL_NVFP4_ABI),
        ("q_proj.weight_scale_2", (1,), "F32", "scalar", 4, MODEL_NVFP4_ABI),
        ("k_proj.weight", (256, 1280), "U8", "packed_e2m1_row_major", 327680, MODEL_NVFP4_ABI),
        ("k_proj.weight_scale", (40960,), "F8_E4M3", "cutlass_sm121_sfb", 40960, MODEL_NVFP4_ABI),
        ("k_proj.weight_scale_2", (1,), "F32", "scalar", 4, MODEL_NVFP4_ABI),
        ("v_proj.weight", (256, 1280), "U8", "packed_e2m1_row_major", 327680, MODEL_NVFP4_ABI),
        ("v_proj.weight_scale", (40960,), "F8_E4M3", "cutlass_sm121_sfb", 40960, MODEL_NVFP4_ABI),
        ("v_proj.weight_scale_2", (1,), "F32", "scalar", 4, MODEL_NVFP4_ABI),
        ("o_proj.weight", (2560, 1536), "U8", "packed_e2m1_row_major", 3932160, MODEL_NVFP4_ABI),
        ("o_proj.weight_scale", (491520,), "F8_E4M3", "cutlass_sm121_sfb", 491520, MODEL_NVFP4_ABI),
        ("o_proj.weight_scale_2", (1,), "F32", "scalar", 4, MODEL_NVFP4_ABI),
        ("q_norm.weight", (256,), "BF16", "checkpoint", 512, "native"),
        ("k_norm.weight", (256,), "BF16", "checkpoint", 512, "native"),
        ("indexer.q_layernorm.weight", (128,), "BF16", "checkpoint", 256, "native"),
        ("indexer.k_layernorm.weight", (128,), "BF16", "checkpoint", 256, "native"),
    )
    local_extents = tuple(
        _extent(local_slab, local_by_name, f"{prefix}.{suffix}", rank, shape,
                dtype, layout, length, artifact, abi)
        for suffix, shape, dtype, layout, length, abi in contracts
    )
    index_name = f"{prefix}.indexer.index_qk_proj.weight"
    halves = (
        _extent(_slab(manifest, 0),
                {e.get("name"): e for e in _slab(manifest, 0)["entries"]},
                index_name, 0, (INDEX_QK_HALF_ROWS, HIDDEN), "BF16",
                "checkpoint", INDEX_QK_HALF_ROWS * HIDDEN * 2, artifact),
        _extent(_slab(manifest, 1),
                {e.get("name"): e for e in _slab(manifest, 1)["entries"]},
                index_name, 1, (INDEX_QK_HALF_ROWS, HIDDEN), "BF16",
                "checkpoint", INDEX_QK_HALF_ROWS * HIDDEN * 2, artifact),
    )
    local_chunk = _one_chunk(local_slab, (*local_extents, halves[rank]))
    peer_chunk = _one_chunk(peer_slab, (halves[1 - rank],))
    _authenticate_chunk(local_slab, local_chunk, artifact)
    _authenticate_chunk(peer_slab, peer_chunk, artifact)
    return AuthenticatedFullAttentionLayer(
        PINNED_CONTRACT.revision,
        artifact_key,
        rank,
        layer,
        local_chunk["sha256"],
        peer_chunk["sha256"],
        local_chunk["offset_bytes"],
        peer_chunk["offset_bytes"],
        local_extents,
        halves,
    )


def reconstruct_index_qk_weight(binding: AuthenticatedFullAttentionLayer) -> bytes:
    """Read rank-0 then rank-1 authenticated halves into canonical row order."""

    if not isinstance(binding, AuthenticatedFullAttentionLayer):
        raise FullAttentionBindingError("authenticated full-attention binding required")
    blobs: list[bytes] = []
    for extent in binding.index_qk_halves:
        fd = os.open(extent.slab_path, os.O_RDONLY)
        try:
            blob = os.pread(fd, extent.length_bytes, extent.offset_bytes)
        finally:
            os.close(fd)
        if len(blob) != extent.length_bytes:
            raise FullAttentionBindingError("short index-QK half read")
        blobs.append(blob)
    result = b"".join(blobs)
    if len(result) != INDEX_QK_ROWS * HIDDEN * 2:
        raise FullAttentionBindingError("reconstructed index-QK extent changed")
    return result


def _slab(manifest: dict, rank: int) -> dict:
    slab = manifest.get("slabs", {}).get(f"rank{rank}-target")
    if (
        not isinstance(slab, dict)
        or not isinstance(slab.get("entries"), list)
        or not isinstance(slab.get("chunks"), list)
    ):
        raise FullAttentionBindingError("rank target slab inventory is missing")
    return slab


def _extent(slab, by_name, name, rank, shape, dtype, layout, length, artifact,
            abi="native"):
    entry = by_name.get(name)
    if (
        not isinstance(entry, dict)
        or tuple(entry.get("shape", ())) != shape
        or entry.get("dtype") != dtype
        or entry.get("layout") != layout
        or entry.get("length_bytes") != length
        or entry.get("abi") != abi
    ):
        raise FullAttentionBindingError(f"full-attention tensor contract changed: {name}")
    offset = entry.get("offset_bytes")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or offset % 256:
        raise FullAttentionBindingError(f"full-attention tensor offset changed: {name}")
    path = artifact / str(slab.get("file", ""))
    try:
        observed = path.stat().st_size
    except OSError as exc:
        raise FullAttentionBindingError("rank target slab file is missing") from exc
    if observed != slab.get("bytes"):
        raise FullAttentionBindingError("rank target slab byte count changed")
    return TensorExtent(name, rank, path, offset, length, shape, dtype, layout)


def _one_chunk(slab, extents):
    touched = [
        chunk for chunk in slab["chunks"]
        if any(extent.offset_bytes < chunk["offset_bytes"] + chunk["length_bytes"] and
               extent.offset_bytes + extent.length_bytes > chunk["offset_bytes"]
               for extent in extents)
    ]
    if len(touched) != 1:
        raise FullAttentionBindingError("one layer must bind one authenticated chunk")
    return touched[0]


def _authenticate_chunk(slab, chunk, artifact):
    path = artifact / str(slab.get("file", ""))
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        remaining = chunk["length_bytes"]
        offset = chunk["offset_bytes"]
        while remaining:
            block = os.pread(fd, min(8 * 1024 * 1024, remaining), offset)
            if not block:
                raise FullAttentionBindingError("short authenticated chunk read")
            digest.update(block)
            offset += len(block)
            remaining -= len(block)
    finally:
        os.close(fd)
    if digest.hexdigest() != chunk.get("sha256"):
        raise FullAttentionBindingError("full-attention chunk digest mismatch")

# SPDX-License-Identifier: Apache-2.0
"""Authenticated TP2 rank-local slab contract for every fixed GDN graph."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import MODEL_NVFP4_ABI, PINNED_CONTRACT, SCHEMA, canonical_bytes

GDN_SCHEMA = "qwen3.8-flash-next:tp2:rank-local:gdn-nvfp4:v2"
GDN_STATE_FAMILIES = ("target_gdn_conv", "target_gdn_recurrent")


class LinearAttentionSlabError(RuntimeError):
    """The fixed GDN slab identity or tensor inventory changed."""


@dataclass(frozen=True)
class GdnComponent:
    name: str
    offset: int
    length: int
    shape: tuple[int, ...]
    dtype: str
    layout: str
    abi: str


@dataclass(frozen=True)
class GdnAuthenticatedChunk:
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class RankGdnLayer:
    schema: str
    artifact_key: str
    rank: int
    layer: int
    slab_path: Path
    slab_bytes: int
    chunks: tuple[GdnAuthenticatedChunk, ...]
    components: tuple[GdnComponent, ...]


# Exact offsets make a descriptor from another plan fail before a CUDA pointer
# can be formed. Source scalar input scales are authenticated even though the
# fixed graph derives per-token activation scales at launch.
_NATIVE_ABI = "native"
GDN_COMPONENT_CONTRACTS: tuple[
    tuple[str, int, tuple[int, ...], str, str, str], ...
] = (
    ("A_log", 48, (24,), "BF16", "checkpoint", _NATIVE_ABI),
    ("conv1d.weight", 40_960, (5_120, 1, 4), "BF16", "checkpoint", _NATIVE_ABI),
    ("dt_bias", 48, (24,), "BF16", "checkpoint", _NATIVE_ABI),
    ("in_proj_a.weight", 30_720, (24, 1_280), "U8", "packed_e2m1_row_major", MODEL_NVFP4_ABI),
    ("in_proj_a.weight_scale", 20_480, (20_480,), "F8_E4M3", "cutlass_sm121_sfb", MODEL_NVFP4_ABI),
    ("in_proj_a.weight_scale_2", 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_a.input_scale", 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_b.weight", 30_720, (24, 1_280), "U8", "packed_e2m1_row_major", MODEL_NVFP4_ABI),
    ("in_proj_b.weight_scale", 20_480, (20_480,), "F8_E4M3", "cutlass_sm121_sfb", MODEL_NVFP4_ABI),
    ("in_proj_b.weight_scale_2", 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_b.input_scale", 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_qkv.weight", 6_553_600, (5_120, 1_280), "U8", "packed_e2m1_row_major", MODEL_NVFP4_ABI),
    ("in_proj_qkv.weight_scale", 819_200, (819_200,), "F8_E4M3", "cutlass_sm121_sfb", MODEL_NVFP4_ABI),
    ("in_proj_qkv.weight_scale_2", 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_qkv.input_scale", 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_z.weight", 3_932_160, (3_072, 1_280), "U8", "packed_e2m1_row_major", MODEL_NVFP4_ABI),
    ("in_proj_z.weight_scale", 491_520, (491_520,), "F8_E4M3", "cutlass_sm121_sfb", MODEL_NVFP4_ABI),
    ("in_proj_z.weight_scale_2", 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_z.input_scale", 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("norm.weight", 256, (128,), "BF16", "checkpoint", _NATIVE_ABI),
    ("out_proj.weight", 3_932_160, (2_560, 1_536), "U8", "packed_e2m1_row_major", MODEL_NVFP4_ABI),
    ("out_proj.weight_scale", 491_520, (491_520,), "F8_E4M3", "cutlass_sm121_sfb", MODEL_NVFP4_ABI),
    ("out_proj.weight_scale_2", 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("out_proj.input_scale", 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
)


def validate_gdn_inventory(
    manifest: dict[str, Any], rank: int, layer: int
) -> tuple[GdnComponent, ...]:
    """Validate every fixed tensor before state pointers or graph state mutate."""

    expected = {
        "schema": SCHEMA,
        "revision": PINNED_CONTRACT.revision,
        "overlay_artifact_key": PINNED_CONTRACT.artifact_key,
        "overlay_sha256": PINNED_CONTRACT.overlay_sha256,
        "tp_size": 2,
        "tensor_alignment_bytes": PINNED_CONTRACT.tensor_alignment_bytes,
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise LinearAttentionSlabError("rank slab manifest contract drift")
    if isinstance(rank, bool) or rank not in (0, 1):
        raise LinearAttentionSlabError("GDN rank must be 0 or 1")
    if isinstance(layer, bool) or not 0 <= layer < 48 or layer % 4 == 3:
        raise LinearAttentionSlabError("GDN layer must use the 36-layer topology")
    slab_key = f"rank{rank}-target"
    prefix = f"model.language_model.layers.{layer}.linear_attn"
    slab = manifest.get("slabs", {}).get(slab_key)
    if not isinstance(slab, dict) or not isinstance(slab.get("entries"), list):
        raise LinearAttentionSlabError(f"{slab_key} slab inventory is missing")
    valid_entries = [entry for entry in slab["entries"] if isinstance(entry, dict)]
    entries = {entry.get("name"): entry for entry in valid_entries}
    if len(entries) != len(valid_entries):
        raise LinearAttentionSlabError(f"{slab_key} contains duplicate tensor names")
    components: list[GdnComponent] = []
    for suffix, length, shape, dtype, layout, abi in GDN_COMPONENT_CONTRACTS:
        name = f"{prefix}.{suffix}"
        entry = entries.get(name)
        offset = entry.get("offset_bytes") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or entry.get("length_bytes") != length
            or tuple(entry.get("shape", ())) != shape
            or entry.get("dtype") != dtype
            or entry.get("layout") != layout
            or entry.get("abi") != abi
            or offset % PINNED_CONTRACT.tensor_alignment_bytes
        ):
            raise LinearAttentionSlabError(f"GDN tensor contract drift: {name}")
        components.append(GdnComponent(name, offset, length, shape, dtype, layout, abi))
    return tuple(components)


def load_gdn_layer(artifact: Path, rank: int, layer: int) -> RankGdnLayer:
    if not isinstance(artifact, Path):
        raise LinearAttentionSlabError("rank slab artifact must be a Path")
    try:
        raw = (artifact / "manifest.json").read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LinearAttentionSlabError("cannot load rank slab manifest") from exc
    digest_input = dict(manifest) if isinstance(manifest, dict) else {}
    claimed = digest_input.pop("artifact_key", None)
    observed = hashlib.sha256(canonical_bytes(digest_input)).hexdigest()
    if claimed != observed or artifact.name != observed:
        raise LinearAttentionSlabError("rank slab manifest content address mismatch")
    components = validate_gdn_inventory(manifest, rank, layer)
    slab_key = f"rank{rank}-target"
    slab = manifest["slabs"][slab_key]
    slab_path = artifact / str(slab.get("file", ""))
    slab_bytes = slab.get("bytes")
    if isinstance(slab_bytes, bool) or not isinstance(slab_bytes, int) or slab_bytes <= 0:
        raise LinearAttentionSlabError(f"{slab_key} byte count is invalid")
    try:
        if slab_path.stat().st_size != slab_bytes:
            raise LinearAttentionSlabError(f"{slab_key} byte count drift")
    except OSError as exc:
        raise LinearAttentionSlabError(f"{slab_key} file is missing") from exc
    chunks = slab.get("chunks")
    if not isinstance(chunks, list):
        raise LinearAttentionSlabError("rank0 target slab chunks are missing")
    touched = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise LinearAttentionSlabError("GDN chunk record is invalid")
        chunk_offset, chunk_length = chunk.get("offset_bytes"), chunk.get("length_bytes")
        if (
            isinstance(chunk_offset, bool) or not isinstance(chunk_offset, int)
            or isinstance(chunk_length, bool) or not isinstance(chunk_length, int)
            or chunk_offset < 0 or chunk_length <= 0
        ):
            raise LinearAttentionSlabError("GDN chunk extent is invalid")
        if any(
            component.offset < chunk_offset + chunk_length
            and component.offset + component.length > chunk_offset
            for component in components
        ):
            touched.append(chunk)
    if not touched:
        raise LinearAttentionSlabError("GDN tensors occupy no authenticated chunk")
    authenticated = []
    try:
        with slab_path.open("rb") as stream:
            for chunk in touched:
                offset, length = chunk.get("offset_bytes"), chunk.get("length_bytes")
                claimed_digest = chunk.get("sha256")
                if (
                    isinstance(offset, bool) or not isinstance(offset, int)
                    or isinstance(length, bool) or not isinstance(length, int)
                    or not isinstance(claimed_digest, str)
                ):
                    raise LinearAttentionSlabError("GDN chunk extent is invalid")
                digest = hashlib.sha256()
                stream.seek(offset)
                remaining = length
                while remaining:
                    data = stream.read(min(8 * 1024 * 1024, remaining))
                    if not data:
                        raise LinearAttentionSlabError("short GDN chunk read")
                    digest.update(data)
                    remaining -= len(data)
                if digest.hexdigest() != claimed_digest:
                    raise LinearAttentionSlabError("GDN slab chunk digest mismatch")
                authenticated.append(GdnAuthenticatedChunk(offset, length, claimed_digest))
    except OSError as exc:
        raise LinearAttentionSlabError("cannot authenticate GDN slab chunk") from exc
    return RankGdnLayer(
        GDN_SCHEMA, claimed, rank, layer, slab_path, slab_bytes,
        tuple(authenticated), components
    )

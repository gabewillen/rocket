# SPDX-License-Identifier: Apache-2.0
"""Authenticated rank-0/layer-0 slab contract for the fixed GDN graph."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import MODEL_NVFP4_ABI, PINNED_CONTRACT, SCHEMA, canonical_bytes

GDN_SCHEMA = "qwen3.8-flash-next:tp2:rank0:layer0:gdn-w4a4:v1"
GDN_PREFIX = "model.language_model.layers.0.linear_attn"
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
class Rank0Layer0Gdn:
    schema: str
    artifact_key: str
    slab_path: Path
    slab_bytes: int
    chunk_offset: int
    chunk_length: int
    chunk_sha256: str
    components: tuple[GdnComponent, ...]


# Exact offsets make a descriptor from another plan fail before a CUDA pointer
# can be formed. Source scalar input scales are authenticated even though the
# fixed graph derives per-token activation scales at launch.
_NATIVE_ABI = "native"
GDN_COMPONENT_CONTRACTS: tuple[
    tuple[str, int, int, tuple[int, ...], str, str, str], ...
] = (
    ("A_log", 1_297_735_680, 48, (24,), "BF16", "checkpoint", _NATIVE_ABI),
    ("conv1d.weight", 1_297_735_936, 40_960, (5_120, 1, 4), "BF16", "checkpoint", _NATIVE_ABI),
    ("dt_bias", 1_297_776_896, 48, (24,), "BF16", "checkpoint", _NATIVE_ABI),
    ("in_proj_a.weight", 1_297_777_152, 30_720, (24, 1_280), "U8", "packed_e2m1_row_major", MODEL_NVFP4_ABI),
    ("in_proj_a.weight_scale", 1_297_807_872, 20_480, (20_480,), "F8_E4M3", "cutlass_sm121_sfb", MODEL_NVFP4_ABI),
    ("in_proj_a.weight_scale_2", 1_297_828_352, 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_a.input_scale", 1_297_828_608, 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_b.weight", 1_297_828_864, 30_720, (24, 1_280), "U8", "packed_e2m1_row_major", MODEL_NVFP4_ABI),
    ("in_proj_b.weight_scale", 1_297_859_584, 20_480, (20_480,), "F8_E4M3", "cutlass_sm121_sfb", MODEL_NVFP4_ABI),
    ("in_proj_b.weight_scale_2", 1_297_880_064, 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_b.input_scale", 1_297_880_320, 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_qkv.weight", 1_297_880_576, 6_553_600, (5_120, 1_280), "U8", "packed_e2m1_row_major", MODEL_NVFP4_ABI),
    ("in_proj_qkv.weight_scale", 1_304_434_176, 819_200, (819_200,), "F8_E4M3", "cutlass_sm121_sfb", MODEL_NVFP4_ABI),
    ("in_proj_qkv.weight_scale_2", 1_305_253_376, 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_qkv.input_scale", 1_305_253_632, 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_z.weight", 1_305_253_888, 3_932_160, (3_072, 1_280), "U8", "packed_e2m1_row_major", MODEL_NVFP4_ABI),
    ("in_proj_z.weight_scale", 1_309_186_048, 491_520, (491_520,), "F8_E4M3", "cutlass_sm121_sfb", MODEL_NVFP4_ABI),
    ("in_proj_z.weight_scale_2", 1_309_677_568, 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("in_proj_z.input_scale", 1_309_677_824, 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("norm.weight", 1_309_678_080, 256, (128,), "BF16", "checkpoint", _NATIVE_ABI),
    ("out_proj.weight", 1_309_678_336, 3_932_160, (2_560, 1_536), "U8", "packed_e2m1_row_major", MODEL_NVFP4_ABI),
    ("out_proj.weight_scale", 1_313_610_496, 491_520, (491_520,), "F8_E4M3", "cutlass_sm121_sfb", MODEL_NVFP4_ABI),
    ("out_proj.weight_scale_2", 1_314_102_016, 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    ("out_proj.input_scale", 1_314_102_272, 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
)


def validate_rank0_layer0_gdn_inventory(manifest: dict[str, Any]) -> tuple[GdnComponent, ...]:
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
    slab = manifest.get("slabs", {}).get("rank0-target")
    if not isinstance(slab, dict) or not isinstance(slab.get("entries"), list):
        raise LinearAttentionSlabError("rank0 target slab inventory is missing")
    valid_entries = [entry for entry in slab["entries"] if isinstance(entry, dict)]
    entries = {entry.get("name"): entry for entry in valid_entries}
    if len(entries) != len(valid_entries):
        raise LinearAttentionSlabError("rank0 target slab contains duplicate tensor names")
    components: list[GdnComponent] = []
    for suffix, offset, length, shape, dtype, layout, abi in GDN_COMPONENT_CONTRACTS:
        name = f"{GDN_PREFIX}.{suffix}"
        entry = entries.get(name)
        if (
            not isinstance(entry, dict)
            or entry.get("offset_bytes") != offset
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


def load_rank0_layer0_gdn(artifact: Path) -> Rank0Layer0Gdn:
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
    components = validate_rank0_layer0_gdn_inventory(manifest)
    slab = manifest["slabs"]["rank0-target"]
    slab_path = artifact / str(slab.get("file", ""))
    slab_bytes = slab.get("bytes")
    if isinstance(slab_bytes, bool) or not isinstance(slab_bytes, int) or slab_bytes <= 0:
        raise LinearAttentionSlabError("rank0 target slab byte count is invalid")
    try:
        if slab_path.stat().st_size != slab_bytes:
            raise LinearAttentionSlabError("rank0 target slab byte count drift")
    except OSError as exc:
        raise LinearAttentionSlabError("rank0 target slab file is missing") from exc
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
    if len(touched) != 1:
        raise LinearAttentionSlabError("GDN tensors must occupy one authenticated chunk")
    chunk = touched[0]
    offset, length = chunk.get("offset_bytes"), chunk.get("length_bytes")
    if not isinstance(offset, int) or not isinstance(length, int):
        raise LinearAttentionSlabError("GDN chunk extent is invalid")
    digest = hashlib.sha256()
    try:
        with slab_path.open("rb") as stream:
            stream.seek(offset)
            remaining = length
            while remaining:
                data = stream.read(min(8 * 1024 * 1024, remaining))
                if not data:
                    raise LinearAttentionSlabError("short GDN chunk read")
                digest.update(data)
                remaining -= len(data)
    except OSError as exc:
        raise LinearAttentionSlabError("cannot authenticate GDN slab chunk") from exc
    if digest.hexdigest() != chunk.get("sha256"):
        raise LinearAttentionSlabError("GDN slab chunk digest mismatch")
    return Rank0Layer0Gdn(
        GDN_SCHEMA, claimed, slab_path, slab_bytes, offset, length,
        chunk["sha256"], components
    )

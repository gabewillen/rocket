# SPDX-License-Identifier: Apache-2.0
"""Authenticated TP2 slab contract for every target QSA layer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import MODEL_NVFP4_ABI, PINNED_CONTRACT, SCHEMA, canonical_bytes

QSA_WEIGHTS_SCHEMA = "qwen3.8-flash-next:tp2:rank-local:qsa-weights:v1"
QSA_INDEXER_SIDECAR_SCHEMA = "qwen3.8-flash-next:qsa-indexer-replica-sidecar:v1"
_NATIVE_ABI = "native"


class QsaWeightsError(RuntimeError):
    """The fixed QSA slab identity or tensor inventory changed."""


@dataclass(frozen=True)
class QsaWeightComponent:
    name: str
    offset: int
    length: int
    shape: tuple[int, ...]
    dtype: str
    layout: str
    abi: str
    storage: str


@dataclass(frozen=True)
class QsaAuthenticatedChunk:
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class RankQsaWeights:
    schema: str
    artifact_key: str
    rank: int
    layer: int
    slab_path: Path
    slab_bytes: int
    indexer_sidecar_key: str
    indexer_sidecar_path: Path
    chunks: tuple[QsaAuthenticatedChunk, ...]
    components: tuple[QsaWeightComponent, ...]


def _nvfp4(family: str, rows: int, columns: int):
    return (
        (f"{family}.weight", rows * columns // 2, (rows, columns // 2), "U8", "packed_e2m1_row_major", MODEL_NVFP4_ABI),
        (f"{family}.weight_scale", rows * columns // 16, (rows * columns // 16,), "F8_E4M3", "cutlass_sm121_sfb", MODEL_NVFP4_ABI),
        (f"{family}.weight_scale_2", 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
        (f"{family}.input_scale", 4, (1,), "F32", "scalar", MODEL_NVFP4_ABI),
    )


# Source checkpoint BF16 and serving NVFP4 families stay explicit. The indexer
# projection is BF16 in the slab and must not be routed through the W4A4 path.
QSA_WEIGHT_CONTRACTS = (
    *_nvfp4("q_proj", 6144, 2560),
    *_nvfp4("k_proj", 256, 2560),
    *_nvfp4("v_proj", 256, 2560),
    *_nvfp4("o_proj", 2560, 3072),
    ("q_norm.weight", 512, (256,), "BF16", "checkpoint", _NATIVE_ABI),
    ("k_norm.weight", 512, (256,), "BF16", "checkpoint", _NATIVE_ABI),
    ("indexer.q_layernorm.weight", 256, (128,), "BF16", "checkpoint", _NATIVE_ABI),
    ("indexer.k_layernorm.weight", 256, (128,), "BF16", "checkpoint", _NATIVE_ABI),
)


def validate_qsa_weight_inventory(
    manifest: dict[str, Any], rank: int, layer: int
) -> tuple[QsaWeightComponent, ...]:
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
        raise QsaWeightsError("rank slab manifest contract drift")
    if isinstance(rank, bool) or rank not in (0, 1):
        raise QsaWeightsError("QSA rank must be 0 or 1")
    if isinstance(layer, bool) or not 0 <= layer < 48 or layer % 4 != 3:
        raise QsaWeightsError("QSA layer must use the 12-layer topology")
    slab_key = f"rank{rank}-target"
    slab = manifest.get("slabs", {}).get(slab_key)
    if not isinstance(slab, dict) or not isinstance(slab.get("entries"), list):
        raise QsaWeightsError(f"{slab_key} slab inventory is missing")
    valid_entries = [entry for entry in slab["entries"] if isinstance(entry, dict)]
    entries = {entry.get("name"): entry for entry in valid_entries}
    if len(entries) != len(valid_entries):
        raise QsaWeightsError(f"{slab_key} contains duplicate tensor names")
    prefix = f"model.language_model.layers.{layer}.self_attn"
    components = []
    for suffix, length, shape, dtype, layout, abi in QSA_WEIGHT_CONTRACTS:
        name = f"{prefix}.{suffix}"
        entry = entries.get(name)
        offset = entry.get("offset_bytes") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or offset % PINNED_CONTRACT.tensor_alignment_bytes
            or entry.get("length_bytes") != length
            or tuple(entry.get("shape", ())) != shape
            or entry.get("dtype") != dtype
            or entry.get("layout") != layout
            or entry.get("abi") != abi
        ):
            raise QsaWeightsError(f"QSA tensor contract drift: {name}")
        components.append(QsaWeightComponent(name, offset, length, shape, dtype, layout, abi, "base-slab"))
    return tuple(components)


def _load_indexer_sidecar(
    sidecar: Path, base_key: str, rank: int, layer: int
) -> tuple[str, Path, QsaWeightComponent]:
    if not isinstance(sidecar, Path):
        raise QsaWeightsError("QSA indexer sidecar must be a Path")
    try:
        raw = (sidecar / "manifest.json").read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QsaWeightsError("cannot load QSA indexer sidecar") from exc
    digest_input = dict(manifest) if isinstance(manifest, dict) else {}
    claimed = digest_input.pop("artifact_key", None)
    if (
        claimed != hashlib.sha256(canonical_bytes(digest_input)).hexdigest()
        or sidecar.name != claimed
        or manifest.get("schema") != QSA_INDEXER_SIDECAR_SCHEMA
        or manifest.get("revision") != PINNED_CONTRACT.revision
        or manifest.get("base_artifact_key") != base_key
        or manifest.get("old_sharded_index_qk_proj") != "rejected"
    ):
        raise QsaWeightsError("QSA indexer sidecar identity drift")
    rank_key = f"rank{rank}-target"
    name = f"model.language_model.layers.{layer}.self_attn.indexer.index_qk_proj.weight"
    if name not in manifest.get("rank_bindings", {}).get(rank_key, []):
        raise QsaWeightsError("QSA indexer sidecar rank binding is missing")
    matches = [item for item in manifest.get("components", []) if item.get("name") == name]
    if len(matches) != 1:
        raise QsaWeightsError("QSA indexer sidecar tensor inventory drift")
    item = matches[0]
    offset, length = item.get("offset_bytes"), item.get("length_bytes")
    if (
        item.get("dtype") != "BF16"
        or item.get("shape") != [640, 2560]
        or item.get("layout") != "checkpoint"
        or item.get("abi") != "native-replicated"
        or isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or offset % PINNED_CONTRACT.tensor_alignment_bytes
        or length != 3_276_800
        or not isinstance(item.get("sha256"), str)
    ):
        raise QsaWeightsError("QSA indexer sidecar tensor contract drift")
    payload = manifest.get("payload", {})
    payload_path = sidecar / str(payload.get("file", ""))
    try:
        if payload_path.stat().st_size != payload.get("bytes"):
            raise QsaWeightsError("QSA indexer sidecar byte count drift")
        with payload_path.open("rb") as stream:
            stream.seek(offset)
            tensor = stream.read(length)
    except OSError as exc:
        raise QsaWeightsError("cannot authenticate QSA indexer sidecar") from exc
    if len(tensor) != length or hashlib.sha256(tensor).hexdigest() != item["sha256"]:
        raise QsaWeightsError("QSA indexer sidecar tensor digest mismatch")
    return claimed, payload_path, QsaWeightComponent(
        name, offset, length, (640, 2560), "BF16", "checkpoint",
        "native-replicated", "indexer-sidecar",
    )


def load_qsa_weights(
    artifact: Path, indexer_sidecar: Path, rank: int, layer: int
) -> RankQsaWeights:
    if not isinstance(artifact, Path):
        raise QsaWeightsError("rank slab artifact must be a Path")
    try:
        manifest = json.loads((artifact / "manifest.json").read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QsaWeightsError("cannot load rank slab manifest") from exc
    digest_input = dict(manifest) if isinstance(manifest, dict) else {}
    claimed = digest_input.pop("artifact_key", None)
    if claimed != hashlib.sha256(canonical_bytes(digest_input)).hexdigest() or artifact.name != claimed:
        raise QsaWeightsError("rank slab manifest content address mismatch")
    components = validate_qsa_weight_inventory(manifest, rank, layer)
    old_indexer = next(
        (
            item
            for item in manifest["slabs"][f"rank{rank}-target"]["entries"]
            if item.get("name")
            == f"model.language_model.layers.{layer}.self_attn.indexer.index_qk_proj.weight"
        ),
        None,
    )
    if not isinstance(old_indexer, dict) or old_indexer.get("shape") != [320, 2560]:
        raise QsaWeightsError("expected old sharded QSA index extent is absent")
    sidecar_key, sidecar_path, indexer = _load_indexer_sidecar(
        indexer_sidecar, claimed, rank, layer
    )
    components = (*components, indexer)
    slab_key = f"rank{rank}-target"
    slab = manifest["slabs"][slab_key]
    slab_path = artifact / str(slab.get("file", ""))
    slab_bytes = slab.get("bytes")
    if isinstance(slab_bytes, bool) or not isinstance(slab_bytes, int) or slab_bytes <= 0:
        raise QsaWeightsError(f"{slab_key} byte count is invalid")
    try:
        if slab_path.stat().st_size != slab_bytes:
            raise QsaWeightsError(f"{slab_key} byte count drift")
    except OSError as exc:
        raise QsaWeightsError(f"{slab_key} file is missing") from exc
    chunks = slab.get("chunks")
    if not isinstance(chunks, list):
        raise QsaWeightsError(f"{slab_key} chunks are missing")
    touched = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise QsaWeightsError("QSA chunk record is invalid")
        offset, length = chunk.get("offset_bytes"), chunk.get("length_bytes")
        if isinstance(offset, bool) or not isinstance(offset, int) or isinstance(length, bool) or not isinstance(length, int) or offset < 0 or length <= 0:
            raise QsaWeightsError("QSA chunk extent is invalid")
        if any(item.offset < offset + length and item.offset + item.length > offset for item in components):
            touched.append(chunk)
    if not touched:
        raise QsaWeightsError("QSA tensors occupy no authenticated chunk")
    authenticated = []
    try:
        with slab_path.open("rb") as stream:
            for chunk in touched:
                offset, length, digest = chunk["offset_bytes"], chunk["length_bytes"], chunk.get("sha256")
                if not isinstance(digest, str):
                    raise QsaWeightsError("QSA chunk digest is invalid")
                observed = hashlib.sha256()
                stream.seek(offset)
                remaining = length
                while remaining:
                    data = stream.read(min(8 * 1024 * 1024, remaining))
                    if not data:
                        raise QsaWeightsError("short QSA chunk read")
                    observed.update(data)
                    remaining -= len(data)
                if observed.hexdigest() != digest:
                    raise QsaWeightsError("QSA slab chunk digest mismatch")
                authenticated.append(QsaAuthenticatedChunk(offset, length, digest))
    except OSError as exc:
        raise QsaWeightsError("cannot authenticate QSA slab chunk") from exc
    return RankQsaWeights(
        QSA_WEIGHTS_SCHEMA, claimed, rank, layer, slab_path, slab_bytes,
        sidecar_key, sidecar_path, tuple(authenticated), components,
    )


def validate_all_qsa_bindings(
    artifact: Path, indexer_sidecar: Path
) -> tuple[RankQsaWeights, ...]:
    """Authenticate the 24 rank-local QSA layer bindings in topology order."""

    return tuple(
        load_qsa_weights(artifact, indexer_sidecar, rank, layer)
        for rank in (0, 1)
        for layer in range(3, 48, 4)
    )

# SPDX-License-Identifier: Apache-2.0
"""Authenticated compact native descriptors for all 48 target layers."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from .contract import PINNED_CONTRACT, SCHEMA, canonical_bytes
from .linear_attention import validate_gdn_inventory
from .qsa_weights import validate_qsa_weight_inventory
from .routed_moe import (
    LOCAL_EXPERTS, _ROUTER_CONTRACT, _SHARED_CONTRACT,
    _expert_contract, _validated_extent,
)

SCHEMA_ID = "rocket.qwen38.target-layer-native-descriptor.v1"
TARGET_ARTIFACT = "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
SIDECAR_ARTIFACT = "bdbebd4f45c398f090a41ab98cd3881b969d958d8ae0bc42f3411844d3262edd"
_HC_CONTRACTS = {
    "hc_norm.weight": (20_480, (10_240,)),
    "input_mix_weight_down.weight": (6_553_600, (320, 10_240)),
    "block_inject_weight.weight": (81_920, (4, 10_240)),
    "input_mix_weight_up.weight": (6_553_600, (10_240, 320)),
}
_NVFP4_ABI = "modelopt_nvfp4_group16_cutlass_sm121_sfb"
_PLE_EMBEDDING_SHARDS = 64


class TargetLayerDescriptorError(RuntimeError):
    pass


def _strides(shape: object) -> list[int]:
    values = list(shape) if isinstance(shape, (list, tuple)) else []
    stride = 1
    result = []
    for dimension in reversed(values):
        result.append(stride)
        stride *= int(dimension)
    return list(reversed(result))


def _extent(item: object, *, storage: str = "target_slab") -> dict[str, object]:
    if not isinstance(item, dict):
        raise TargetLayerDescriptorError("target layer extent is absent")
    return {
        "name": item["name"], "offset_bytes": item["offset_bytes"],
        "length_bytes": item["length_bytes"], "shape": item["shape"],
        "dtype": item["dtype"], "layout": item["layout"],
        "abi": item["abi"], "storage": storage,
        "strides": _strides(item["shape"]),
    }


@lru_cache(maxsize=1)
def _manifest(artifact: Path) -> dict[str, object]:
    try:
        value = json.loads((artifact / "manifest.json").read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetLayerDescriptorError("target slab manifest is unavailable") from exc
    authenticated = dict(value) if isinstance(value, dict) else {}
    claimed = authenticated.pop("artifact_key", None)
    if (
        claimed != TARGET_ARTIFACT or artifact.name != claimed
        or hashlib.sha256(canonical_bytes(authenticated)).hexdigest() != claimed
        or value.get("schema") != SCHEMA
        or value.get("revision") != PINNED_CONTRACT.revision
        or value.get("tp_size") != 2
        or value.get("tensor_alignment_bytes") != 256
    ):
        raise TargetLayerDescriptorError("target slab manifest identity changed")
    return value


@lru_cache(maxsize=1)
def _sidecar(sidecar: Path) -> dict[str, object]:
    try:
        value = json.loads((sidecar / "manifest.json").read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetLayerDescriptorError("QSA sidecar manifest is unavailable") from exc
    authenticated = dict(value) if isinstance(value, dict) else {}
    claimed = authenticated.pop("artifact_key", None)
    if (
        claimed != SIDECAR_ARTIFACT or sidecar.name != claimed
        or hashlib.sha256(canonical_bytes(authenticated)).hexdigest() != claimed
        or value.get("base_artifact_key") != TARGET_ARTIFACT
    ):
        raise TargetLayerDescriptorError("QSA sidecar identity changed")
    return value


def _hc(entries: Mapping[str, object], layer: int) -> list[dict[str, object]]:
    result = []
    for family in ("attn_hyper_connection", "mlp_hyper_connection"):
        for suffix, (length, shape) in _HC_CONTRACTS.items():
            name = f"model.language_model.layers.{layer}.{family}.{suffix}"
            item = entries.get(name)
            if (
                not isinstance(item, dict) or item.get("length_bytes") != length
                or tuple(item.get("shape", ())) != shape
                or item.get("dtype") != "BF16" or item.get("layout") != "checkpoint"
                or item.get("abi") != "native"
                or not isinstance(item.get("offset_bytes"), int)
                or item["offset_bytes"] % 256
            ):
                raise TargetLayerDescriptorError(f"HC extent changed: {name}")
            result.append(_extent(item))
    return result


def _moe(entries: Mapping[str, object], rank: int,
         layer: int) -> list[dict[str, object]]:
    prefix = f"model.language_model.layers.{layer}.mlp"
    selected = [
        _validated_extent(entries, prefix, suffix, contract)
        for suffix, contract in _ROUTER_CONTRACT.items()
    ]
    first = rank * LOCAL_EXPERTS
    for expert in range(first, first + LOCAL_EXPERTS):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for leaf in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                suffix = f"experts.{expert}.{projection}.{leaf}"
                selected.append(_validated_extent(
                    entries, prefix, suffix, _expert_contract(projection, leaf)))
    for suffix, (length, shape) in _SHARED_CONTRACT.items():
        selected.append(_validated_extent(
            entries, prefix, suffix,
            (length, shape, "BF16", "checkpoint", "native")))
    return [{
        "name": item.name, "offset_bytes": item.offset,
        "length_bytes": item.length, "shape": list(item.shape),
        "dtype": item.dtype, "layout": item.layout, "abi": item.abi,
        "storage": "target_slab", "strides": _strides(item.shape),
    } for item in selected]


def _ple(entries: Mapping[str, object], rank: int,
         layer: int) -> list[dict[str, object]]:
    """Return the complete rank-local layer-2 PLE inventory."""

    if layer != 1:
        return []
    prefix = "model.language_model.layers.1.ple."
    contracts: list[tuple[str, int, tuple[int, ...], str, str, str]] = [
        ("conv1d.weight", 81_920, (10_240, 1, 4), "BF16", "checkpoint", "native"),
        ("key_proj.weight", 13_107_200, (10_240, 1_280), "U8", "packed_e2m1_row_major", _NVFP4_ABI),
        ("key_proj.weight_scale", 1_638_400, (1_638_400,), "F8_E4M3", "cutlass_sm121_sfb", _NVFP4_ABI),
        ("key_proj.weight_scale_2", 4, (1,), "F32", "scalar", _NVFP4_ABI),
        ("key_proj.input_scale", 4, (1,), "F32", "scalar", _NVFP4_ABI),
        ("value_proj.weight", 3_276_800, (2_560, 1_280), "U8", "packed_e2m1_row_major", _NVFP4_ABI),
        ("value_proj.weight_scale", 409_600, (409_600,), "F8_E4M3", "cutlass_sm121_sfb", _NVFP4_ABI),
        ("value_proj.weight_scale_2", 4, (1,), "F32", "scalar", _NVFP4_ABI),
        ("value_proj.input_scale", 4, (1,), "F32", "scalar", _NVFP4_ABI),
        ("norm_key.weight", 20_480, (10_240,), "BF16", "checkpoint", "native"),
        ("norm_query.weight", 20_480, (10_240,), "BF16", "checkpoint", "native"),
        ("norm_conv.weight", 20_480, (10_240,), "BF16", "checkpoint", "native"),
        ("ple_embedding.layer_multipliers", 24, (3,), "I64", "checkpoint", "native"),
        ("ple_embedding.ngram_heads_offsets", 128, (16,), "I64", "checkpoint", "native"),
        ("ple_embedding.ngram_heads_vocab_sizes", 128, (16,), "I64", "checkpoint", "native"),
        ("ple_embedding.ngram_embedding.weight_scale", 2, (1,), "BF16", "checkpoint", "native"),
    ]
    contracts.extend(
        (f"ple_embedding.ngram_embedding.shard_{index}.weight", 400_001_920,
         (2_500_012, 160), "F8_E4M3", "checkpoint", "native")
        for index in range(rank * _PLE_EMBEDDING_SHARDS,
                           (rank + 1) * _PLE_EMBEDDING_SHARDS)
    )
    result = []
    for suffix, length, shape, dtype, layout, abi in contracts:
        name = prefix + suffix
        item = entries.get(name)
        if (
            not isinstance(item, dict) or item.get("length_bytes") != length
            or tuple(item.get("shape", ())) != shape or item.get("dtype") != dtype
            or item.get("layout") != layout or item.get("abi") != abi
            or not isinstance(item.get("offset_bytes"), int)
            or item["offset_bytes"] % 256
        ):
            raise TargetLayerDescriptorError(f"PLE extent changed: {name}")
        result.append(_extent(item))
    return result


def target_layer_descriptor(artifact: Path, sidecar: Path, rank: int,
                            layer: int) -> dict[str, object]:
    """Build one pointer-free descriptor from authenticated publication metadata."""

    if rank not in (0, 1) or isinstance(layer, bool) or not 0 <= layer < 48:
        raise TargetLayerDescriptorError("TP2 rank and target layer 0..47 are required")
    manifest = _manifest(artifact)
    slab = manifest.get("slabs", {}).get(f"rank{rank}-target")
    if not isinstance(slab, dict) or not isinstance(slab.get("entries"), list):
        raise TargetLayerDescriptorError("target slab inventory is absent")
    entries = {item.get("name"): item for item in slab["entries"] if isinstance(item, dict)}
    if len(entries) != len(slab["entries"]):
        raise TargetLayerDescriptorError("target slab entries are invalid or duplicate")
    extents = _hc(entries, layer)
    kind = "qsa" if layer % 4 == 3 else "gdn"
    scalar_bits: dict[str, str] = {}
    slab_path = artifact / str(slab.get("file", ""))
    if kind == "qsa":
        components = validate_qsa_weight_inventory(manifest, rank, layer)
        extents.extend(_extent(entries[item.name]) for item in components)
        side_manifest = _sidecar(sidecar)
        name = f"model.language_model.layers.{layer}.self_attn.indexer.index_qk_proj.weight"
        matches = [item for item in side_manifest.get("components", ())
                   if isinstance(item, dict) and item.get("name") == name]
        if len(matches) != 1 or matches[0].get("length_bytes") != 3_276_800:
            raise TargetLayerDescriptorError("QSA sidecar extent changed")
        extents.append(_extent(matches[0], storage="indexer_sidecar"))
        fd = os.open(slab_path, os.O_RDONLY)
        try:
            for family in ("q", "k", "v", "o"):
                scalar = entries[f"model.language_model.layers.{layer}.self_attn.{family}_proj.weight_scale_2"]
                raw = os.pread(fd, 4, scalar["offset_bytes"])
                if len(raw) != 4:
                    raise TargetLayerDescriptorError("QSA scalar read was short")
                scalar_bits[family] = raw.hex()
        finally:
            os.close(fd)
    else:
        extents.extend(_extent(entries[item.name])
                       for item in validate_gdn_inventory(manifest, rank, layer))
        fd = os.open(slab_path, os.O_RDONLY)
        try:
            for family, suffix in (
                ("qkv", "in_proj_qkv"), ("z", "in_proj_z"),
                ("b", "in_proj_b"), ("a", "in_proj_a"),
                ("out", "out_proj"),
            ):
                scalar = entries[
                    f"model.language_model.layers.{layer}.linear_attn.{suffix}.weight_scale_2"]
                raw = os.pread(fd, 4, scalar["offset_bytes"])
                if len(raw) != 4:
                    raise TargetLayerDescriptorError("GDN scalar read was short")
                scalar_bits[family] = raw.hex()
        finally:
            os.close(fd)
    extents.extend(_ple(entries, rank, layer))
    moe_extents = _moe(entries, rank, layer)
    extents.extend(moe_extents)
    extents.sort(key=lambda item: str(item["name"]))
    if len({item["name"] for item in extents}) != len(extents):
        raise TargetLayerDescriptorError("target layer extent names overlap")
    publication = {key: slab[key] for key in ("file", "bytes", "chunks")}
    descriptor: dict[str, object] = {
        "schema": SCHEMA_ID, "rank": rank, "peer_rank": 1 - rank,
        "layer": layer, "attention_kind": kind,
        "artifact_key": TARGET_ARTIFACT,
        "slab_key": f"rank{rank}-target", "slab_bytes": slab["bytes"],
        "slab_publication_layout_sha256": hashlib.sha256(
            canonical_bytes(publication)).hexdigest(),
        "indexer_sidecar_key": SIDECAR_ARTIFACT if kind == "qsa" else "",
        "attention_projection_global_le_hex": scalar_bits,
        "moe_layout_sha256": hashlib.sha256(canonical_bytes([
            {
                "name": item["name"], "offset": item["offset_bytes"],
                "length": item["length_bytes"], "shape": item["shape"],
                "dtype": item["dtype"], "layout": item["layout"],
                "abi": item["abi"],
            }
            for item in moe_extents
        ])).hexdigest(),
        "extents": extents,
    }
    descriptor["native_binding_inventory_sha256"] = hashlib.sha256(
        canonical_bytes(extents)).hexdigest()
    descriptor["descriptor_sha256"] = hashlib.sha256(
        canonical_bytes(descriptor)).hexdigest()
    return descriptor


def descriptor_identity(descriptor: Mapping[str, object]) -> dict[str, object]:
    return {key: descriptor[key] for key in (
        "rank", "layer", "attention_kind", "descriptor_sha256",
        "native_binding_inventory_sha256", "slab_publication_layout_sha256",
        "moe_layout_sha256",
    )}


def load_descriptor_allowlist(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetLayerDescriptorError("target layer allowlist is unavailable") from exc
    if not isinstance(value, list) or len(value) != 96:
        raise TargetLayerDescriptorError("target layer allowlist cardinality changed")
    result: list[Mapping[str, object]] = []
    identities: set[tuple[int, int]] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "rank", "layer", "attention_kind", "descriptor_sha256",
            "native_binding_inventory_sha256", "slab_publication_layout_sha256",
            "moe_layout_sha256",
        }:
            raise TargetLayerDescriptorError("target layer allowlist shape changed")
        rank, layer = item["rank"], item["layer"]
        expected_kind = "qsa" if isinstance(layer, int) and layer % 4 == 3 else "gdn"
        digests = (item["descriptor_sha256"],
                   item["native_binding_inventory_sha256"],
                   item["slab_publication_layout_sha256"],
                   item["moe_layout_sha256"])
        if (
            rank not in (0, 1) or isinstance(layer, bool)
            or not isinstance(layer, int) or not 0 <= layer < 48
            or item["attention_kind"] != expected_kind
            or any(not isinstance(value, str) or len(value) != 64
                   or set(value) - set("0123456789abcdef") for value in digests)
            or (rank, layer) in identities
        ):
            raise TargetLayerDescriptorError("target layer allowlist identity changed")
        identities.add((rank, layer))
        result.append(item)
    return tuple(result)


def authenticate_descriptor_identity(
    identity: Mapping[str, object], allowlist: tuple[Mapping[str, object], ...]
) -> bool:
    """Allocation-free semantic check after the allowlist has been loaded."""

    return any(dict(expected) == dict(identity) for expected in allowlist)

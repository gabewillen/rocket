# SPDX-License-Identifier: Apache-2.0
"""Authenticated CPU preflight for the two-rank layer-3 physical factory."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .contract import PINNED_CONTRACT, SCHEMA, canonical_bytes
from .qsa_weights import RankQsaWeights, load_qsa_weights
from .routed_moe import OwnerLocalMoeSlab, load_owner_local_moe

ORACLE_MANIFEST_SHA256 = (
    "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b"
)
LAYER02_SHA256 = "6503eeeb3c70c9c1c163aca08dcdb4e1df7ca997699f2f20946265a72069d2fe"
LAYER03_SHA256 = "aa2d2a1454f0654ea2082d12e7e3284cad9304a7ede124303f99c52bbf9cddbe"
TARGET_ARTIFACT = "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
INDEXER_SIDECAR = "bdbebd4f45c398f090a41ab98cd3881b969d958d8ae0bc42f3411844d3262edd"
PLAN_SCHEMA = "rocket.qwen38.layer3-physical-plan.v1"
ROWS = 35
HC_WIDTH = 10_240
HIDDEN = 2_560
LAYER = 3


class Layer3FactoryError(RuntimeError):
    """A physical layer-3 launch plan could not be authenticated."""


def _failure_class(exc: Exception) -> str:
    if isinstance(exc, OSError):
        return "io"
    if isinstance(exc, (Layer3FactoryError, ValueError, TypeError)):
        return "contract"
    return "dependency"


@dataclass(frozen=True)
class Layer3RankPlan:
    rank: int
    qsa: RankQsaWeights
    moe: OwnerLocalMoeSlab
    hyperconnection_layout_sha256: str


@dataclass(frozen=True)
class Layer3PhysicalPlan:
    schema: str
    oracle_manifest_sha256: str
    layer02_path: Path
    layer03_path: Path
    token_ids: tuple[int, ...]
    replay_rows: tuple[int, ...]
    compare_row: int
    replicated_hc_shape: tuple[int, int]
    ranks: tuple[Layer3RankPlan, Layer3RankPlan]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _oracle(capture: Path) -> tuple[Path, Path, tuple[int, ...]]:
    manifest_path = capture / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Layer3FactoryError("layer-3 oracle manifest is unavailable") from exc
    artifacts = {
        item.get("name"): item
        for item in manifest.get("artifacts", ()) if isinstance(item, dict)
    }
    before, after = artifacts.get("layer.02"), artifacts.get("layer.03")
    tokens = manifest.get("input_token_ids")
    if (
        _sha256(manifest_path) != ORACLE_MANIFEST_SHA256
        or manifest.get("schema") != "rocket.qwen38.k0-target-oracle.v1"
        or manifest.get("valid") is not True
        or manifest.get("complete") is not True
        or len(artifacts) != 51
        or manifest.get("greedy_token_id") != 248_046
        or not isinstance(tokens, list) or len(tokens) != ROWS
        or tokens[-1] != 13
        or not isinstance(before, dict) or not isinstance(after, dict)
    ):
        raise Layer3FactoryError("layer-3 oracle identity changed")
    paths = []
    for item, digest in ((before, LAYER02_SHA256), (after, LAYER03_SHA256)):
        path = capture / str(item.get("file", ""))
        if (
            item.get("dtype") != "bfloat16"
            or item.get("shape") != [ROWS, HC_WIDTH]
            or item.get("strides") != [HC_WIDTH, 1]
            or item.get("bytes") != ROWS * HC_WIDTH * 2
            or item.get("sha256") != digest
            or not path.is_file() or path.stat().st_size != item["bytes"]
            or _sha256(path) != digest
        ):
            raise Layer3FactoryError("layer-3 oracle boundary changed")
        paths.append(path.resolve())
    return paths[0], paths[1], tuple(tokens)


def _hyperconnection_layout(artifact: Path, rank: int) -> str:
    try:
        manifest = json.loads((artifact / "manifest.json").read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Layer3FactoryError("layer-3 HC manifest is unavailable") from exc
    digest_input = dict(manifest) if isinstance(manifest, dict) else {}
    claimed = digest_input.pop("artifact_key", None)
    slab = manifest.get("slabs", {}).get(f"rank{rank}-target", {})
    entries = {
        item.get("name"): item
        for item in slab.get("entries", ()) if isinstance(item, dict)
    }
    if (
        claimed != TARGET_ARTIFACT or artifact.name != claimed
        or hashlib.sha256(canonical_bytes(digest_input)).hexdigest() != claimed
        or manifest.get("schema") != SCHEMA
        or manifest.get("revision") != PINNED_CONTRACT.revision
        or not isinstance(slab, dict) or len(entries) != len(slab.get("entries", ()))
    ):
        raise Layer3FactoryError("layer-3 HC slab identity changed")
    contracts = {
        "hc_norm.weight": ([HC_WIDTH], HC_WIDTH * 2),
        "input_mix_weight_down.weight": ([320, HC_WIDTH], 320 * HC_WIDTH * 2),
        "block_inject_weight.weight": ([4, HC_WIDTH], 4 * HC_WIDTH * 2),
        "input_mix_weight_up.weight": ([HC_WIDTH, 320], HC_WIDTH * 320 * 2),
    }
    selected = []
    for boundary in ("attn_hyper_connection", "mlp_hyper_connection"):
        for suffix, (shape, length) in contracts.items():
            name = f"model.language_model.layers.3.{boundary}.{suffix}"
            item = entries.get(name)
            if (
                not isinstance(item, dict) or item.get("shape") != shape
                or item.get("dtype") != "BF16" or item.get("layout") != "checkpoint"
                or item.get("length_bytes") != length
                or not isinstance(item.get("offset_bytes"), int)
                or item["offset_bytes"] % 256
            ):
                raise Layer3FactoryError("layer-3 HC extent changed")
            selected.append(item)
    chunks = []
    for chunk in slab.get("chunks", ()):
        if not isinstance(chunk, dict):
            raise Layer3FactoryError("layer-3 HC chunk table changed")
        start, length = chunk.get("offset_bytes"), chunk.get("length_bytes")
        if not isinstance(start, int) or not isinstance(length, int):
            raise Layer3FactoryError("layer-3 HC chunk extent changed")
        if any(
            item["offset_bytes"] < start + length
            and item["offset_bytes"] + item["length_bytes"] > start
            for item in selected
        ):
            chunks.append(chunk)
    slab_path = artifact / str(slab.get("file", ""))
    if not chunks or not slab_path.is_file() or slab_path.stat().st_size != slab.get("bytes"):
        raise Layer3FactoryError("layer-3 HC slab bytes changed")
    fd = os.open(slab_path, os.O_RDONLY)
    try:
        for chunk in chunks:
            payload = os.pread(fd, chunk["length_bytes"], chunk["offset_bytes"])
            if (
                len(payload) != chunk["length_bytes"]
                or hashlib.sha256(payload).hexdigest() != chunk.get("sha256")
            ):
                raise Layer3FactoryError("layer-3 HC chunk digest changed")
    finally:
        os.close(fd)
    return hashlib.sha256(canonical_bytes([
        {key: item[key] for key in ("name", "offset_bytes", "length_bytes",
                                    "shape", "dtype", "layout")}
        for item in selected
    ])).hexdigest()


def prepare_layer3_physical_plan(
    *, artifact: Path, indexer_sidecar: Path, oracle_capture: Path,
    tracer: object,
) -> Layer3PhysicalPlan:
    if tracer is None or not callable(getattr(tracer, "start_as_current_span", None)):
        raise Layer3FactoryError("layer-3 factory requires OpenTelemetry")
    with tracer.start_as_current_span("rocket.qwen38.layer3_factory.prepare") as span:
        span.set_attribute("phase", "prepare")
        try:
            before, after, tokens = _oracle(oracle_capture)
            if artifact.name != TARGET_ARTIFACT or indexer_sidecar.name != INDEXER_SIDECAR:
                raise Layer3FactoryError("layer-3 slab or sidecar key changed")
            ranks = tuple(
                Layer3RankPlan(
                    rank,
                    load_qsa_weights(artifact, indexer_sidecar, rank, LAYER),
                    load_owner_local_moe(artifact, rank, LAYER),
                    _hyperconnection_layout(artifact, rank),
                )
                for rank in (0, 1)
            )
        except Exception as exc:
            span.set_attribute("outcome", "failure")
            span.set_attribute("failure.class", _failure_class(exc))
            span.record_exception(exc)
            raise
        span.set_attribute("outcome", "success")
        span.set_attribute("failure.class", "none")
    return Layer3PhysicalPlan(
        PLAN_SCHEMA, ORACLE_MANIFEST_SHA256, before, after, tokens,
        tuple(range(ROWS)), ROWS - 1, (ROWS, HC_WIDTH), ranks,
    )


def public_plan(plan: Layer3PhysicalPlan) -> Mapping[str, object]:
    return MappingProxyType({
        "schema": plan.schema,
        "valid": True,
        "complete": True,
        "oracle_manifest_sha256": plan.oracle_manifest_sha256,
        "rows": len(plan.replay_rows),
        "replay": "sequential_rows_0_34",
        "compare_row": plan.compare_row,
        "replicated_hc_shape": list(plan.replicated_hc_shape),
        "ranks": [item.rank for item in plan.ranks],
        "layer": LAYER,
        "generation": 0,
        "next": "materialize_cuda_graph",
    })


__all__ = ["Layer3FactoryError", "Layer3PhysicalPlan", "Layer3RankPlan",
           "prepare_layer3_physical_plan", "public_plan"]

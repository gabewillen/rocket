# SPDX-License-Identifier: Apache-2.0
"""Authenticated CPU preflight for the two-rank layer-3 physical factory."""

from __future__ import annotations

import hashlib
import ctypes
import ipaddress
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

from .contract import PINNED_CONTRACT, SCHEMA, canonical_bytes
from .cuda_slab_loader import (
    LoadedRankSlabs, NativeTargetSlabFinalizer, accepted_native_handoff,
)
from .qsa_weights import RankQsaWeights, load_qsa_weights
from .routed_moe import OwnerLocalMoeSlab, load_owner_local_moe
from .layer3_runtime import ARENA_SPECS

ORACLE_MANIFEST_SHA256 = (
    "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b"
)
LAYER02_SHA256 = "6503eeeb3c70c9c1c163aca08dcdb4e1df7ca997699f2f20946265a72069d2fe"
LAYER03_SHA256 = "aa2d2a1454f0654ea2082d12e7e3284cad9304a7ede124303f99c52bbf9cddbe"
TARGET_ARTIFACT = "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
INDEXER_SIDECAR = "bdbebd4f45c398f090a41ab98cd3881b969d958d8ae0bc42f3411844d3262edd"
PLAN_SCHEMA = "rocket.qwen38.layer3-physical-plan.v1"
NATIVE_PLAN_SCHEMA = "rocket.qwen38.layer3-native-plan.v1"
TARGET_LAYER_NATIVE_DESCRIPTOR_SCHEMA = (
    "rocket.qwen38.target-layer-native-descriptor.v1"
)
TARGET_MANIFEST_SHA256 = (
    "a44a450d9c0b6fe3df904ad1a78ecee959f28d9f055301195181e986bdc7028b"
)


def _accepted_native_descriptor_schema(value: object) -> bool:
    return value in (NATIVE_PLAN_SCHEMA, TARGET_LAYER_NATIVE_DESCRIPTOR_SCHEMA)
ROWS = 35
HC_WIDTH = 10_240
HIDDEN = 2_560
LAYER = 3

# The accepted loader owns CUDA allocations through Torch tensors. Native
# consumers may retain raw aliases after handoff, including after an
# unprovable CUDA fence, so these owners are intentionally never released
# before process exit. Registration is append-only and rejects replacement.
_PROCESS_LIFETIME_SLAB_OWNERS: dict[int, tuple[LoadedRankSlabs, str]] = {}


class Layer3FactoryError(RuntimeError):
    """A physical layer-3 launch plan could not be authenticated."""


class NativeTargetSlabFinalizeError(Layer3FactoryError):
    """Bounded native accepted-loader finalization failure."""

    def __init__(self, stage: str):
        self.stage = stage
        super().__init__("native target slab finalization rejected")


def _native_finalize_stage(result: int) -> str:
    return {
        1: "native_contract", 2: "native_replacement",
        31: "cuda_device", 32: "cuda_pointer",
        33: "cuda_allocation_range", 34: "cuda_ready_event",
        35: "publication_receipt",
        351: "receipt_header", 352: "publication_pointer_event",
        353: "publication_bytes", 354: "publication_rank_device",
        355: "slab_key", 356: "artifact_manifest",
        357: "layout_identity", 358: "chunks_authenticated",
        359: "peak_pinned_bytes", 360: "open_duration",
        361: "probe_memory_type", 362: "probe_device",
        363: "allocation_base", 364: "allocation_extent",
        365: "receipt_chunk", 366: "receipt_bytes",
    }.get(result, "native_status")


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
    extents: tuple["Layer3ExtentPlan", ...]


@dataclass(frozen=True)
class Layer3SourceChunk:
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class Layer3ExtentPlan:
    name: str
    offset: int
    length: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    layout: str
    abi: str
    storage: str
    source_chunks: tuple[Layer3SourceChunk, ...]


@dataclass(frozen=True)
class Layer3PairReducePlan:
    rank: int
    peer_rank: int
    bootstrap_host: str
    bootstrap_port: int
    timeout_ms: int
    session_sha256: str
    rails: tuple[str, str]
    gid_index: int
    calls: int


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
    pair_reduce: tuple[Layer3PairReducePlan, Layer3PairReducePlan]


@dataclass(frozen=True)
class NativeTargetSlabHandoff:
    """Owner-retaining, zero-copy startup handoff to the native factory.

    ``owner`` is also pinned in the module's append-only process-lifetime
    registry, keeping the Torch tensor and readiness event alive for every
    native borrower and quarantine. Native execution and cleanup receive only
    the native lease handle and make no Python or Torch call after handoff.
    The Python registry is released only as part of process teardown.
    """

    owner: LoadedRankSlabs
    native_lease: object
    device_base: int
    ready_event: int
    bytes: int
    device: int
    rank: int
    artifact_key: str
    slab_key: str
    manifest_sha256: str
    layout_sha256: str
    open_to_publish_ns: int
    chunks_authenticated: int
    peak_host_pinned_bytes: int
    receipt_sha256: str


class CtypesNativeTargetSlabLeaseFactory:
    """Native finalizer used only by ``CudaRankSlabLoader._load_locked``."""

    def __init__(self, library: Path):
        class _ChunkReceipt(ctypes.Structure):
            _fields_ = (
                ("index", ctypes.c_uint64), ("bytes", ctypes.c_uint64),
                ("direct_read_ns", ctypes.c_uint64),
                ("sha256_ns", ctypes.c_uint64),
                ("h2d_fence_ns", ctypes.c_uint64),
            )
        self._chunk_receipt = _ChunkReceipt
        self._native = ctypes.CDLL(str(library))
        self._retain = self._native.qwen38_target_slab_retain_accepted_loader
        self._retain.argtypes = (
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_uint64, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(_ChunkReceipt), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self._retain.restype = ctypes.c_int

    def retain_accepted_loader(self, **publication: object) -> object:
        receipt = publication["receipt"]
        chunks = tuple(receipt.chunks)
        chunk_array = (self._chunk_receipt * len(chunks))(*(
            self._chunk_receipt(
                chunk.index, chunk.bytes, chunk.direct_read_ns,
                chunk.sha256_ns, chunk.h2d_fence_ns,
            ) for chunk in chunks
        ))
        handle = ctypes.c_void_p()
        result = self._retain(
            int(publication["device_base"]), int(publication["ready_event"]),
            int(publication["bytes"]), int(publication["device"]),
            int(publication["rank"]),
            str(publication["slab_key"]).encode("ascii"),
            str(publication["layout_sha256"]).encode("ascii"),
            str(publication["receipt_sha256"]).encode("ascii"),
            int(publication["open_to_publish_ns"]),
            int(publication["chunks_authenticated"]),
            int(publication["peak_host_pinned_bytes"]),
            int(receipt.started_ns), int(receipt.completed_ns),
            chunk_array, len(chunks), ctypes.byref(handle),
        )
        if result != 0 or not handle.value:
            raise NativeTargetSlabFinalizeError(_native_finalize_stage(result))
        return handle


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


def _strides(shape: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    result = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return tuple(reversed(result))


def _extent(entry: Mapping[str, object], chunks: Sequence[Mapping[str, object]],
            *, storage: str) -> Layer3ExtentPlan:
    offset = int(entry["offset_bytes"])
    length = int(entry["length_bytes"])
    sources = tuple(
        Layer3SourceChunk(int(chunk["offset_bytes"]),
                          int(chunk["length_bytes"]), str(chunk["sha256"]))
        for chunk in chunks
        if offset < int(chunk["offset_bytes"]) + int(chunk["length_bytes"])
        and offset + length > int(chunk["offset_bytes"])
    )
    if not sources:
        raise Layer3FactoryError("layer-3 extent has no authenticated source chunk")
    shape = tuple(int(value) for value in entry["shape"])
    return Layer3ExtentPlan(
        str(entry["name"]), offset, length, shape, _strides(shape),
        str(entry["dtype"]), str(entry["layout"]), str(entry["abi"]),
        storage, sources,
    )


def _hyperconnection_layout(
    artifact: Path, rank: int,
) -> tuple[str, tuple[Layer3ExtentPlan, ...]]:
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
    layout = hashlib.sha256(canonical_bytes([
        {key: item[key] for key in ("name", "offset_bytes", "length_bytes",
                                    "shape", "dtype", "layout")}
        for item in selected
    ])).hexdigest()
    return layout, tuple(_extent(item, chunks, storage="target_slab")
                         for item in selected)


def _rank_extents(rank: int, qsa: RankQsaWeights,
                  moe: OwnerLocalMoeSlab,
                  hc: tuple[Layer3ExtentPlan, ...]) -> tuple[Layer3ExtentPlan, ...]:
    manifest = json.loads((qsa.slab_path.parent / "manifest.json").read_bytes())
    slab = manifest["slabs"][f"rank{rank}-target"]
    entries = {item["name"]: item for item in slab["entries"]}
    chunks = slab["chunks"]
    sidecar_manifest = json.loads(
        (qsa.indexer_sidecar_path.parent / "manifest.json").read_bytes()
    )
    sidecar_entries = {
        item["name"]: item for item in sidecar_manifest["components"]
    }
    result = list(hc)
    for component in qsa.components:
        if component.storage == "indexer-sidecar":
            source = sidecar_entries.get(component.name)
            if not isinstance(source, dict) or source.get("sha256") is None:
                raise Layer3FactoryError("layer-3 sidecar source identity changed")
            result.append(Layer3ExtentPlan(
                component.name, component.offset, component.length,
                component.shape, _strides(component.shape), component.dtype,
                component.layout, component.abi, component.storage,
                (Layer3SourceChunk(component.offset, component.length,
                 str(source["sha256"])),),
            ))
        else:
            result.append(_extent(entries[component.name], chunks,
                                  storage="target_slab"))
    for component in (*moe.router, *moe.routed, *moe.shared):
        result.append(_extent(entries[component.name], chunks,
                              storage="target_slab"))
    names = [item.name for item in result]
    if len(names) != len(set(names)):
        raise Layer3FactoryError("layer-3 native extent inventory overlaps by name")
    return tuple(sorted(result, key=lambda item: item.name))


def prepare_layer3_physical_plan(
    *, artifact: Path, indexer_sidecar: Path, oracle_capture: Path,
    tracer: object, bootstrap_host: str = "192.168.100.10",
    bootstrap_port: int = 18839, timeout_ms: int = 120_000,
) -> Layer3PhysicalPlan:
    if tracer is None or not callable(getattr(tracer, "start_as_current_span", None)):
        raise Layer3FactoryError("layer-3 factory requires OpenTelemetry")
    with tracer.start_as_current_span("rocket.qwen38.layer3_factory.prepare") as span:
        span.set_attribute("phase", "prepare")
        try:
            try:
                ipaddress.IPv4Address(bootstrap_host)
            except (ipaddress.AddressValueError, TypeError) as exc:
                raise Layer3FactoryError("layer-3 bootstrap IPv4 changed") from exc
            if (
                isinstance(bootstrap_port, bool)
                or not isinstance(bootstrap_port, int)
                or not 1 <= bootstrap_port <= 65_535
                or isinstance(timeout_ms, bool)
                or not isinstance(timeout_ms, int)
                or not 100 <= timeout_ms <= 120_000
            ):
                raise Layer3FactoryError("layer-3 PairReduce bounds changed")
            before, after, tokens = _oracle(oracle_capture)
            if artifact.name != TARGET_ARTIFACT or indexer_sidecar.name != INDEXER_SIDECAR:
                raise Layer3FactoryError("layer-3 slab or sidecar key changed")
            rank_items = []
            for rank in (0, 1):
                qsa = load_qsa_weights(artifact, indexer_sidecar, rank, LAYER)
                moe = load_owner_local_moe(artifact, rank, LAYER)
                hc_layout, hc = _hyperconnection_layout(artifact, rank)
                rank_items.append(Layer3RankPlan(
                    rank, qsa, moe, hc_layout,
                    _rank_extents(rank, qsa, moe, hc),
                ))
            ranks = tuple(rank_items)
        except Exception as exc:
            span.set_attribute("outcome", "failure")
            span.set_attribute("failure.class", _failure_class(exc))
            span.record_exception(exc)
            raise
        span.set_attribute("outcome", "success")
        span.set_attribute("failure.class", "none")
    pair_reduce = tuple(
        Layer3PairReducePlan(
            rank=rank, peer_rank=1 - rank, bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port, timeout_ms=timeout_ms,
            session_sha256=ORACLE_MANIFEST_SHA256,
            rails=("rocep1s0f1", "roceP2p1s0f1"), gid_index=3,
            calls=2 * ROWS,
        )
        for rank in (0, 1)
    )
    return Layer3PhysicalPlan(
        PLAN_SCHEMA, ORACLE_MANIFEST_SHA256, before, after, tokens,
        tuple(range(ROWS)), ROWS - 1, (ROWS, HC_WIDTH), ranks, pair_reduce,
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
        "pair_reduce": {
            "bootstrap_host": plan.pair_reduce[0].bootstrap_host,
            "bootstrap_port": plan.pair_reduce[0].bootstrap_port,
            "timeout_ms": plan.pair_reduce[0].timeout_ms,
            "session_sha256": plan.pair_reduce[0].session_sha256,
            "rails": list(plan.pair_reduce[0].rails),
            "gid_index": plan.pair_reduce[0].gid_index,
            "calls_per_rank": plan.pair_reduce[0].calls,
            "schedule": "35x(attention_m1,moe_m1)",
        },
        "next": "materialize_cuda_graph",
    })


_DTYPE_BYTES = MappingProxyType({
    "uint8": 1, "int32": 4, "int64": 8, "float32": 4,
    "bfloat16": 2,
})


def _buffer(name: str, shape: Sequence[int], dtype: str) -> dict[str, object]:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return {
        "name": name, "shape": list(shape), "strides": list(_strides(shape)),
        "dtype": dtype, "bytes": elements * _DTYPE_BYTES[dtype],
    }


def _qsa_projection_globals(item: Layer3RankPlan) -> dict[str, object]:
    """Authenticate and carry the four host scalar inputs to native init."""

    by_name = {extent.name: extent for extent in item.extents}
    prefix = f"model.language_model.layers.{LAYER}.self_attn."
    result = {}
    cached_chunks: dict[tuple[int, int, str], bytes] = {}
    fd = os.open(item.qsa.slab_path, os.O_RDONLY)
    try:
        for family in ("q", "k", "v", "o"):
            name = f"{prefix}{family}_proj.weight_scale_2"
            extent = by_name.get(name)
            if (
                extent is None or extent.storage != "target_slab"
                or extent.dtype != "F32" or extent.shape != (1,)
                or extent.strides != (1,) or extent.length != 4
                or len(extent.source_chunks) != 1
            ):
                raise Layer3FactoryError(
                    f"layer-3 {family} projection scalar extent changed"
                )
            chunk = extent.source_chunks[0]
            key = (chunk.offset, chunk.length, chunk.sha256)
            payload = cached_chunks.get(key)
            if payload is None:
                payload = os.pread(fd, chunk.length, chunk.offset)
                if (
                    len(payload) != chunk.length
                    or hashlib.sha256(payload).hexdigest() != chunk.sha256
                ):
                    raise Layer3FactoryError(
                        f"layer-3 {family} projection scalar source changed"
                    )
                cached_chunks[key] = payload
            relative = extent.offset - chunk.offset
            raw = payload[relative:relative + extent.length]
            if len(raw) != 4:
                raise Layer3FactoryError(
                    f"layer-3 {family} projection scalar read was short"
                )
            value = struct.unpack("<f", raw)[0]
            if not math.isfinite(value) or value <= 0.0:
                raise Layer3FactoryError(
                    f"layer-3 {family} projection scalar changed"
                )
            result[family] = {
                "extent_name": name,
                "dtype": "F32",
                "value_le_hex": raw.hex(),
                "source_chunk_sha256": chunk.sha256,
            }
    finally:
        os.close(fd)
    return result


def native_rank_descriptor(plan: Layer3PhysicalPlan, rank: int) -> Mapping[str, object]:
    """Canonical native handoff. It contains no device pointers."""

    if rank not in (0, 1) or plan.schema != PLAN_SCHEMA:
        raise Layer3FactoryError("layer-3 native descriptor rank changed")
    item = plan.ranks[rank]
    pair = plan.pair_reduce[rank]
    manifest = json.loads((item.qsa.slab_path.parent / "manifest.json").read_bytes())
    slab_manifest = manifest["slabs"][f"rank{rank}-target"]
    slab_publication_layout = hashlib.sha256(canonical_bytes({
        key: slab_manifest[key] for key in ("file", "bytes", "chunks")
    })).hexdigest()
    extents = [
        {
            "name": extent.name, "offset_bytes": extent.offset,
            "length_bytes": extent.length, "shape": list(extent.shape),
            "strides": list(extent.strides), "dtype": extent.dtype,
            "layout": extent.layout, "abi": extent.abi,
            "storage": extent.storage,
            "source_chunks": [
                {"offset_bytes": chunk.offset, "length_bytes": chunk.length,
                 "sha256": chunk.sha256}
                for chunk in extent.source_chunks
            ],
        }
        for extent in item.extents
    ]
    qsa_arena = [
        _buffer(name, shape, dtype)
        for name, (shape, dtype) in sorted(ARENA_SPECS.items())
    ]
    qsa_state = [
        _buffer("main_key_cache", (1, 1600, 256), "bfloat16"),
        _buffer("main_value_cache", (1, 1600, 256), "bfloat16"),
        _buffer("raw_key_cache", (1, 8, 140), "bfloat16"),
        _buffer("compressed_key_cache", (1, 400, 128), "bfloat16"),
        *[_buffer(name, (164,), "int32") for name in
          ("main_block_table", "compressed_block_table")],
        *[_buffer(name, (1,), "int64") for name in
          ("positions", "logical_positions")],
        *[_buffer(name, (1,), "int32") for name in (
            "main_slot_mapping", "raw_slot_mapping", "raw_block_table",
            "compressed_slot_mapping", "query_start_locations",
            "sequence_lengths", "token_to_request", "compression_work",
        )],
    ]
    # Fixed FlashInfer 91bda04 compact-c1 workspace with E11 state slots,
    # E10 staged weights, max_rows=10, H2560, physical N768, top-k10.
    moe_workspace = [
        _buffer("router_logits", (512,), "float32"),
        _buffer("global_ids", (10,), "int32"),
        _buffer("routing_weights", (10,), "float32"),
        _buffer("local_ids", (10,), "int32"),
        _buffer("local_weights", (10,), "float32"),
        _buffer("source_generation", (1,), "int64"),
        _buffer("requested_generation", (1,), "int64"),
        _buffer("route_summary", (16,), "uint8"),
        _buffer("staged_w13_packed", (10, 1536, 1280), "uint8"),
        _buffer("staged_w13_scale", (10, 1536, 160), "uint8"),
        _buffer("staged_down_packed", (10, 2560, 384), "uint8"),
        _buffer("staged_down_scale", (10, 2560, 48), "uint8"),
        *[_buffer(name, (10,), "float32") for name in (
            "staged_input_global_scale", "staged_folded_w1_alpha",
            "staged_w2_alpha", "staged_down_input_scale",
        )],
        _buffer("staged_source_expert_ids", (10,), "int32"),
        _buffer("staged_compact_expert_ids", (10,), "int32"),
        _buffer("staged_compact_routing_weights", (10,), "float32"),
        _buffer("staged_evidence", (16,), "uint8"),
        _buffer("packed_a", (11, 10, 1280), "uint8"),
        _buffer("packed_a_scale", (11, 128, 160), "uint8"),
        _buffer("route_output_scratch", (10, 3, 2560), "bfloat16"),
        _buffer("barrier_count", (1,), "int32"),
        _buffer("barrier_epoch", (1,), "int32"),
        _buffer("row_counts", (11,), "int32"),
        _buffer("active_expert_count", (1,), "int32"),
        _buffer("weight_expert_ids", (11,), "int32"),
        _buffer("global_to_local_expert", (10,), "int32"),
        _buffer("virtual_route_scratch", (28,), "int32"),
        _buffer("token_map", (11, 10), "int32"),
        _buffer("token_weights", (11, 10), "float32"),
        _buffer("shared_gate_scratch", (160,), "float32"),
        _buffer("shared_up_scratch", (160,), "float32"),
        _buffer("shared_gate_scalar", (1,), "float32"),
        _buffer("rank_local_partial", (1, HIDDEN), "bfloat16"),
    ]
    row_buffers = [
        _buffer("replicated_layer02", (ROWS, HC_WIDTH), "bfloat16"),
        _buffer("replicated_layer03", (ROWS, HC_WIDTH), "bfloat16"),
        _buffer("attention_input", (1, HIDDEN), "bfloat16"),
        _buffer("attention_injection", (1, 4), "bfloat16"),
        _buffer("reduced_attention", (1, HIDDEN), "float32"),
        _buffer("post_attention_hidden", (1, HC_WIDTH), "bfloat16"),
        _buffer("moe_input", (1, HIDDEN), "bfloat16"),
        _buffer("moe_injection", (1, 4), "bfloat16"),
        _buffer("reduced_moe", (1, HIDDEN), "float32"),
    ]
    qsa_projection_globals = _qsa_projection_globals(item)
    descriptor = {
        "schema": NATIVE_PLAN_SCHEMA, "rank": rank, "peer_rank": 1 - rank,
        "layer": LAYER, "artifact_key": TARGET_ARTIFACT,
        "manifest_sha256": TARGET_MANIFEST_SHA256,
        "slab_key": f"rank{rank}-target", "slab_bytes": item.qsa.slab_bytes,
        "layout_sha256": item.moe.layout_sha256,
        "slab_publication_layout_sha256": slab_publication_layout,
        "indexer_sidecar_key": INDEXER_SIDECAR,
        "oracle_manifest_sha256": plan.oracle_manifest_sha256,
        "oracle_layer02_sha256": LAYER02_SHA256,
        "oracle_layer03_sha256": LAYER03_SHA256,
        "replay_rows": list(plan.replay_rows), "compare_row": plan.compare_row,
        "extents": extents,
        "qsa_projection_globals": qsa_projection_globals,
        "buffers": {
            "qsa_arena": qsa_arena, "qsa_state": qsa_state,
            "target_moe_workspace": moe_workspace,
            "row": row_buffers,
        },
        "pair_reduce": {
            "bootstrap_host": pair.bootstrap_host,
            "bootstrap_port": pair.bootstrap_port,
            "timeout_ms": pair.timeout_ms,
            "session_sha256": pair.session_sha256,
            "rails": list(pair.rails), "gid_index": pair.gid_index,
            "calls": pair.calls,
        },
        "native_abis": {
            "target_slab_publication": 1, "qsa_c1": 1,
            "hyperconnection": 1, "target_moe_device_stage": 1,
            "target_full_moe_c1": 2,
            "pair_reduce_bootstrap": 3, "oracle_comparator": 1,
        },
    }
    descriptor["extent_inventory_sha256"] = hashlib.sha256(
        canonical_bytes(extents)
    ).hexdigest()
    descriptor["native_binding_inventory_sha256"] = hashlib.sha256(
        canonical_bytes([
            {key: extent[key] for key in
             ("name", "offset_bytes", "length_bytes", "storage")}
            for extent in extents
        ])
    ).hexdigest()
    descriptor["buffer_inventory_sha256"] = hashlib.sha256(
        canonical_bytes(descriptor["buffers"])
    ).hexdigest()
    descriptor["qsa_projection_globals_sha256"] = hashlib.sha256(
        canonical_bytes(qsa_projection_globals)
    ).hexdigest()
    descriptor["descriptor_sha256"] = hashlib.sha256(
        canonical_bytes(descriptor)
    ).hexdigest()
    return MappingProxyType(descriptor)


def native_target_slab_handoff(
    descriptor: Mapping[str, object], loaded: LoadedRankSlabs,
) -> NativeTargetSlabHandoff:
    """Alias the accepted CUDA loader's target allocation without copying it."""

    capability = accepted_native_handoff(loaded)
    if (
        not isinstance(loaded, LoadedRankSlabs)
        or capability is None
        or not _accepted_native_descriptor_schema(descriptor.get("schema"))
        or descriptor.get("artifact_key") != TARGET_ARTIFACT
        or descriptor.get("manifest_sha256") != TARGET_MANIFEST_SHA256
    ):
        raise Layer3FactoryError("native target slab handoff identity changed")
    rank = descriptor.get("rank")
    slab_key = descriptor.get("slab_key")
    if rank not in (0, 1) or loaded.receipt.rank != rank or slab_key != f"rank{rank}-target":
        raise Layer3FactoryError("native target slab handoff rank changed")
    target = loaded.slabs.get(slab_key)
    pointer = int(getattr(target, "data_ptr", lambda: 0)())
    elements = int(getattr(target, "numel", lambda: 0)())
    dtype = str(getattr(target, "dtype", ""))
    device_text = str(getattr(target, "device", ""))
    event = loaded.ready_event
    event_handle = int(getattr(event, "cuda_event", 0))
    try:
        device = int(device_text.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise Layer3FactoryError("native target slab CUDA device changed") from exc
    receipt = loaded.receipt.target
    expected_bytes = int(descriptor.get("slab_bytes", 0))
    if (
        pointer <= 0 or elements != expected_bytes or "uint8" not in dtype
        or not device_text.startswith("cuda:") or event_handle <= 0
        or receipt.key != slab_key or receipt.bytes_read != expected_bytes
        or receipt.h2d_bytes != expected_bytes or receipt.direct_reads != 236
        or receipt.h2d_copies != 236 or len(receipt.chunks) != 236
        or receipt.completed_ns <= receipt.started_ns
    ):
        raise Layer3FactoryError("native target slab publication changed")
    observed_receipt_sha256 = hashlib.sha256(canonical_bytes({
        "rank": rank, "slab_key": receipt.key,
        "bytes_read": receipt.bytes_read, "h2d_bytes": receipt.h2d_bytes,
        "direct_reads": receipt.direct_reads,
        "h2d_copies": receipt.h2d_copies,
        "started_ns": receipt.started_ns,
        "completed_ns": receipt.completed_ns,
        "chunks": [vars(chunk) for chunk in receipt.chunks],
    })).hexdigest()
    native_lease, receipt_sha256 = capability
    if receipt_sha256 != observed_receipt_sha256:
        raise Layer3FactoryError("native target slab receipt changed after load")
    prior = _PROCESS_LIFETIME_SLAB_OWNERS.get(rank)
    if prior is not None and (prior[0] is not loaded or prior[1] != receipt_sha256):
        raise Layer3FactoryError("process-lifetime target slab owner replaced")
    _PROCESS_LIFETIME_SLAB_OWNERS[rank] = (loaded, receipt_sha256)
    if native_lease is None:
        raise Layer3FactoryError("native process-lifetime slab lease absent")
    return NativeTargetSlabHandoff(
        owner=loaded, native_lease=native_lease,
        device_base=pointer, ready_event=event_handle,
        bytes=expected_bytes, device=device, rank=rank,
        artifact_key=TARGET_ARTIFACT, slab_key=slab_key,
        manifest_sha256=TARGET_MANIFEST_SHA256,
        layout_sha256=str(descriptor["slab_publication_layout_sha256"]),
        open_to_publish_ns=receipt.completed_ns - receipt.started_ns,
        chunks_authenticated=receipt.direct_reads,
        peak_host_pinned_bytes=4 * (268_435_456 + 65_536 - 1),
        receipt_sha256=receipt_sha256,
    )


__all__ = ["Layer3FactoryError", "NativeTargetSlabFinalizeError",
           "Layer3ExtentPlan", "Layer3PairReducePlan",
           "Layer3PhysicalPlan", "Layer3RankPlan", "native_rank_descriptor",
           "NativeTargetSlabHandoff", "NativeTargetSlabFinalizer",
           "CtypesNativeTargetSlabLeaseFactory",
           "native_target_slab_handoff",
           "prepare_layer3_physical_plan", "public_plan"]

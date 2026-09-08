# SPDX-License-Identifier: Apache-2.0
"""Slab-backed binding for the native c1 target QSA graph C ABI."""

from __future__ import annotations

import ctypes
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .qsa_weights import QSA_WEIGHTS_SCHEMA, RankQsaWeights

ARENA_FIELDS = (
    "qkv_packed", "qkv_sfa", "raw_main_qkv", "index_projected_qk",
    "main_query", "attention_gate", "index_query", "index_logits",
    "visible_blocks", "selected_blocks", "selected_tokens",
    "attention_partial", "attention_lse", "attention_output",
    "gated_attention", "output_packed", "output_sfa", "projected_output",
)
STATE_POINTER_FIELDS = (
    "main_key_cache", "main_value_cache", "raw_key_cache",
    "compressed_key_cache", "positions", "main_slot_mapping",
    "main_block_table", "raw_slot_mapping", "raw_block_table",
    "compressed_slot_mapping", "compressed_block_table",
    "query_start_locations", "logical_positions", "sequence_lengths",
    "token_to_request", "compression_work",
)
STATE_DTYPES = {
    **{name: "bfloat16" for name in STATE_POINTER_FIELDS[:4]},
    "positions": "int64",
    "logical_positions": "int64",
    **{
        name: "int32"
        for name in STATE_POINTER_FIELDS[4:]
        if name not in ("positions", "logical_positions")
    },
}


class NativeQsaBindingError(RuntimeError):
    """The authenticated slab, graph arena, or native QSA ABI changed."""


class ProjectionWeights(ctypes.Structure):
    _fields_ = [
        ("q_weight", ctypes.c_void_p), ("q_scale", ctypes.c_void_p),
        ("q_global", ctypes.c_float), ("k_weight", ctypes.c_void_p),
        ("k_scale", ctypes.c_void_p), ("k_global", ctypes.c_float),
        ("v_weight", ctypes.c_void_p), ("v_scale", ctypes.c_void_p),
        ("v_global", ctypes.c_float), ("o_weight", ctypes.c_void_p),
        ("o_scale", ctypes.c_void_p), ("o_global", ctypes.c_float),
    ]


class PreprocessWeights(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "main_q_norm", "main_k_norm", "index_qk", "index_q_norm",
        "index_k_norm", "rope_cos_sin",
    )]


class GraphArena(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in ARENA_FIELDS]


class StateView(ctypes.Structure):
    _fields_ = [
        *[(name, ctypes.c_void_p) for name in STATE_POINTER_FIELDS],
        ("main_blocks", ctypes.c_int),
        ("compressed_blocks", ctypes.c_int),
        ("compression_work_items", ctypes.c_int),
        ("rows", ctypes.c_int),
        ("rank", ctypes.c_int),
        ("layer", ctypes.c_int),
        ("uses_mrope", ctypes.c_bool),
        ("main_kv_dtype", ctypes.c_uint8),
        ("side_cache_dtype", ctypes.c_uint8),
        ("generation", ctypes.c_uint64),
        ("expected_generation", ctypes.c_uint64),
    ]


def _pointer(tensor: object, minimum_bytes: int, label: str) -> int:
    pointer = getattr(tensor, "data_ptr", None)
    numel = getattr(tensor, "numel", None)
    if (
        not callable(pointer)
        or not callable(numel)
        or numel() < minimum_bytes
        or "uint8" not in str(getattr(tensor, "dtype", ""))
        or not str(getattr(tensor, "device", "")).startswith("cuda:")
    ):
        raise NativeQsaBindingError(f"{label} must be a resident CUDA byte slab")
    value = int(pointer())
    if value <= 0:
        raise NativeQsaBindingError(f"{label} has no device pointer")
    return value


def _scalar(path: Path, offset: int) -> float:
    with path.open("rb") as stream:
        stream.seek(offset)
        payload = stream.read(4)
    if len(payload) != 4:
        raise NativeQsaBindingError("QSA global scale read was short")
    value = struct.unpack("<f", payload)[0]
    if not math.isfinite(value) or not value > 0.0:
        raise NativeQsaBindingError("QSA global scale must be positive")
    return value


@dataclass(frozen=True)
class NativeQsaSlabPointers:
    projection: ProjectionWeights
    preprocess: PreprocessWeights
    arena: GraphArena


def bind_state_view(
    tensors: Mapping[str, object], *, rank: int, layer: int,
    generation: int, main_blocks: int, compressed_blocks: int,
    compression_work_items: int,
) -> StateView:
    """Bind one exact caller-owned c1 BF16 causal-state generation."""

    if set(tensors) != set(STATE_POINTER_FIELDS):
        raise NativeQsaBindingError("native QSA state inventory changed")
    if (
        rank not in (0, 1) or layer not in range(3, 48, 4)
        or generation <= 0 or main_blocks <= 0 or compressed_blocks <= 0
        or compression_work_items <= 0
    ):
        raise NativeQsaBindingError("native QSA state identity changed")
    pointers = []
    devices = set()
    for name in STATE_POINTER_FIELDS:
        tensor = tensors[name]
        pointer = int(getattr(tensor, "data_ptr", lambda: 0)())
        device = str(getattr(tensor, "device", ""))
        if (
            pointer <= 0 or not device.startswith("cuda:")
            or STATE_DTYPES[name] not in str(getattr(tensor, "dtype", ""))
        ):
            raise NativeQsaBindingError(f"native QSA state pointer changed: {name}")
        pointers.append(pointer)
        devices.add(device)
    if len(devices) != 1:
        raise NativeQsaBindingError("native QSA state spans devices")
    return StateView(
        *pointers, main_blocks, compressed_blocks, compression_work_items,
        1, rank, layer, False, 0, 0, generation, generation,
    )


def bind_slab_pointers(
    descriptor: RankQsaWeights,
    target_slab: object,
    indexer_sidecar_slab: object,
    rope_cos_sin: object,
    arena: Mapping[str, object],
) -> NativeQsaSlabPointers:
    """Resolve authenticated offsets into caller-owned resident CUDA storage."""

    if not isinstance(descriptor, RankQsaWeights) or descriptor.schema != QSA_WEIGHTS_SCHEMA:
        raise NativeQsaBindingError("authenticated QSA descriptor is required")
    base = _pointer(target_slab, descriptor.slab_bytes, "target slab")
    sidecar_bytes = descriptor.indexer_sidecar_path.stat().st_size
    sidecar = _pointer(indexer_sidecar_slab, sidecar_bytes, "indexer sidecar")
    rope = int(getattr(rope_cos_sin, "data_ptr", lambda: 0)())
    if rope <= 0 or "bfloat16" not in str(getattr(rope_cos_sin, "dtype", "")):
        raise NativeQsaBindingError("caller-owned BF16 RoPE table is required")
    by_suffix = {}
    for item in descriptor.components:
        suffix = item.name.split(".self_attn.", 1)[1]
        by_suffix[suffix] = item

    def address(suffix: str) -> int:
        item = by_suffix.get(suffix)
        if item is None:
            raise NativeQsaBindingError(f"QSA component is absent: {suffix}")
        return (sidecar if item.storage == "indexer-sidecar" else base) + item.offset

    def global_scale(family: str) -> float:
        item = by_suffix[f"{family}.weight_scale_2"]
        return _scalar(descriptor.slab_path, item.offset)

    projection = ProjectionWeights(
        address("q_proj.weight"), address("q_proj.weight_scale"), global_scale("q_proj"),
        address("k_proj.weight"), address("k_proj.weight_scale"), global_scale("k_proj"),
        address("v_proj.weight"), address("v_proj.weight_scale"), global_scale("v_proj"),
        address("o_proj.weight"), address("o_proj.weight_scale"), global_scale("o_proj"),
    )
    preprocess = PreprocessWeights(
        address("q_norm.weight"), address("k_norm.weight"),
        address("indexer.index_qk_proj.weight"),
        address("indexer.q_layernorm.weight"),
        address("indexer.k_layernorm.weight"), rope,
    )
    if set(arena) != set(ARENA_FIELDS):
        raise NativeQsaBindingError("native QSA graph arena inventory changed")
    arena_pointers = []
    for name in ARENA_FIELDS:
        value = int(getattr(arena[name], "data_ptr", lambda: 0)())
        if value <= 0 or not str(getattr(arena[name], "device", "")).startswith("cuda:"):
            raise NativeQsaBindingError(f"native QSA arena pointer changed: {name}")
        arena_pointers.append(value)
    return NativeQsaSlabPointers(
        projection, preprocess, GraphArena(*arena_pointers)
    )


class NativeQsaGraphHandle:
    """Own one initialized native graph handle outside graph replay."""

    def __init__(self, library: Path, descriptor: RankQsaWeights,
                 pointers: NativeQsaSlabPointers, device: int):
        try:
            self._native = ctypes.CDLL(str(library))
        except OSError as exc:
            raise NativeQsaBindingError("cannot load native QSA C ABI") from exc
        self._native.rocket_qwen38_target_qsa_last_error.restype = ctypes.c_char_p
        self._native.rocket_qwen38_target_qsa_create_c1.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
            ctypes.POINTER(ProjectionWeights), ctypes.POINTER(PreprocessWeights),
            ctypes.POINTER(GraphArena), ctypes.POINTER(ctypes.c_void_p),
        ]
        self._native.rocket_qwen38_target_qsa_create_c1.restype = ctypes.c_int
        self._native.rocket_qwen38_target_qsa_launch_c1.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(StateView),
            ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p,
        ]
        self._native.rocket_qwen38_target_qsa_launch_c1.restype = ctypes.c_int
        self._native.rocket_qwen38_target_qsa_projected_output_c1.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ]
        self._native.rocket_qwen38_target_qsa_projected_output_c1.restype = ctypes.c_int
        self._native.rocket_qwen38_target_qsa_destroy_c1.argtypes = [ctypes.c_void_p]
        self._native.rocket_qwen38_target_qsa_destroy_c1.restype = None
        self._handle = ctypes.c_void_p()
        status = self._native.rocket_qwen38_target_qsa_create_c1(
            device, descriptor.rank, descriptor.layer,
            descriptor.indexer_sidecar_key.encode(), ctypes.byref(pointers.projection),
            ctypes.byref(pointers.preprocess), ctypes.byref(pointers.arena),
            ctypes.byref(self._handle),
        )
        if status or not self._handle.value:
            raw = self._native.rocket_qwen38_target_qsa_last_error()
            detail = raw.decode("utf-8", "replace") if raw else "unknown"
            raise NativeQsaBindingError(f"native QSA create failed: {detail}")
        self._pointers = pointers

    @property
    def pointer(self) -> int:
        if not self._handle.value:
            raise NativeQsaBindingError("native QSA graph handle is closed")
        return int(self._handle.value)

    def launch(self, block_input: object, state: StateView,
               generation: int, stream_pointer: int) -> int:
        """Enqueue one c1 generation. The caller fences before publication."""

        if not isinstance(state, StateView) or generation != state.generation:
            raise NativeQsaBindingError("native QSA launch state changed")
        hidden = int(getattr(block_input, "data_ptr", lambda: 0)())
        if (
            hidden <= 0 or stream_pointer <= 0
            or "bfloat16" not in str(getattr(block_input, "dtype", ""))
            or not str(getattr(block_input, "device", "")).startswith("cuda:")
        ):
            raise NativeQsaBindingError("native QSA launch pointers changed")
        status = self._native.rocket_qwen38_target_qsa_launch_c1(
            self._handle, hidden, ctypes.byref(state), generation, 1,
            stream_pointer,
        )
        if status:
            raw = self._native.rocket_qwen38_target_qsa_last_error()
            detail = raw.decode("utf-8", "replace") if raw else "unknown"
            raise NativeQsaBindingError(f"native QSA launch failed: {detail}")
        output = ctypes.c_void_p()
        status = self._native.rocket_qwen38_target_qsa_projected_output_c1(
            self._handle, ctypes.byref(output)
        )
        if status or not output.value:
            raise NativeQsaBindingError("native QSA published no output pointer")
        return int(output.value)

    def close(self) -> None:
        if self._handle.value:
            self._native.rocket_qwen38_target_qsa_destroy_c1(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "NativeQsaGraphHandle":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["ARENA_FIELDS", "STATE_DTYPES", "STATE_POINTER_FIELDS", "NativeQsaBindingError",
           "NativeQsaGraphHandle", "NativeQsaSlabPointers", "StateView",
           "bind_slab_pointers", "bind_state_view"]

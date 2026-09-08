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

    def close(self) -> None:
        if self._handle.value:
            self._native.rocket_qwen38_target_qsa_destroy_c1(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "NativeQsaGraphHandle":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["ARENA_FIELDS", "NativeQsaBindingError", "NativeQsaGraphHandle",
           "NativeQsaSlabPointers", "bind_slab_pointers"]

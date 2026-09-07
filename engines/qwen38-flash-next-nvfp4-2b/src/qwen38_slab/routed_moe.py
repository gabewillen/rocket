# SPDX-License-Identifier: Apache-2.0
"""Fixed Qwen3.8 TP2 routed/shared MoE graph contract.

The production backend is an adapter around FlashInfer 91bda04's b12x SM121
kernel family. Runtime shape and backend selection are removed. Each rank owns
256 routed experts and 160 columns of the replicated BF16 shared expert before
the existing TP2 PairReduce.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from .contract import MODEL_NVFP4_ABI, PINNED_CONTRACT, SCHEMA, canonical_bytes

HIDDEN = 2_560
INTERMEDIATE = 640
SHARED_INTERMEDIATE = 320
GLOBAL_EXPERTS = 512
LOCAL_EXPERTS = 256
TOP_K = 10
BUCKET_SELECTORS: Mapping[int, str] = MappingProxyType(
    {1: "static", 2: "static", 4: "static_tail", 8: "dynamic", 16: "dynamic"}
)
SEQUENCE_BUCKETS = (1, 2, 4, 8, 16)
MAX_VERIFY_WIDTH = 8
RESIDENT_K0_ROWS = SEQUENCE_BUCKETS
LAZY_VERIFIER_ROWS = (32, 64, 80, 128)
FLASHINFER_COMMIT = "91bda04c66f7cb851e1ab3b78b9fecea644b9844"
SLAB_ARTIFACT_KEY = "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
MOE_SCHEMA = "qwen3.8-flash-next:tp2:routed-moe-w4a4:v1"
_NATIVE_ABI = "native"
_HEX = frozenset("0123456789abcdef")


class RoutedMoeGraphError(RuntimeError):
    """Fixed MoE identity, routing, backend, or graph contract changed."""


@dataclass(frozen=True)
class MoeExtent:
    name: str
    offset: int
    length: int
    shape: tuple[int, ...]
    dtype: str
    layout: str
    abi: str


@dataclass(frozen=True)
class MoeShape:
    sequences: int
    verify_width: int = 1

    def __post_init__(self) -> None:
        if self.sequences not in SEQUENCE_BUCKETS or isinstance(self.verify_width, bool) or not 1 <= self.verify_width <= MAX_VERIFY_WIDTH:
            raise RoutedMoeGraphError("MoE shape must bind c1/c2/c4/c8/c16 and K0..K7")

    @property
    def token_rows(self) -> int:
        return self.sequences * self.verify_width


def selector_for(shape: MoeShape) -> str:
    if not isinstance(shape, MoeShape):
        raise RoutedMoeGraphError("MoE shape is required")
    if shape.verify_width == 1:
        return BUCKET_SELECTORS[shape.sequences]
    return "dynamic"


def compact_owner_pairs(
    global_ids: tuple[tuple[int, ...], ...],
    routing_weights: tuple[tuple[float, ...], ...],
    rank: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...]]:
    """CPU reference for the production device-side EP2 route compaction."""

    if rank not in (0, 1) or len(global_ids) != len(routing_weights):
        raise RoutedMoeGraphError("EP2 route compaction input changed")
    first = rank * LOCAL_EXPERTS
    ids_out, weights_out = [], []
    for ids, weights in zip(global_ids, routing_weights, strict=True):
        if len(ids) != TOP_K or len(weights) != TOP_K:
            raise RoutedMoeGraphError("every token must carry exactly top-k10 pairs")
        local_ids, local_weights = [], []
        for expert, weight in zip(ids, weights, strict=True):
            if isinstance(expert, bool) or not isinstance(expert, int) or not 0 <= expert < GLOBAL_EXPERTS:
                raise RoutedMoeGraphError("global expert id is outside E512")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise RoutedMoeGraphError("routing weight is invalid")
            owned = first <= expert < first + LOCAL_EXPERTS
            local_ids.append(expert - first if owned else 0)
            local_weights.append(float(weight) if owned else 0.0)
        ids_out.append(tuple(local_ids))
        weights_out.append(tuple(local_weights))
    return tuple(ids_out), tuple(weights_out)


@dataclass(frozen=True)
class OwnerLocalMoeSlab:
    schema: str
    artifact_key: str
    revision: str
    rank: int
    layer: int
    first_expert: int
    last_expert: int
    slab_path: Path
    slab_bytes: int
    layout_sha256: str
    chunk_sha256: tuple[str, ...]
    routed: tuple[MoeExtent, ...]
    shared: tuple[MoeExtent, ...]


class RoutedMoeBackend(Protocol):
    def has_bucket(self, shape: MoeShape, selector: str) -> bool: ...
    def launch(
        self,
        hidden: object,
        global_ids: object,
        routing_weights: object,
        *,
        rank: int,
        shape: MoeShape,
        selector: str,
    ) -> object: ...


@dataclass(frozen=True)
class FlashInferMoeWeights:
    """Graph-resident tensors assembled once from authenticated slab views."""

    w1_weight: object
    w1_scale: object
    w1_alpha: object
    w2_weight: object
    w2_scale: object
    w2_alpha: object
    input_scale: object
    fc2_input_scale: object
    shared_gate: object
    shared_up: object
    shared_down: object
    shared_expert_gate: object


class FlashInferRoutedMoeBackend:
    """Exact E256 adapter for FlashInfer's measured SM121 b12x kernels.

    The caller materializes contiguous owner-local tensors once from the
    authenticated extents. No E512 weight plane or generic selector is kept.
    """

    def __init__(self, weights: FlashInferMoeWeights, *, torch_api=None):
        if torch_api is None:
            import torch as torch_api  # type: ignore[no-redef]
        try:
            from flashinfer import B12xMoEWrapper
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
        except ImportError as exc:
            raise RoutedMoeGraphError("pinned FlashInfer b12x runtime is absent") from exc
        self._torch = torch_api
        self._dispatch = moe_dispatch
        self._weights = weights
        self._validate_weights()
        device = weights.w1_weight.device
        self._supported_rows = frozenset(
            c * width
            for c in SEQUENCE_BUCKETS
            for width in range(1, MAX_VERIFY_WIDTH + 1)
        )
        # K0 is the resident hot path. Verifier widths K1..K7 are lazy so a
        # c16 K0 process never pays workspace residency for rows 32..128.
        self._wrapper_type = B12xMoEWrapper
        self._device = device
        self._wrappers = {}
        for m in RESIDENT_K0_ROWS:
            selector = BUCKET_SELECTORS[m]
            with self._forced_selector(selector):
                self._wrappers[(m, selector)] = self._new_wrapper(m)

    def _validate_weights(self) -> None:
        w = self._weights
        expected = (
            (w.w1_weight, (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2), "uint8"),
            (w.w2_weight, (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // 2), "uint8"),
            (w.w1_alpha, (LOCAL_EXPERTS,), "float32"),
            (w.w2_alpha, (LOCAL_EXPERTS,), "float32"),
            (w.input_scale, (LOCAL_EXPERTS,), "float32"),
            (w.fc2_input_scale, (LOCAL_EXPERTS,), "float32"),
            (w.shared_gate, (SHARED_INTERMEDIATE, HIDDEN), "bfloat16"),
            (w.shared_up, (SHARED_INTERMEDIATE, HIDDEN), "bfloat16"),
            (w.shared_down, (HIDDEN, SHARED_INTERMEDIATE), "bfloat16"),
            (w.shared_expert_gate, (1, HIDDEN), "bfloat16"),
        )
        devices = set()
        for tensor, shape, dtype in expected:
            if tuple(getattr(tensor, "shape", ())) != shape or dtype not in str(getattr(tensor, "dtype", "")):
                raise RoutedMoeGraphError("owner-local MoE tensor shape or dtype changed")
            device = getattr(tensor, "device", None)
            if device is None or getattr(tensor, "is_cuda", False) is not True:
                raise RoutedMoeGraphError("owner-local MoE weights must reside on CUDA")
            devices.add(str(device))
        if len(devices) != 1:
            raise RoutedMoeGraphError("owner-local MoE weights span CUDA devices")

    def has_bucket(self, shape: MoeShape, selector: str) -> bool:
        return selector_for(shape) == selector and shape.token_rows in self._supported_rows

    def _new_wrapper(self, m: int):
        return self._wrapper_type(
            num_experts=LOCAL_EXPERTS,
            top_k=TOP_K,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            use_cuda_graph=True,
            max_num_tokens=m,
            quant_mode="nvfp4",
            source_format="modelopt",
            device=self._device,
        )

    def _wrapper(self, m: int, selector: str):
        key = (m, selector)
        wrapper = self._wrappers.get(key)
        if wrapper is None:
            wrapper = self._new_wrapper(m)
            self._wrappers[key] = wrapper
        return wrapper

    @contextmanager
    def _forced_selector(self, selector: str):
        with FLASHINFER_SELECTOR_LOCK:
            previous = self._dispatch._FORCED_BACKEND
            previous_tail = getattr(self._dispatch, "_EXACT_N640_RETAINED_TAIL", False)
            try:
                if selector == "static_tail":
                    if not hasattr(self._dispatch, "_EXACT_N640_RETAINED_TAIL"):
                        raise RoutedMoeGraphError("exact-N640 retained-tail source is absent")
                    self._dispatch._EXACT_N640_RETAINED_TAIL = True
                    forced = "static"
                else:
                    if hasattr(self._dispatch, "_EXACT_N640_RETAINED_TAIL"):
                        self._dispatch._EXACT_N640_RETAINED_TAIL = False
                    forced = selector
                self._dispatch._FORCED_BACKEND = forced
                yield
            finally:
                self._dispatch._FORCED_BACKEND = previous
                if hasattr(self._dispatch, "_EXACT_N640_RETAINED_TAIL"):
                    self._dispatch._EXACT_N640_RETAINED_TAIL = previous_tail

    def routed_only(
        self,
        hidden: object,
        global_ids: object,
        routing_weights: object,
        *,
        rank: int,
        shape: MoeShape,
        selector: str,
    ) -> object:
        torch = self._torch
        m = shape.token_rows
        if (
            tuple(getattr(hidden, "shape", ())) != (m, HIDDEN)
            or "bfloat16" not in str(getattr(hidden, "dtype", ""))
            or tuple(getattr(global_ids, "shape", ())) != (m, TOP_K)
            or "int32" not in str(getattr(global_ids, "dtype", ""))
            or tuple(getattr(routing_weights, "shape", ())) != (m, TOP_K)
            or "float32" not in str(getattr(routing_weights, "dtype", ""))
            or any(str(getattr(value, "device", "")) != str(self._weights.w1_weight.device)
                   for value in (hidden, global_ids, routing_weights))
        ):
            raise RoutedMoeGraphError("MoE activation or routing tensor contract changed")
        # These capture-safe device assertions surface at the transactional
        # graph fence. Invalid routes never become silently remote/zero pairs.
        torch._assert_async(
            ((global_ids >= 0) & (global_ids < GLOBAL_EXPERTS)).all(),
            "global expert id is outside E512",
        )
        torch._assert_async(
            torch.isfinite(routing_weights).all(),
            "routing weights contain a non-finite value",
        )
        first = rank * LOCAL_EXPERTS
        local = (global_ids >= first) & (global_ids < first + LOCAL_EXPERTS)
        local_ids = torch.where(local, global_ids - first, torch.zeros_like(global_ids))
        local_weights = torch.where(local, routing_weights, torch.zeros_like(routing_weights))
        w = self._weights
        with self._forced_selector(selector):
                routed = self._wrapper(m, selector).run(
                    x=hidden,
                    w1_weight=w.w1_weight,
                    w1_weight_sf=w.w1_scale,
                    w1_alpha=w.w1_alpha,
                    w2_weight=w.w2_weight,
                    w2_weight_sf=w.w2_scale,
                    w2_alpha=w.w2_alpha,
                    input_global_scale=w.input_scale,
                    fc2_input_scale=w.fc2_input_scale,
                    token_selected_experts=local_ids,
                    token_final_scales=local_weights,
                )
        return routed

    def launch(
        self,
        hidden: object,
        global_ids: object,
        routing_weights: object,
        *,
        rank: int,
        shape: MoeShape,
        selector: str,
    ) -> object:
        routed = self.routed_only(
            hidden, global_ids, routing_weights,
            rank=rank, shape=shape, selector=selector,
        )
        shared = self.shared_partial(hidden, rank)
        # The disjoint N160 partials sum to the full replicated N320 expert.
        return routed.add_(shared)

    def shared_partial(self, hidden: object, rank: int) -> object:
        """Return the rank-owned N160 shared-expert contribution."""

        if rank not in (0, 1):
            raise RoutedMoeGraphError("shared expert rank is outside TP2")
        torch = self._torch
        w = self._weights
        begin, end = shared_intermediate_bounds(rank)
        gate = torch.nn.functional.silu(hidden @ w.shared_gate[begin:end].T)
        up = hidden @ w.shared_up[begin:end].T
        partial = (gate * up) @ w.shared_down[:, begin:end].T
        return partial * torch.sigmoid(hidden @ w.shared_expert_gate.T)

    def shared_reference(self, hidden: object) -> object:
        """Full N320 control used only by live parity proof."""

        torch = self._torch
        w = self._weights
        gate = torch.nn.functional.silu(hidden @ w.shared_gate.T)
        up = hidden @ w.shared_up.T
        full = (gate * up) @ w.shared_down.T
        return full * torch.sigmoid(hidden @ w.shared_expert_gate.T)


def shared_intermediate_bounds(rank: int) -> tuple[int, int]:
    """Return the disjoint EP2 intermediate shard for the BF16 shared expert."""

    if rank not in (0, 1):
        raise RoutedMoeGraphError("shared expert rank is outside TP2")
    width = SHARED_INTERMEDIATE // 2
    return rank * width, (rank + 1) * width


def materialize_flashinfer_weights(
    slab: OwnerLocalMoeSlab,
    *,
    torch_api=None,
    device: object = "cuda",
) -> FlashInferMoeWeights:
    """Assemble the authenticated E256 slab views into b12x tensor planes.

    This is a one-time load operation. It never creates the generic E512
    control plane and performs no packing on a graph replay.
    """

    if torch_api is None:
        import torch as torch_api  # type: ignore[no-redef]
    torch = torch_api
    if not isinstance(slab, OwnerLocalMoeSlab) or len(slab.routed) != LOCAL_EXPERTS * 12:
        raise RoutedMoeGraphError("authenticated E256 slab descriptor is required")
    by_name = {extent.name: extent for extent in (*slab.routed, *slab.shared)}
    if len(by_name) != len(slab.routed) + len(slab.shared):
        raise RoutedMoeGraphError("MoE slab extents overlap by name")

    def read_extent(name: str) -> bytearray:
        extent = by_name.get(name)
        if extent is None:
            raise RoutedMoeGraphError(f"MoE materialization extent is absent: {name}")
        try:
            fd = os.open(slab.slab_path, os.O_RDONLY)
            try:
                payload = os.pread(fd, extent.length, extent.offset)
            finally:
                os.close(fd)
        except OSError as exc:
            raise RoutedMoeGraphError(f"cannot read MoE slab extent: {name}") from exc
        if len(payload) != extent.length:
            raise RoutedMoeGraphError(f"short MoE slab extent: {name}")
        return bytearray(payload)

    prefix = f"model.language_model.layers.{slab.layer}.mlp"
    first = slab.first_expert
    w1, w1_sf, w2, w2_sf = bytearray(), bytearray(), bytearray(), bytearray()
    w1_alpha: list[float] = []
    w2_alpha: list[float] = []
    input_scale: list[float] = []
    fc2_input_scale: list[float] = []
    for expert in range(first, first + LOCAL_EXPERTS):
        root = f"{prefix}.experts.{expert}"
        # FlashInfer's ModelOpt FC1 ABI is [up, gate], not checkpoint name order.
        for projection in ("up_proj", "gate_proj"):
            w1.extend(read_extent(f"{root}.{projection}.weight"))
            w1_sf.extend(read_extent(f"{root}.{projection}.weight_scale"))
        gate_alpha = _read_f32(read_extent(f"{root}.gate_proj.weight_scale_2"))
        up_alpha = _read_f32(read_extent(f"{root}.up_proj.weight_scale_2"))
        gate_input = _read_f32(read_extent(f"{root}.gate_proj.input_scale"))
        up_input = _read_f32(read_extent(f"{root}.up_proj.input_scale"))
        if gate_alpha != up_alpha or gate_input != up_input:
            raise RoutedMoeGraphError("fused FC1 gate/up scalar scales differ")
        w1_alpha.append(gate_alpha)
        input_scale.append(gate_input)
        w2.extend(read_extent(f"{root}.down_proj.weight"))
        w2_sf.extend(read_extent(f"{root}.down_proj.weight_scale"))
        w2_alpha.append(_read_f32(read_extent(f"{root}.down_proj.weight_scale_2")))
        fc2_input_scale.append(_read_f32(read_extent(f"{root}.down_proj.input_scale")))

    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    def u8(blob: bytearray, shape: tuple[int, ...]):
        return torch.frombuffer(blob, dtype=torch.uint8).reshape(shape).to(device)

    w1_weight = u8(w1, (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2))
    w2_weight = u8(w2, (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // 2))
    w1_scale = convert_sf_to_mma_layout(
        torch.frombuffer(w1_sf, dtype=torch.uint8).view(torch.float8_e4m3fn).to(device),
        m=2 * INTERMEDIATE, k=HIDDEN, num_groups=LOCAL_EXPERTS,
    )
    w2_scale = convert_sf_to_mma_layout(
        torch.frombuffer(w2_sf, dtype=torch.uint8).view(torch.float8_e4m3fn).to(device),
        m=HIDDEN, k=INTERMEDIATE, num_groups=LOCAL_EXPERTS,
    )

    def f32(values: list[float]):
        return torch.tensor(values, dtype=torch.float32, device=device)

    def bf16(suffix: str, shape: tuple[int, ...]):
        blob = read_extent(f"{prefix}.{suffix}")
        return torch.frombuffer(blob, dtype=torch.uint8).view(torch.bfloat16).reshape(shape).to(device)

    return FlashInferMoeWeights(
        w1_weight, w1_scale, f32(w1_alpha), w2_weight, w2_scale,
        f32(w2_alpha), f32(input_scale), f32(fc2_input_scale),
        bf16("shared_expert.gate_proj.weight", (SHARED_INTERMEDIATE, HIDDEN)),
        bf16("shared_expert.up_proj.weight", (SHARED_INTERMEDIATE, HIDDEN)),
        bf16("shared_expert.down_proj.weight", (HIDDEN, SHARED_INTERMEDIATE)),
        bf16("shared_expert_gate.weight", (1, HIDDEN)),
    )


def _read_f32(payload: bytearray) -> float:
    if len(payload) != 4:
        raise RoutedMoeGraphError("MoE scalar extent is not F32")
    return struct.unpack("<f", payload)[0]


def _expert_contract(projection: str, leaf: str) -> tuple[int, tuple[int, ...], str, str, str]:
    rows, columns = {
        "gate_proj": (INTERMEDIATE, HIDDEN),
        "up_proj": (INTERMEDIATE, HIDDEN),
        "down_proj": (HIDDEN, INTERMEDIATE),
    }[projection]
    if leaf == "weight":
        return rows * columns // 2, (rows, columns // 2), "U8", "checkpoint", MODEL_NVFP4_ABI
    if leaf == "weight_scale":
        return rows * columns // 16, (rows * columns // 16,), "F8_E4M3", "cutlass_sm121_sfb", MODEL_NVFP4_ABI
    if leaf in ("weight_scale_2", "input_scale"):
        return 4, (), "F32", "checkpoint", MODEL_NVFP4_ABI
    raise RoutedMoeGraphError("unknown routed-expert component")


_SHARED_CONTRACT = {
    "shared_expert.down_proj.weight": (HIDDEN * SHARED_INTERMEDIATE * 2, (HIDDEN, SHARED_INTERMEDIATE)),
    "shared_expert.gate_proj.weight": (SHARED_INTERMEDIATE * HIDDEN * 2, (SHARED_INTERMEDIATE, HIDDEN)),
    "shared_expert.up_proj.weight": (SHARED_INTERMEDIATE * HIDDEN * 2, (SHARED_INTERMEDIATE, HIDDEN)),
    "shared_expert_gate.weight": (HIDDEN * 2, (1, HIDDEN)),
}


def load_owner_local_moe(artifact: Path, rank: int, layer: int) -> OwnerLocalMoeSlab:
    """Authenticate every routed and shared extent for one frozen layer."""

    if (
        not isinstance(artifact, Path)
        or rank not in (0, 1)
        or isinstance(layer, bool)
        or not isinstance(layer, int)
        or not 0 <= layer < 48
    ):
        raise RoutedMoeGraphError("artifact Path, TP2 rank, and layer 0..47 are required")
    try:
        manifest = json.loads((artifact / "manifest.json").read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoutedMoeGraphError("cannot read rank-slab manifest") from exc
    if not isinstance(manifest, dict):
        raise RoutedMoeGraphError("rank-slab manifest root changed")
    digest_input = dict(manifest)
    claimed = digest_input.pop("artifact_key", None)
    observed = hashlib.sha256(canonical_bytes(digest_input)).hexdigest()
    if (
        claimed != observed
        or artifact.name != claimed
        or manifest.get("schema") != SCHEMA
        or manifest.get("revision") != PINNED_CONTRACT.revision
        or manifest.get("overlay_artifact_key") != PINNED_CONTRACT.artifact_key
        or manifest.get("overlay_sha256") != PINNED_CONTRACT.overlay_sha256
        or manifest.get("tp_size") != 2
        or manifest.get("tensor_alignment_bytes") != 256
        or manifest.get("shared_payload_policy") != "target-owned-mtp-reference-read-once"
    ):
        raise RoutedMoeGraphError("rank-slab identity changed")
    slab = manifest.get("slabs", {}).get(f"rank{rank}-target")
    if not isinstance(slab, dict) or not isinstance(slab.get("entries"), list):
        raise RoutedMoeGraphError("owner-local target slab is absent")
    entries = {entry.get("name"): entry for entry in slab["entries"] if isinstance(entry, dict)}
    if len(entries) != len(slab["entries"]):
        raise RoutedMoeGraphError("target slab entries are duplicate or invalid")
    first = rank * LOCAL_EXPERTS
    prefix = f"model.language_model.layers.{layer}.mlp"
    routed: list[MoeExtent] = []
    for expert in range(first, first + LOCAL_EXPERTS):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for leaf in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                suffix = f"experts.{expert}.{projection}.{leaf}"
                routed.append(_validated_extent(entries, prefix, suffix, _expert_contract(projection, leaf)))
    shared = tuple(
        _validated_extent(
            entries, prefix,
            suffix,
            (length, shape, "BF16", "checkpoint", _NATIVE_ABI),
        )
        for suffix, (length, shape) in _SHARED_CONTRACT.items()
    )
    selected = tuple(routed) + shared
    layout_sha = hashlib.sha256(
        canonical_bytes([
            {"name": x.name, "offset": x.offset, "length": x.length,
             "shape": x.shape, "dtype": x.dtype, "layout": x.layout, "abi": x.abi}
            for x in selected
        ])
    ).hexdigest()
    chunks = slab.get("chunks")
    if not isinstance(chunks, list):
        raise RoutedMoeGraphError("target slab chunk table is absent")
    touched: list[tuple[int, int, str]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise RoutedMoeGraphError("target slab chunk record changed")
        start, length, digest = chunk.get("offset_bytes"), chunk.get("length_bytes"), chunk.get("sha256")
        if not isinstance(start, int) or not isinstance(length, int) or not _sha256(digest):
            raise RoutedMoeGraphError("target slab chunk identity changed")
        if any(x.offset < start + length and x.offset + x.length > start for x in selected):
            touched.append((start, length, digest))
    if not touched:
        raise RoutedMoeGraphError("MoE extents have no authenticated chunks")
    slab_path = artifact / str(slab.get("file", ""))
    slab_bytes = slab.get("bytes")
    try:
        valid_size = isinstance(slab_bytes, int) and not isinstance(slab_bytes, bool) and slab_bytes > 0 and slab_path.stat().st_size == slab_bytes
    except OSError as exc:
        raise RoutedMoeGraphError("owner-local slab is absent") from exc
    if not valid_size:
        raise RoutedMoeGraphError("owner-local slab byte identity changed")
    try:
        fd = os.open(slab_path, os.O_RDONLY)
        try:
            for start, length, digest in touched:
                payload = os.pread(fd, length, start)
                if len(payload) != length or hashlib.sha256(payload).hexdigest() != digest:
                    raise RoutedMoeGraphError("owner-local MoE chunk digest changed")
        finally:
            os.close(fd)
    except OSError as exc:
        raise RoutedMoeGraphError("cannot authenticate owner-local MoE chunks") from exc
    return OwnerLocalMoeSlab(
        MOE_SCHEMA, claimed, PINNED_CONTRACT.revision, rank, layer, first,
        first + LOCAL_EXPERTS - 1, slab_path, slab_bytes, layout_sha,
        tuple(item[2] for item in touched), tuple(routed), shared,
    )


def _validated_extent(entries: Mapping[str, object], prefix: str, suffix: str, contract: tuple[int, tuple[int, ...], str, str, str]) -> MoeExtent:
    name = f"{prefix}.{suffix}"
    entry = entries.get(name)
    length, shape, dtype, layout, abi = contract
    if not isinstance(entry, dict):
        raise RoutedMoeGraphError(f"MoE extent is absent: {name}")
    offset = entry.get("offset_bytes")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or offset % 256
        or entry.get("length_bytes") != length
        or tuple(entry.get("shape", ())) != shape
        or entry.get("dtype") != dtype
        or entry.get("layout") != layout
        or entry.get("abi") != abi
    ):
        raise RoutedMoeGraphError(f"MoE extent contract changed: {name}")
    return MoeExtent(name, offset, length, shape, dtype, layout, abi)


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


class RoutedMoeGraph:
    """One rank-local MoE node with immutable selector and output ownership."""

    _SPAN = "rocket.qwen38.decode.routed_moe"

    def __init__(self, slab: OwnerLocalMoeSlab, backend: RoutedMoeBackend, tracer):
        if (
            not isinstance(slab, OwnerLocalMoeSlab)
            or slab.schema != MOE_SCHEMA
            or slab.artifact_key != SLAB_ARTIFACT_KEY
            or slab.revision != PINNED_CONTRACT.revision
            or slab.rank not in (0, 1)
            or isinstance(slab.layer, bool)
            or not 0 <= slab.layer < 48
            or slab.first_expert != slab.rank * LOCAL_EXPERTS
            or slab.last_expert != slab.first_expert + LOCAL_EXPERTS - 1
            or not _sha256(slab.layout_sha256)
            or not slab.chunk_sha256
            or any(not _sha256(value) for value in slab.chunk_sha256)
            or len(slab.routed) != LOCAL_EXPERTS * 3 * 4
            or len(slab.shared) != len(_SHARED_CONTRACT)
            or any(
                not extent.name.startswith(
                    f"model.language_model.layers.{slab.layer}.mlp."
                )
                for extent in (*slab.routed, *slab.shared)
            )
        ):
            raise RoutedMoeGraphError("authenticated owner-local MoE slab changed")
        if backend is None or tracer is None:
            raise RoutedMoeGraphError("MoE backend and tracer are required")
        for sequences in SEQUENCE_BUCKETS:
            for width in range(1, MAX_VERIFY_WIDTH + 1):
                shape = MoeShape(sequences, width)
                if not backend.has_bucket(shape, selector_for(shape)):
                    raise RoutedMoeGraphError("all bounded MoE shapes must be available")
        self._slab = slab
        self._backend = backend
        self._tracer = tracer
        self._faulted = False

    @property
    def shared_intermediate_shard(self) -> tuple[int, int]:
        return shared_intermediate_bounds(self.rank)

    @property
    def rank(self) -> int:
        return self._slab.rank

    @property
    def identity(self) -> tuple[str, str, int, str, tuple[str, ...]]:
        return (
            self._slab.revision,
            self._slab.artifact_key,
            self._slab.layer,
            self._slab.layout_sha256,
            self._slab.chunk_sha256,
        )

    def launch(self, hidden: object, global_ids: object, routing_weights: object, shape: MoeShape, *, request_id: str = "") -> object:
        if self._faulted:
            raise RoutedMoeGraphError("faulted MoE graph cannot be replayed")
        selector = selector_for(shape)
        m = shape.token_rows
        if not isinstance(request_id, str) or len(request_id) > 128:
            raise RoutedMoeGraphError("MoE bucket or request identity changed")
        tensors = (
            (hidden, (m, HIDDEN), "bfloat16"),
            (global_ids, (m, TOP_K), "int32"),
            (routing_weights, (m, TOP_K), "float32"),
        )
        devices = set()
        for tensor, expected_shape, dtype in tensors:
            if tuple(getattr(tensor, "shape", ())) != expected_shape or dtype not in str(getattr(tensor, "dtype", "")) or getattr(tensor, "is_cuda", False) is not True:
                raise RoutedMoeGraphError("MoE activation or routing tensor contract changed")
            devices.add(str(getattr(tensor, "device", "")))
        if len(devices) != 1:
            raise RoutedMoeGraphError("MoE launch tensors span CUDA devices")
        try:
            with self._observed(m, selector, request_id) as span:
                output = self._backend.launch(
                    hidden, global_ids, routing_weights,
                    rank=self.rank, shape=shape, selector=selector,
                )
                span.set_attribute("outcome", "success")
                if tuple(getattr(output, "shape", ())) != (m, HIDDEN) or "bfloat16" not in str(getattr(output, "dtype", "")) or str(getattr(output, "device", "")) not in devices:
                    raise RoutedMoeGraphError("rank-local MoE output contract changed")
                return output
        except BaseException:
            self._faulted = True
            raise

    @contextmanager
    def _observed(self, m: int, selector: str, request_id: str):
        with self._tracer.start_as_current_span(self._SPAN) as span:
            span.set_attribute("rank", self.rank)
            span.set_attribute("m_bucket", m)
            span.set_attribute("selector", selector)
            if request_id:
                span.set_attribute("request_id", request_id)
            try:
                yield span
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                raise


FLASHINFER_SELECTOR_LOCK = threading.RLock()

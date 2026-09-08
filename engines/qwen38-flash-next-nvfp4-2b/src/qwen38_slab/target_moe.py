# SPDX-License-Identifier: Apache-2.0
"""Fail-closed K0 target router/MoE participant.

The participant is the Python composition boundary around one native captured
target-MoE launch. The graph owns every mutable activation, route, generation,
local-route, and output buffer. The router port must write stable global top-10
routes. ``TargetMoeBinding`` localizes E512 IDs on device into its dense E256
route planes and calls the measured NVFP4 B12x target primitive plus the BF16
shared-expert partial on the same borrowed stream.

Reference implementations inspected before this boundary was written:

* pinned vLLM 54da70c1eb6976fe655efd535f14a01378cd08bc
* upstream vLLM 9ea8f3ffc354901b740f0b31988900897b7221d7

Both Qwen3Next implementations select softmax top-10 and renormalize the
selected probabilities. Rocket preserves those semantics while using the
accepted target-slab NVFP4 router mapping.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, Sequence

from .contract import MODEL_NVFP4_ABI
from .k0_composition import K0_DOMAIN
from .routed_moe import (
    GLOBAL_EXPERTS,
    HIDDEN,
    MOE_SCHEMA,
    TOP_K,
    MoeShape,
    OwnerLocalMoeSlab,
)

ROUTING_SEMANTICS = "softmax_top10_renormalized_stable_global_ids:v1"
SHARED_ABI = "BF16/native"
PINNED_VLLM_COMMIT = "54da70c1eb6976fe655efd535f14a01378cd08bc"
REFERENCE_VLLM_COMMIT = "9ea8f3ffc354901b740f0b31988900897b7221d7"
_SEQUENCE_BUCKETS = frozenset((1, 2, 4, 8, 16))


class TargetMoeError(RuntimeError):
    """Target router/MoE binding or execution violated its fixed contract."""


class NativeTargetMoePort(Protocol):
    """Raw no-fallback adapter to the measured target B12x primitive."""

    @property
    def router_abi(self) -> str: ...

    @property
    def routed_abi(self) -> str: ...

    @property
    def shared_abi(self) -> str: ...

    @property
    def localizes_global_ids(self) -> bool: ...

    @property
    def generation_checked(self) -> bool: ...

    def enqueue(
        self,
        *,
        hidden_bf16: object,
        global_ids_i32: object,
        routing_weights_f32: object,
        local_ids_i32: object,
        local_weights_f32: object,
        generation: object,
        stream: object,
        rank_local_bf16: object,
        rank: int,
        layer: int,
        slab: OwnerLocalMoeSlab,
    ) -> object: ...


class NativeTargetRouterPort(Protocol):
    """Raw no-fallback adapter to the authenticated target NVFP4 router."""

    @property
    def abi(self) -> str: ...

    @property
    def routing_semantics(self) -> str: ...

    @property
    def writes_generation(self) -> bool: ...

    def enqueue(
        self,
        *,
        hidden_bf16: object,
        router_logits_f32: object,
        global_ids_i32: object,
        routing_weights_f32: object,
        generation: "TargetMoeGeneration",
        stream: object,
        rank: int,
        layer: int,
        slab: OwnerLocalMoeSlab,
    ) -> None: ...


@dataclass(frozen=True)
class TargetMoeGeneration:
    """Distinct graph-owned source/requested device generation scalars."""

    source: object
    requested: object


@dataclass(frozen=True)
class TargetMoeBinding:
    """One graph-owned dense-route binding for one rank/layer/K0 bucket."""

    rank: int
    layer: int
    sequences: int
    target_slab: object
    slab: OwnerLocalMoeSlab
    native: NativeTargetMoePort
    router_logits: object
    global_expert_ids: object
    routing_weights: object
    generation: TargetMoeGeneration
    stream: object
    local_expert_ids: object
    local_routing_weights: object
    local_partial: object

    def __post_init__(self) -> None:
        rows = self.sequences
        if (
            self.rank not in (0, 1)
            or isinstance(self.layer, bool)
            or not 0 <= self.layer < 48
            or rows not in _SEQUENCE_BUCKETS
            or self.target_slab is None
            or not isinstance(self.slab, OwnerLocalMoeSlab)
            or self.slab.rank != self.rank
            or self.slab.layer != self.layer
            or self.native.router_abi != MODEL_NVFP4_ABI
            or self.native.routed_abi != MODEL_NVFP4_ABI
            or self.native.shared_abi != SHARED_ABI
            or self.native.localizes_global_ids is not True
            or self.native.generation_checked is not True
            or not _tensor(self.router_logits, (rows, GLOBAL_EXPERTS), "float32")
            or not _tensor(self.global_expert_ids, (rows, TOP_K), "int32")
            or not _tensor(self.routing_weights, (rows, TOP_K), "float32")
            or not _tensor(self.local_expert_ids, (rows, TOP_K), "int32")
            or not _tensor(self.local_routing_weights, (rows, TOP_K), "float32")
            or not _tensor(self.local_partial, (rows, HIDDEN), "bfloat16")
            or not isinstance(self.generation, TargetMoeGeneration)
            or self.generation.source is None
            or self.generation.requested is None
            or self.generation.source is self.generation.requested
            or self.stream is None
        ):
            raise TargetMoeError("invalid graph-owned target MoE binding")
        devices = {
            str(getattr(value, "device", ""))
            for value in (
                self.router_logits,
                self.global_expert_ids,
                self.routing_weights,
                self.local_expert_ids,
                self.local_routing_weights,
                self.local_partial,
            )
        }
        if len(devices) != 1:
            raise TargetMoeError("target MoE binding buffers span devices")

    def enqueue(
        self,
        hidden_bf16: object,
        global_ids_i32: object,
        routing_weights_f32: object,
        generation: TargetMoeGeneration,
        stream: object,
    ) -> object:
        """Enqueue device localization and target MoE into fixed storage."""

        if (
            global_ids_i32 is not self.global_expert_ids
            or routing_weights_f32 is not self.routing_weights
            or generation is not self.generation
            or stream is not self.stream
        ):
            raise TargetMoeError("target MoE enqueue replaced graph-owned inputs")
        return self.native.enqueue(
            hidden_bf16=hidden_bf16,
            global_ids_i32=global_ids_i32,
            routing_weights_f32=routing_weights_f32,
            local_ids_i32=self.local_expert_ids,
            local_weights_f32=self.local_routing_weights,
            generation=generation,
            stream=stream,
            rank_local_bf16=self.local_partial,
            rank=self.rank,
            layer=self.layer,
            slab=self.slab,
        )


class _Span(Protocol):
    def __enter__(self) -> "_Span": ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
    def set_attribute(self, key: str, value: str | int) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...


class OtelTracer(Protocol):
    def start_as_current_span(self, name: str) -> _Span: ...


def stable_softmax_top10_reference(
    logits: Sequence[Sequence[float]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...]]:
    """CPU parity oracle for the fixed global-ID routing contract.

    Equal logits are ordered by ascending global expert ID. The selected ten
    softmax probabilities are renormalized, matching vLLM's Qwen3Next router.
    """

    ids_out: list[tuple[int, ...]] = []
    weights_out: list[tuple[float, ...]] = []
    for row in logits:
        if len(row) != GLOBAL_EXPERTS or any(not math.isfinite(x) for x in row):
            raise TargetMoeError("router logits must be finite E512 rows")
        selected = sorted(range(GLOBAL_EXPERTS), key=lambda expert: (-row[expert], expert))[:TOP_K]
        peak = row[selected[0]]
        exponentials = [math.exp(row[expert] - peak) for expert in selected]
        denominator = sum(exponentials)
        ids_out.append(tuple(selected))
        weights_out.append(tuple(value / denominator for value in exponentials))
    if not ids_out:
        raise TargetMoeError("router logits must contain at least one row")
    return tuple(ids_out), tuple(weights_out)


class TargetMoeLayerParticipant:
    """One immutable rank/layer participant for the target-only K0 domain.

    Spans use only bounded dimensions: phase(1), outcome(2), failure.class(6),
    rank(2), layer(48), and sequence.bucket(5). Paths, hashes, pointers, route
    IDs, weights, generations, and request identities are never attributes.
    """

    execution_domain = K0_DOMAIN
    _SPAN = "rocket.qwen38.k0.target_moe"

    def __init__(
        self,
        *,
        slab: OwnerLocalMoeSlab,
        target_slab: object,
        router: NativeTargetRouterPort,
        tracer: OtelTracer,
    ) -> None:
        if not isinstance(slab, OwnerLocalMoeSlab) or slab.schema != MOE_SCHEMA:
            raise TargetMoeError("authenticated target MoE descriptor is required")
        if target_slab is None or router is None or tracer is None:
            raise TargetMoeError("target slab, native router, and tracer are required")
        if (
            router.abi != MODEL_NVFP4_ABI
            or router.routing_semantics != ROUTING_SEMANTICS
            or router.writes_generation is not True
        ):
            raise TargetMoeError("native target MoE implementation identity changed")
        if any(extent.abi != MODEL_NVFP4_ABI for extent in slab.router):
            raise TargetMoeError("target router is not the authenticated NVFP4 family")
        if any(extent.abi != MODEL_NVFP4_ABI for extent in slab.routed):
            raise TargetMoeError("target routed experts are not the authenticated NVFP4 family")
        if any(extent.abi != "native" or extent.dtype != "BF16" for extent in slab.shared):
            raise TargetMoeError("target shared expert is not the authenticated BF16 family")
        self._slab = slab
        self._target_slab = target_slab
        self._router = router
        self._tracer = tracer
        self._bindings: dict[int, tuple[object, TargetMoeBinding]] = {}
        self._faulted = False

    @property
    def rank(self) -> int:
        return self._slab.rank

    @property
    def layer(self) -> int:
        return self._slab.layer

    @property
    def identity(self) -> tuple[str, str, int, int, str]:
        return (
            self._slab.revision,
            self._slab.artifact_key,
            self.rank,
            self.layer,
            self._slab.layout_sha256,
        )

    def execute(
        self, layer: int, hidden: object, shape: MoeShape, graph: object,
    ) -> object:
        if self._faulted:
            raise TargetMoeError("faulted target MoE participant cannot replay")
        failure_class = "contract"
        with self._observed(shape) as span:
            try:
                buffers = self._validate_call(layer, hidden, shape, graph)
                failure_class = "router_launch"
                self._router.enqueue(
                    hidden_bf16=hidden,
                    router_logits_f32=buffers.router_logits,
                    global_ids_i32=buffers.global_expert_ids,
                    routing_weights_f32=buffers.routing_weights,
                    generation=buffers.generation,
                    stream=buffers.stream,
                    rank=self.rank,
                    layer=self.layer,
                    slab=self._slab,
                )
                failure_class = "moe_launch"
                output = buffers.enqueue(
                    hidden,
                    buffers.global_expert_ids,
                    buffers.routing_weights,
                    buffers.generation,
                    buffers.stream,
                )
                if output is not buffers.local_partial:
                    failure_class = "output_identity"
                    raise TargetMoeError("native target MoE returned foreign output storage")
                span.set_attribute("outcome", "success")
                span.set_attribute("failure.class", "none")
                return output
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.set_attribute("failure.class", failure_class)
                span.record_exception(exc)
                if failure_class != "contract":
                    self._faulted = True
                raise

    def _validate_call(
        self, layer: int, hidden: object, shape: MoeShape, graph: object,
    ) -> TargetMoeBinding:
        if layer != self.layer or not isinstance(shape, MoeShape) or shape.verify_width != 1:
            raise TargetMoeError("K0 target MoE rank/layer/shape identity changed")
        if (
            graph is None
            or getattr(graph, "target_slab", None) is not self._target_slab
            or getattr(graph, "mtp_slab", None) is not None
        ):
            raise TargetMoeError("K0 graph target-slab identity changed")
        m = shape.token_rows
        if not _tensor(hidden, (m, HIDDEN), "bfloat16"):
            raise TargetMoeError("K0 target MoE hidden activation changed")
        getter = getattr(graph, "target_moe_binding", None)
        if not callable(getter):
            raise TargetMoeError("K0 graph exposes no target MoE binding")
        buffers = getter(self.rank, self.layer, shape)
        self._validate_buffers(buffers, hidden, shape.sequences)
        key = id(graph)
        prior = self._bindings.get(key)
        if prior is None:
            self._bindings[key] = (graph, buffers)
        elif prior[0] is not graph or prior[1] is not buffers:
            raise TargetMoeError("K0 graph replaced fixed target MoE buffers")
        return buffers

    def _validate_buffers(
        self, buffers: object, hidden: object, expected_sequences: int,
    ) -> None:
        m = self._slab.rank  # keep rank validation visibly tied to the descriptor
        if not isinstance(buffers, TargetMoeBinding):
            raise TargetMoeError("K0 graph target MoE binding is absent")
        rows = buffers.sequences
        if (
            buffers.rank != m
            or buffers.layer != self.layer
            or rows != expected_sequences
            or buffers.target_slab is not self._target_slab
            or buffers.slab is not self._slab
            or buffers.native.router_abi != MODEL_NVFP4_ABI
            or buffers.native.routed_abi != MODEL_NVFP4_ABI
            or buffers.native.shared_abi != SHARED_ABI
            or buffers.native.localizes_global_ids is not True
            or buffers.native.generation_checked is not True
            or not _tensor(buffers.router_logits, (rows, GLOBAL_EXPERTS), "float32")
            or not _tensor(buffers.global_expert_ids, (rows, TOP_K), "int32")
            or not _tensor(buffers.routing_weights, (rows, TOP_K), "float32")
            or not _tensor(buffers.local_expert_ids, (rows, TOP_K), "int32")
            or not _tensor(buffers.local_routing_weights, (rows, TOP_K), "float32")
            or not _tensor(buffers.local_partial, (rows, HIDDEN), "bfloat16")
            or not isinstance(buffers.generation, TargetMoeGeneration)
            or buffers.generation.source is None
            or buffers.generation.requested is None
            or buffers.generation.source is buffers.generation.requested
            or buffers.stream is None
        ):
            raise TargetMoeError("K0 graph target MoE buffers changed")
        device = str(getattr(hidden, "device", ""))
        tensors = (
            buffers.router_logits,
            buffers.global_expert_ids,
            buffers.routing_weights,
            buffers.local_expert_ids,
            buffers.local_routing_weights,
            buffers.local_partial,
        )
        if any(str(getattr(value, "device", "")) != device for value in tensors):
            raise TargetMoeError("K0 target MoE buffers span devices")

    @contextmanager
    def _observed(self, shape: object):
        sequences = shape.sequences if isinstance(shape, MoeShape) else 0
        with self._tracer.start_as_current_span(self._SPAN) as span:
            span.set_attribute("phase", "execute")
            span.set_attribute("rank", self.rank)
            span.set_attribute("layer", self.layer)
            span.set_attribute(
                "sequence.bucket", sequences if sequences in _SEQUENCE_BUCKETS else 0
            )
            yield span


def _tensor(value: object, shape: tuple[int, ...], dtype: str) -> bool:
    return (
        tuple(getattr(value, "shape", ())) == shape
        and dtype in str(getattr(value, "dtype", ""))
        and getattr(value, "is_cuda", False) is True
    )


__all__ = [
    "NativeTargetMoePort",
    "NativeTargetRouterPort",
    "PINNED_VLLM_COMMIT",
    "REFERENCE_VLLM_COMMIT",
    "ROUTING_SEMANTICS",
    "SHARED_ABI",
    "TargetMoeError",
    "TargetMoeBinding",
    "TargetMoeGeneration",
    "TargetMoeLayerParticipant",
    "stable_softmax_top10_reference",
]

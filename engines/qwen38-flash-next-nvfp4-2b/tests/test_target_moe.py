# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path

from qwen38_slab.contract import MODEL_NVFP4_ABI, PINNED_CONTRACT
from qwen38_slab.k0_composition import K0_DOMAIN
from qwen38_slab.routed_moe import (
    GLOBAL_EXPERTS,
    HIDDEN,
    LOCAL_EXPERTS,
    MOE_SCHEMA,
    SLAB_ARTIFACT_KEY,
    TOP_K,
    MoeExtent,
    MoeShape,
    OwnerLocalMoeSlab,
)
from qwen38_slab.target_moe import (
    ROUTING_SEMANTICS,
    SHARED_ABI,
    TargetMoeBinding,
    TargetMoeError,
    TargetMoeGeneration,
    TargetMoeLayerParticipant,
    stable_softmax_top10_reference,
)


class Tensor:
    def __init__(self, shape, dtype, device="cuda:0"):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.is_cuda = True


class Span:
    def __init__(self): self.attributes = {}; self.exceptions = []
    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, exception): self.exceptions.append(exception)


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, _name):
        span = Span(); self.spans.append(span); return span


class Router:
    abi = MODEL_NVFP4_ABI
    routing_semantics = ROUTING_SEMANTICS
    writes_generation = True

    def __init__(self): self.calls = []; self.fail = False
    def enqueue(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail: raise RuntimeError("router failure")


class Moe:
    router_abi = MODEL_NVFP4_ABI
    routed_abi = MODEL_NVFP4_ABI
    shared_abi = SHARED_ABI
    localizes_global_ids = True
    generation_checked = True

    def __init__(self): self.calls = []; self.fail = False
    def enqueue(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail: raise RuntimeError("MoE failure")
        return kwargs["rank_local_bf16"]


def descriptor(rank=0, layer=0):
    prefix = f"model.language_model.layers.{layer}.mlp"
    router = tuple(
        MoeExtent(f"{prefix}.gate.{i}", i * 256, 256, (1,), "U8", "layout", MODEL_NVFP4_ABI)
        for i in range(3)
    )
    routed = tuple(
        MoeExtent(
            f"{prefix}.experts.{rank * LOCAL_EXPERTS + expert}.p{part}.x{leaf}",
            4096 + (expert * 12 + part * 4 + leaf) * 256,
            256, (1,), "U8", "layout", MODEL_NVFP4_ABI,
        )
        for expert in range(LOCAL_EXPERTS)
        for part in range(3)
        for leaf in range(4)
    )
    shared = tuple(
        MoeExtent(f"{prefix}.shared.{i}", 900_000 + i * 256, 256, (1,), "BF16", "checkpoint", "native")
        for i in range(4)
    )
    return OwnerLocalMoeSlab(
        MOE_SCHEMA, SLAB_ARTIFACT_KEY, PINNED_CONTRACT.revision, rank, layer,
        rank * LOCAL_EXPERTS, rank * LOCAL_EXPERTS + LOCAL_EXPERTS - 1,
        Path("rank.slab"), 1, "a" * 64, ("b" * 64,), router, routed, shared,
    )


def binding(slab, target_slab, moe, sequences=1):
    return TargetMoeBinding(
        slab.rank, slab.layer, sequences, target_slab, slab, moe,
        Tensor((sequences, GLOBAL_EXPERTS), "float32"),
        Tensor((sequences, TOP_K), "int32"),
        Tensor((sequences, TOP_K), "float32"),
        TargetMoeGeneration(object(), object()), object(),
        Tensor((sequences, TOP_K), "int32"),
        Tensor((sequences, TOP_K), "float32"),
        Tensor((sequences, HIDDEN), "bfloat16"),
    )


class Graph:
    def __init__(self, target_slab, binding, *, mtp_slab=None):
        self.target_slab = target_slab
        self.mtp_slab = mtp_slab
        self.binding = binding

    def target_moe_binding(self, rank, layer, shape):
        self.request = (rank, layer, shape)
        return self.binding


class TargetMoeTests(unittest.TestCase):
    def test_stable_softmax_top10_preserves_global_order_and_normalizes(self):
        row = [0.0] * GLOBAL_EXPERTS
        row[11] = 3.0
        row[7] = 3.0
        row[300] = 2.0
        ids, weights = stable_softmax_top10_reference((row,))
        self.assertEqual(ids[0][:3], (7, 11, 300))
        self.assertEqual(ids[0][3:], tuple(range(7)))
        self.assertAlmostEqual(sum(weights[0]), 1.0)
        self.assertTrue(all(weights[0][i] >= weights[0][i + 1] for i in range(9)))
        with self.assertRaisesRegex(TargetMoeError, "finite E512"):
            stable_softmax_top10_reference(([math.nan] * GLOBAL_EXPERTS,))

    def test_every_rank_and_layer_binds_exact_target_families(self):
        for rank in (0, 1):
            target = object()
            for layer in range(48):
                participant = TargetMoeLayerParticipant(
                    slab=descriptor(rank, layer), target_slab=target,
                    router=Router(), tracer=Tracer(),
                )
                self.assertEqual(participant.execution_domain, K0_DOMAIN)
                self.assertEqual(participant.identity[2:4], (rank, layer))

    def test_execute_uses_graph_owned_routes_and_pair_reduce_output(self):
        slab, target, router, moe, tracer = descriptor(1, 47), object(), Router(), Moe(), Tracer()
        fixed = binding(slab, target, moe, 16)
        graph = Graph(target, fixed)
        participant = TargetMoeLayerParticipant(
            slab=slab, target_slab=target, router=router, tracer=tracer,
        )
        hidden = Tensor((16, HIDDEN), "bfloat16")
        output = participant.execute(47, hidden, MoeShape(16), graph)
        self.assertIs(output, fixed.local_partial)
        self.assertEqual(graph.request, (1, 47, MoeShape(16)))
        self.assertIs(router.calls[0]["global_ids_i32"], fixed.global_expert_ids)
        self.assertIs(moe.calls[0]["local_ids_i32"], fixed.local_expert_ids)
        self.assertIs(moe.calls[0]["generation"], fixed.generation)
        self.assertEqual(
            tracer.spans[-1].attributes,
            {"phase": "execute", "rank": 1, "layer": 47,
             "sequence.bucket": 16, "outcome": "success", "failure.class": "none"},
        )

    def test_replaced_binding_and_mtp_graph_fail_before_native_launch(self):
        slab, target, router, moe = descriptor(), object(), Router(), Moe()
        fixed = binding(slab, target, moe)
        graph = Graph(target, fixed)
        participant = TargetMoeLayerParticipant(
            slab=slab, target_slab=target, router=router, tracer=Tracer(),
        )
        hidden = Tensor((1, HIDDEN), "bfloat16")
        participant.execute(0, hidden, MoeShape(1), graph)
        graph.binding = binding(slab, target, moe)
        with self.assertRaisesRegex(TargetMoeError, "replaced fixed"):
            participant.execute(0, hidden, MoeShape(1), graph)
        self.assertEqual(len(router.calls), 1)
        other = TargetMoeLayerParticipant(
            slab=slab, target_slab=target, router=Router(), tracer=Tracer(),
        )
        with self.assertRaisesRegex(TargetMoeError, "target-slab identity"):
            other.execute(0, hidden, MoeShape(1), Graph(target, fixed, mtp_slab=object()))

    def test_launch_failure_is_terminal_and_telemetry_dimensions_are_bounded(self):
        slab, target, router, moe, tracer = descriptor(), object(), Router(), Moe(), Tracer()
        moe.fail = True
        participant = TargetMoeLayerParticipant(
            slab=slab, target_slab=target, router=router, tracer=tracer,
        )
        with self.assertRaisesRegex(RuntimeError, "MoE failure"):
            participant.execute(
                0, Tensor((1, HIDDEN), "bfloat16"), MoeShape(1),
                Graph(target, binding(slab, target, moe)),
            )
        self.assertEqual(
            tracer.spans[-1].attributes,
            {"phase": "execute", "rank": 0, "layer": 0,
             "sequence.bucket": 1, "outcome": "failure", "failure.class": "moe_launch"},
        )
        with self.assertRaisesRegex(TargetMoeError, "faulted"):
            participant.execute(0, Tensor((1, HIDDEN), "bfloat16"), MoeShape(1), object())

    def test_mtp_fp8_or_foreign_output_cannot_masquerade_as_target(self):
        slab, target = descriptor(), object()
        with self.assertRaisesRegex(TargetMoeError, "implementation identity"):
            TargetMoeLayerParticipant(
                slab=slab, target_slab=target,
                router=replace_router_abi(Router(), "fp8"), tracer=Tracer(),
            )
        moe = Moe()
        moe.routed_abi = "fp8"
        with self.assertRaisesRegex(TargetMoeError, "invalid graph-owned"):
            binding(slab, target, moe)


def replace_router_abi(router, abi):
    router.abi = abi
    return router


if __name__ == "__main__":
    unittest.main()

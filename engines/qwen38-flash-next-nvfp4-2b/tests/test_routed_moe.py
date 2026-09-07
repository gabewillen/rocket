# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from qwen38_slab.routed_moe import (
    BUCKET_SELECTORS,
    FLASHINFER_COMMIT,
    HIDDEN,
    LAZY_VERIFIER_ROWS,
    LOCAL_EXPERTS,
    MOE_SCHEMA,
    RESIDENT_K0_ROWS,
    SHARED_INTERMEDIATE,
    SLAB_ARTIFACT_KEY,
    TOP_K,
    MoeExtent,
    MoeShape,
    OwnerLocalMoeSlab,
    RoutedMoeGraph,
    RoutedMoeGraphError,
    compact_owner_pairs,
    shared_intermediate_bounds,
    selector_for,
    validate_external_mtp_nvfp4,
)
from qwen38_slab.contract import PINNED_CONTRACT
from qwen38_slab.mtp_source import ExternalMtpSource, MTP_NONEXPERT_TENSORS


class Tensor:
    def __init__(self, shape, dtype, device="cuda:0"):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.is_cuda = True


class Span:
    def __init__(self): self.attributes = {}
    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, _exception): pass


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, _name):
        span = Span(); self.spans.append(span); return span


class Backend:
    def __init__(self): self.calls = []; self.fail = False
    def has_bucket(self, shape, selector): return selector_for(shape) == selector
    def launch(self, hidden, ids, weights, *, rank, shape, selector):
        self.calls.append((rank, shape, selector, hidden, ids, weights))
        if self.fail: raise RuntimeError("injected backend failure")
        return Tensor((shape.token_rows, HIDDEN), "bfloat16")


def descriptor(rank=0, layer=0):
    prefix = f"model.language_model.layers.{layer}.mlp"
    routed = tuple(
        MoeExtent(
            f"{prefix}.experts.{rank * LOCAL_EXPERTS + expert}.p{part}.x{leaf}",
            (expert * 12 + part * 4 + leaf) * 256, 256, (1,), "U8", "layout", "abi",
        )
        for expert in range(LOCAL_EXPERTS)
        for part in range(3)
        for leaf in range(4)
    )
    shared = tuple(
        MoeExtent(f"{prefix}.shared.{index}", 900_000 + index * 256, 256,
                  (1,), "BF16", "checkpoint", "native")
        for index in range(4)
    )
    return OwnerLocalMoeSlab(
        MOE_SCHEMA, SLAB_ARTIFACT_KEY, PINNED_CONTRACT.revision, rank, layer,
        rank * LOCAL_EXPERTS, rank * LOCAL_EXPERTS + LOCAL_EXPERTS - 1,
        Path("rank.slab"), 1, "a" * 64, ("b" * 64,), routed, shared,
    )


class RoutedMoeGraphTests(unittest.TestCase):
    def test_external_mtp_candidate_requires_identical_nonexperts(self):
        hashes = {name: "a" * 64 for name in MTP_NONEXPERT_TENSORS}
        source = ExternalMtpSource(
            "b" * 64, PINNED_CONTRACT.revision, 0,
            "modelopt_nvfp4_group16_cutlass_sm121_sfb", (0, 255),
            "c" * 64, hashes, "d" * 64,
        )
        self.assertIsNone(validate_external_mtp_nvfp4(source))
        with self.assertRaisesRegex(RoutedMoeGraphError, "source health"):
            validate_external_mtp_nvfp4(replace(source, nonexpert_differences=("mtp.fc_hidden.weight",)))

    def test_production_graph_exists_for_every_immutable_bucket(self):
        self.assertEqual(
            BUCKET_SELECTORS,
            {1: "static", 2: "static", 4: "static_tail", 8: "dynamic", 16: "dynamic"},
        )
        self.assertTrue(callable(RoutedMoeGraph))
        self.assertTrue(issubclass(RoutedMoeGraphError, RuntimeError))
        self.assertEqual(FLASHINFER_COMMIT[:8], "91bda04c")

    def test_k4_c16_is_explicit_dynamic_80_row_shape(self):
        self.assertEqual(RESIDENT_K0_ROWS, (1, 2, 4, 8, 16))
        self.assertEqual(LAZY_VERIFIER_ROWS, (32, 64, 80, 128))
        expected = {2: 32, 5: 80, 8: 128}
        for width, rows in expected.items():
            with self.subTest(width=width):
                shape = MoeShape(16, width)
                self.assertEqual(shape.token_rows, rows)
                self.assertEqual(selector_for(shape), "dynamic")

    def test_shared_expert_is_disjoint_n160_shard_not_replicated_half(self):
        self.assertEqual(shared_intermediate_bounds(0), (0, 160))
        self.assertEqual(shared_intermediate_bounds(1), (160, 320))
        covered = set(range(*shared_intermediate_bounds(0))) | set(
            range(*shared_intermediate_bounds(1))
        )
        overlap = set(range(*shared_intermediate_bounds(0))) & set(
            range(*shared_intermediate_bounds(1))
        )
        self.assertEqual(covered, set(range(SHARED_INTERMEDIATE)))
        self.assertEqual(overlap, set())
        with self.assertRaisesRegex(RoutedMoeGraphError, "rank"):
            shared_intermediate_bounds(2)

    def test_layer47_is_bound_and_cross_layer_extent_reuse_fails(self):
        backend, tracer = Backend(), Tracer()
        graph = RoutedMoeGraph(descriptor(layer=47), backend, tracer)
        self.assertEqual(graph.identity[2], 47)
        with self.assertRaisesRegex(RoutedMoeGraphError, "authenticated"):
            RoutedMoeGraph(replace(descriptor(layer=0), layer=47), backend, tracer)

    def test_rank_local_launch_returns_bf16_and_emits_bounded_dimensions(self):
        backend, tracer = Backend(), Tracer()
        graph = RoutedMoeGraph(descriptor(rank=1, layer=47), backend, tracer)
        shape = MoeShape(16, 5)
        hidden = Tensor((80, HIDDEN), "bfloat16")
        ids = Tensor((80, TOP_K), "int32")
        weights = Tensor((80, TOP_K), "float32")
        output = graph.launch(hidden, ids, weights, shape, request_id="request")
        self.assertEqual(output.shape, (80, HIDDEN))
        self.assertEqual(backend.calls[0][:3], (1, shape, "dynamic"))
        self.assertEqual(
            tracer.spans[-1].attributes,
            {"rank": 1, "m_bucket": 80, "selector": "dynamic",
             "request_id": "request", "outcome": "success"},
        )

    def test_global_pairs_compact_to_each_owner_with_invalid_zero_weight(self):
        ids = ((0, 255, 256, 511, 17, 300, 2, 400, 9, 500),)
        weights = ((0.1,) * TOP_K,)
        ids0, weights0 = compact_owner_pairs(ids, weights, 0)
        ids1, weights1 = compact_owner_pairs(ids, weights, 1)
        self.assertEqual(ids0[0][:4], (0, 255, 0, 0))
        self.assertEqual(weights0[0][:4], (0.1, 0.1, 0.0, 0.0))
        self.assertEqual(ids1[0][:4], (0, 0, 0, 255))
        self.assertEqual(weights1[0][:4], (0.0, 0.0, 0.1, 0.1))

    def test_layout_mismatch_fails_before_backend_and_fault_is_terminal(self):
        backend, tracer = Backend(), Tracer()
        graph = RoutedMoeGraph(descriptor(), backend, tracer)
        with self.assertRaisesRegex(RoutedMoeGraphError, "tensor contract"):
            graph.launch(
                Tensor((1, HIDDEN), "bfloat16"),
                Tensor((1, TOP_K), "int64"),
                Tensor((1, TOP_K), "float32"), MoeShape(1),
            )
        self.assertEqual(backend.calls, [])
        backend.fail = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            graph.launch(
                Tensor((1, HIDDEN), "bfloat16"), Tensor((1, TOP_K), "int32"),
                Tensor((1, TOP_K), "float32"), MoeShape(1),
            )
        with self.assertRaisesRegex(RoutedMoeGraphError, "faulted"):
            graph.launch(
                Tensor((1, HIDDEN), "bfloat16"), Tensor((1, TOP_K), "int32"),
                Tensor((1, TOP_K), "float32"), MoeShape(1),
            )


if __name__ == "__main__":
    unittest.main()

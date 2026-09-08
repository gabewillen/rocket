# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from pathlib import Path

from qwen38_slab.contract import MODEL_NVFP4_ABI, PINNED_CONTRACT
from qwen38_slab.routed_moe import (
    HIDDEN,
    LOCAL_EXPERTS,
    MOE_SCHEMA,
    SLAB_ARTIFACT_KEY,
    TOP_K,
    MoeExtent,
    MoeShape,
    OwnerLocalMoeSlab,
    RoutedMoeGraphError,
)
from qwen38_slab.target_moe_validation import (
    EAGER_VALIDATION_IDENTITY,
    EagerValidationTargetMoeAdapter,
    compare_pair_reduced_partial,
)


class Tensor:
    def __init__(self, shape, dtype, device="cuda:0"):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.is_cuda = True


class Array:
    def __init__(self, values):
        self.values = tuple(tuple(float(value) for value in row) for row in values)
        self.shape = (len(self.values), len(self.values[0]))
    def float(self): return self
    def __add__(self, other):
        return Array([[a + b for a, b in zip(x, y)] for x, y in zip(self.values, other.values)])
    def __sub__(self, other):
        return Array([[a - b for a, b in zip(x, y)] for x, y in zip(self.values, other.values)])
    def __truediv__(self, other):
        return Array([[a / b for a, b in zip(x, y)] for x, y in zip(self.values, other.values)])
    def abs(self): return Array([[abs(value) for value in row] for row in self.values])
    def clamp_min(self, minimum):
        return Array([[max(value, minimum) for value in row] for row in self.values])
    def max(self): return Scalar(max(max(row) for row in self.values))


class Scalar:
    def __init__(self, value): self.value = value
    def item(self): return self.value


class TorchApi:
    @staticmethod
    def allclose(actual, expected, *, atol, rtol):
        return all(
            abs(a - b) <= atol + rtol * abs(b)
            for x, y in zip(actual.values, expected.values)
            for a, b in zip(x, y)
        )


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


class Backend:
    implementation_identity = EAGER_VALIDATION_IDENTITY
    def __init__(self): self.calls = []
    def launch(self, hidden, ids, weights, **kwargs):
        self.calls.append((hidden, ids, weights, kwargs))
        return Tensor((kwargs["shape"].token_rows, HIDDEN), "bfloat16")


def descriptor():
    prefix = "model.language_model.layers.47.mlp"
    router = tuple(
        MoeExtent(f"{prefix}.gate.{i}", i * 256, 256, (1,), "U8", "layout", MODEL_NVFP4_ABI)
        for i in range(3)
    )
    routed = tuple(
        MoeExtent(
            f"{prefix}.experts.{expert}.p{part}.x{leaf}",
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
        MOE_SCHEMA, SLAB_ARTIFACT_KEY, PINNED_CONTRACT.revision, 0, 47, 0,
        255, Path("rank.slab"), 1, "a" * 64, ("b" * 64,),
        router, routed, shared,
    )


class EagerValidationTests(unittest.TestCase):
    def test_real_backend_identity_is_required_and_adapter_is_never_production(self):
        with self.assertRaisesRegex(RoutedMoeGraphError, "identity changed"):
            EagerValidationTargetMoeAdapter(descriptor(), object(), Tracer())
        adapter = EagerValidationTargetMoeAdapter(descriptor(), Backend(), Tracer())
        self.assertTrue(adapter.validation_only)
        self.assertFalse(adapter.production_eligible)
        self.assertFalse(adapter.graph_safe)
        self.assertFalse(hasattr(adapter, "execution_domain"))

    def test_eager_adapter_returns_backend_rank_local_partial(self):
        backend, tracer = Backend(), Tracer()
        adapter = EagerValidationTargetMoeAdapter(descriptor(), backend, tracer)
        shape = MoeShape(16)
        hidden = Tensor((16, HIDDEN), "bfloat16")
        ids = Tensor((16, TOP_K), "int32")
        weights = Tensor((16, TOP_K), "float32")
        output = adapter.execute(hidden, ids, weights, shape)
        self.assertEqual(output.shape, (16, HIDDEN))
        self.assertEqual(backend.calls[0][3]["rank"], 0)
        self.assertEqual(backend.calls[0][3]["selector"], "dynamic")
        self.assertEqual(
            tracer.spans[-1].attributes,
            {"phase": "execute", "rank": 0, "layer": 47,
             "sequence.bucket": 16, "outcome": "success"},
        )

    def test_factory_source_is_exact_authenticated_b12x_path(self):
        source = Path(
            EagerValidationTargetMoeAdapter.from_authenticated_slab.__code__.co_filename
        ).read_text()
        self.assertIn("materialize_flashinfer_weights", source)
        self.assertIn("FlashInferRoutedMoeBackend(weights", source)
        self.assertNotIn("execution_domain =", source)

    def test_pair_reduce_parity_evidence_uses_explicit_tolerances(self):
        rank0 = Array([[1.0, 2.0] + [0.0] * (HIDDEN - 2)])
        rank1 = Array([[3.0, 4.0] + [0.0] * (HIDDEN - 2)])
        reference = Array([[4.0, 6.0] + [0.0] * (HIDDEN - 2)])
        result = compare_pair_reduced_partial(
            rank0, rank1, reference, atol=0.0, rtol=0.0, torch_api=TorchApi
        )
        self.assertTrue(result.accepted)
        self.assertEqual((result.rows, result.max_abs, result.max_rel), (1, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()

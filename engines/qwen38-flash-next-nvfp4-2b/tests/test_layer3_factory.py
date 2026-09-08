# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen38_slab.layer3_factory import (
    Layer3FactoryError,
    prepare_layer3_physical_plan,
    public_plan,
)

ARTIFACT = Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/"
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
)
SIDECAR = Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/qsa-indexer-sidecars/"
    "bdbebd4f45c398f090a41ab98cd3881b969d958d8ae0bc42f3411844d3262edd"
)
ORACLE = Path("/home/glwillen/calibration/qwen38-k0-oracle-a1794d5-01/capture")


class Span:
    def __init__(self):
        self.attributes = {}
        self.exception = None
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, exception): self.exception = exception


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, name):
        span = Span(); self.spans.append(span); return span


class Layer3FactoryTests(unittest.TestCase):
    def test_missing_oracle_fails_closed_with_bounded_telemetry(self):
        tracer = Tracer()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(Layer3FactoryError, "oracle manifest"):
                prepare_layer3_physical_plan(
                    artifact=ARTIFACT, indexer_sidecar=SIDECAR,
                    oracle_capture=Path(directory), tracer=tracer,
                )
        self.assertEqual(tracer.spans[-1].attributes, {
            "phase": "prepare", "outcome": "failure",
            "failure.class": "contract",
        })

    @unittest.skipUnless(
        ARTIFACT.is_dir() and SIDECAR.is_dir() and ORACLE.is_dir(),
        "authenticated layer-3 artifacts are unavailable",
    )
    def test_real_two_rank_plan_authenticates_replicated_prefill_boundary(self):
        tracer = Tracer()
        plan = prepare_layer3_physical_plan(
            artifact=ARTIFACT, indexer_sidecar=SIDECAR,
            oracle_capture=ORACLE, tracer=tracer,
        )
        record = dict(public_plan(plan))
        self.assertEqual(record["ranks"], [0, 1])
        self.assertEqual(record["replicated_hc_shape"], [35, 10240])
        self.assertEqual(record["replay"], "sequential_rows_0_34")
        self.assertEqual(record["compare_row"], 34)
        self.assertEqual(record["pair_reduce"]["calls_per_rank"], 70)
        self.assertEqual(record["pair_reduce"]["session_sha256"],
                         "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b")
        self.assertEqual(tuple(item.rank for item in plan.ranks), (0, 1))
        self.assertEqual(tracer.spans[-1].attributes, {
            "phase": "prepare", "outcome": "success",
            "failure.class": "none",
        })


if __name__ == "__main__":
    unittest.main()

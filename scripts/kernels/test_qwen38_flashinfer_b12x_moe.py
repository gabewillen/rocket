#!/usr/bin/env python3
"""CPU contract tests for the exact-shape FlashInfer b12x benchmark."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("qwen38-flashinfer-b12x-moe.py")
SPEC = importlib.util.spec_from_file_location("qwen38_flashinfer_b12x_moe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class B12xReferenceContractTest(unittest.TestCase):
    def test_dynamic_barrier_telemetry_is_bounded(self):
        self.assertEqual(
            BENCH.DYNAMIC_BARRIER_PHASES,
            (
                "post_init",
                "post_histogram",
                "post_prefix",
                "post_producer",
                "post_publish",
            ),
        )
        self.assertNotIn("static_tail", BENCH.BACKENDS)
        self.assertIn("static_tail", BENCH.DIAGNOSTIC_BACKENDS)

    def test_geometry_and_modelopt_bytes_are_exact(self):
        self.assertEqual(
            (BENCH.HIDDEN, BENCH.INTERMEDIATE, BENCH.EXPERTS, BENCH.TOP_K),
            (2560, 640, 512, 10),
        )
        sizes = BENCH.modelopt_storage_bytes()
        self.assertEqual(sizes["w1_weight"], 838_860_800)
        self.assertEqual(sizes["w2_weight"], 419_430_400)
        self.assertEqual(sizes["w1_weight_scale"], 104_857_600)
        self.assertEqual(sizes["w2_weight_scale"], 52_428_800)
        self.assertEqual(sizes["runtime_scalars"], 8192)
        self.assertEqual(sizes["total"], 1_415_585_792)

    def test_natural_dispatch_matches_pinned_cutovers(self):
        self.assertEqual(
            {m: BENCH.natural_backend(m) for m in BENCH.BUCKETS},
            {1: "micro", 2: "micro", 4: "micro", 8: "static", 16: "static"},
        )

    def test_measured_n640_dispatch_selects_tail_only_at_c4(self):
        self.assertEqual(
            {m: BENCH.selected_n640_backend(m) for m in (4, 8, 16)},
            {4: "static_tail", 8: "dynamic", 16: "dynamic"},
        )
        for tokens in (1, 2, 3, True, 4.0):
            with self.assertRaisesRegex(ValueError, "overlay bucket"):
                BENCH.selected_n640_backend(tokens)
        tail_n, tail_bytes = BENCH.kernel_touched_bytes(4, "static_tail")
        self.assertEqual(tail_n, 640)
        self.assertEqual(tail_bytes, BENCH.touched_bytes(4))

    def test_forced_backend_contract_fails_closed(self):
        for tokens in BENCH.BUCKETS:
            eligible, reason = BENCH.backend_eligibility("direct_micro", tokens)
            self.assertFalse(eligible)
            self.assertIn("640", reason)
        self.assertTrue(BENCH.backend_eligibility("micro", 8)[0])
        self.assertFalse(BENCH.backend_eligibility("micro", 16)[0])
        for tokens in BENCH.BUCKETS:
            self.assertTrue(BENCH.backend_eligibility("static", tokens)[0])
            self.assertTrue(BENCH.backend_eligibility("dynamic", tokens)[0])

    def test_touched_bytes_count_unique_routes(self):
        per_expert = BENCH.modelopt_storage_bytes(1)["total"]
        expected = 160 * per_expert + 16 * 2560 * 4 + 16 * 10 * 8
        self.assertEqual(BENCH.touched_bytes(16), expected)
        static_n, static_bytes = BENCH.kernel_touched_bytes(16, "static")
        dynamic_n, dynamic_bytes = BENCH.kernel_touched_bytes(16, "dynamic")
        self.assertEqual(static_n, 768)
        self.assertEqual(dynamic_n, 640)
        self.assertGreater(static_bytes, dynamic_bytes)
        self.assertEqual(dynamic_bytes, expected)
        with self.assertRaisesRegex(ValueError, "bucket"):
            BENCH.touched_bytes(3)

    def test_component_contract_matches_rank_slab_abi(self):
        self.assertEqual(
            BENCH._component_contract("gate_proj", "weight"),
            ("U8", (640, 1280), 819_200, "checkpoint"),
        )
        self.assertEqual(
            BENCH._component_contract("down_proj", "weight_scale"),
            ("F8_E4M3", (102_400,), 102_400, "cutlass_sm121_sfb"),
        )
        self.assertEqual(
            BENCH._component_contract("up_proj", "input_scale"),
            ("F32", (), 4, "checkpoint"),
        )
        with self.assertRaisesRegex(ValueError, "projection"):
            BENCH._component_contract("other", "weight")

    def test_unknown_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "bucket"):
            BENCH.natural_backend(3)
        with self.assertRaisesRegex(ValueError, "unknown"):
            BENCH.backend_eligibility("other", 1)
        for bad in (True, 0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                BENCH.modelopt_storage_bytes(bad)


if __name__ == "__main__":
    unittest.main()

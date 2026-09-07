#!/usr/bin/env python3
"""Cold-start terminal-boundary tests for the Qwen CUDA slab harness."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("qwen38-cuda-slab-load.py")
SPEC = importlib.util.spec_from_file_location("qwen38_cuda_slab_load", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
loader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = loader
SPEC.loader.exec_module(loader)


class TotalColdStartContractTests(unittest.TestCase):
    def test_pointer_publication_cannot_count_as_total_ready(self):
        status = loader.total_cold_start_status(100, {}, None)
        self.assertEqual(status["status"], "incomplete_engine")
        self.assertIsNone(status["total_cold_load_ns"])
        self.assertFalse(status["comparison_scope_match"])
        self.assertEqual(
            status["missing_evidence"], loader.TOTAL_READY_STAGES + ("first_token_id",)
        )

    def test_first_token_closes_total_only_after_every_required_stage(self):
        stages = {
            stage: 200 + index * 10
            for index, stage in enumerate(loader.TOTAL_READY_STAGES)
        }
        status = loader.total_cold_start_status(100, stages, 17)
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["first_token_id"], 17)
        self.assertEqual(status["total_cold_load_ns"], stages["first_token_generated"] - 100)
        self.assertTrue(status["comparison_scope_match"])

    def test_missing_token_or_nonterminal_token_fails_closed(self):
        stages = {
            stage: 200 + index * 10
            for index, stage in enumerate(loader.TOTAL_READY_STAGES)
        }
        self.assertEqual(
            loader.total_cold_start_status(100, stages, None)["status"],
            "incomplete_engine",
        )
        stages["api_ready"] = stages["first_token_generated"] + 1
        with self.assertRaisesRegex(ValueError, "terminal"):
            loader.total_cold_start_status(100, stages, 17)

    def test_unknown_stage_and_invalid_clock_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "evidence"):
            loader.total_cold_start_status(100, {"pointer_published": 101}, None)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            loader.total_cold_start_status(100, {"weights_loaded": 99}, None)


if __name__ == "__main__":
    unittest.main()

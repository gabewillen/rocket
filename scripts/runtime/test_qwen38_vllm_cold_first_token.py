#!/usr/bin/env python3
"""Tests for the vLLM exec-to-first-generated-token control."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("qwen38-vllm-cold-first-token.py")
SPEC = importlib.util.spec_from_file_location("qwen38_vllm_cold", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cold = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cold
SPEC.loader.exec_module(cold)


class ColdFirstTokenTests(unittest.TestCase):
    def test_role_and_health_do_not_close_first_token_boundary(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
            b'data: {"choices":[{"delta":{"content":""}}]}\n',
            b'data: {"choices":[{"delta":{"reasoning_content":"7"}}]}\n',
            b"data: [DONE]\n",
        ]
        self.assertEqual(cold.first_generated_token(lines), ("reasoning_content", "7"))

    def test_stream_without_generated_token_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "without a generated token"):
            cold.first_generated_token(
                [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n']
            )

    def test_hardware_summary_requires_both_ranks_in_timed_window(self) -> None:
        rows = []
        for rank in (0, 1):
            rows.append(
                {
                    "rank": rank,
                    "started_unix_ns": 110,
                    "finished_unix_ns": 120,
                    "pair_started_unix_ns": 105,
                    "pair_finished_unix_ns": 125,
                    "clock_mhz": 2400 + rank,
                    "power_w": 40 + rank,
                    "utilization_percent": 90 + rank,
                }
            )
        summary = cold.hardware_summary(rows, 100, 130)
        self.assertEqual([row["rank"] for row in summary["ranks"]], [0, 1])
        self.assertEqual(summary["max_pair_span_ms"], 0.00002)
        with self.assertRaisesRegex(RuntimeError, "no rank 1"):
            cold.hardware_summary(rows[:1], 100, 130)

    def test_variance_is_sample_variance_and_rejects_one_run(self) -> None:
        fields = {
            "exec_to_model_ready_seconds": (10.0, 14.0),
            "model_ready_to_first_token_seconds": (1.0, 3.0),
            "exec_to_first_token_seconds": (11.0, 17.0),
        }
        runs = [
            {name: values[index] for name, values in fields.items()}
            for index in range(2)
        ]
        summary = cold.variance_summary(runs)
        self.assertEqual(summary["runs"], 2)
        self.assertAlmostEqual(
            summary["exec_to_model_ready_seconds"]["sample_stdev"], 2.8284271247
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            cold.variance_summary(runs[:1])

    def test_result_schema_names_exec_and_first_token_boundaries(self) -> None:
        self.assertEqual(cold.SCHEMA, "rocket.qwen38.vllm-cold-first-token.v1")
        payload = json.dumps(
            {
                "start": "service_exec_started_monotonic_ns",
                "terminal": "first_token_monotonic_ns",
            }
        )
        self.assertIn("first_token_monotonic_ns", payload)

    def test_prepared_contract_requires_k1_production_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("run.json").write_text(json.dumps({"mtp_depth": 1}))
            for name in ("launch-head.sh", "launch-worker.sh"):
                root.joinpath(name).write_text("docker run --gpus all image --mtp K1\n")
            self.assertEqual(cold.validate_production_prepared(root)["mtp_depth"], 1)
            root.joinpath("launch-head.sh").write_text("docker run --enforce-eager\n")
            with self.assertRaisesRegex(ValueError, "not a production launch"):
                cold.validate_production_prepared(root)
            root.joinpath("launch-head.sh").write_text("docker run image\n")
            root.joinpath("run.json").write_text(json.dumps({"mtp_depth": 7}))
            with self.assertRaisesRegex(ValueError, "K1 ceiling"):
                cold.validate_production_prepared(root)


if __name__ == "__main__":
    unittest.main()

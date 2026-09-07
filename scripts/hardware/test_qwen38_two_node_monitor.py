#!/usr/bin/env python3
"""Focused tests for synchronized two-node GPU evidence."""

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("qwen38-two-node-monitor.py")
SPEC = importlib.util.spec_from_file_location("qwen38_hardware", SCRIPT)
MONITOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MONITOR
SPEC.loader.exec_module(MONITOR)


class HardwareMonitorTest(unittest.TestCase):
    def test_query_parses_bounded_fields_and_fails_closed(self):
        def good(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, "2483, 89.5, 95\n", "")

        row = MONITOR.query(["nvidia-smi"], 0, good)
        self.assertEqual((row["rank"], row["clock_mhz"], row["power_w"],
                          row["utilization_percent"]), (0, 2483.0, 89.5, 95.0))

        def bad(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 1, "", "unavailable")

        with self.assertRaisesRegex(RuntimeError, "rank 1 nvidia-smi failed"):
            MONITOR.query(["ssh"], 1, bad)

        def timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(["ssh"], 10)

        with self.assertRaisesRegex(RuntimeError, "rank 1 nvidia-smi timed out"):
            MONITOR.query(["ssh"], 1, timeout)

    def test_summary_has_case_boundaries_and_rank_distributions(self):
        samples = [
            {"rank": rank, "started_unix_ns": tick, "finished_unix_ns": tick + 1,
             "pair_started_unix_ns": tick, "pair_finished_unix_ns": tick + 2,
             "clock_mhz": 2400 + tick, "power_w": 80 + tick,
             "utilization_percent": 90 + tick}
            for tick in (10, 20) for rank in (0, 1)
        ]
        benchmark = [{"concurrency": 16, "started_at": "2026-09-07T00:00:00Z",
                      "finished_at": "2026-09-07T00:00:01Z",
                      "started_unix_ns": 5, "finished_unix_ns": 25}]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            sample_path, benchmark_path = root / "samples.jsonl", root / "throughput.json"
            sample_path.write_text("".join(json.dumps(row) + "\n" for row in samples))
            benchmark_path.write_text(json.dumps(benchmark))
            output = MONITOR.summarize(sample_path, benchmark_path)
        self.assertEqual(output["schema"], "rocket.qwen38.two-node-hardware.v1")
        self.assertEqual(output["max_pair_span_ms"], 0.000002)
        self.assertEqual([rank["rank"] for rank in output["cases"][0]["ranks"]], [0, 1])
        self.assertEqual(output["cases"][0]["ranks"][0]["clock_mhz"],
                         {"median": 2415.0, "max": 2420})
        self.assertIn("started_at", output["cases"][0])

    def test_summary_rejects_missing_rank_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            samples, benchmark = root / "samples", root / "benchmark"
            samples.write_text(json.dumps({"rank": 0, "started_unix_ns": 1,
                                           "finished_unix_ns": 2, "clock_mhz": 1,
                                           "pair_started_unix_ns": 1,
                                           "pair_finished_unix_ns": 2,
                                           "power_w": 1, "utilization_percent": 1}) + "\n")
            benchmark.write_text(json.dumps([{"concurrency": 1, "started_at": "a",
                                              "finished_at": "b", "started_unix_ns": 1,
                                              "finished_unix_ns": 3}]))
            with self.assertRaisesRegex(ValueError, "no rank 1 samples"):
                MONITOR.summarize(samples, benchmark)


if __name__ == "__main__":
    unittest.main()

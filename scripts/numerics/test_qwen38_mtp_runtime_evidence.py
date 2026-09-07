#!/usr/bin/env python3
"""Focused offline tests for Qwen3.8 MTP runtime log evidence."""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("qwen38-mtp-runtime-evidence.py")


def metric_line(timestamp, accepted=47, drafted=60):
    rate = accepted / drafted * 100
    return (
        f"{timestamp} (APIServer pid=1) INFO metrics.py:120] "
        "SpecDecoding metrics: Mean acceptance length: 3.35, "
        "Accepted throughput: 0.81 tokens/s, Drafted throughput: 1.04 tokens/s, "
        f"Accepted: {accepted} tokens, Drafted: {drafted} tokens, "
        "Per-position acceptance rate: 0.950, 0.750, 0.650, "
        f"Avg Draft acceptance rate: {rate:.1f}%\n"
    )


class MtpRuntimeEvidenceTest(unittest.TestCase):
    def run_script(self, text, *extra):
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / "head-workload.log"
            log.write_text(text)
            return subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--log",
                    str(log),
                    "--not-before",
                    "2026-09-07T03:04:00Z",
                    *extra,
                ],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_repeated_fresh_metrics_prove_mtp(self):
        result = self.run_script(
            metric_line("2026-09-07T03:04:45.747222197Z")
            + metric_line("2026-09-07T03:04:55.748826888Z", 87, 207),
            "--min-records",
            "2",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema"], "rocket.qwen38.mtp-runtime-evidence.v1")
        self.assertEqual(payload["verdict"], "proven")
        self.assertEqual(payload["evidence"]["records"], 2)
        self.assertEqual(payload["totals"]["accepted_tokens"], 134)
        self.assertEqual(payload["totals"]["drafted_tokens"], 267)
        self.assertEqual(len(payload["input_sha256"]), 64)

    def test_absent_metrics_are_rejected(self):
        result = self.run_script("ordinary workload log line\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no SpecDecoding metrics", result.stderr)

    def test_malformed_metric_line_is_rejected(self):
        result = self.run_script(
            "2026-09-07T03:04:45Z SpecDecoding metrics: Accepted: nope\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("malformed SpecDecoding metrics", result.stderr)

    def test_stale_only_metrics_are_rejected(self):
        result = self.run_script(
            metric_line("2026-09-07T03:03:45.747222197Z")
            + metric_line("2026-09-07T03:03:55.748826888Z")
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stale", result.stderr)

    def test_stale_metrics_are_excluded_from_fresh_summary(self):
        result = self.run_script(
            metric_line("2026-09-07T03:03:55.748826888Z", 59, 60)
            + metric_line("2026-09-07T03:04:45.747222197Z", 47, 60)
            + metric_line("2026-09-07T03:04:55.748826888Z", 87, 207)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["evidence"]["records"], 2)
        self.assertEqual(payload["evidence"]["stale_records_ignored"], 1)
        self.assertEqual(payload["totals"]["accepted_tokens"], 134)

    def test_too_few_fresh_records_are_rejected(self):
        result = self.run_script(
            metric_line("2026-09-07T03:04:45.747222197Z"),
            "--min-records",
            "2",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("only 1 fresh", result.stderr)

    def test_invalid_counter_relationship_is_rejected(self):
        result = self.run_script(
            metric_line("2026-09-07T03:04:45.747222197Z", 61, 60)
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("accepted tokens exceed drafted tokens", result.stderr)


if __name__ == "__main__":
    unittest.main()

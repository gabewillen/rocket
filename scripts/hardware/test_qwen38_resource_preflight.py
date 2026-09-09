#!/usr/bin/env python3
"""Focused tests for the read-only Qwen3.8 resource preflight."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).with_name("qwen38-resource-preflight.py")
SPEC = importlib.util.spec_from_file_location("qwen38_resource_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fake_runner(outputs: list[subprocess.CompletedProcess[str]]):
    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(command)
        if not outputs:
            raise AssertionError("unexpected command")
        return outputs.pop(0)

    runner.calls = calls
    return runner


def completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class ResourcePreflightTest(unittest.TestCase):
    def run_guard(self, runner, **overrides):
        options = dict(
            min_mem_available_bytes=16 * MODULE.BYTES_PER_GIB,
            min_swap_free_bytes=4 * MODULE.BYTES_PER_GIB,
            expected_gpu_count=1,
            container_names=("rocket-capture",),
            meminfo_reader=lambda: (
                "MemAvailable: 33554432 kB\nSwapFree: 8388608 kB\n"
            ),
            runner=runner,
        )
        options.update(overrides)
        return MODULE.run_preflight(**options)

    def test_success_reports_bounded_telemetry(self):
        runner = fake_runner(
            [completed("0\n"), completed(""), completed("", 1, "Error: No such object: rocket-capture")]
        )
        payload = self.run_guard(runner)
        self.assertEqual(payload["schema"], MODULE.SCHEMA)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["failures"], [])
        self.assertEqual(payload["observations"]["gpu_inventory"], {"observed_gpu_count": 1})
        self.assertEqual(payload["observations"]["compute_owners"], {"owner_count": 0})
        self.assertEqual(payload["observations"]["containers"], [{"name": "rocket-capture", "state": "absent"}])
        self.assertNotIn("pid", json.dumps(payload))

    def test_memory_thresholds_fail_individually(self):
        runner = fake_runner(
            [completed("0\n"), completed(""), completed("", 1, "Error: No such object: rocket-capture")]
        )
        payload = self.run_guard(
            runner,
            meminfo_reader=lambda: "MemAvailable: 1 kB\nSwapFree: 2 kB\n",
        )
        self.assertEqual(
            payload["failures"],
            [
                {"check": "memory.mem_available", "reason": "below_minimum"},
                {"check": "memory.swap_free", "reason": "below_minimum"},
            ],
        )

    def test_gpu_count_failure(self):
        runner = fake_runner(
            [completed("0\n1\n"), completed(""), completed("", 1, "Error: No such object: rocket-capture")]
        )
        payload = self.run_guard(runner)
        self.assertIn(
            {"check": "gpu_inventory", "reason": "count_mismatch"},
            payload["failures"],
        )

    def test_compute_owner_failure(self):
        runner = fake_runner(
            [completed("0\n"), completed("1234\n"), completed("", 1, "Error: No such object: rocket-capture")]
        )
        payload = self.run_guard(runner)
        self.assertIn(
            {"check": "compute_owners", "reason": "owners_present"},
            payload["failures"],
        )
        self.assertNotIn("1234", json.dumps(payload))

    def test_named_container_failure(self):
        runner = fake_runner(
            [completed("0\n"), completed(""), completed("exited\n")]
        )
        payload = self.run_guard(runner)
        self.assertEqual(
            payload["failures"],
            [{"check": "container:rocket-capture", "reason": "present"}],
        )
        self.assertEqual(
            payload["observations"]["containers"],
            [{"name": "rocket-capture", "state": "exited"}],
        )

    def test_malformed_readings_fail_closed(self):
        runner = fake_runner(
            [completed("GPU 0\n"), completed("no owners\n"), completed("unknown\n")]
        )
        payload = self.run_guard(
            runner,
            meminfo_reader=lambda: "MemAvailable: nope kB\nSwapFree: 1 kB\n",
        )
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(
            payload["failures"],
            [
                {"check": "memory", "reason": "malformed_meminfo"},
                {"check": "gpu_inventory", "reason": "malformed_gpu_inventory"},
                {"check": "compute_owners", "reason": "malformed_compute_inventory"},
                {"check": "container:rocket-capture", "reason": "malformed_container_state"},
            ],
        )

    def test_missing_named_container_is_distinguished_from_docker_failure(self):
        missing = completed(
            "", 1, "Error: No such object: rocket-capture"
        )
        self.assertIsNone(
            MODULE.read_container_state("rocket-capture", fake_runner([missing]))
        )
        with self.assertRaisesRegex(MODULE.ResourcePreflightError, "command_failed"):
            MODULE.read_container_state(
                "rocket-capture",
                fake_runner([completed("", 1, "Cannot connect to Docker daemon")]),
            )

    def test_command_failure_publishes_structured_failure(self):
        runner = fake_runner(
            [completed("", returncode=1, stderr="driver unavailable"), completed(""), completed("")]
        )
        payload = self.run_guard(runner)
        self.assertEqual(payload["status"], "failed")
        self.assertIn(
            {"check": "gpu_inventory", "reason": "command_failed"},
            payload["failures"],
        )
        self.assertNotIn("driver unavailable", json.dumps(payload))

    def test_cli_returns_nonzero_for_stale_container_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            meminfo = pathlib.Path(directory) / "meminfo"
            meminfo.write_text(
                "MemAvailable: 33554432 kB\nSwapFree: 8388608 kB\n",
                encoding="ascii",
            )
            runner = fake_runner([completed("0\n"), completed(""), completed("dead\n")])
            with mock.patch.object(MODULE, "print") as printer:
                code = MODULE.main(
                    ["--container-name", "rocket-capture"],
                    meminfo_path=meminfo,
                    runner=runner,
                )
            self.assertEqual(code, 1)
            payload = json.loads(printer.call_args.args[0])
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(
                runner.calls,
                [
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                    ["docker", "container", "inspect", "--format", "{{.State.Status}}", "rocket-capture"],
                ],
            )


if __name__ == "__main__":
    unittest.main()

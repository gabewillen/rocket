#!/usr/bin/env python3
"""Focused tests for the bounded safetensors loader phase benchmark."""

from __future__ import annotations

import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("safetensors-loader-phases.py")
SPEC = importlib.util.spec_from_file_location("safetensors_loader_phases", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class StepClock:
    def __init__(self, step: float = 0.25) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


class StepFaults:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return benchmark.FaultCounts(minor=self.calls * 2, major=self.calls)


def write_safetensors(path: Path, payload_bytes: int = 150_000) -> int:
    header = {
        "weight": {
            "dtype": "U8",
            "shape": [payload_bytes],
            "data_offsets": [0, payload_bytes],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        block = bytes(range(256))
        full, tail = divmod(payload_bytes, len(block))
        file.write(block * full + block[:tail])
    return 8 + len(encoded)


class LoaderPhaseBenchmarkTests(unittest.TestCase):
    def test_anonymous_buffer_is_64k_aligned(self) -> None:
        with benchmark.AlignedAnonymousBuffer(benchmark.ALIGNMENT_BYTES) as buffer:
            self.assertEqual(buffer.address % benchmark.ALIGNMENT_BYTES, 0)
            self.assertEqual(len(buffer.view), benchmark.ALIGNMENT_BYTES)

    def test_checkpoint_index_selects_first_referenced_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_safetensors(root / "model-00002-of-00002.safetensors")
            write_safetensors(root / "model-00001-of-00002.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "b": "model-00002-of-00002.safetensors",
                            "a": "model-00001-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            selected = benchmark.resolve_checkpoint_file(root)
            self.assertEqual(selected.name, "model-00001-of-00002.safetensors")

    def test_all_phases_transfer_exact_bounded_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.safetensors"
            payload_offset = write_safetensors(path)
            report = benchmark.run_benchmark(
                benchmark.BenchmarkConfig(
                    file=path,
                    sample_bytes_requested=100_000,
                    iterations=2,
                    chunk_bytes=benchmark.ALIGNMENT_BYTES,
                ),
                clock=StepClock(),
                fault_reader=StepFaults(),
            )

        self.assertEqual(report["schema"], benchmark.SCHEMA)
        self.assertEqual(report["payload_offset_bytes"], payload_offset)
        self.assertEqual(report["sample_bytes"], 100_000)
        self.assertEqual(report["anonymous_alignment_bytes"], 65_536)
        self.assertEqual(
            report["anonymous_scratch_upper_bound_bytes"],
            4 * benchmark.ALIGNMENT_BYTES,
        )
        self.assertEqual(
            [phase["name"] for phase in report["phases"]],
            ["preadv_aligned", "mmap_fault_copy", "hot_reused_memcpy"],
        )
        self.assertEqual(report["phase_order"], [
            "preadv_aligned",
            "mmap_fault_copy",
            "hot_reused_memcpy",
        ])
        self.assertEqual(report["hot_copy_working_set_bytes"], benchmark.ALIGNMENT_BYTES)
        self.assertEqual(
            report["cache_note"],
            "preadv runs before mmap, so mmap sees pages warmed by this process under ambient cache",
        )
        for phase in report["phases"]:
            self.assertEqual(len(phase["iterations"]), 2)
            self.assertEqual(
                [item["bytes"] for item in phase["iterations"]],
                [100_000, 100_000],
            )
            self.assertEqual(
                [item["elapsed_seconds"] for item in phase["iterations"]],
                [0.25, 0.25],
            )

    def test_sample_limit_clamps_to_tensor_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.safetensors"
            write_safetensors(path, payload_bytes=4096)
            report = benchmark.run_benchmark(
                benchmark.BenchmarkConfig(
                    file=path,
                    sample_bytes_requested=8192,
                    iterations=1,
                    chunk_bytes=benchmark.ALIGNMENT_BYTES,
                ),
                clock=StepClock(),
                fault_reader=StepFaults(),
            )
        self.assertEqual(report["sample_bytes_requested"], 8192)
        self.assertEqual(report["sample_bytes"], 4096)

    def test_renderers_have_stable_order_and_shape(self) -> None:
        report = {
            "schema": benchmark.SCHEMA,
            "file": "/checkpoint/model.safetensors",
            "file_size_bytes": 70000,
            "payload_offset_bytes": 80,
            "payload_bytes": 69920,
            "sample_bytes_requested": 65536,
            "sample_bytes": 65536,
            "iterations": 1,
            "chunk_bytes": 65536,
            "phase_order": ["phase_a"],
            "hot_copy_working_set_bytes": 65536,
            "anonymous_alignment_bytes": 65536,
            "host_page_bytes": 65536,
            "anonymous_scratch_upper_bound_bytes": 262144,
            "serialized_header_limit_bytes": 67108864,
            "file_mapping_bytes_per_mmap_iteration": 65616,
            "cache_control": "none",
            "cache_note": "preadv runs before mmap",
            "phases": [
                {
                    "name": "phase_a",
                    "operation": "fixed operation",
                    "iterations": [
                        {
                            "iteration": 1,
                            "bytes": 65536,
                            "elapsed_seconds": 0.5,
                            "gib_per_second": 0.00012207,
                            "minor_faults": 2,
                            "major_faults": 0,
                        }
                    ],
                    "summary": {
                        "median_elapsed_seconds": 0.5,
                        "median_gib_per_second": 0.00012207,
                        "min_gib_per_second": 0.00012207,
                        "max_gib_per_second": 0.00012207,
                    },
                }
            ],
        }
        rendered_json = benchmark.render_json(report)
        self.assertLess(rendered_json.index('"schema"'), rendered_json.index('"phases"'))
        self.assertEqual(json.loads(rendered_json), report)
        self.assertEqual(
            benchmark.render_table(report),
            "metric\tvalue\n"
            f"schema\t\"{benchmark.SCHEMA}\"\n"
            "file\t\"/checkpoint/model.safetensors\"\n"
            "file_size_bytes\t70000\n"
            "payload_offset_bytes\t80\n"
            "payload_bytes\t69920\n"
            "sample_bytes_requested\t65536\n"
            "sample_bytes\t65536\n"
            "iterations\t1\n"
            "chunk_bytes\t65536\n"
            'phase_order\t["phase_a"]\n'
            "hot_copy_working_set_bytes\t65536\n"
            "anonymous_alignment_bytes\t65536\n"
            "host_page_bytes\t65536\n"
            "anonymous_scratch_upper_bound_bytes\t262144\n"
            "serialized_header_limit_bytes\t67108864\n"
            "file_mapping_bytes_per_mmap_iteration\t65616\n"
            "cache_control\t\"none\"\n"
            "cache_note\t\"preadv runs before mmap\"\n"
            "\n"
            "phase\titeration\tbytes\telapsed_s\tGiB_s\tminor_faults\tmajor_faults\n"
            "phase_a\t1\t65536\t0.500000000\t0.000122070\t2\t0\n"
            "\n"
            "phase\tmedian_elapsed_s\tmedian_GiB_s\tmin_GiB_s\tmax_GiB_s\n"
            "phase_a\t0.500000000\t0.000122070\t0.000122070\t0.000122070\n",
        )

    def test_malformed_header_fails_before_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            path.write_bytes(struct.pack("<Q", 10_000) + b"{}")
            with self.assertRaisesRegex(
                benchmark.BenchmarkError, "header extends beyond the file"
            ):
                benchmark.inspect_safetensors(path)

    def test_cli_json_uses_exact_file_argument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cli.safetensors"
            write_safetensors(path, payload_bytes=8192)
            stdout = io.StringIO()
            stderr = io.StringIO()
            result = benchmark.main(
                [
                    "--file",
                    str(path),
                    "--sample-bytes",
                    "4096",
                    "--iterations",
                    "1",
                    "--chunk-bytes",
                    "65536",
                    "--format",
                    "json",
                ],
                stdout=stdout,
                stderr=stderr,
            )
        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["sample_bytes"], 4096)
        self.assertEqual(len(payload["phases"]), 3)

    def test_invalid_chunk_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.safetensors"
            write_safetensors(path)
            with self.assertRaisesRegex(
                benchmark.BenchmarkError, "multiple of 65536"
            ):
                benchmark.validate_config(
                    benchmark.BenchmarkConfig(
                        file=path,
                        sample_bytes_requested=1024,
                        iterations=1,
                        chunk_bytes=benchmark.ALIGNMENT_BYTES + 1,
                    )
                )


if __name__ == "__main__":
    unittest.main()

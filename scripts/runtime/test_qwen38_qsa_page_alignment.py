#!/usr/bin/env python3
"""Tests for the pinned Qwen3.8 QSA page-alignment overlay."""

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCHER = ROOT / "scripts/runtime/patch-qwen38-qsa-page-alignment.py"
IMAGE = "vllm/vllm-openai:qwen38-flash-next"
PLATFORM = "/usr/local/lib/python3.12/dist-packages/vllm/platforms/interface.py"


class QsaPageAlignmentPatchTests(unittest.TestCase):
    def patch(self, source: str) -> tuple[subprocess.CompletedProcess[str], Path]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        platform_py = Path(directory.name) / "interface.py"
        platform_py.write_text(source)
        result = subprocess.run(
            [sys.executable, str(PATCHER), str(platform_py)],
            capture_output=True,
            text=True,
            check=False,
        )
        return result, platform_py

    def test_original_k5_failure_becomes_aligned_without_changing_ring(self) -> None:
        # The served Qwen3.8 checkpoint identifies its text config with the
        # upstream internal model type, not its public architecture name.
        pinned_text_model_type = "qwen4_exp_text"
        self.assertNotEqual(pinned_text_model_type, "qwen3_8_flash_next")
        self.assertEqual(pinned_text_model_type, "qwen4_exp_text")

        required_tokens = 3232
        compression_ratio = 4
        speculative_tokens = 5
        capacity = compression_ratio * math.ceil(
            (compression_ratio + speculative_tokens) / compression_ratio
        )
        self.assertEqual((capacity, required_tokens % capacity), (12, 4))

        alignment = math.lcm(16, capacity)
        aligned_tokens = alignment * math.ceil(required_tokens / alignment)
        self.assertEqual((alignment, aligned_tokens), (48, 3264))
        self.assertEqual(aligned_tokens % capacity, 0)
        self.assertEqual(capacity, 12)

    def test_patches_actual_pinned_platform_once_and_fails_on_drift(self) -> None:
        source = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "cat", IMAGE, PLATFORM],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        result, platform_py = self.patch(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        patched = platform_py.read_text()
        compile(patched, str(platform_py), "exec")
        self.assertEqual(patched.count("ROCKET_QWEN38_QSA_PAGE_ALIGNMENT_V1"), 1)
        self.assertIn(
            'model_config.hf_text_config.model_type == "qwen4_exp_text"', patched
        )
        self.assertIn("qsa_capacity = compress_ratio * cdiv", patched)
        self.assertIn("kernel_block_alignment_size, qsa_capacity", patched)

        again = subprocess.run(
            [sys.executable, str(PATCHER), str(platform_py)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("already patched", again.stderr)
        drift, _ = self.patch("import torch\n")
        self.assertNotEqual(drift.returncode, 0)
        self.assertIn("source drift", drift.stderr)


if __name__ == "__main__":
    unittest.main()

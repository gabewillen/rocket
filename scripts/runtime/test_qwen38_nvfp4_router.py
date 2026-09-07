#!/usr/bin/env python3
"""Tests for the fail-closed pinned Qwen3.8 router reconstruction patch."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCHER = ROOT / "scripts/runtime/patch-qwen38-nvfp4-router.py"
IMAGE = "vllm/vllm-openai:qwen38-flash-next"
MODEL = (
    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
    "qwen3_8_flash_next/nvidia/model.py"
)


class RouterPatchTests(unittest.TestCase):
    def patch(self, source):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        model_py = Path(directory.name) / "model.py"
        model_py.write_text(source)
        result = subprocess.run(
            [sys.executable, str(PATCHER), str(model_py)],
            capture_output=True,
            text=True,
            check=False,
        )
        return result, model_py

    def test_patches_only_the_exact_router_construction(self):
        source = (
            "from vllm.model_executor.layers.logits_processor import LogitsProcessor\n"
            "class Block:\n"
            "    def __init__(self, vllm_config, prefix):\n"
            "        super().__init__(vllm_config=vllm_config, prefix=prefix)\n"
        )
        result, model_py = self.patch(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        patched = model_py.read_text()
        compile(patched, str(model_py), "exec")
        self.assertIn("ROCKET_QWEN38_NVFP4_ROUTER_V1", patched)
        self.assertIn("ReplicatedLinear", patched)
        self.assertIn('router_algo == "NVFP4"', patched)
        self.assertIn("self.experts.gate = self.gate", patched)
        self.assertIn("unsupported policy", patched)

    def test_refuses_repatch_and_source_drift(self):
        result, model_py = self.patch(
            "from vllm.model_executor.layers.logits_processor import LogitsProcessor\n"
            "class Block:\n"
            "    def __init__(self, vllm_config, prefix):\n"
            "        super().__init__(vllm_config=vllm_config, prefix=prefix)\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        again = subprocess.run(
            [sys.executable, str(PATCHER), str(model_py)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("already patched", again.stderr)
        drift, _ = self.patch("import torch\n")
        self.assertNotEqual(drift.returncode, 0)
        self.assertIn("source drift", drift.stderr)

    def test_patches_actual_pinned_nvidia_model(self):
        source = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "cat", IMAGE, MODEL],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        upstream = subprocess.run(
            [
                "docker", "run", "--rm", "--entrypoint", "cat", IMAGE,
                "/usr/local/lib/python3.12/dist-packages/vllm/"
                "model_executor/models/qwen3_next.py",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertIn("quant_config=None", upstream)
        result, model_py = self.patch(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        patched = model_py.read_text()
        compile(patched, str(model_py), "exec")
        self.assertEqual(patched.count("ROCKET_QWEN38_NVFP4_ROUTER_V1"), 1)


if __name__ == "__main__":
    unittest.main()

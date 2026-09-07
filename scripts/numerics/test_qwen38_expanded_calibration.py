#!/usr/bin/env python3
"""Focused contract tests for the two-node expanded-calibration launcher."""

import pathlib
import subprocess
import unittest


SCRIPT = pathlib.Path(__file__).with_name("qwen38-expanded-calibration.sh")


class ExpandedCalibrationLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text()

    def test_shell_parses_and_help_does_not_prepare_or_launch(self):
        syntax = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        help_result = subprocess.run(
            ["bash", str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertEqual(help_result.stderr, "")
        self.assertIn("--launch", help_result.stdout)
        self.assertIn("without launching", help_result.stdout)

    def test_pins_every_external_identity(self):
        self.assertIn(
            "sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a",
            self.source,
        )
        self.assertIn("c2325b22602b51a5faf55fc2bebccc34f3f80b9f", self.source)
        self.assertIn("fc694b54fb0174e0913e6adf86691ef85a4ead47", self.source)
        self.assertIn("actual_image_id", self.source)
        self.assertIn("remote_image_id", self.source)

    def test_preserves_64k_and_runtime_overlay_contract(self):
        self.assertIn("getconf PAGESIZE", self.source)
        self.assertIn("patch-vllm-64k-loader.py", self.source)
        self.assertIn("patch-qwen38-activation-telemetry.py", self.source)
        for name in (
            "ple_layer_patched.py",
            "modelopt_patched.py",
            "weight_utils_64k.py",
            "qsa_ops_patched.py",
            "qsa_nvidia_patched.py",
            "config_patched.json",
            "hf_quant_config_patched.json",
        ):
            self.assertIn(name, self.source)

    def test_launch_is_explicit_and_reduction_requires_expanded_coverage(self):
        self.assertIn('if [[ "$LAUNCH" != true ]]', self.source)
        self.assertIn("qwen38-attention-calibration.py", self.source)
        self.assertIn("--require-expanded", self.source)
        self.assertIn("--recurrent-state-layers 36", self.source)
        self.assertIn("--speculative-config", self.source)

    def test_uses_only_durable_output_for_generated_launch_scripts(self):
        self.assertNotIn("/tmp/", self.source)
        self.assertIn('$OUTPUT_DIR/launch-worker.sh', self.source)
        self.assertIn('$OUTPUT_DIR/launch-head.sh', self.source)
        self.assertIn('$LOG_DIR/head.log', self.source)
        self.assertIn('$LOG_DIR/worker.log', self.source)

    def test_no_credential_or_direct_storage_flags(self):
        forbidden = ("HF_TOKEN", "API_KEY", "--env-file", "/dev/nvidia-fs", "cuFile")
        for token in forbidden:
            self.assertNotIn(token, self.source)


if __name__ == "__main__":
    unittest.main()

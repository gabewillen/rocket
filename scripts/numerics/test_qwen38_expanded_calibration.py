#!/usr/bin/env python3
"""Focused contract tests for the two-node expanded-calibration launcher."""

import pathlib
import subprocess
import tempfile
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
        self.assertIn("--startup-timeout-seconds", help_result.stdout)
        self.assertIn("default: 3600", help_result.stdout)
        self.assertIn("--fp8-artifact-dir", help_result.stdout)

    def test_startup_timeout_requires_positive_integer(self):
        for invalid in ("0", "-1", "1.5", "nope"):
            with self.subTest(value=invalid):
                result = subprocess.run(
                    [
                        "bash",
                        str(SCRIPT),
                        "--output-dir",
                        "/unused/rocket-timeout-validation",
                        "--startup-timeout-seconds",
                        invalid,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("positive integer", result.stderr)

    def test_timeout_budget_is_recorded_and_used_in_failure(self):
        self.assertIn(
            '"startup_timeout_seconds":$STARTUP_TIMEOUT_SECONDS', self.source
        )
        self.assertIn(
            'deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))', self.source
        )
        self.assertIn(
            'within ${STARTUP_TIMEOUT_SECONDS} seconds', self.source
        )

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
        self.assertIn("ROCKET_QWEN38_LOAD_TRACE=1", self.source)
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
        self.assertLess(
            self.source.index("patch-vllm-64k-loader.py"),
            self.source.index("patch-qwen38-fp8-overlay-loader.py"),
        )

    def test_fp8_launch_is_opt_in_read_only_and_keeps_default_config(self):
        self.assertIn('if [[ -n "$FP8_ARTIFACT_DIR" ]]', self.source)
        self.assertIn("/rocket/qwen38-linear-fp8:ro", self.source)
        self.assertIn("ROCKET_QWEN38_FP8_OVERLAY_MANIFEST", self.source)
        self.assertIn("ROCKET_QWEN38_FP8_QUANT_CONFIG", self.source)
        self.assertIn('quant_config_source="$artifact_dir/hf_quant_config_patched.json"', self.source)
        self.assertIn('quant_config_source="$fp8_host_dir/hf_quant_config.json"', self.source)
        self.assertIn("assert len(result['selected']) == 180", self.source)
        self.assertIn('if [[ -z "$FP8_ARTIFACT_DIR" ]]', self.source)
        self.assertEqual(self.source.count("linear-attention-fp8.safetensors"), 1)
        self.assertIn('basename "$FP8_ARTIFACT_DIR"', self.source)
        self.assertNotIn('\n$fp8_options\n', self.source)

    def test_generated_launch_scripts_default_and_fp8_are_single_commands(self):
        function = self.source[
            self.source.index("write_launch_script() {"):
            self.source.index('\nREMOTE_FP8_ARTIFACT=""')
        ]
        with tempfile.TemporaryDirectory(dir=pathlib.Path.cwd()) as directory:
            root = pathlib.Path(directory)
            default_script = root / "default.sh"
            fp8_script = root / "fp8.sh"
            harness = root / "generate.sh"
            harness.write_text(
                "set -euo pipefail\n"
                'HEAD_CONTAINER=head\nWORKER_CONTAINER=worker\n'
                'CONTAINER_MODEL_DIR=/vllm/model\nCONTAINER_VLLM_DIR=/vllm\n'
                'MODEL_CACHE_NAME=model-cache\nMODEL_REVISION=revision\n'
                'GID_INDEX=3\nIMAGE_TAG=image\nMODEL_ID=model\nHEAD_IP=10.0.0.1\nMASTER_PORT=50000\n'
                + function
                + f'\nwrite_launch_script {default_script} 0 10.0.0.1 eth0 hca /cache /generated "--host 0.0.0.0" ro ""\n'
                + f'write_launch_script {fp8_script} 1 10.0.0.2 eth1 hca /cache /generated --headless ro /durable/fp8\n'
            )
            subprocess.run(["bash", str(harness)], check=True)
            for generated in (default_script, fp8_script):
                parsed = subprocess.run(
                    ["bash", "-n", str(generated)], capture_output=True, text=True
                )
                self.assertEqual(parsed.returncode, 0, parsed.stderr)
                source = generated.read_text()
                self.assertNotIn("$fp8_options", source)
                command = source[source.index("exec docker run"):].splitlines()
                self.assertTrue(all(line.rstrip().endswith("\\") for line in command[:-1]))
                self.assertFalse(command[-1].rstrip().endswith("\\"))
            default = default_script.read_text()
            self.assertNotIn("/rocket/qwen38-linear-fp8", default)
            self.assertIn("/generated/hf_quant_config_patched.json", default)
            fp8 = fp8_script.read_text()
            self.assertIn("-v /durable/fp8:/rocket/qwen38-linear-fp8:ro", fp8)
            self.assertIn(
                "-e ROCKET_QWEN38_FP8_OVERLAY_MANIFEST=/rocket/qwen38-linear-fp8/manifest.json",
                fp8,
            )
            self.assertIn(
                "-e ROCKET_QWEN38_FP8_QUANT_CONFIG=/rocket/qwen38-linear-fp8/hf_quant_config.json",
                fp8,
            )
            self.assertIn(
                "/durable/fp8/hf_quant_config.json:/root/.cache/huggingface/hub/model-cache/snapshots/revision/hf_quant_config.json:ro",
                fp8,
            )

    def test_real_artifact_passes_accepted_runtime_preflight(self):
        artifact = pathlib.Path(
            "/home/glwillen/calibration/qwen38-linear-fp8-artifacts/"
            "dbefeae04f00118080ce821909786b0c84941b3ac39f1854941a2d2bf4cd516d"
        )
        snapshot = pathlib.Path(
            "/home/glwillen/.cache/huggingface/hub/"
            "models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
            "fc694b54fb0174e0913e6adf86691ef85a4ead47"
        )
        with tempfile.TemporaryDirectory(dir=pathlib.Path.cwd()) as directory:
            weight_utils = pathlib.Path(directory) / "weight_utils.py"
            container = subprocess.run(
                ["docker", "create", "vllm/vllm-openai:qwen38-flash-next"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            try:
                subprocess.run([
                    "docker", "cp",
                    f"{container}:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py",
                    str(weight_utils),
                ], check=True)
            finally:
                subprocess.run(["docker", "rm", container], check=True, capture_output=True)
            subprocess.run([
                "python3", str(SCRIPT.parents[1] / "runtime/patch-vllm-64k-loader.py"), str(weight_utils)
            ], check=True)
            subprocess.run([
                "python3", str(SCRIPT.parents[1] / "runtime/patch-qwen38-fp8-overlay-loader.py"), str(weight_utils)
            ], check=True)
            code = (
                "import glob; from vllm.model_executor.model_loader.weight_utils import "
                "_rocket_qwen38_fp8_overlay_preflight as check; "
                f"r=check(sorted(glob.glob('/cache/snapshots/{snapshot.name}/*.safetensors')),'lazy'); "
                "assert len(r['selected']) == 180; print('validated=180')"
            )
            result = subprocess.run([
                "docker", "run", "--rm",
                "-v", f"{snapshot.parents[1]}:/cache:ro",
                "-v", f"{artifact}:/rocket/qwen38-linear-fp8:ro",
                "-v", f"{weight_utils}:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py:ro",
                "-e", "ROCKET_QWEN38_FP8_OVERLAY_MANIFEST=/rocket/qwen38-linear-fp8/manifest.json",
                "-e", "ROCKET_QWEN38_FP8_QUANT_CONFIG=/rocket/qwen38-linear-fp8/hf_quant_config.json",
                "--entrypoint", "python3", "vllm/vllm-openai:qwen38-flash-next", "-c", code,
            ], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("validated=180", result.stdout)

    def test_modelopt_checksum_matches_dual_spelling_generator(self):
        self.assertIn(
            "89c54b49756e3fe9def912e22c6721e576c03b93d0238cbe061c029d8a6c84e0",
            self.source,
        )
        self.assertNotIn(
            "5cb67475490badba79ab2b8f6d2527a985a52848bb63055cba84e0b2ea89163a",
            self.source,
        )
        self.assertIn("patch-qwen38-modelopt-fp8-block-moe.py", self.source)

    def test_launch_is_explicit_and_reduction_requires_expanded_coverage(self):
        self.assertIn('if [[ "$LAUNCH" != true ]]', self.source)
        self.assertIn("qwen38-attention-calibration.py", self.source)
        self.assertIn("--require-expanded", self.source)
        self.assertIn("--expanded-v2-only", self.source)
        self.assertIn("--min-emission-call 8", self.source)
        self.assertIn("--recurrent-state-layers 36", self.source)
        self.assertIn("--speculative-config", self.source)
        self.assertIn("qwen38-mtp-runtime-evidence.py", self.source)
        self.assertIn('$OUTPUT_DIR/mtp-runtime-evidence.json', self.source)
        self.assertIn('--not-before "$head_workload_since"', self.source)
        self.assertIn("--min-records 2 --positions 3", self.source)
        workload = self.source.index(
            'python3 "$SCRIPT_DIR/qwen38-attention-calibration.py"'
        )
        mtp_proof = self.source.index(
            'python3 "$SCRIPT_DIR/qwen38-mtp-runtime-evidence.py"'
        )
        self.assertLess(workload, mtp_proof)
        self.assertLess(
            mtp_proof,
            self.source.index("for node in head worker combined"),
        )
        self.assertIn("set -euo pipefail", self.source)

    def test_reducer_input_is_bounded_to_post_health_workload_logs(self):
        self.assertIn('head_workload_since=$(date --iso-8601=seconds)', self.source)
        self.assertIn(
            'worker_workload_since=$(ssh -o BatchMode=yes "$SSH_TARGET" date --iso-8601=seconds)',
            self.source,
        )
        self.assertIn("Move past the health-check second", self.source)
        self.assertIn('--since "$head_workload_since"', self.source)
        self.assertIn("--since '$worker_workload_since'", self.source)
        self.assertIn('$LOG_DIR/head-workload.log', self.source)
        self.assertIn('$LOG_DIR/worker-workload.log', self.source)
        self.assertIn('$LOG_DIR/$node-workload.log', self.source)

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

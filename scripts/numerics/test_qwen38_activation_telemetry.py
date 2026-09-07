#!/usr/bin/env python3

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCHER = ROOT / "scripts/numerics/patch-qwen38-activation-telemetry.py"
REDUCER = ROOT / "scripts/numerics/qwen38-activation-maxima.py"


def load_reducer():
    spec = importlib.util.spec_from_file_location("qwen38_activation_maxima", REDUCER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PatchTests(unittest.TestCase):
    def test_patches_expected_nvidia_model_anchors(self):
        source = (
            "from itertools import islice\n\nimport torch\n"
            "from vllm.model_executor.layers.logits_processor import LogitsProcessor\n"
            "class Qwen3_8FlashNextSparseMoeBlock(Qwen3NextSparseMoeBlock):\n"
            "    def __init__(self, vllm_config, prefix):\n"
            "        super().__init__(vllm_config=vllm_config, prefix=prefix)\n"
            "class Wrapper:\n"
            "    def __init__(self):\n"
            "        enable_qwen38next_low_latency_gemm(self, self.model_config.dtype)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            model_py = Path(directory) / "model.py"
            model_py.write_text(source)
            result = subprocess.run(
                [sys.executable, str(PATCHER), str(model_py)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            patched = model_py.read_text()
        self.assertIn('rocket.qwen38.activation-telemetry.v2', patched)
        self.assertIn('ROCKET_QWEN38_LINEAR_LOAD', patched)
        self.assertIn('getattr(value, "shard_id", None)', patched)
        self.assertIn('type(self.quant_method).__name__', patched)
        self.assertIn('sorted(self._parameters)', patched)
        self.assertIn('getattr(layer, "linear_attn", None)', patched)
        self.assertIn('getattr(layer, "self_attn", None)', patched)
        self.assertIn('getattr(layer, "ple", None)', patched)
        self.assertIn('getattr(mlp, "gate", None)', patched)
        self.assertNotIn("ROCKET_QWEN38_NVFP4_ROUTER_V1", patched)
        self.assertIn("chunk_gated_delta_rule", patched)
        compile(patched, "model.py", "exec")

    def test_refuses_source_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            model_py = Path(directory) / "model.py"
            model_py.write_text("import torch\n")
            result = subprocess.run(
                [sys.executable, str(PATCHER), str(model_py)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source drift", result.stderr)

class ReducerTests(unittest.TestCase):
    def legacy_lines(self, layers=2):
        for layer in range(layers):
            for projection in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
                yield (
                    "ROCKET_NVFP4_CALIBRATION\t"
                    f"layer.{layer}.linear_attn.{projection}\t{layer + 1}.0\n"
                )

    def record(self, channel, call=1):
        return "ROCKET_NVFP4_TELEMETRY\t" + json.dumps(
            {
                "schema": "rocket.qwen38.activation-telemetry.v2",
                "channel": channel,
                "kind": "test",
                "call": call,
                "source_numel": 4,
                "sample_numel": 4,
                "absmax": 2.0,
                "mean": 0.0,
                "rms": 1.0,
                "abs_p50": 0.5,
                "abs_p90": 1.0,
                "abs_p99": 2.0,
                "histogram_log2": [1] * 10,
            }
        ) + "\n"

    def expanded_records(self, layers=36, full_layers=12, call=8):
        lines = []
        for layer in range(layers):
            for projection in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
                lines += [
                    self.record(f"layer.{layer}.linear_attn.{projection}.input", call),
                    self.record(f"layer.{layer}.linear_attn.{projection}.output", call),
                ]
            lines += [
                self.record(f"layer.{layer}.linear_attn.output", call),
                self.record(
                    f"layer.{layer}.linear_attn.recurrent_state.output", call
                ),
            ]
        for layer in range(layers, layers + full_layers):
            lines += [
                self.record(f"layer.{layer}.full_attn.qkv_proj.output", call),
                self.record(f"layer.{layer}.full_attn.o_proj.output", call),
                self.record(f"layer.{layer}.full_attn.output", call),
            ]
        lines += [self.record("layer.1.ple.output", call)]
        for layer in range(layers + full_layers):
            lines += [self.record(f"layer.{layer}.router.topk.output", call)]
        return lines

    def test_v2_only_gate_accepts_complete_call8_with_106_legacy_channels(self):
        legacy = list(self.legacy_lines(36))[:-2]
        lines = legacy + self.expanded_records()
        result = subprocess.run(
            [
                sys.executable,
                str(REDUCER),
                "--require-expanded",
                "--expanded-v2-only",
                "--min-emission-call", "8",
                "--ple-layers", "1",
                "--router-layers", "48",
                "--recurrent-state-layers", "36",
            ],
            input="".join(lines),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["gate"]["source"], "v2_only")
        self.assertEqual(payload["gate"]["min_emission_call"], 8)
        self.assertEqual(len(payload["channels"]), 106)

    def test_v2_only_gate_rejects_startup_only_and_incomplete_call8(self):
        lines = list(self.legacy_lines(36))
        lines += self.expanded_records(call=4)
        real_records = self.expanded_records(call=8)
        lines += real_records[:-1]
        result = subprocess.run(
            [
                sys.executable,
                str(REDUCER),
                "--require-expanded",
                "--expanded-v2-only",
                "--min-emission-call", "8",
                "--ple-layers", "1",
                "--router-layers", "48",
                "--recurrent-state-layers", "36",
            ],
            input="".join(lines),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("router_layers=47/48", result.stderr)

    def test_default_mode_still_requires_complete_legacy_channels(self):
        lines = list(self.legacy_lines(36))[:-2] + self.expanded_records()
        result = subprocess.run(
            [sys.executable, str(REDUCER), "--require-expanded"],
            input="".join(lines),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("106/108 channels", result.stderr)

    def test_preserves_legacy_108_channel_compatibility(self):
        reducer = load_reducer()
        maxima, telemetry = reducer.parse_stream(self.legacy_lines(36))
        _, complete = reducer.legacy_layers(maxima)
        self.assertEqual(len(maxima), 108)
        self.assertEqual(len(complete), 36)
        self.assertEqual(telemetry, {})

    def test_keeps_latest_bounded_v2_record_per_channel(self):
        reducer = load_reducer()
        lines = list(self.legacy_lines())
        lines += [self.record("layer.0.full_attn.output", 1)]
        lines += [self.record("layer.0.full_attn.output", 2)]
        _, telemetry = reducer.parse_stream(lines)
        self.assertEqual(telemetry["layer.0.full_attn.output"]["call"], 2)

    def test_reducer_expanded_gate(self):
        lines = list(self.legacy_lines(2))
        for layer in range(2):
            for projection in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
                lines += [
                    self.record(f"layer.{layer}.linear_attn.{projection}.input"),
                    self.record(f"layer.{layer}.linear_attn.{projection}.output"),
                ]
        lines += [self.record("layer.2.full_attn.qkv_proj.output")]
        lines += [self.record("layer.2.full_attn.o_proj.output")]
        lines += [self.record("layer.2.full_attn.output")]
        lines += [self.record("layer.0.ple.output")]
        lines += [self.record("layer.0.router.topk.output")]
        result = subprocess.run(
            [
                sys.executable,
                str(REDUCER),
                "--layers", "2",
                "--require-expanded",
                "--full-attention-layers", "1",
                "--ple-layers", "1",
                "--router-layers", "1",
                "--recurrent-state-layers", "0",
            ],
            input="".join(lines),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["coverage"]["full_attention_layers"], 1)
        self.assertEqual(payload["schema"], "rocket.qwen38.activation-summary.v2")

    def test_rejects_malformed_v2_json(self):
        reducer = load_reducer()
        with self.assertRaisesRegex(ValueError, "invalid v2 telemetry JSON"):
            reducer.parse_stream(["ROCKET_NVFP4_TELEMETRY\t{bad}\n"])

    def test_rejects_incomplete_v2_record(self):
        reducer = load_reducer()
        line = "ROCKET_NVFP4_TELEMETRY\t" + json.dumps(
            {
                "schema": "rocket.qwen38.activation-telemetry.v2",
                "channel": "layer.0.ple.output",
                "call": 1,
            }
        )
        with self.assertRaisesRegex(ValueError, "missing fields"):
            reducer.parse_stream([line])

    def test_expanded_gate_can_require_recurrent_state_coverage(self):
        lines = list(self.legacy_lines(1))
        for projection in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
            lines += [
                self.record(f"layer.0.linear_attn.{projection}.input"),
                self.record(f"layer.0.linear_attn.{projection}.output"),
            ]
        result = subprocess.run(
            [
                sys.executable,
                str(REDUCER),
                "--layers", "1",
                "--require-expanded",
                "--full-attention-layers", "0",
                "--recurrent-state-layers", "1",
            ],
            input="".join(lines),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("recurrent_state_layers=0/1", result.stderr)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Focused tests for the Qwen3.8 FP8 ModelOpt config overlay."""

import importlib.util
import json
import pathlib
import tempfile
import unittest


PATH = pathlib.Path(__file__).with_name("patch-qwen38-fp8-quant-config.py")
SPEC = importlib.util.spec_from_file_location("config_patch", PATH)
config_patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(config_patch)
PINNED_CONFIG = pathlib.Path(
    "/home/glwillen/.cache/huggingface/hub/"
    "models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
    "fc694b54fb0174e0913e6adf86691ef85a4ead47/hf_quant_config.json"
)


def fixture():
    return {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "exclude_modules": [
                *(f"model.language_model.layers.{layer}.linear_attn*" for layer in range(36)),
                "lm_head",
            ],
            "quantized_layers": {"existing": {"quant_algo": "NVFP4"}},
        }
    }


class ConfigPatchTest(unittest.TestCase):
    def test_exact_180_entries_and_other_policy_preserved(self):
        result = config_patch.patched(fixture())
        quant = result["quantization"]
        fp8 = {name for name, value in quant["quantized_layers"].items() if value == {"quant_algo": "FP8"}}
        self.assertEqual(len(fp8), 180)
        self.assertEqual(quant["exclude_modules"], ["lm_head"])
        self.assertEqual(quant["quantized_layers"]["existing"], {"quant_algo": "NVFP4"})

    def test_incomplete_and_preexisting_configs_fail_closed(self):
        value = fixture()
        value["quantization"]["exclude_modules"].pop(0)
        with self.assertRaisesRegex(ValueError, "35/36"):
            config_patch.patched(value)
        value = fixture()
        value["quantization"]["quantized_layers"]["model.language_model.layers.0.linear_attn.out_proj"] = {"quant_algo": "FP8"}
        with self.assertRaisesRegex(ValueError, "already exists"):
            config_patch.patched(value)

    def test_output_is_deterministic_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source, first, second = root / "source.json", root / "a.json", root / "b.json"
            source.write_text(json.dumps(fixture()))
            self.assertEqual(config_patch.main.__name__, "main")
            data = config_patch.canonical_bytes(config_patch.patched(fixture()))
            first.write_bytes(data)
            second.write_bytes(data)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_actual_pinned_config_produces_exact_policy(self):
        source = json.loads(PINNED_CONFIG.read_text())
        result = config_patch.patched(source)
        quant = result["quantization"]
        selected = [name for name, value in quant["quantized_layers"].items() if value == {"quant_algo": "FP8"} and ".linear_attn." in name]
        self.assertEqual(len(selected), 180)
        self.assertFalse(any(config_patch.LINEAR_EXCLUDE.fullmatch(item) for item in quant["exclude_modules"]))


if __name__ == "__main__":
    unittest.main()

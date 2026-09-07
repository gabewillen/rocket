#!/usr/bin/env python3

import importlib.util
import json
import pathlib
import unittest


PATH = pathlib.Path(__file__).with_name("qwen38-embed-fp8-config.py")
SPEC = importlib.util.spec_from_file_location("qwen38_embed_fp8_config", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EmbedConfigTest(unittest.TestCase):
    def test_real_artifact_replaces_runtime_policy(self):
        base = json.loads(pathlib.Path(
            "/home/glwillen/calibration/qwen38-linear-fp8-live-20260907-07/"
            "artifacts/config_patched.json"
        ).read_text())
        sidecar = json.loads(pathlib.Path(
            "/home/glwillen/calibration/qwen38-linear-fp8-artifacts/"
            "dbefeae04f00118080ce821909786b0c84941b3ac39f1854941a2d2bf4cd516d/"
            "hf_quant_config.json"
        ).read_text())
        result = MODULE.embed(base, sidecar)
        policy = result["quantization_config"]
        self.assertEqual(policy["quant_method"], "modelopt")
        self.assertEqual(policy["quant_algo"], "MIXED_PRECISION")
        self.assertEqual(
            sum(bool(MODULE.SELECTED.fullmatch(key)) for key in policy["quantized_layers"]),
            180,
        )
        self.assertFalse(any("linear_attn" in item for item in policy["exclude_modules"]))

    def test_partial_policy_fails_closed(self):
        base = {"quantization_config": {"quant_algo": "MIXED_PRECISION", "quant_method": "modelopt"}}
        sidecar = {"quantization": {"quant_algo": "MIXED_PRECISION", "exclude_modules": [], "quantized_layers": {}}}
        with self.assertRaisesRegex(ValueError, "0/180"):
            MODULE.embed(base, sidecar)

    def test_full_attention_and_combined_policies_are_independently_selectable(self):
        base = {
            "quantization_config": {
                "quant_algo": "MIXED_PRECISION",
                "quant_method": "modelopt",
            }
        }
        full_layers = {
            f"model.language_model.layers.{layer}.self_attn.{projection}": {
                "quant_algo": "NVFP4"
            }
            for layer in range(3, 48, 4)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        linear_layers = {
            f"model.language_model.layers.{layer}.linear_attn.{projection}": {
                "quant_algo": "NVFP4"
            }
            for layer in set(range(48)) - set(range(3, 48, 4))
            for projection in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
        }
        full = {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "exclude_modules": [
                    f"model.language_model.layers.{layer}.linear_attn*"
                    for layer in set(range(48)) - set(range(3, 48, 4))
                ],
                "quantized_layers": full_layers,
            }
        }
        result = MODULE.embed(base, full, "NVFP4", ("full_attention",))
        self.assertEqual(len(result["quantization_config"]["quantized_layers"]), 48)
        combined = json.loads(json.dumps(full))
        combined["quantization"]["quantized_layers"].update(linear_layers)
        combined["quantization"]["exclude_modules"] = []
        result = MODULE.embed(
            base,
            combined,
            "NVFP4",
            ("linear_attention", "full_attention"),
        )
        self.assertEqual(len(result["quantization_config"]["quantized_layers"]), 228)


if __name__ == "__main__":
    unittest.main()

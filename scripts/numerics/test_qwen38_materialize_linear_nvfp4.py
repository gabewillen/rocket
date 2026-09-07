#!/usr/bin/env python3
"""Offline contract tests for the Qwen linear-attention NVFP4 overlay."""

import importlib.util
import json
import pathlib
import tempfile
import unittest


PATH = pathlib.Path(__file__).with_name("qwen38-materialize-linear-nvfp4.py")
SPEC = importlib.util.spec_from_file_location("nvfp4_materializer", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
REAL_SNAPSHOT = pathlib.Path.home() / ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots" / module.REVISION
REAL_TRACE = pathlib.Path("/home/glwillen/calibration/qwen38-expanded-mtp3-20260907-03/combined-activation-summary.json")


class Nvfp4MaterializerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = module.build_plan(REAL_SNAPSHOT, REAL_TRACE, REAL_SNAPSHOT / "hf_quant_config.json")

    def test_real_family_and_exact_bytes(self):
        plan = self.plan
        self.assertEqual(len(plan["tensors"]), 180)
        self.assertEqual(sum(x["source_bytes"] for x in plan["tensors"]), 4_170_055_680)
        self.assertEqual(sum(x["encoded_bytes"] for x in plan["tensors"]), 1_172_829_600)
        self.assertLessEqual(plan["bytes_hashed"], 4_170_055_680 + 64 * 2**20)

    def test_nvidia_dense_linear_shapes(self):
        for item in self.plan["tensors"]:
            n, k = item["shape"]
            entries = module.output_entries(item)
            self.assertEqual(entries[0][1:], ("U8", [n, k // 2], n * k // 2))
            self.assertEqual(entries[1][1:], ("F8_E4M3", [n, k // 16], n * k // 16))
            self.assertEqual(entries[2][1:], ("F32", [1], 4))
            self.assertEqual(entries[3][1:], ("F32", [1], 4))
            self.assertGreater(item["input_scale"], 0)

    def test_config_selects_only_180_nvfp4_projections(self):
        layers = self.plan["quant_config"]["quantization"]["quantized_layers"]
        selected = [name for name, policy in layers.items() if policy.get("quant_algo") == "NVFP4" and ".linear_attn." in name]
        self.assertEqual(len(selected), 180)
        self.assertEqual(self.plan["quant_config"]["quantization"]["group_size"], 16)

    def test_header_is_720_tensors_and_payload_exact(self):
        raw = module.safetensors_header(self.plan["tensors"])
        header = json.loads(raw.rstrip())
        self.assertEqual(len(header), 720)
        final = max(meta["data_offsets"][1] for meta in header.values())
        self.assertEqual(final, sum(x["encoded_bytes"] for x in self.plan["tensors"]))

    def test_interrupted_output_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path.cwd()) as directory:
            root = pathlib.Path(directory)
            (root / f".{module.REVISION}.linear-nvfp4.building").mkdir()
            with self.assertRaisesRegex(module.MaterializeError, "interrupted output"):
                module.materialize({"tensors": []}, root)


if __name__ == "__main__":
    unittest.main()

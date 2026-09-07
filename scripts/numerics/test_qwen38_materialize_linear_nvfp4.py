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
        cls.full_plan = module.build_plan(
            REAL_SNAPSHOT,
            REAL_TRACE,
            REAL_SNAPSHOT / "hf_quant_config.json",
            ("full_attention",),
        )
        cls.combined_plan = module.build_plan(
            REAL_SNAPSHOT,
            REAL_TRACE,
            REAL_SNAPSHOT / "hf_quant_config.json",
            ("linear_attention", "full_attention"),
        )
        cls.all_eligible_plan = module.build_plan(
            REAL_SNAPSHOT,
            REAL_TRACE,
            REAL_SNAPSHOT / "hf_quant_config.json",
            ("linear_attention", "full_attention", "base_routers", "base_ple"),
        )

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

    def test_full_attention_family_is_exact_and_mtp_is_isolated(self):
        plan = self.full_plan
        self.assertEqual(plan["families"], ("full_attention",))
        self.assertEqual(len(plan["tensors"]), 48)
        self.assertEqual(sum(x["source_bytes"] for x in plan["tensors"]), 1_195_376_640)
        self.assertEqual(sum(x["encoded_bytes"] for x in plan["tensors"]), 336_200_064)
        self.assertEqual({x["layer"] for x in plan["tensors"]}, set(range(3, 48, 4)))
        self.assertFalse(any(x["name"].startswith("mtp.") for x in plan["tensors"]))
        self.assertTrue(all(x["family"] == "full_attention" for x in plan["tensors"]))
        for layer in range(3, 48, 4):
            scales = {
                item["input_scale"]
                for item in plan["tensors"]
                if item["layer"] == layer and item["projection"] != "o_proj"
            }
            self.assertEqual(len(scales), 1)

    def test_combined_map_preserves_both_complete_families(self):
        plan = self.combined_plan
        self.assertEqual(plan["families"], ("full_attention", "linear_attention"))
        self.assertEqual(len(plan["tensors"]), 228)
        self.assertEqual(sum(x["source_bytes"] for x in plan["tensors"]), 5_365_432_320)
        self.assertEqual(sum(x["encoded_bytes"] for x in plan["tensors"]), 1_509_029_664)
        algorithms = plan["quant_config"]["quantization"]["quantized_layers"]
        selected = [
            name
            for name, value in algorithms.items()
            if value.get("quant_algo") == "NVFP4"
            and (".linear_attn." in name or ".self_attn." in name)
        ]
        self.assertEqual(len(selected), 228)

    def test_all_eligible_families_are_complete_and_calibrated(self):
        plan = self.all_eligible_plan
        self.assertEqual(
            plan["families"],
            ("base_ple", "base_routers", "full_attention", "linear_attention"),
        )
        self.assertEqual(len(plan["tensors"]), 278)
        self.assertEqual(sum(x["source_bytes"] for x in plan["tensors"]), 5_556_797_440)
        self.assertEqual(sum(x["encoded_bytes"] for x in plan["tensors"]), 1_562_851_504)
        routers = [x for x in plan["tensors"] if x["family"] == "base_routers"]
        ple = [x for x in plan["tensors"] if x["family"] == "base_ple"]
        self.assertEqual(len(routers), 48)
        self.assertEqual({x["layer"] for x in routers}, set(range(48)))
        self.assertEqual({x["projection"] for x in routers}, {"gate"})
        self.assertEqual(len(ple), 2)
        self.assertEqual(
            {(x["layer"], x["projection"]) for x in ple},
            {(1, "key_proj"), (1, "value_proj")},
        )
        expected_scale = module.FP8.float32(160.0 / module.NVFP4_DENOMINATOR)
        self.assertEqual({x["input_scale"] for x in ple}, {expected_scale})

    def test_router_family_requires_complete_v2_coverage(self):
        trace = json.loads(REAL_TRACE.read_text())
        trace["coverage"]["router_layers"] = 47
        with tempfile.TemporaryDirectory(dir=pathlib.Path.cwd()) as directory:
            path = pathlib.Path(directory) / "trace.json"
            path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(module.MaterializeError, "router_layers=47/48"):
                module.build_plan(
                    REAL_SNAPSHOT,
                    path,
                    REAL_SNAPSHOT / "hf_quant_config.json",
                    ("base_routers",),
                )

    def test_full_trace_or_family_policy_cannot_be_partial(self):
        trace = json.loads(REAL_TRACE.read_text())
        trace["coverage"]["full_attention_layers"] = 11
        with tempfile.TemporaryDirectory(dir=pathlib.Path.cwd()) as directory:
            path = pathlib.Path(directory) / "trace.json"
            path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(module.MaterializeError, "full-attention trace coverage"):
                module.build_plan(
                    REAL_SNAPSHOT,
                    path,
                    REAL_SNAPSHOT / "hf_quant_config.json",
                    ("full_attention",),
                )

    def test_quant_config_full_family_is_complete_and_isolated(self):
        source = json.loads((REAL_SNAPSHOT / "hf_quant_config.json").read_text())
        result = module.CONFIG.patched(source, ("full_attention",))
        quant = result["quantization"]
        selected = {
            name
            for name, policy in quant["quantized_layers"].items()
            if ".self_attn." in name and policy.get("quant_algo") == "NVFP4"
        }
        self.assertEqual(len(selected), 48)
        self.assertTrue(any("linear_attn" in item for item in quant["exclude_modules"]))
        partial = json.loads(json.dumps(source))
        partial["quantization"]["exclude_modules"].remove(
            "model.language_model.layers.3.self_attn*"
        )
        with self.assertRaisesRegex(ValueError, "11/12"):
            module.CONFIG.patched(partial, ("full_attention",))

    def test_interrupted_output_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path.cwd()) as directory:
            root = pathlib.Path(directory)
            (root / f".{module.REVISION}.linear-nvfp4.building").mkdir()
            with self.assertRaisesRegex(module.MaterializeError, "interrupted output"):
                module.materialize(
                    {"families": ("linear_attention",), "tensors": []}, root
                )


if __name__ == "__main__":
    unittest.main()

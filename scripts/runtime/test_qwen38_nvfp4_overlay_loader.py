#!/usr/bin/env python3
"""Source-contract tests for the Qwen NVFP4 overlay loader."""

import importlib.util
import pathlib
import unittest


PATH = pathlib.Path(__file__).with_name("patch-qwen38-nvfp4-overlay-loader.py")
SPEC = importlib.util.spec_from_file_location("nvfp4_loader", PATH)
patcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patcher)
BASE_SPEC = importlib.util.spec_from_file_location("fp8_loader_test", pathlib.Path(__file__).with_name("test_qwen38_fp8_overlay_loader.py"))
fixture = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(fixture)


class Nvfp4OverlayLoaderTest(unittest.TestCase):
    def test_exact_nvfp4_abi_and_selected_family_preflight(self):
        result = patcher.patched(fixture.SOURCE)
        self.assertIn(patcher.MARKER, result)
        self.assertIn('"rocket.qwen38.linear-nvfp4-overlay.v1"', result)
        self.assertIn('"rocket.qwen38.nvfp4-overlay.v2"', result)
        self.assertIn('get("quant_algo") != "NVFP4"', result)
        self.assertIn('name: ("U8", [n, k // 2])', result)
        self.assertIn('prefix + ".weight_scale": ("F8_E4M3", [n, k // 16])', result)
        self.assertIn('prefix + ".weight_scale_2": ("F32", [1])', result)
        self.assertIn('"linear_attention": 180, "full_attention": 48', result)
        self.assertIn('"base_routers": 48, "base_ple": 2', result)
        self.assertIn('"full_attention": (set(range(3, 48, 4))', result)
        self.assertIn('"q_proj", "k_proj", "v_proj", "o_proj"', result)
        self.assertIn('"base_routers": (set(range(48)), {"gate"})', result)
        self.assertIn('"base_ple": ({1}, {"key_proj", "value_proj"})', result)
        self.assertIn("tensor outside selected families", result)
        self.assertIn("partial {family} family", result)
        self.assertIn("family policy does not match selected tensors", result)
        self.assertIn("unselected {family} is not excluded", result)

    def test_yields_all_four_tensors_and_preserves_clone(self):
        result = patcher.patched(fixture.SOURCE)
        self.assertIn('prefix + ".weight_scale_2"', result)
        self.assertEqual(result.count(".clone()"), 2)
        self.assertNotIn("cuFile", result)
        self.assertNotIn("nvidia-fs", result)

    def test_source_drift_and_repatch_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "64 KiB clone"):
            patcher.patched(fixture.SOURCE.replace("param = f.get_tensor(name).clone()", "param = f.get_tensor(name)"))
        with self.assertRaisesRegex(ValueError, "already patched"):
            patcher.patched(patcher.patched(fixture.SOURCE))


if __name__ == "__main__":
    unittest.main()

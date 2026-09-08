#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import hashlib
import pathlib
import unittest


ENGINE = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ENGINE / "vendor" / "flashinfer-91bda04"


class GdnFlashInferCutlassContract(unittest.TestCase):
    def test_vendored_source_is_byte_exact(self) -> None:
        expected = {
            "LICENSE": "cb67c224f503e0a063908950b12f89a7280c6e527dcffac972aa114e4bf3c5de",
            "include/flashinfer/arch_condition.h":
                "99d6b54df9bd2b9095bfe64401581122542ea65d84aa609e24e2ffc1be522418",
            "include/flashinfer/cutlass_utils.cuh":
                "2c2f6d2a96710c7418797643ab108c62cce944273b63420447030aacd0f1f4b7",
            "include/flashinfer/gemm/cutlass_gemm_configs.h":
                "114f3a3ef96b914150475d2e20212309b6cac8a0b55443cf5114b84b092239be",
            "include/flashinfer/gemm/fp4_gemm_template_sm120.h":
                "a70a47370ed14ee8f88b4d93e54547db6a23891af2451f78ca1159de0eda312c",
        }
        actual = {
            relative: hashlib.sha256((VENDOR / relative).read_bytes()).hexdigest()
            for relative in expected
        }
        self.assertEqual(actual, expected)

    def test_only_measured_fallback_tactic_is_bound(self) -> None:
        source = (ENGINE / "src/linear_attention/gdn_flashinfer_cutlass.cu").read_text()
        self.assertEqual(source.count("INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER"), 1)
        self.assertIn(
            "INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 1, 1,",
            source,
        )
        self.assertIn("_1SM, false)", source)
        self.assertNotIn("genericFp4GemmKernelLauncherStreamK<", source)

    def test_default_backend_requires_imported_source(self) -> None:
        header = (ENGINE / "src/linear_attention/gdn_cutlass.h").read_text()
        cmake = (ENGINE / "CMakeLists.txt").read_text()
        graph = (ENGINE / "src/linear_attention/gdn_cutlass.cu").read_text()
        smoke = (ENGINE / "bench/gdn_graph_smoke.cu").read_text()
        self.assertIn("GdnPrefillInputBackend::kFlashInferCutlass", header)
        self.assertIn("message(FATAL_ERROR", cmake)
        self.assertIn("add_library(qwen38_gdn_flashinfer_cutlass STATIC", cmake)
        self.assertIn("CUDA_SEPARABLE_COMPILATION OFF", cmake)
        self.assertIn("GdnFlashInferCutlassGemm qkvz_gemm", graph)
        self.assertIn("GdnFlashInferCutlassGemm ba_gemm", graph)
        self.assertIn("bucket->qkvz_gemm.run(stream)", graph)
        self.assertIn("bucket->ba_gemm.run(stream)", graph)
        self.assertIn('"flashinfer_cutlass_91bda04"', smoke)


if __name__ == "__main__":
    unittest.main()

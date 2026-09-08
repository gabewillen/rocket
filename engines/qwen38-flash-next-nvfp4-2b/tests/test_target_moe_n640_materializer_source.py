# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TargetMoeN640MaterializerSource(unittest.TestCase):
    def test_transform_is_native_init_only(self):
        source = (ROOT / "src/moe/target_moe_n640_materializer.cu").read_text()
        self.assertNotIn("torch", source.lower())
        self.assertNotIn("python", source.lower())
        self.assertIn("cudaStreamCreateWithFlags", source)
        self.assertIn("cudaEventSynchronize", source)
        self.assertIn("target_moe_n640_sha256(source)", source)
        self.assertIn("target_moe_n768_sha256(host)", source)

    def test_captured_b12x_has_no_materializer_dependency(self):
        source = (ROOT / "src/moe/target_moe_b12x_aot.cc").read_text()
        self.assertNotIn("target_moe_n640_materializer", source)
        self.assertNotIn("cudaMalloc", source)
        self.assertNotIn("cudaMemcpy", source)

    def test_versioned_digest_abi_is_an_explicit_build_input(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        self.assertIn('MATCHES "libcrypto\\\\.so\\\\.3$"', cmake)
        self.assertIn("add_library(rocket_qwen38_openssl3_crypto SHARED IMPORTED", cmake)
        self.assertIn("OpenSSL_version_num() >> 28", (
            ROOT / "src/moe/target_moe_n640_materializer.cu").read_text())

    def test_physical_comparator_invokes_pinned_flashinfer(self):
        script = (ROOT.parents[1] / "scripts/moe/compare-qwen38-target-moe-n768.py").read_text()
        self.assertIn("moe_dispatch._pad_intermediate_to_tile", script)
        self.assertIn("moe_dispatch._get_weight_views", script)
        self.assertNotIn("def _python_physical", script)
        self.assertIn('FLASHINFER_COMMIT = "91bda04c66f7cb851e1ab3b78b9fecea644b9844"', script)
        self.assertIn("pinned FlashInfer source identity changed", script)
        self.assertIn("FLASHINFER_FP4_HELPERS_SHA256", script)
        self.assertIn("FLASHINFER_W4A16_HOST_SHA256", script)

    def test_rank_layout_digests_are_fixed(self):
        source = (ROOT / "src/moe/target_moe_n640_materializer.cu").read_text()
        self.assertIn("ebf6db24c257c3516f7ff8c94bb2ba70d692c62c4cbf1b2577ebe99a3f56875b", source)
        self.assertIn("6e20c303b336f980e94e7aa3d897009ac527e1810279c09c8cbab1bd7867f841", source)
        self.assertIn("identity.source_layout_sha256 != expected.source_layout_sha256", source)


if __name__ == "__main__":
    unittest.main()

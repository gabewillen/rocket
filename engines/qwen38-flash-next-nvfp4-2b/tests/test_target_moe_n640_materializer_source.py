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


if __name__ == "__main__":
    unittest.main()

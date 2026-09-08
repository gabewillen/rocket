#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import hashlib
import pathlib
import unittest


ENGINE = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ENGINE / "vendor" / "flashinfer-91bda04"
CUTLASS_VENDOR = ENGINE / "vendor" / "cutlass-b46b16d"


class GdnFlashInferCutlassContract(unittest.TestCase):
    def test_cutlass_dependency_closure_is_byte_exact(self) -> None:
        manifest = CUTLASS_VENDOR / "SHA256SUMS"
        self.assertEqual(
            hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "3109195c7eec9395a2ee46a624d34549f5c57e61e30ff41695af6916b2b06f0f",
        )
        expected = {}
        for line in manifest.read_text().splitlines():
            digest, relative = line.split("  ", 1)
            expected[relative.removeprefix("./")] = digest
        actual = {
            relative: hashlib.sha256((CUTLASS_VENDOR / relative).read_bytes()).hexdigest()
            for relative in expected
        }
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 525)
        version = (CUTLASS_VENDOR / "include/cutlass/version.h").read_text()
        self.assertIn("#define CUTLASS_MAJOR 4", version)
        self.assertIn("#define CUTLASS_MINOR 5", version)
        self.assertIn("#define CUTLASS_PATCH 0", version)

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
        self.assertIn("vendor/cutlass-b46b16d", cmake)
        imported_target = cmake.split(
            "add_library(qwen38_gdn_flashinfer_cutlass STATIC", 1
        )[1].split("add_library(qwen38_gdn_graph SHARED", 1)[0]
        self.assertIn("ROCKET_QWEN38_FLASHINFER_CUTLASS_DIR", imported_target)
        self.assertNotIn("ROCKET_QWEN38_CUTLASS_DIR", imported_target)
        self.assertIn("CUDA_SEPARABLE_COMPILATION OFF", cmake)
        self.assertIn('CUDA_ARCHITECTURES "120f"', cmake)
        self.assertIn("CUDA_STANDARD 17", cmake)
        for flag in (
            "-use_fast_math",
            "-Xfatbin=-compress-all",
            "--compress-mode=size",
            "-static-global-template-stub=false",
        ):
            self.assertIn(flag, cmake)
        for definition in (
            "ENABLE_BF16=1",
            "ENABLE_FP4=1",
            "FLASHINFER_ENABLE_FP4_E2M1=1",
            "FLASHINFER_ENABLE_FP8_E8M0=1",
            "CUTLASS_ENABLE_GDC_FOR_SM100=1",
        ):
            self.assertIn(definition, cmake)
        decode_target = cmake.split(
            "add_library(qwen38_decode_execution STATIC", 1
        )[1].split("add_library(qwen38_layer_pair_reduce SHARED", 1)[0]
        self.assertIn("POSITION_INDEPENDENT_CODE ON", decode_target)
        self.assertIn("GdnFlashInferCutlassGemm qkvz_gemm", graph)
        self.assertIn("GdnFlashInferCutlassGemm ba_gemm", graph)
        self.assertIn("bucket->qkvz_gemm.run(stream)", graph)
        self.assertIn("bucket->ba_gemm.run(stream)", graph)
        self.assertIn('"flashinfer_cutlass_91bda04"', smoke)

    def test_wheel_runner_is_diagnostic_and_fails_closed(self) -> None:
        header = (ENGINE / "src/linear_attention/gdn_flashinfer_wheel.h").read_text()
        adapter = (ENGINE / "src/linear_attention/gdn_flashinfer_wheel.cc").read_text()
        graph_header = (ENGINE / "src/linear_attention/gdn_cutlass.h").read_text()
        smoke = (ENGINE / "bench/gdn_graph_smoke.cu").read_text()
        cmake = (ENGINE / "CMakeLists.txt").read_text()
        self.assertIn("Diagnostic-only adapter", header)
        self.assertIn("flashinfer-jit-cache==0.6.17+cu130", header)
        self.assertIn("13.0.88", header)
        self.assertIn("sm_120f", header)
        self.assertIn("b46b16d003484063bca4ed365e44095c4c6ed633", header)
        self.assertIn(
            "dfd9f2076fda819e45d169c88774bd26a9fc93bdc714592ab5dfe12b90bbf5ae",
            header,
        )
        self.assertIn("posix_spawn", adapter)
        self.assertIn("O_RDONLY | O_CLOEXEC | O_NOFOLLOW", adapter)
        self.assertIn('const_cast<char*>("-")', adapter)
        self.assertIn('"/proc/self/fd/"', adapter)
        self.assertIn("expected.st_ino == observed.st_ino", adapter)
        self.assertIn("RTLD_LAZY | RTLD_LOCAL", adapter)
        self.assertIn("gdn_flashinfer_cutlass_fallback_symbol()", adapter)
        self.assertIn("dlsym(impl_->library, symbol)", adapter)
        self.assertIn("fallback runner ABI mismatch", adapter)
        self.assertIn("fallback symbol interposed", adapter)
        self.assertNotIn("torch::", adapter)
        self.assertNotIn("tvm::", adapter)
        self.assertIn("kFlashInferWheelBenchmark", graph_header)
        self.assertIn(
            "GdnPrefillInputBackend::kFlashInferCutlass,\n"
            "                              std::string_view wheel_shared_object = {}",
            graph_header,
        )
        self.assertIn("--prefill-projection-flashinfer-wheel", smoke)
        self.assertIn("fallback_tactic=-1 fallback_cta=128x128x256", smoke)
        self.assertIn("fallback_scheduler=dp fallback_swap_ab=false", smoke)
        self.assertIn("fallback_cluster=1x1x1", smoke)
        self.assertIn("src/linear_attention/gdn_flashinfer_wheel.cc", cmake)
        self.assertIn("${CMAKE_DL_LIBS}", cmake)

    def test_cross_runtime_fixture_is_hashed_and_byte_preserving(self) -> None:
        smoke = (ENGINE / "bench/gdn_graph_smoke.cu").read_text()
        header = (ENGINE / "src/linear_attention/gdn_cutlass.h").read_text()
        profiler = (
            ENGINE.parent.parent / "scripts/attention/qwen38-gdn-prefill-phase.py"
        ).read_text()
        self.assertIn("rocket-gdn-fp4-fixture-v1", smoke)
        self.assertIn("--prefill-projection-flashinfer-wheel-fixture", smoke)
        self.assertIn("--prefill-projection-flashinfer-wheel-qkvz-cross", smoke)
        self.assertIn("capture_measure_python_scope", smoke)
        self.assertIn('\\"samples_us\\":[', smoke)
        self.assertIn("gdn_sha256_file(path.string())", smoke)
        for getter in (
            "qkvz_weight()", "qkvz_sfb()", "ba_weight()", "ba_sfb()",
            "projection_alpha()",
        ):
            self.assertIn(getter, header)
        self.assertIn("--dump-projection-fixtures", profiler)
        self.assertIn("--import-projection-fixtures", profiler)
        self.assertIn("QuantizedActivation(qkvz_a, qkvz_sfa", profiler)
        self.assertIn("projection fixture identity mismatch", profiler)
        self.assertIn('"ba_logical_mnk": [tokens, 48, 2560]', profiler)
        self.assertIn('"ba_physical_mnk": [tokens, 64, 2560]', profiler)
        self.assertIn(r'\"ba_physical_n\": 48', smoke)
        self.assertIn('Entry{"ba_b.bin", projection.ba_weight(), 48ULL', smoke)

    def test_qkvz_cross_reuses_runner_output_workspace_and_alpha(self) -> None:
        smoke = (ENGINE / "bench/gdn_graph_smoke.cu").read_text()
        wheel_header = (
            ENGINE / "src/linear_attention/gdn_flashinfer_wheel.h"
        ).read_text()
        self.assertEqual(smoke.count("GdnFlashInferWheelGemm qkvz;"), 1)
        self.assertIn("qkvz.bind_inputs(crossing.a, crossing.sfa", smoke)
        self.assertIn('Crossing{"synthetic", "synthetic"', smoke)
        self.assertIn('Crossing{"synthetic", "authenticated"', smoke)
        self.assertIn('Crossing{"authenticated", "synthetic"', smoke)
        self.assertIn('Crossing{"authenticated", "authenticated"', smoke)
        self.assertIn(r'\"output_workspace_alpha_fixed\":true', smoke)
        self.assertIn("bind_inputs", wheel_header)


if __name__ == "__main__":
    unittest.main()

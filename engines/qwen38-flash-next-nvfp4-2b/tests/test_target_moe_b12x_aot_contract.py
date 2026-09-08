#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "moe" / "export-qwen38-target-moe-b12x-aot.py"
ADAPTER = (
    ROOT
    / "engines"
    / "qwen38-flash-next-nvfp4-2b"
    / "src"
    / "moe"
    / "target_moe_b12x_aot.cc"
)
SPEC = importlib.util.spec_from_file_location("target_moe_b12x_aot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TargetMoeB12xAotContract(unittest.TestCase):
    def test_plan_matches_measured_fixed_c1_object(self) -> None:
        plan = MODULE.PLAN
        self.assertEqual(
            (
                plan.tokens,
                plan.hidden,
                plan.logical_intermediate,
                plan.physical_intermediate,
                plan.top_k,
                plan.max_rows,
                plan.local_experts,
                plan.state_experts,
                plan.tile_m,
                plan.tile_n,
                plan.max_active_clusters,
            ),
            (1, 2560, 640, 768, 10, 10, 256, 257, 64, 128, 20),
        )
        self.assertEqual(plan.activation, "silu")
        self.assertTrue(plan.fast_math)

    def test_source_and_measured_object_are_pinned(self) -> None:
        self.assertEqual(
            MODULE.PINNED_FLASHINFER_COMMIT,
            "91bda04c66f7cb851e1ab3b78b9fecea644b9844",
        )
        self.assertEqual(
            MODULE.PINNED_TVM_FFI_OBJECT_SHA256,
            "8cc49bdb4163b07338818bb7db482aea812d91cc1cbf3ef1eeecf0ee2756fef4",
        )

    def test_export_replaces_tvm_ffi_stream_lookup(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("use_tvm_ffi_env_stream=False", source)
        self.assertIn('__annotations__["stream"] = cuda.CUstream', source)
        self.assertNotIn("--enable-tvm-ffi", source)

    def test_exported_abi_rejects_implicit_stream_or_ffi(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            header = root / "kernel.h"
            object_file = root / "kernel.o"
            header.write_text(
                "qwen38_target_moe_b12x_c1_Kernel_Module_t\n"
                "cute_dsl_qwen38_target_moe_b12x_c1_wrapper\n"
                "cudaStream_t stream\n",
                encoding="utf-8",
            )
            object_file.write_bytes(b"explicit-stream-object")
            MODULE.authenticate_exported_abi(header, object_file)
            object_file.write_bytes(b"TVMFFIEnvGetStream")
            with self.assertRaisesRegex(RuntimeError, "TVM-FFI"):
                MODULE.authenticate_exported_abi(header, object_file)

    def test_real_manifest_recomputes_and_generates_ascii_identity(self) -> None:
        default = pathlib.Path(
            "/home/glwillen/calibration/qwen38-rank-slabs-fc694/"
            "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4/"
            "manifest.json"
        )
        manifest_path = pathlib.Path(
            os.environ.get("ROCKET_QWEN38_TARGET_SLAB_MANIFEST", default)
        )
        if not manifest_path.is_file():
            self.skipTest("authenticated target slab manifest is not installed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        canonical = dict(manifest)
        claimed = canonical.pop("artifact_key")
        recomputed = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(recomputed, claimed)
        expected = MODULE.authenticate_artifact_manifest(manifest_path)
        with tempfile.TemporaryDirectory() as directory:
            header = MODULE.emit_artifact_key_header(pathlib.Path(directory), expected)
            text = header.read_text(encoding="utf-8")
        self.assertIn(f' = "{expected}";', text)
        self.assertNotIn("std::array", text)

        for position in range(64):
            mutated = (
                expected[:position]
                + ("1" if expected[position] == "0" else "0")
                + expected[position + 1:]
            )
            self.assertNotEqual(mutated, expected)

    def test_production_enqueue_has_no_ownership_or_host_transfer_calls(self) -> None:
        source = ADAPTER.read_text(encoding="utf-8")
        begin = source.index("TargetMoeOutcome TargetMoeB12xAot::enqueue(")
        end = source.index("const TargetMoeB12xIdentity&", begin)
        hot_path = source[begin:end]
        for forbidden in (
            "new ",
            "cudaMalloc",
            "cudaFree",
            "cudaMemcpy",
            "cudaMemset",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "torch",
            "PyObject",
        ):
            self.assertNotIn(forbidden, hot_path)
        self.assertIn("launch.stream", hot_path)

    def test_production_has_no_manual_digest_byte_list(self) -> None:
        source = ADAPTER.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"0x[0-9a-fA-F]{2}")
        self.assertIn("target_moe_artifact_key_ascii()", source)


if __name__ == "__main__":
    unittest.main()

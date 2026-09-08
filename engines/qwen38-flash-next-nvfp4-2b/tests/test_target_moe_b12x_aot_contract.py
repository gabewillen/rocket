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
CONFIG = ROOT / "scripts" / "moe" / "qwen38-target-moe-compact-e10.json"
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
                plan.weight_experts,
                plan.state_experts,
                plan.tile_m,
                plan.tile_n,
                plan.max_active_clusters,
            ),
            (1, 2560, 640, 768, 10, 10, 10, 11, 64, 128, 20),
        )
        self.assertEqual(plan.activation, "silu")
        self.assertTrue(plan.fast_math)

    def test_compact_config_authenticates_exact_stage_consumer_abi(self) -> None:
        config = MODULE.authenticate_compact_config(
            CONFIG,
            "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4",
        )
        self.assertEqual(sum(item["bytes"] for item in config["planes"]), 33_177_760)
        self.assertEqual(sum(item["bytes"] for item in config["control"]), 136)
        self.assertEqual(config["shape"]["weight_experts"], 10)
        self.assertEqual(config["shape"]["state_experts"], 11)
        self.assertEqual(config["route_remap_abi"], MODULE.ROUTE_REMAP_ABI)
        self.assertEqual(
            MODULE.canonical_digest(config), MODULE.PINNED_COMPACT_CONFIG_SHA256
        )

    def test_compact_config_rejects_every_identity_and_layout_family(self) -> None:
        original = json.loads(CONFIG.read_text(encoding="utf-8"))
        artifact = original["target_artifact_key"]
        mutations = []
        for field in ("target_artifact_key", "source_abi", "transform_abi", "route_remap_abi"):
            changed = json.loads(json.dumps(original))
            changed[field] += "-changed"
            mutations.append(changed)
        for field in ("weight_experts", "state_experts", "physical_intermediate"):
            changed = json.loads(json.dumps(original))
            changed["shape"][field] += 1
            mutations.append(changed)
        for collection in ("planes", "control"):
            changed = json.loads(json.dumps(original))
            changed[collection][0]["bytes"] += 1
            mutations.append(changed)
        changed = json.loads(json.dumps(original))
        changed["fc1_physical_rows"][1] = "nonzero[640:768]"
        mutations.append(changed)
        for rank in range(2):
            for field in (
                "descriptor_sha256",
                "binding_inventory_sha256",
                "publication_layout_sha256",
            ):
                changed = json.loads(json.dumps(original))
                value = changed["ranks"][rank][field]
                changed["ranks"][rank][field] = (
                    ("1" if value[0] == "0" else "0") + value[1:]
                )
                mutations.append(changed)
        changed = json.loads(json.dumps(original))
        changed["unexpected"] = True
        mutations.append(changed)

        for changed in mutations:
            with self.subTest(changed=changed):
                with tempfile.TemporaryDirectory() as directory:
                    path = pathlib.Path(directory) / "config.json"
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        MODULE.authenticate_compact_config(path, artifact)

    def test_compact_header_carries_all_fixed_identities(self) -> None:
        config = MODULE.authenticate_compact_config(
            CONFIG,
            "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4",
        )
        with tempfile.TemporaryDirectory() as directory:
            header, config_sha = MODULE.emit_compact_config_header(
                pathlib.Path(directory), config
            )
            text = header.read_text(encoding="utf-8")
        self.assertIn("TargetMoeWeightExperts = 10", text)
        self.assertIn("TargetMoeStateExperts = 11", text)
        self.assertIn("TargetMoePhysicalIntermediate = 768", text)
        self.assertIn(config_sha, text)
        self.assertIn(MODULE.compact_layout_identity(config), text)
        for rank in config["ranks"]:
            self.assertIn(rank["descriptor_sha256"], text)
            self.assertIn(rank["binding_inventory_sha256"], text)
            self.assertIn(rank["publication_layout_sha256"], text)

    def test_unmodified_physical_n768_source_is_pinned(self) -> None:
        self.assertEqual(
            MODULE.PINNED_FLASHINFER_COMMIT,
            "91bda04c66f7cb851e1ab3b78b9fecea644b9844",
        )
        self.assertEqual(
            MODULE.PINNED_DISPATCH_SHA256,
            "c518e65d6bfd7f08db1e5261e20795fd020e82e681699729171bc2fd5331239a",
        )
        self.assertEqual(
            MODULE.PINNED_KERNEL_SHA256,
            "c7b6f24b94d7939cc0eb917ab15cef4c34f3dc12bf75e15fc9dc316ee7327f3f",
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

    def test_build_requires_generated_compact_config(self) -> None:
        cmake = (ROOT / "engines" / "qwen38-flash-next-nvfp4-2b" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("target_moe_compact_config.h", cmake)
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--compact-config", type=Path)', source)
        self.assertIn("--compact-config is required", source)

    def test_production_has_no_manual_digest_byte_list(self) -> None:
        source = ADAPTER.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"0x[0-9a-fA-F]{2}")
        self.assertIn("target_moe_artifact_key_ascii()", source)


if __name__ == "__main__":
    unittest.main()

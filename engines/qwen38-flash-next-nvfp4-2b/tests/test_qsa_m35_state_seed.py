# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from qwen38_slab.qsa_m35_state_seed import (
    C1_SCHEMA,
    IMPLEMENTATION,
    ORACLE_MANIFEST_SHA256,
    QsaM35StateSeedError,
    SCHEMA,
    _C1_LAYOUTS,
    _LAYOUTS,
    _Bundle,
    _Config,
    _Record,
    _StateView,
    authenticate_qsa_c1_input_bundle,
    authenticate_qsa_m35_state_bundle,
)


class QsaM35StateSeedTests(unittest.TestCase):
    def test_ctypes_layout_matches_native_static_contract(self):
        import ctypes

        self.assertEqual(ctypes.sizeof(_Record), 6)
        self.assertEqual(ctypes.sizeof(_Config), 32)
        self.assertEqual(ctypes.sizeof(_Bundle), 64)
        self.assertEqual(ctypes.sizeof(_StateView), 176)

    def bundle(self, root: Path) -> Path:
        entries = []
        for index, (name, (dtype, shape, element_bytes)) in enumerate(_LAYOUTS.items()):
            payload = bytes([index + 1]) * (__import__("math").prod(shape) * element_bytes)
            filename = name + ".bin"
            (root / filename).write_bytes(payload)
            entries.append({
                "name": name, "file": filename, "dtype": dtype,
                "shape": list(shape), "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        manifest = {
            "schema": SCHEMA, "oracle_manifest_sha256": ORACLE_MANIFEST_SHA256,
            "implementation": IMPLEMENTATION, "source_sha256": "e" * 64,
            "rank": 0, "layer": 3, "rows": 35, "generation_index": 0,
            "tensors": entries,
        }
        key = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        bundle = root / key
        bundle.mkdir()
        for entry in entries:
            (root / entry["file"]).rename(bundle / entry["file"])
        manifest["artifact_key"] = key
        (bundle / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        return bundle

    def c1_bundle(self, root: Path) -> Path:
        entries = []
        for index, (name, (dtype, shape, element_bytes)) in enumerate(_C1_LAYOUTS.items()):
            size = __import__("math").prod(shape) * element_bytes
            payload = bytes([index + 1]) * size
            if name == "row35_positions":
                payload = __import__("struct").pack("<qqq", 35, 35, 35)
            filename = name + ".bin"
            (root / filename).write_bytes(payload)
            entries.append({
                "name": name, "file": filename, "dtype": dtype,
                "shape": list(shape), "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        parent_manifest = {
            "schema": SCHEMA,
            "oracle_manifest_sha256": ORACLE_MANIFEST_SHA256,
            "implementation": IMPLEMENTATION, "source_sha256": "e" * 64,
            "rank": 0, "layer": 3, "rows": 35, "generation_index": 0,
            "tensors": entries[:len(_LAYOUTS)],
        }
        parent_key = hashlib.sha256(json.dumps(
            parent_manifest, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        manifest = {
            "schema": C1_SCHEMA,
            "parent_artifact_key": parent_key,
            "oracle_manifest_sha256": ORACLE_MANIFEST_SHA256,
            "implementation": IMPLEMENTATION, "source_sha256": "e" * 64,
            "indexer_source_sha256":
                "e2f398a2fe29466c9681627651ccb5b8eb2b5980445c9e44b270ff45bcb61066",
            "rank": 0, "layer": 3, "rows": 35, "c1_position": 35,
            "generation_index": 0, "tensors": entries,
        }
        key = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        bundle = root / key
        bundle.mkdir()
        for entry in entries:
            (root / entry["file"]).rename(bundle / entry["file"])
        manifest["artifact_key"] = key
        (bundle / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        return bundle

    def test_authenticates_exact_v2_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.bundle(Path(directory))
            loaded = authenticate_qsa_m35_state_bundle(bundle)
            self.assertEqual(set(loaded), set(_LAYOUTS))

    def test_rejects_payload_and_identity_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.bundle(Path(directory))
            (bundle / "raw_state.bin").write_bytes(b"changed")
            with self.assertRaises(QsaM35StateSeedError):
                authenticate_qsa_m35_state_bundle(bundle)

    def test_authenticates_c1_extension_and_rejects_parent_or_position(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.c1_bundle(Path(directory))
            loaded = authenticate_qsa_c1_input_bundle(bundle)
            self.assertEqual(set(loaded), set(_C1_LAYOUTS))
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.c1_bundle(Path(directory))
            manifest = json.loads((bundle / "manifest.json").read_text())
            manifest["parent_artifact_key"] = "0" * 64
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(QsaM35StateSeedError):
                authenticate_qsa_c1_input_bundle(bundle)
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.c1_bundle(Path(directory))
            (bundle / "row35_positions.bin").write_bytes(
                __import__("struct").pack("<qqq", 34, 34, 34)
            )
            with self.assertRaises(QsaM35StateSeedError):
                authenticate_qsa_c1_input_bundle(bundle)
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.bundle(Path(directory))
            manifest = json.loads((bundle / "manifest.json").read_text())
            manifest["rows"] = 87
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(QsaM35StateSeedError):
                authenticate_qsa_m35_state_bundle(bundle)


if __name__ == "__main__":
    unittest.main()

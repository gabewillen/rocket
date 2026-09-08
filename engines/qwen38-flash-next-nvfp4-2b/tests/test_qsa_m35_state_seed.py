# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from qwen38_slab.qsa_m35_state_seed import (
    IMPLEMENTATION,
    ORACLE_MANIFEST_SHA256,
    QsaM35StateSeedError,
    SCHEMA,
    _LAYOUTS,
    _Bundle,
    _Config,
    _Record,
    _StateView,
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
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.bundle(Path(directory))
            manifest = json.loads((bundle / "manifest.json").read_text())
            manifest["rows"] = 87
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(QsaM35StateSeedError):
                authenticate_qsa_m35_state_bundle(bundle)


if __name__ == "__main__":
    unittest.main()

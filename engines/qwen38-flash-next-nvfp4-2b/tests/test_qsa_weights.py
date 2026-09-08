# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from qwen38_slab.qsa_weights import (
    QSA_WEIGHTS_SCHEMA,
    QsaWeightsError,
    load_qsa_weights,
    validate_all_qsa_bindings,
    validate_qsa_weight_inventory,
)

REAL_SLAB = Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/"
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
)


class QsaWeightsTests(unittest.TestCase):
    @unittest.skipUnless(REAL_SLAB.is_dir(), "production rank slab unavailable")
    def test_all_24_target_bindings_are_authenticated(self):
        bindings = validate_all_qsa_bindings(REAL_SLAB)
        self.assertEqual(len(bindings), 24)
        self.assertEqual({item.rank for item in bindings}, {0, 1})
        self.assertEqual({item.layer for item in bindings}, set(range(3, 48, 4)))
        self.assertTrue(all(item.schema == QSA_WEIGHTS_SCHEMA for item in bindings))
        self.assertTrue(all(len(item.components) == 21 for item in bindings))
        indexers = [
            item
            for binding in bindings
            for item in binding.components
            if item.name.endswith("indexer.index_qk_proj.weight")
        ]
        self.assertTrue(all(item.dtype == "BF16" and item.abi == "native" for item in indexers))
        main = [
            item
            for binding in bindings
            for item in binding.components
            if item.name.endswith("self_attn.q_proj.weight")
        ]
        self.assertTrue(all(item.dtype == "U8" and item.abi != "native" for item in main))
        with self.assertRaises(FrozenInstanceError):
            bindings[0].rank = 1

    @unittest.skipUnless(REAL_SLAB.is_dir(), "production rank slab unavailable")
    def test_identity_and_topology_fail_closed(self):
        with self.assertRaisesRegex(QsaWeightsError, "Path"):
            load_qsa_weights(str(REAL_SLAB), 0, 3)
        manifest = json.loads((REAL_SLAB / "manifest.json").read_bytes())
        with self.assertRaisesRegex(QsaWeightsError, "rank"):
            validate_qsa_weight_inventory(manifest, 2, 3)
        with self.assertRaisesRegex(QsaWeightsError, "topology"):
            validate_qsa_weight_inventory(manifest, 0, 4)
        entry = next(
            row for row in manifest["slabs"]["rank0-target"]["entries"]
            if row["name"].endswith("layers.3.self_attn.indexer.index_qk_proj.weight")
        )
        entry["dtype"] = "U8"
        with self.assertRaisesRegex(QsaWeightsError, "index_qk_proj"):
            validate_qsa_weight_inventory(manifest, 0, 3)


if __name__ == "__main__":
    unittest.main()

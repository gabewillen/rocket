# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import os
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from qwen38_slab.linear_attention import (
    GDN_COMPONENT_CONTRACTS,
    GDN_PREFIX,
    GDN_SCHEMA,
    GDN_STATE_FAMILIES,
    LinearAttentionSlabError,
    load_rank0_layer0_gdn,
    validate_rank0_layer0_gdn_inventory,
)
from qwen38_slab.contract import PINNED_CONTRACT, SCHEMA
from qwen38_slab.state_txn import STATE_FAMILIES

REAL_SLAB = Path(os.environ.get("ROCKET_QWEN38_RANK_SLAB", "/nonexistent"))


class LinearAttentionSlabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = {
            "schema": SCHEMA,
            "revision": PINNED_CONTRACT.revision,
            "overlay_artifact_key": PINNED_CONTRACT.artifact_key,
            "overlay_sha256": PINNED_CONTRACT.overlay_sha256,
            "tp_size": 2,
            "tensor_alignment_bytes": PINNED_CONTRACT.tensor_alignment_bytes,
            "slabs": {"rank0-target": {"entries": [
                {"name": f"{GDN_PREFIX}.{suffix}", "offset_bytes": offset,
                 "length_bytes": length, "shape": list(shape), "dtype": dtype,
                 "layout": layout, "abi": abi}
                for suffix, offset, length, shape, dtype, layout, abi
                in GDN_COMPONENT_CONTRACTS
            ]}},
        }

    def _entry(self, manifest, suffix):
        name = f"{GDN_PREFIX}.{suffix}"
        return next(x for x in manifest["slabs"]["rank0-target"]["entries"] if x["name"] == name)

    @unittest.skipUnless(REAL_SLAB.is_dir(), "production rank slab unavailable")
    def test_exact_production_descriptor_is_immutable(self):
        descriptor = load_rank0_layer0_gdn(REAL_SLAB)
        self.assertEqual(descriptor.schema, GDN_SCHEMA)
        self.assertEqual(len(descriptor.components), 24)
        self.assertEqual(descriptor.components[0].offset, 1_297_735_680)
        self.assertEqual(descriptor.components[-1].offset, 1_314_102_272)
        with self.assertRaises(FrozenInstanceError):
            descriptor.slab_bytes = 0

    def test_wrong_a_log_dtype_fails_before_binding(self):
        manifest = copy.deepcopy(self.manifest)
        self._entry(manifest, "A_log")["dtype"] = "F32"
        with self.assertRaisesRegex(LinearAttentionSlabError, "A_log"):
            validate_rank0_layer0_gdn_inventory(manifest)

    def test_recurrent_families_remain_in_nine_family_transaction(self):
        self.assertEqual(GDN_STATE_FAMILIES, STATE_FAMILIES[3:5])

    def test_missing_family_fails_before_binding(self):
        manifest = copy.deepcopy(self.manifest)
        entries = manifest["slabs"]["rank0-target"]["entries"]
        entries.remove(self._entry(manifest, "conv1d.weight"))
        with self.assertRaisesRegex(LinearAttentionSlabError, "conv1d.weight"):
            validate_rank0_layer0_gdn_inventory(manifest)

    def test_wrong_offset_fails_before_binding(self):
        manifest = copy.deepcopy(self.manifest)
        self._entry(manifest, "dt_bias")["offset_bytes"] += 256
        with self.assertRaisesRegex(LinearAttentionSlabError, "dt_bias"):
            validate_rank0_layer0_gdn_inventory(manifest)

    def test_missing_projection_fails_before_binding(self):
        manifest = copy.deepcopy(self.manifest)
        entries = manifest["slabs"]["rank0-target"]["entries"]
        entries.remove(self._entry(manifest, "in_proj_qkv.weight"))
        with self.assertRaisesRegex(LinearAttentionSlabError, "in_proj_qkv.weight"):
            validate_rank0_layer0_gdn_inventory(manifest)

    def test_core_selects_owner_device_before_stream_tail_clear(self):
        source = (
            Path(__file__).parents[1] / "src" / "linear_attention" / "gdn_core.cu"
        ).read_text()
        launch = source[source.index("void CorePlan::launch(") :]
        self.assertLess(
            launch.index('cudaSetDevice(impl_->device)'),
            launch.index('"clear inactive GDN rows"'),
        )


if __name__ == "__main__":
    unittest.main()

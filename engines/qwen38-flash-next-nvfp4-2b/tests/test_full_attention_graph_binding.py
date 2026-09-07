from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from qwen38_slab.full_attention_graph import (
    CURRENT_ARTIFACT_KEY,
    FULL_ATTENTION_LAYERS,
    FullAttentionBindingError,
    load_full_attention_layer_binding,
    reconstruct_index_qk_weight,
)

REAL_SLAB = Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/"
    f"{CURRENT_ARTIFACT_KEY}"
)


class FullAttentionGraphBindingTests(unittest.TestCase):
    def test_rank_layer_and_path_fail_before_io(self):
        with self.assertRaisesRegex(FullAttentionBindingError, "must be a Path"):
            load_full_attention_layer_binding(str(REAL_SLAB), rank=0, layer=3)
        with self.assertRaisesRegex(FullAttentionBindingError, "rank"):
            load_full_attention_layer_binding(REAL_SLAB, rank=2, layer=3)
        with self.assertRaisesRegex(FullAttentionBindingError, "layer"):
            load_full_attention_layer_binding(REAL_SLAB, rank=0, layer=4)

    @unittest.skipUnless(REAL_SLAB.is_dir(), "published rank slab unavailable")
    def test_layer3_authenticates_both_legacy_halves(self):
        bindings = [
            load_full_attention_layer_binding(REAL_SLAB, rank=rank, layer=3)
            for rank in (0, 1)
        ]
        self.assertEqual(tuple(item.layer for item in bindings), (3, 3))
        self.assertEqual(tuple(item.rank for item in bindings), (0, 1))
        self.assertTrue(all(item.artifact_key == CURRENT_ARTIFACT_KEY for item in bindings))
        self.assertTrue(all(len(item.local_extents) == 16 for item in bindings))
        self.assertEqual(
            bindings[0].index_qk_halves,
            bindings[1].index_qk_halves,
        )
        rebuilt0 = reconstruct_index_qk_weight(bindings[0])
        rebuilt1 = reconstruct_index_qk_weight(bindings[1])
        self.assertEqual(len(rebuilt0), 640 * 2560 * 2)
        self.assertEqual(hashlib.sha256(rebuilt0).digest(), hashlib.sha256(rebuilt1).digest())

    def test_fixed_layer_set_is_exact(self):
        self.assertEqual(FULL_ATTENTION_LAYERS, tuple(range(3, 48, 4)))


if __name__ == "__main__":
    unittest.main()

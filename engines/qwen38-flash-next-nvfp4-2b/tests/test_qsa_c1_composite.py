# SPDX-License-Identifier: Apache-2.0
import unittest

from qwen38_slab.native_qsa import ARENA_FIELDS
from qwen38_slab.qsa_c1_composite import (
    ARENA_BYTES,
    GRAPH_STORAGE_BYTES,
    arena_offsets,
    mismatch_summary,
)


class QsaC1CompositeTests(unittest.TestCase):
    def test_exact_arena_inventory_fits_one_owner_extent(self):
        offsets = arena_offsets()
        self.assertEqual(tuple(offsets), ARENA_FIELDS)
        self.assertEqual(tuple(ARENA_BYTES), ARENA_FIELDS)
        for offset, size in offsets.values():
            self.assertEqual(offset % 256, 0)
            self.assertGreater(size, 0)
            self.assertLessEqual(offset + size, GRAPH_STORAGE_BYTES)
        self.assertEqual(GRAPH_STORAGE_BYTES, 812_544)

    def test_arena_views_do_not_overlap(self):
        spans = sorted(arena_offsets().values())
        for left, right in zip(spans, spans[1:]):
            self.assertLessEqual(left[0] + left[1], right[0])

    def test_bounded_exact_byte_mismatch_summary(self):
        self.assertEqual(mismatch_summary(b"abc", b"abc"), (0, -1))
        self.assertEqual(mismatch_summary(b"abc", b"axd"), (2, 1))
        with self.assertRaisesRegex(RuntimeError, "comparison_extent"):
            mismatch_summary(b"a", b"ab")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path

from qwen38_slab.hyperconnection import (
    HC_LOW_RANK,
    HC_SCHEMA,
    HC_STREAMS,
    HC_WIDTH,
    load_layer3_hyperconnection,
)

REAL_SLAB = Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/"
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
)


class HyperConnectionPayloadTests(unittest.TestCase):
    @unittest.skipUnless(REAL_SLAB.is_dir(), "production rank slab unavailable")
    def test_layer3_rank0_weights_are_exact_and_authenticated(self):
        payload = load_layer3_hyperconnection(REAL_SLAB)
        self.assertEqual(payload.schema, HC_SCHEMA)
        self.assertEqual((payload.descriptor.layer, payload.descriptor.rank), (3, 0))
        for family in (payload.attention, payload.mlp):
            self.assertEqual(len(family.norm_bf16), HC_WIDTH * 2)
            self.assertEqual(len(family.down_bf16), HC_LOW_RANK * HC_WIDTH * 2)
            self.assertEqual(len(family.injection_bf16), HC_STREAMS * HC_WIDTH * 2)
            self.assertEqual(len(family.up_bf16), HC_WIDTH * HC_LOW_RANK * 2)


if __name__ == "__main__":
    unittest.main()

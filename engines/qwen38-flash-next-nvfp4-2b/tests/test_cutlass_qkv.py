from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from qwen38_slab.cutlass_qkv import CutlassQkvRuntime
from qwen38_slab.device_decode import DeviceDecodeError
from qwen38_slab.projection import (
    load_full_projection_payload,
    load_rank0_layer3_projection,
)

REAL_SLAB = Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/"
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
)


class CutlassQkvContractTests(unittest.TestCase):
    def test_runtime_rejects_non_payload_before_loading_cuda(self):
        with self.assertRaisesRegex(DeviceDecodeError, "payload ABI"):
            CutlassQkvRuntime(object())

    @unittest.skipUnless(REAL_SLAB.is_dir(), "production rank slab unavailable")
    def test_runtime_rejects_truncated_production_extent_before_loading_cuda(self):
        payload = load_full_projection_payload(load_rank0_layer3_projection(REAL_SLAB))
        broken = replace(
            payload,
            packed_weights=(payload.packed_weights[0][:-1], *payload.packed_weights[1:]),
        )
        with self.assertRaisesRegex(DeviceDecodeError, "payload extent"):
            CutlassQkvRuntime(broken)


if __name__ == "__main__":
    unittest.main()

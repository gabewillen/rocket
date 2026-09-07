from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from qwen38_slab.device_decode import Cuda13GraphRuntime, DeviceDecodeError
from qwen38_slab.projection import (
    PROJECTION_K,
    PROJECTION_OUTPUTS,
    PROJECTION_SCHEMA,
    ProjectionError,
    load_projection_payload,
    load_rank0_layer3_projection,
    reference_projection,
)

REAL_SLAB = Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/"
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
)


class ProjectionContractTests(unittest.TestCase):
    def test_artifact_path_type_fails_closed(self):
        with self.assertRaisesRegex(ProjectionError, "must be a Path"):
            load_rank0_layer3_projection(str(REAL_SLAB))

    @unittest.skipUnless(REAL_SLAB.is_dir(), "production rank slab unavailable")
    def test_production_descriptor_and_payload_are_fixed(self):
        descriptor = load_rank0_layer3_projection(REAL_SLAB)
        self.assertEqual(descriptor.schema, PROJECTION_SCHEMA)
        self.assertEqual((descriptor.rank, descriptor.layer), (0, 3))
        self.assertEqual(len(descriptor.components), 9)
        with self.assertRaises(FrozenInstanceError):
            descriptor.rank = 1
        payload = load_projection_payload(descriptor)
        self.assertEqual(len(payload.packed_weights), PROJECTION_OUTPUTS * PROJECTION_K // 2)
        self.assertEqual(len(payload.linear_scales), PROJECTION_OUTPUTS * PROJECTION_K // 16)
        self.assertEqual(len(payload.activations_bf16), 16 * PROJECTION_K * 2)
        self.assertEqual(reference_projection(payload, 0), (0.0,) * (16 * PROJECTION_OUTPUTS))
        with self.assertRaisesRegex(ProjectionError, "descriptor"):
            load_projection_payload(replace(descriptor, schema="drift"))
        with self.assertRaisesRegex(DeviceDecodeError, "payload ABI"):
            Cuda13GraphRuntime(
                projection=replace(payload, packed_weights=payload.packed_weights[:-1])
            )

    @unittest.skipUnless(REAL_SLAB.is_dir(), "production rank slab unavailable")
    def test_scalar_reference_is_deterministic_and_pads_rows(self):
        payload = load_projection_payload(load_rank0_layer3_projection(REAL_SLAB))
        first = reference_projection(payload, 1)
        second = reference_projection(payload, 1)
        self.assertEqual(first, second)
        self.assertTrue(any(value != 0 for value in first[:PROJECTION_OUTPUTS]))
        self.assertTrue(all(value == 0 for value in first[PROJECTION_OUTPUTS:]))


if __name__ == "__main__":
    unittest.main()

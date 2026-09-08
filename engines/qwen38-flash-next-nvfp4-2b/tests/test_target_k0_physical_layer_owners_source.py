# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from pathlib import Path


class TargetK0PhysicalLayerOwnersSourceTest(unittest.TestCase):
    def test_shared_qsa_assets_have_one_factory_owner(self) -> None:
        root = Path(__file__).parents[1] / "src"
        factory = (root / "decode/target_k0_physical_layer_owners.cu").read_text()
        qsa = (root / "decode/target_qsa_k0_layer_owner.cu").read_text()
        header = (root / "decode/target_k0_physical_layer_owners.h").read_text()
        self.assertEqual(factory.count("make_unique<attention::QsaSidecarDeviceOwner>"), 1)
        self.assertEqual(factory.count("make_unique<attention::Layer3RopeDeviceOwner>"), 1)
        self.assertNotIn("make_unique<attention::QsaSidecarDeviceOwner>", qsa)
        self.assertNotIn("make_unique<attention::Layer3RopeDeviceOwner>", qsa)
        self.assertIn("sidecar.identity.layer3_sha256 != expected_sidecar.layer3_sha256", qsa)
        self.assertLess(header.rindex("qsa_sidecar_"), header.rindex("inventory_"))
        self.assertLess(header.rindex("qsa_rope_"), header.rindex("inventory_"))
        self.assertIn("TargetK0OracleQsaStateOwner", header)

    def test_oracle35_scope_is_hard_cut(self) -> None:
        root = Path(__file__).parents[1] / "src"
        state = (root / "attention/target_k0_qsa_state_owner.cu").read_text()
        factory = (root / "decode/target_k0_physical_layer_owners.cu").read_text()
        self.assertIn("max_rows != 35", state)
        self.assertIn("device, rank, 35", factory)
        self.assertIn("stage_telemetry, 35", factory)


if __name__ == "__main__":
    unittest.main()

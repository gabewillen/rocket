# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from pathlib import Path

from qwen38_slab.target_layer_descriptor import (
    authenticate_descriptor_identity, load_descriptor_allowlist,
)


class TargetLayerDescriptorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.allowlist = load_descriptor_allowlist(
            Path(__file__).parents[1] / "src/decode/target_layer_descriptor_identities.json")

    def test_all_96_rank_layer_identities_are_accepted(self) -> None:
        self.assertEqual({(item["rank"], item["layer"]) for item in self.allowlist},
                         {(rank, layer) for rank in (0, 1) for layer in range(48)})
        for identity in self.allowlist:
            with self.subTest(rank=identity["rank"], layer=identity["layer"]):
                self.assertTrue(authenticate_descriptor_identity(identity, self.allowlist))

    def test_each_of_96_descriptor_mutations_is_rejected(self) -> None:
        for identity in self.allowlist:
            mutated = dict(identity)
            mutated["descriptor_sha256"] = "0" * 64
            with self.subTest(rank=identity["rank"], layer=identity["layer"]):
                self.assertFalse(authenticate_descriptor_identity(mutated, self.allowlist))


if __name__ == "__main__":
    unittest.main()

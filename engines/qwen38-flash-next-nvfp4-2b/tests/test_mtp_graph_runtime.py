# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

from qwen38_slab.decode import Depth
from qwen38_slab.mtp_graph_runtime import MtpGraphKey, MtpGraphRuntimeError, Residency
from pathlib import Path


class MtpReferenceOracleContractTests(unittest.TestCase):
    def test_source_excludes_oracle_from_production(self):
        source = (
            Path(__file__).parents[1] / "src" / "qwen38_slab" / "mtp_graph_runtime.py"
        ).read_text()
        self.assertIn("excluded from the production", source)
        self.assertIn("class MtpReferenceOracle", source)

    def test_hot_draft_is_one_graph_replay_without_step_loop(self):
        source = (
            Path(__file__).parents[1] / "src" / "qwen38_slab" / "mtp_graph_runtime.py"
        ).read_text()
        body = source.split("    def draft(", 1)[1].split("    def stage_accept", 1)[0]
        self.assertEqual(body.count("proposal_graph.replay()"), 1)
        self.assertNotIn("for step", body)

    def test_k1_k3_are_resident_and_k4_is_lazy_at_c16(self):
        for depth in (Depth.K1, Depth.K2, Depth.K3):
            key = MtpGraphKey(depth, 16)
            self.assertEqual(
                Residency.RESIDENT if int(key.depth) <= 3 else Residency.LAZY,
                Residency.RESIDENT,
            )
        self.assertEqual(
            Residency.RESIDENT if int(MtpGraphKey(Depth.K4, 16).depth) <= 3 else Residency.LAZY,
            Residency.LAZY,
        )

    def test_k5_k7_are_low_concurrency_only(self):
        for depth in (Depth.K5, Depth.K6, Depth.K7):
            for sequences in (1, 2, 4):
                self.assertEqual(MtpGraphKey(depth, sequences).sequences, sequences)
            with self.assertRaisesRegex(MtpGraphRuntimeError, "K1-K7 policy"):
                MtpGraphKey(depth, 8)


if __name__ == "__main__":
    unittest.main()

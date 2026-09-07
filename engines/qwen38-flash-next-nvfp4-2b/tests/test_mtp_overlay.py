# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

from qwen38_slab.mtp_overlay import (
    FP8_SOURCE_BYTES_PER_EXPERT,
    NVFP4_SERVING_BYTES_PER_EXPERT,
    INTERACTION_FAMILIES,
    PROJECTION_FAMILIES,
    MtpOverlayError,
    QuantizationEvidence,
    compare_mtp_expert_overlay,
)


class MtpOverlayTests(unittest.TestCase):
    def test_actual_route_counts_keep_source_and_serving_dtype_separate(self):
        result = compare_mtp_expert_overlay((4, 7, 3, 9))
        self.assertEqual(result.source_weight_bytes, 23 * FP8_SOURCE_BYTES_PER_EXPERT)
        self.assertEqual(result.serving_weight_bytes, 23 * NVFP4_SERVING_BYTES_PER_EXPERT)
        self.assertEqual(result.saved_weight_bytes, 49_472_448)
        self.assertAlmostEqual(result.current_control_tokens_per_second, 457.524496688903)
        self.assertAlmostEqual(result.hypothetical_tokens_per_second, 469.3926469156607)
        self.assertAlmostEqual(result.current_control_stream_tokens_per_second, 28.59528104305644)
        self.assertAlmostEqual(result.hypothetical_stream_tokens_per_second, 29.337040432228793)
        self.assertFalse(result.quality_evidence_complete)

    def test_quality_claim_fails_closed_until_all_interactions_are_measured(self):
        incomplete = {"gate_proj": 0.0}
        evidence = QuantizationEvidence(
            {"gate_proj": (-1.0, 1.0)}, incomplete, 0.0, 0.0, 1.0, incomplete
        )
        with self.assertRaisesRegex(MtpOverlayError, "every MTP family"):
            compare_mtp_expert_overlay((8,), evidence)

        projections = {family: 0.0 for family in PROJECTION_FAMILIES}
        ranges = {family: (-1.0, 1.0) for family in PROJECTION_FAMILIES}
        interactions = {family: 0.0 for family in INTERACTION_FAMILIES}
        evidence = QuantizationEvidence(
            ranges, projections, 0.0, 0.0, 1.0, interactions
        )
        self.assertTrue(compare_mtp_expert_overlay((8,), evidence).quality_evidence_complete)


if __name__ == "__main__":
    unittest.main()

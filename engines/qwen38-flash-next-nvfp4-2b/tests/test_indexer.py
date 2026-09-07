from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from qwen38_slab.indexer import (
    QSA_BLOCK_TOPK,
    QSA_EXPANDED_WIDTH,
    QsaIndexerContract,
    QsaIndexerError,
    reference_expand_qsa_topk,
    reference_select_qsa_blocks,
)


class QsaIndexerContractTests(unittest.TestCase):
    def test_fixed_qwen_shape_is_immutable(self):
        contract = QsaIndexerContract()
        self.assertEqual((contract.block_topk, contract.output_width), (512, 2051))
        with self.assertRaises(FrozenInstanceError):
            contract.block_topk = 2048

    def test_expansion_is_causal_and_appends_open_tail(self):
        blocks = (tuple(range(QSA_BLOCK_TOPK)),)
        output = reference_expand_qsa_topk(blocks, (10,), (11,), (0,))[0]
        self.assertEqual(output[:8], tuple(range(8)))
        self.assertEqual(output[8:11], (8, 9, 10))
        self.assertTrue(all(value == -1 for value in output[11:]))

    def test_invalid_request_fills_minus_one_and_extent_drift_fails(self):
        blocks = (tuple(range(QSA_BLOCK_TOPK)),)
        output = reference_expand_qsa_topk(blocks, (7,), (8,), (-1,))[0]
        self.assertEqual(output, (-1,) * QSA_EXPANDED_WIDTH)
        with self.assertRaisesRegex(QsaIndexerError, "input ABI"):
            reference_expand_qsa_topk(((0,),), (0,), (1,), (0,))

    def test_score_select_is_relu_scaled_stable_and_minus_one_padded(self):
        zero = (0.0,) * 128
        positive = (1.0,) + (0.0,) * 127
        negative = (-2.0,) + (0.0,) * 127
        query = ((positive, positive, negative, zero),)
        keys = ((1.0,) + zero[1:], (2.0,) + zero[1:], (2.0,) + zero[1:])
        selected = reference_select_qsa_blocks(query, keys, (3,))[0]
        self.assertEqual(selected[:3], (1, 2, 0))
        self.assertEqual(selected[3:], (-1,) * (QSA_BLOCK_TOPK - 3))


if __name__ == "__main__":
    unittest.main()

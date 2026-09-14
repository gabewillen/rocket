#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest

import torch

MODULE = pathlib.Path(__file__).with_name("glm53-rosa-memory-lane.py")
spec = importlib.util.spec_from_file_location("rosa_lane", MODULE)
rosa = importlib.util.module_from_spec(spec)
assert spec.loader
import sys
sys.modules[spec.name] = rosa
spec.loader.exec_module(rosa)


class ExactSuffixMemoryTest(unittest.TestCase):
    def test_exact_recall_is_causal(self):
        memory = rosa.ExactSuffixMemory(max_ngram=4)
        memory.extend([10, 11, 12, 90, 10, 11, 12])
        result = memory.retrieve(width=1, min_ngram=3)
        self.assertEqual(result.tokens, (90,))
        self.assertLessEqual(result.source_end + 1, len(memory.tokens) - 3)

    def test_no_future_continuation(self):
        memory = rosa.ExactSuffixMemory(max_ngram=3)
        memory.extend([1, 2, 3, 1, 2, 3])
        self.assertIsNone(memory.retrieve(width=1))


class MemoryLaneTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = rosa.RosaMemoryLane(hidden=8, rank=2)
        self.streams = torch.randn(2, 4, 8)
        self.retrieval = torch.randn(2, 8)
        self.features = torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.7, 1.0]])

    def test_zero_gate_is_bit_exact(self):
        output, gate = self.model(self.streams, self.retrieval, self.features, layer=3)
        self.assertTrue(torch.equal(output, self.streams))
        self.assertEqual(torch.count_nonzero(gate), 0)

    def test_only_existing_fourth_stream_is_injected(self):
        output, gate = self.model(self.streams, self.retrieval, self.features, layer=3,
                                  fixed_gate=0.125)
        self.assertTrue(torch.equal(output[:, :3], self.streams[:, :3]))
        self.assertFalse(torch.equal(output[0, 3], self.streams[0, 3]))
        self.assertTrue(torch.equal(output[1, 3], self.streams[1, 3]))
        self.assertEqual(float(gate[1]), 0.0)

    def test_false_retrieval_has_suppression_gradient(self):
        output, _ = self.model(self.streams, self.retrieval, self.features, layer=3,
                               fixed_gate=0.25)
        delta = output[:, 3] - self.streams[:, 3]
        trust = torch.tensor([1.0, 0.0])
        target = torch.ones_like(delta)
        loss = (trust * ((delta - target) ** 2).mean(-1)
                + (1 - trust) * delta.square().mean(-1)).mean()
        loss.backward()
        self.assertIsNotNone(self.model.up.weight.grad)


if __name__ == "__main__":
    unittest.main()

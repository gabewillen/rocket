#!/usr/bin/env python3
"""Unit tests for qwen38-roofline.py."""

import importlib.util
import math
import pathlib
import unittest

PATH = pathlib.Path(__file__).with_name("qwen38-roofline.py")
SPEC = importlib.util.spec_from_file_location("roofline", PATH)
roofline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roofline)


def tensor(start, end):
    return {"dtype": "U8", "shape": [end - start], "data_offsets": [start, end]}


class RooflineTest(unittest.TestCase):
    def setUp(self):
        self.headers = {
            "model.language_model.layers.0.mlp.experts.0.up_proj.weight": tensor(0, 100),
            "model.language_model.layers.0.linear_attn.in_proj.weight": tensor(0, 200),
            "model.language_model.embed_tokens.weight": tensor(0, 20),
            "lm_head.weight": tensor(0, 30),
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": tensor(0, 40),
            "mtp.layers.0.mlp.experts.0.up_proj.weight": tensor(0, 50),
            "mtp.layers.0.self_attn.q_proj.weight": tensor(0, 60),
            "model.visual.blocks.0.weight": tensor(0, 70),
        }

    def test_inventory_is_exhaustive(self):
        got = roofline.inventory(self.headers)
        self.assertEqual(got["base_all_experts"], 100)
        self.assertEqual(got["base_dense_other"], 200)
        self.assertEqual(got["vision_excluded"], 70)

    def test_uniform_union(self):
        self.assertEqual(roofline.expected_union(512, 10, 1), 10)
        self.assertAlmostEqual(roofline.expected_union(512, 10, 16), 138.56926479073724)

    def test_same_route_exact_roofline(self):
        got = roofline.model(self.headers, batch=2, experts=10, top_k=2,
                             nodes=2, gb_s_per_node=1, ple_rows=1,
                             hidden_size=1, ple_bytes=1, embedding_bytes=1,
                             route="same", mtp_proposals=0,
                             accepted_tokens_per_step=1)
        self.assertEqual(got["step_bytes"]["total"], 254)
        self.assertAlmostEqual(got["ceiling"]["aggregate_accepted_tokens_per_s"], 4e9 / 254)

    def test_mtp_traffic_is_not_free(self):
        got = roofline.model(self.headers, batch=2, experts=10, top_k=2,
                             nodes=2, gb_s_per_node=1, ple_rows=1,
                             hidden_size=1, ple_bytes=1, embedding_bytes=1,
                             route="same", mtp_proposals=3,
                             accepted_tokens_per_step=2)
        self.assertEqual(got["step_bytes"]["mtp_proposal_weights_per_pass"], 70)
        self.assertEqual(got["step_bytes"]["total"], 464)
        self.assertAlmostEqual(got["ceiling"]["aggregate_accepted_tokens_per_s"], 8e9 / 464)

    def test_bad_offsets_fail(self):
        self.headers["lm_head.weight"] = {"data_offsets": [0]}
        with self.assertRaisesRegex(ValueError, "data_offsets"):
            roofline.inventory(self.headers)


if __name__ == "__main__":
    unittest.main()

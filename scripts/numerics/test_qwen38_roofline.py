#!/usr/bin/env python3
"""Unit tests for qwen38-roofline.py."""

import argparse
import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest

PATH = pathlib.Path(__file__).with_name("qwen38-roofline.py")
SPEC = importlib.util.spec_from_file_location("roofline", PATH)
roofline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roofline)


def tensor(start, end, dtype="U8"):
    return {"dtype": dtype, "shape": [end - start], "data_offsets": [start, end]}


class RooflineTest(unittest.TestCase):
    def setUp(self):
        self.headers = {
            "model.language_model.layers.0.mlp.experts.0.up_proj.weight": tensor(0, 100),
            "model.language_model.layers.0.linear_attn.in_proj.weight": tensor(0, 200),
            "model.language_model.layers.3.self_attn.q_proj.weight": tensor(0, 80),
            "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight": tensor(0, 40),
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": tensor(0, 30),
            "model.language_model.layers.0.mlp.gate.weight": tensor(0, 10),
            "model.language_model.layers.1.ple.key_proj.weight": tensor(0, 20),
            "model.language_model.embed_tokens.weight": tensor(0, 50, "BF16"),
            "lm_head.weight": tensor(0, 60, "BF16"),
            (
                "model.language_model.layers.1.ple.ple_embedding."
                "ngram_embedding.shard_0.weight"
            ): tensor(
                0, 70, "F8_E4M3"
            ),
            "mtp.layers.0.mlp.experts.0.up_proj.weight": tensor(
                0, 100, "F8_E4M3"
            ),
            "mtp.layers.0.self_attn.q_proj.weight": tensor(0, 40),
            "mtp.layers.0.attn_hyper_connection.hc_norm.weight": tensor(0, 20),
            "mtp.layers.0.mlp.shared_expert.up_proj.weight": tensor(0, 10),
            "mtp.layers.0.mlp.gate.weight": tensor(0, 5),
            "mtp.fc_hidden.weight": tensor(0, 15),
            "model.visual.blocks.0.weight": tensor(0, 90),
        }
        self.args = {
            "experts": 10,
            "top_k": 2,
            "nodes": 2,
            "gb_s_per_node": 1,
            "ple_rows": 1,
            "hidden_size": 1,
            "ple_bytes": 1,
            "embedding_bytes": 1,
            "route": "same",
            "mtp_proposals": 0,
            "accepted_tokens_per_step": 1,
        }

    def test_inventory_is_exhaustive_and_dtype_aware(self):
        got = roofline.inventory_details(self.headers)
        self.assertEqual(got["base_routed_experts"]["bytes"], 100)
        self.assertEqual(got["base_linear_attention"]["bytes"], 200)
        self.assertEqual(got["base_ple_dense"]["bytes"], 20)
        self.assertEqual(got["vision_excluded"]["bytes"], 90)
        self.assertEqual(got["lm_head"]["dtype_bytes"], {"BF16": 60})

    def test_uniform_union_uses_token_positions(self):
        self.assertEqual(roofline.expected_union(512, 10, 1), 10)
        self.assertAlmostEqual(
            roofline.expected_union(512, 10, 16), 138.56926479073724
        )

    def test_non_speculative_same_route_ceiling(self):
        got = roofline.model(self.headers, batch=2, **self.args)
        expected_total = 200 + 80 + 40 + 30 + 10 + 20 + 20 + 60 + 4
        self.assertEqual(got["step_bytes"]["total"], expected_total)
        self.assertAlmostEqual(
            got["ceiling"]["aggregate_accepted_tokens_per_s"],
            4e9 / expected_total,
        )
        self.assertEqual(
            got["assumptions"]["mtp_continuation_expected_unique_experts"], 0
        )

    def test_mtp_charges_verifier_width_drafts_and_shared_lm_head(self):
        args = dict(self.args)
        args.update(mtp_proposals=3, accepted_tokens_per_step=2.5)
        got = roofline.model(self.headers, batch=2, **args)
        by_family = got["step_bytes"]["by_family"]
        self.assertEqual(got["assumptions"]["verifier_positions_per_stream"], 4)
        self.assertEqual(got["assumptions"]["verifier_expected_unique_experts"], 2)
        self.assertEqual(by_family["mtp_routed_experts"], 60)
        self.assertEqual(by_family["mtp_attention"], 120)
        self.assertEqual(by_family["mtp_lm_head"], 180)
        self.assertEqual(by_family["mtp_embedding_rows"], 12)
        self.assertEqual(got["step_bytes"]["mtp_first_pass_weights"], 178)
        self.assertEqual(got["step_bytes"]["mtp_continuation_pass_weights"], 172)
        self.assertEqual(got["step_bytes"]["mtp_proposal_traffic"], 522)
        self.assertAlmostEqual(
            got["ceiling"]["aggregate_accepted_tokens_per_s"],
            2e9 * 5 / got["step_bytes"]["total"],
        )

    def test_uniform_speculation_expands_verifier_expert_union(self):
        args = dict(self.args)
        args.update(route="uniform", mtp_proposals=3, accepted_tokens_per_step=2)
        got = roofline.model(self.headers, batch=2, **args)
        self.assertAlmostEqual(
            got["assumptions"]["verifier_expected_unique_experts"],
            roofline.expected_union(10, 2, 8),
        )
        self.assertAlmostEqual(
            got["assumptions"]["mtp_continuation_expected_unique_experts"],
            roofline.expected_union(10, 2, 2),
        )
        expected_mtp_union_sum = (
            roofline.expected_union(10, 2, 8)
            + 2 * roofline.expected_union(10, 2, 2)
        )
        self.assertAlmostEqual(
            got["step_bytes"]["by_family"]["mtp_routed_experts"],
            100 * expected_mtp_union_sum / 10,
        )

    def test_traffic_ranking_is_complete_and_descending(self):
        got = roofline.model(self.headers, batch=2, **self.args)
        ranking = got["traffic_ranking"]
        values = [entry["bytes_per_step"] for entry in ranking]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(sum(values), got["step_bytes"]["total"])
        self.assertEqual(
            [entry["rank"] for entry in ranking],
            list(range(1, len(ranking) + 1)),
        )

    def test_lever_ranking_uses_dtype_and_reports_unmeasured_quality(self):
        args = dict(self.args)
        args.update(mtp_proposals=1, accepted_tokens_per_step=1.5)
        got = roofline.model(self.headers, batch=2, **args)
        ranking = got["lever_ranking"]
        lm_head = next(entry for entry in ranking if entry["family"] == "lm_head")
        self.assertEqual(lm_head["quality_effect"], "unmeasured")
        self.assertEqual(
            lm_head["estimated_bytes_removed_per_step"],
            120 * (1 - roofline.NVFP4_BYTES_PER_BF16_BYTE),
        )
        fp8_families = {
            entry["family"]
            for entry in ranking
            if entry["action"] == "quantize_fp8_to_checkpoint_nvfp4_layout"
        }
        self.assertEqual(fp8_families, {"mtp_routed_experts", "ple_table"})

    def test_residency_ranking_is_complete_and_descending(self):
        got = roofline.model(self.headers, batch=2, **self.args)
        ranking = got["residency_ranking"]
        values = [entry["bytes"] for entry in ranking]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(sum(values), sum(roofline.inventory(self.headers).values()))

    def test_remove_mtp_lever_resets_accepted_tokens_to_one(self):
        args = dict(self.args)
        args.update(mtp_proposals=3, accepted_tokens_per_step=2.5)
        got = roofline.model(self.headers, batch=2, **args)
        lever = next(
            item
            for item in got["lever_ranking"]
            if item["action"] == "remove_mtp_speculation"
        )
        expected = (
            2e9 * 2 / lever["estimated_step_bytes_after"]
        )
        self.assertAlmostEqual(
            lever["estimated_aggregate_accepted_tokens_per_s_after"], expected
        )

    def test_default_sweep_has_requested_concurrencies(self):
        got = roofline.model_sweep(
            self.headers, roofline.DEFAULT_CONCURRENCIES, **self.args
        )
        self.assertEqual(
            [item["assumptions"]["concurrency"] for item in got],
            [1, 2, 4, 8, 16, 32, 64],
        )

    def test_contract_rejects_impossible_acceptance(self):
        for proposals, accepted in ((0, 1.1), (3, 0.9), (3, 4.1)):
            with self.subTest(proposals=proposals, accepted=accepted):
                args = dict(self.args)
                args.update(
                    mtp_proposals=proposals,
                    accepted_tokens_per_step=accepted,
                )
                with self.assertRaisesRegex(ValueError, "accepted tokens/step"):
                    roofline.model(self.headers, batch=2, **args)

    def test_bad_offsets_fail(self):
        self.headers["lm_head.weight"] = {
            "dtype": "BF16",
            "data_offsets": [2, 1],
        }
        with self.assertRaisesRegex(ValueError, "data_offsets"):
            roofline.inventory(self.headers)

    def test_unknown_tensor_family_fails_closed(self):
        self.headers["unmapped.weight"] = tensor(0, 1)
        with self.assertRaisesRegex(ValueError, "unknown tensor"):
            roofline.inventory(self.headers)

    def test_concurrency_parser_rejects_duplicates(self):
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "unique"):
            roofline.parse_concurrencies("1,2,2")

    def test_local_headers_reads_indexed_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            meta = {"x": tensor(0, 3)}
            encoded = json.dumps(meta).encode()
            (root / "model-00001.safetensors").write_bytes(
                struct.pack("<Q", len(encoded)) + encoded + b"abc"
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "model-00001.safetensors"}}),
                encoding="utf-8",
            )
            self.assertEqual(roofline.local_headers(root), meta)

    def test_local_headers_rejects_shard_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "../outside.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "escapes model directory"):
                roofline.local_headers(root)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Focused tests for qwen38-precision-family-ranking.py."""

import importlib.util
import pathlib
import unittest


PATH = pathlib.Path(__file__).with_name("qwen38-precision-family-ranking.py")
SPEC = importlib.util.spec_from_file_location("ranking", PATH)
ranking = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ranking)


def meta(size, dtype="BF16"):
    return {"dtype": dtype, "data_offsets": [0, size]}


def trace():
    telemetry = {}
    linear = [0]
    full = [3]
    for layer in linear:
        for projection in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
            for end in ("input", "output"):
                channel = f"layer.{layer}.linear_attn.{projection}.{end}"
                telemetry[channel] = record(channel, 8 if end == "output" else 2)
        for suffix in ("output", "recurrent_state.output"):
            channel = f"layer.{layer}.linear_attn.{suffix}"
            telemetry[channel] = record(channel, 3)
    for layer in full:
        for projection in ("qkv_proj", "o_proj"):
            for end in ("input", "output"):
                channel = f"layer.{layer}.full_attn.{projection}.{end}"
                telemetry[channel] = record(channel, 4)
        channel = f"layer.{layer}.full_attn.output"
        telemetry[channel] = record(channel, 5)
    for channel in ("layer.1.ple.embedding.output", "layer.1.ple.output"):
        telemetry[channel] = record(channel, 6)
    for layer in (0, 3):
        for suffix in ("gate.input", "gate.output", "topk.output"):
            channel = f"layer.{layer}.router.{suffix}"
            telemetry[channel] = record(channel, 7)
    return {
        "schema": ranking.TRACE_SCHEMA,
        "gate": {"source": "v2_only", "min_emission_call": 8},
        "coverage": dict(ranking.EXPECTED_COVERAGE),
        "telemetry": telemetry,
    }


def record(channel, maximum):
    return {
        "channel": channel, "absmax": maximum, "abs_p99": maximum / 2,
        "rms": maximum / 4, "sample_numel": 32, "source_numel": 32,
    }


def headers():
    return {
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": meta(400),
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": meta(200),
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": meta(20),
        "model.language_model.layers.0.linear_attn.in_proj_b.weight": meta(20),
        "model.language_model.layers.0.linear_attn.out_proj.weight": meta(100),
        "model.language_model.layers.3.self_attn.q_proj.weight": meta(50),
        "model.language_model.layers.3.self_attn.k_proj.weight": meta(10),
        "model.language_model.layers.3.self_attn.v_proj.weight": meta(10),
        "model.language_model.layers.3.self_attn.o_proj.weight": meta(60),
        "model.language_model.layers.1.ple.key_proj.weight": meta(40),
        "model.language_model.layers.1.ple.value_proj.weight": meta(20),
        "model.language_model.layers.0.mlp.gate.weight": meta(30),
        "model.language_model.layers.3.mlp.gate.weight": meta(30),
        "mtp.layers.0.self_attn.q_proj.weight": meta(50),
        "mtp.layers.0.self_attn.k_proj.weight": meta(10),
        "mtp.layers.0.self_attn.v_proj.weight": meta(10),
        "mtp.layers.0.self_attn.o_proj.weight": meta(60),
        "mtp.layers.0.mlp.gate.weight": meta(30),
        "mtp.fc_embedding.weight": meta(40),
        "mtp.fc_hidden.weight": meta(40),
    }


class RankingTest(unittest.TestCase):
    def test_ranks_whole_family_and_joins_state_interactions(self):
        result = ranking.rank(headers(), trace(), {"accepted_tokens": 3})
        self.assertEqual(result["decision"]["family"], "base_linear_attention")
        self.assertEqual(result["decision"]["bytes_removed_per_c16_decode_step"], 370)
        family = next(x for x in result["families"] if x["family"] == "base_linear_attention")
        self.assertIn("layer.0.linear_attn.recurrent_state.output", family["telemetry_interactions"])
        self.assertEqual(family["calibration"]["risk"], "low")

    def test_mtp_is_reported_but_ineligible_without_activation_range(self):
        result = ranking.rank(headers(), trace(), {"accepted_tokens": 3})
        mtp = next(x for x in result["families"] if x["family"] == "mtp_attention")
        self.assertFalse(mtp["eligible"])
        self.assertEqual(mtp["calibration"]["status"], "unavailable")
        self.assertEqual(result["mtp_runtime_evidence"]["accepted_tokens"], 3)

    def test_missing_required_coverage_fails_closed(self):
        incomplete = trace()
        incomplete["coverage"]["router_layers"] = 47
        with self.assertRaisesRegex(ValueError, "router_layers=47/48"):
            ranking.rank(headers(), incomplete, {"accepted_tokens": 3})

    def test_missing_checkpoint_family_fails_closed(self):
        incomplete = headers()
        del incomplete["mtp.fc_hidden.weight"]
        del incomplete["mtp.fc_embedding.weight"]
        with self.assertRaisesRegex(ValueError, "mtp_input_projection"):
            ranking.rank(incomplete, trace(), {"accepted_tokens": 3})

    def test_coverage_counter_cannot_hide_missing_interaction(self):
        incomplete = trace()
        del incomplete["telemetry"]["layer.0.linear_attn.recurrent_state.output"]
        with self.assertRaisesRegex(ValueError, "missing required telemetry interactions"):
            ranking.rank(headers(), incomplete, {"accepted_tokens": 3})


if __name__ == "__main__":
    unittest.main()

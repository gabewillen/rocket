#!/usr/bin/env python3
"""Focused tests for the Qwen3.8 adaptive lazy-MTP policy."""

import importlib.util
import pathlib
import sys
import unittest


SCRIPT = pathlib.Path(__file__).with_name("qwen38-adaptive-mtp.py")
SPEC = importlib.util.spec_from_file_location("adaptive_mtp", SCRIPT)
MTP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MTP
SPEC.loader.exec_module(MTP)


class AdaptiveMtpTest(unittest.TestCase):
    def test_forced_phases_probe_and_remaining_cap(self):
        state = MTP.CohortState()
        self.assertEqual(state.choose("prefill", 4, 99)["depth"], 0)
        self.assertEqual(state.choose("restoring", 4, 99)["depth"], 0)
        observed = [state.choose("early_decode", 4, 99)["depth"] for _ in range(16)]
        self.assertEqual(observed, list(MTP.PROBE_DEPTHS))
        capped = MTP.CohortState().choose("early_decode", 1, 1)
        self.assertEqual((capped["depth"], capped["labels"]["reason"]), (0, "remaining_cap"))

    def test_explores_one_deeper_every_64_eligible_rounds(self):
        state = MTP.CohortState(depth=1, decode_rounds=16, eligible_rounds=64)
        decision = state.choose("steady_decode", 12, 100)
        self.assertEqual((decision["depth"], decision["labels"]["reason"]), (2, "explore"))

    def test_two_failing_eight_round_windows_demote(self):
        state = MTP.CohortState(depth=3)
        reasons = [state.apply_window(False) for _ in range(16)]
        self.assertEqual(reasons[-1], "demote")
        self.assertEqual(state.depth, 2)

    def test_confident_promotion_uses_three_percent_margin_and_lazy_cost(self):
        evidence = MTP.Evidence([900, 650, 450], [1000, 1000, 1000])
        costs = MTP.CostModel((100.0, 150.0, 205.0, 250.0), 64.0, 64)
        result = MTP.recommendation(evidence, costs)
        self.assertGreaterEqual(result["selected_depth"], 1)
        cold = result["depths"][1]["bytes_per_accepted_token"]["mean"]
        warm = MTP.recommendation(evidence, costs, resident=True)["depths"][1]["bytes_per_accepted_token"]["mean"]
        self.assertGreater(cold, warm)

    def test_metric_dimensions_are_bounded_and_identifier_free(self):
        decision = MTP.CohortState().choose("early_decode", 23, 10)
        self.assertEqual(set(decision["labels"]), {"phase", "concurrency", "depth", "reason"})
        self.assertEqual(decision["labels"]["concurrency"], "c17_plus")
        self.assertFalse({"request_id", "session_id"} & set(decision["labels"]))

    def test_cohorts_do_not_mix_phase_or_concurrency_state(self):
        policy = MTP.AdaptivePolicy()
        policy.choose("early_decode", 4, 10)
        policy.choose("steady_decode", 4, 10)
        policy.choose("early_decode", 8, 10)
        self.assertEqual(
            set(policy.cohorts),
            {("early_decode", "c2_4"), ("steady_decode", "c2_4"),
             ("early_decode", "c5_8")},
        )
        self.assertTrue(all(state.decode_rounds == 1 for state in policy.cohorts.values()))


if __name__ == "__main__":
    unittest.main()

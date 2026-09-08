#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("qwen38-router-cohort-reduce.py")
LIVE = Path(__file__).with_name("qwen38-router-cohort-live.py")
SPEC = importlib.util.spec_from_file_location("router_cohort_reduce", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
LIVE_SPEC = importlib.util.spec_from_file_location("router_cohort_live", LIVE)
LIVE_MODULE = importlib.util.module_from_spec(LIVE_SPEC)
assert LIVE_SPEC.loader is not None
LIVE_SPEC.loader.exec_module(LIVE_MODULE)


class ReducerTests(unittest.TestCase):
    def test_driver_runs_one_cohort_and_publishes_metadata_after_engine_init(self):
        source = LIVE.read_text()
        engine = source.index("engine = LLM(")
        warmup = source.index("warm_outputs = engine.generate(")
        metadata = source.index('"ROCKET_ROUTER_RANK": str(rank)')
        cohort = source.index('"ROCKET_ROUTER_COHORT": cohort')
        self.assertLess(engine, metadata)
        self.assertLess(warmup, metadata)
        self.assertLess(engine, cohort)
        self.assertNotIn("os.environ.update(", source[:engine])
        self.assertIn('"--concurrency", type=int, required=True', source)
        self.assertNotIn("for concurrency in CONCURRENCY", source)
        self.assertIn("args.concurrency * args.divergence_tokens > 8192", source)

    def test_decode_exceeds_observed_c2_three_call_terminal_by_full_iteration(self):
        self.assertEqual(LIVE_MODULE.VERIFY_WIDTH, 5)
        self.assertEqual(LIVE_MODULE.CAPTURE_CALLS, 4)
        self.assertEqual(LIVE_MODULE.OBSERVED_THREE_CALL_TERMINAL_TOKENS, 17)
        self.assertEqual(LIVE_MODULE.CONSERVATIVE_ITERATION_MARGIN, 6)
        self.assertEqual(LIVE_MODULE.MIN_DECODE, 17 + 6 + 1)
        self.assertEqual(LIVE_MODULE.MIN_DECODE, 24)

    def test_exact_target_and_speculative_rank_unions(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for rank in (0, 1):
                path = Path(directory) / f"rank{rank}.log"
                lines = []
                for layer in range(48):
                    for cohort_call in range(1, 5):
                        offset = rank * 256
                        record = {
                            "schema": "rocket.qwen38.activation-telemetry.v3",
                            "channel": f"layer.{layer}.router.topk.output",
                            "cohort": "c1-k4",
                            "rank": rank,
                            "sequences": 1,
                            "verify_width": 5,
                            "route_top_k": 10,
                            "cohort_call": cohort_call,
                            "selected_expert_count": 50,
                            "route_rows": [],
                        }
                        for row in range(5):
                            record["route_rows"].append(
                                {
                                    "row": row,
                                    "sequence": 0,
                                    "position": row,
                                    "position_kind": "target" if row == 0 else "speculative",
                                    "expert_ids": list(range(offset, offset + 10)),
                                    "weights": [0.1] * 10,
                                }
                            )
                        lines.append("prefix ROCKET_NVFP4_TELEMETRY\t" + json.dumps(record))
                path.write_text("\n".join(lines))
                paths.append(path)
            result = MODULE.reduce(paths)
        self.assertEqual(len(result["cases"]), 2)
        case = result["cases"][0]
        sample = case["layers"][0]["samples"][0]
        self.assertEqual(sample["target_unique_local_experts"], 10)
        self.assertEqual(sample["speculative_unique_local_experts"], 10)
        self.assertEqual(sample["union_unique_local_experts"], 10)
        self.assertEqual(sample["local_routes"], 50)
        self.assertEqual(sample["routed_slab_bytes"], MODULE.routed_slab_bytes(10))

    def test_incomplete_four_call_layer_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0.log"
            lines = []
            for layer in range(48):
                record = {
                    "schema": "rocket.qwen38.activation-telemetry.v3",
                    "channel": f"layer.{layer}.router.topk.output",
                    "cohort": "c1-k4",
                    "rank": 0,
                    "sequences": 1,
                    "verify_width": 5,
                    "route_top_k": 10,
                    "cohort_call": 1,
                    "selected_expert_count": 50,
                    "route_rows": [],
                }
                for row in range(5):
                    record["route_rows"].append(
                        {
                            "row": row,
                            "sequence": 0,
                            "position": row,
                            "position_kind": (
                                "target" if row == 0 else "speculative"
                            ),
                            "expert_ids": list(range(10)),
                            "weights": [0.1] * 10,
                        }
                    )
                lines.append("ROCKET_NVFP4_TELEMETRY\t" + json.dumps(record))
            path.write_text("\n".join(lines))
            with self.assertRaisesRegex(ValueError, "calls 1..4"):
                MODULE.reduce([path])


if __name__ == "__main__":
    unittest.main()

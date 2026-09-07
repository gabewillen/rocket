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


class ReducerTests(unittest.TestCase):
    def test_driver_publishes_complete_cohort_only_after_engine_init(self):
        source = LIVE.read_text()
        engine = source.index("engine = LLM(")
        metadata = source.index('"ROCKET_ROUTER_RANK": str(rank)')
        cohort = source.index('"ROCKET_ROUTER_COHORT": "forked-prefix-c1-k4"')
        self.assertLess(engine, metadata)
        self.assertLess(engine, cohort)
        self.assertNotIn("ROCKET_ROUTER_", source[:engine])

    def test_exact_target_and_speculative_rank_unions(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for rank in (0, 1):
                path = Path(directory) / f"rank{rank}.log"
                lines = []
                for layer in range(48):
                    offset = rank * 256
                    record = {
                        "schema": "rocket.qwen38.activation-telemetry.v3",
                        "channel": f"layer.{layer}.router.topk.output",
                        "cohort": "c1-k4",
                        "rank": rank,
                        "sequences": 1,
                        "verify_width": 5,
                        "route_top_k": 10,
                        "cohort_call": 1,
                        "route_rows": [],
                    }
                    for row in range(5):
                        record["route_rows"].append(
                            {
                                "position_kind": "target" if row == 0 else "speculative",
                                "expert_ids": [offset + row] * 10,
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
        self.assertEqual(sample["target_unique_local_experts"], 1)
        self.assertEqual(sample["speculative_unique_local_experts"], 4)
        self.assertEqual(sample["union_unique_local_experts"], 5)
        self.assertEqual(sample["local_routes"], 50)
        self.assertEqual(sample["routed_slab_bytes"], MODULE.routed_slab_bytes(5))


if __name__ == "__main__":
    unittest.main()

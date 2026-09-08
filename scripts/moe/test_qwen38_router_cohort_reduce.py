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


def router_record(rank, layer, cohort_call, widths, *, cohort_sequences=None):
    offset = rank * MODULE.LOCAL_EXPERTS
    rows = sum(widths)
    row_offsets = [0]
    route_rows = []
    for sequence, width in enumerate(widths):
        start = row_offsets[-1]
        row_offsets.append(start + width)
        for position in range(width):
            route_rows.append(
                {
                    "row": start + position,
                    "sequence": sequence,
                    "position": position,
                    "position_kind": "target" if position == 0 else "speculative",
                    "expert_ids": list(range(offset, offset + 10)),
                    "weights": [0.1] * 10,
                }
            )
    return {
        "schema": MODULE.SCHEMA,
        "channel": f"layer.{layer}.router.topk.output",
        "call": 17 + cohort_call,
        "cohort": f"forked-prefix-c{cohort_sequences or len(widths)}-k4",
        "rank": rank,
        "cohort_sequences": cohort_sequences or len(widths),
        "sequences": len(widths),
        "verify_width": 5,
        "request_widths": widths,
        "row_offsets": row_offsets,
        "route_top_k": 10,
        "route_layout": "sequence_major",
        "cohort_call": cohort_call,
        "selected_expert_count": rows * 10,
        "top_experts": [
            {"expert_id": expert_id, "selections": rows}
            for expert_id in range(offset, offset + 10)
        ],
        "route_rows": route_rows,
    }


def write_logs(directory, call_widths, *, ranks=(0, 1), layers=48):
    paths = []
    for rank in ranks:
        path = Path(directory) / f"rank{rank}.log"
        lines = []
        for layer in range(layers):
            for cohort_call, widths in enumerate(call_widths, 1):
                record = router_record(
                    rank,
                    layer,
                    cohort_call,
                    widths,
                    cohort_sequences=max(len(item) for item in call_widths),
                )
                lines.append("ROCKET_NVFP4_TELEMETRY\t" + json.dumps(record))
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


class ReducerTests(unittest.TestCase):
    def test_driver_publishes_metadata_only_after_initialization(self):
        source = LIVE.read_text(encoding="utf-8")
        engine = source.index("engine = LLM(")
        warmup = source.index("warm_outputs = engine.generate(")
        metadata = source.index('"ROCKET_ROUTER_RANK": str(rank)')
        self.assertLess(engine, metadata)
        self.assertLess(warmup, metadata)
        self.assertNotIn("os.environ.update(", source[:engine])
        self.assertIn('"--concurrency", type=int, required=True', source)
        self.assertNotIn("for concurrency in CONCURRENCY", source)
        self.assertIn("args.concurrency * args.divergence_tokens > 8192", source)
        self.assertIn(
            '"telemetry_schema": "rocket.qwen38.activation-telemetry.v4"', source
        )
        self.assertIn('"vllm.forward_context.attn_metadata.query_start_loc"', source)

    def test_c16_primes_complete_prompts_and_proves_cache_before_gate(self):
        source = LIVE.read_text(encoding="utf-8")
        prompts = source.index("prompts = []")
        sequential_prime = source.index("for prompt in prompts:")
        concurrent_barrier = source.index(
            "barrier_outputs = engine.generate(prompts, warm_sampling"
        )
        observable_counts = source.index('"ROCKET_ROUTER_CACHE_BARRIER\\t"')
        continuation = source.index("prompts = measured_prompts")
        cached_check = source.index("output.num_cached_tokens != expected_cached_tokens")
        metadata = source.index("os.environ.update(cohort_metadata)")
        measured = source.index("outputs = engine.generate(prompts, sampling")
        self.assertLess(prompts, sequential_prime)
        self.assertLess(sequential_prime, continuation)
        self.assertLess(continuation, concurrent_barrier)
        self.assertLess(sequential_prime, concurrent_barrier)
        self.assertLess(concurrent_barrier, cached_check)
        self.assertLess(observable_counts, cached_check)
        self.assertLess(cached_check, metadata)
        self.assertLess(metadata, measured)
        self.assertIn('if concurrency == 16:', source)
        self.assertIn(
            'cache_barrier = "two-cache-pages-v2"', source
        )
        self.assertIn(
            'cohort_metadata["ROCKET_ROUTER_CACHE_BARRIER"] = cache_barrier',
            source,
        )
        self.assertIn(
            '"prompt_tokens": len(prompts[0]["prompt_token_ids"])', source
        )
        self.assertIn(
            '"primed_continuation_tokens": 1 if concurrency == 16 else 0', source
        )
        self.assertIn(
            '"pool_with_c1_c8_prompt_distribution": concurrency != 16', source
        )
        self.assertIn("args.prefix_tokens != 6304", source)
        self.assertNotIn(
            "engine.llm_engine.vllm_config.cache_config.block_size", source
        )
        self.assertIn("args.expected_cache_block_size != 3216", source)
        self.assertIn("prompt_tokens != 6433", source)
        self.assertIn("expected_cached_tokens != 6432", source)
        self.assertIn('"cache_pages": 2 if concurrency == 16 else None', source)
        self.assertIn('"two_cache_pages" if concurrency == 16 else None', source)
        self.assertIn(
            '"attention_block_size_proof": "all_request_cache_hit_counts"', source
        )

    def test_decode_exceeds_observed_c2_three_call_terminal_by_full_iteration(self):
        self.assertEqual(LIVE_MODULE.VERIFY_WIDTH, 5)
        self.assertEqual(LIVE_MODULE.CAPTURE_CALLS, 4)
        self.assertEqual(LIVE_MODULE.OBSERVED_THREE_CALL_TERMINAL_TOKENS, 17)
        self.assertEqual(LIVE_MODULE.MIN_DECODE, 24)

    def test_exact_target_and_speculative_rank_unions(self):
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE.reduce(write_logs(directory, [[5]] * 4))
        self.assertEqual(len(result["cases"]), 2)
        sample = result["cases"][0]["layers"][0]["samples"][0]
        self.assertEqual(sample["request_widths"], [5])
        self.assertEqual(sample["row_offsets"], [0, 5])
        self.assertEqual(sample["target_unique_local_experts"], 10)
        self.assertEqual(sample["speculative_unique_local_experts"], 10)
        self.assertEqual(sample["union_unique_local_experts"], 10)
        self.assertEqual(sample["local_routes"], 50)
        self.assertEqual(sample["routed_slab_bytes"], MODULE.routed_slab_bytes(10))

    def test_mixed_request_widths_use_actual_boundaries_and_rows(self):
        call_widths = [
            [5, 3, 1, 4],
            [2, 5, 4, 1],
            [1, 1, 5, 2],
            [4, 2, 3, 5],
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE.reduce(write_logs(directory, call_widths))
        self.assertEqual(result["schema"], MODULE.SUMMARY_SCHEMA)
        samples = result["cases"][0]["layers"][0]["samples"]
        self.assertEqual(samples[0]["request_widths"], [5, 3, 1, 4])
        self.assertEqual(samples[0]["row_offsets"], [0, 5, 8, 9, 13])
        self.assertEqual(samples[0]["route_rows"], 13)
        self.assertEqual(samples[0]["target_rows"], 4)
        self.assertEqual(samples[0]["speculative_rows"], 9)
        self.assertEqual(samples[0]["local_routes"], 130)

    def test_c8_decode24_variable_calls_close_preserved_red(self):
        call_widths = [
            [5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 4, 3, 2],
            [5, 4, 4, 3, 2, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE.reduce(write_logs(directory, call_widths))
        samples = result["cases"][0]["layers"][0]["samples"]
        self.assertEqual(
            [sample["route_rows"] for sample in samples], [40, 34, 21, 8]
        )
        self.assertEqual(
            [sample["cohort_call"] for sample in samples], [1, 2, 3, 4]
        )

    def test_c8_two_exact_calls_remains_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_logs(directory, [[5] * 8] * 2)
            with self.assertRaisesRegex(ValueError, "calls 1..4"):
                MODULE.reduce(paths)

    def test_missing_rank_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_logs(directory, [[5]] * 4, ranks=(0,))
            with self.assertRaisesRegex(ValueError, "authenticate ranks 0 and 1"):
                MODULE.reduce(paths)

    def test_invalid_variable_width_boundary_is_terminal(self):
        record = router_record(0, 0, 1, [5, 2, 1])
        record["row_offsets"] = [0, 5, 6, 8]
        with self.assertRaisesRegex(ValueError, "row offsets"):
            MODULE.validate_record(record)

    def test_invalid_top_expert_counts_are_terminal(self):
        record = router_record(0, 0, 1, [5, 2, 1])
        record["top_experts"][0]["selections"] -= 1
        with self.assertRaisesRegex(ValueError, "top expert count"):
            MODULE.validate_record(record)

    def test_malformed_router_json_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0.log"
            path.write_text(
                "ROCKET_NVFP4_TELEMETRY\t{bad}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "invalid router telemetry JSON"):
                MODULE.reduce([path])

    def test_non_monotonic_raw_calls_are_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_logs(directory, [[5]] * 4)
            lines = paths[0].read_text(encoding="utf-8").splitlines()
            second = json.loads(lines[1].split("\t", 1)[1])
            second["call"] = 18
            lines[1] = "ROCKET_NVFP4_TELEMETRY\t" + json.dumps(second)
            paths[0].write_text("\n".join(lines), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw call ids are not monotonic"):
                MODULE.reduce(paths)

    def test_width_schedule_mismatch_between_ranks_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_logs(directory, [[5, 3]] * 4)
            lines = paths[1].read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0].split("\t", 1)[1])
            first["request_widths"] = [4, 4]
            first["row_offsets"] = [0, 4, 8]
            for row in first["route_rows"]:
                if row["row"] == 4:
                    row["sequence"] = 1
                    row["position"] = 0
                    row["position_kind"] = "target"
                elif row["row"] < 4:
                    row["sequence"] = 0
                    row["position"] = row["row"]
                else:
                    row["sequence"] = 1
                    row["position"] = row["row"] - 4
                    row["position_kind"] = "speculative"
            lines[0] = "ROCKET_NVFP4_TELEMETRY\t" + json.dumps(first)
            paths[1].write_text("\n".join(lines), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schedule changed"):
                MODULE.reduce(paths)


if __name__ == "__main__":
    unittest.main()

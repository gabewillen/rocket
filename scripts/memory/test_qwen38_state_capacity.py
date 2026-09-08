#!/usr/bin/env python3
"""Focused tests for the pinned Qwen3.8 serving-state capacity ledger."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from unittest import mock
from pathlib import Path


PATH = Path(__file__).with_name("qwen38-state-capacity.py")
SPEC = importlib.util.spec_from_file_location("qwen38_state_capacity", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CONFIG = Path(
    "/home/glwillen/.cache/huggingface/hub/"
    "models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
    f"{MODULE.REVISION}/config.json"
)


class StateCapacityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = CONFIG.read_bytes()
        cls.kwargs = dict(
            revision=MODULE.REVISION, context=262144, concurrency=16, tp=2,
            block_size=16, kernel_block_alignment=16, kv_dtype="bfloat16",
            mamba_cache_dtype="bfloat16", mamba_ssm_dtype="float32",
            speculative_tokens=3, mamba_cache_mode="align", host_page=65536,
            image_id=MODULE.IMAGE_ID,
            source_hashes={name: spec["sha256"] for name, spec in MODULE.SOURCE_CONTRACT.items()},
        )

    def plan(self):
        return MODULE.build_plan(self.config, **self.kwargs)

    def test_exact_family_coverage_and_shapes(self) -> None:
        plan = self.plan()
        rows = {row["id"]: row for row in plan["families"]}
        self.assertEqual(set(rows), {
            "target.full_attention.kv", "target.qsa.raw", "target.qsa.compressed",
            "target.linear.conv", "target.linear.recurrent", "target.ple.conv",
            "mtp.full_attention.kv", "mtp.qsa.raw", "mtp.qsa.compressed",
        })
        self.assertEqual(rows["target.full_attention.kv"]["layers"], 12)
        self.assertEqual(rows["target.full_attention.kv"]["logical_shape_per_layer_per_stream"], [262144, 1, 512])
        self.assertEqual(rows["target.qsa.raw"]["logical_shape_per_layer_per_stream"], [8, 1, 140])
        self.assertEqual(rows["target.qsa.compressed"]["logical_shape_per_layer_per_stream"], [65536, 1, 128])
        self.assertEqual(rows["target.linear.conv"]["logical_shape_per_layer_per_stream"], [6, 5120])
        self.assertEqual(rows["target.linear.recurrent"]["logical_shape_per_layer_per_stream"], [24, 128, 128])
        self.assertEqual(rows["target.ple.conv"]["logical_shape_per_layer_per_stream"], [12, 10240])
        self.assertEqual(plan["serving"]["effective_block_size_tokens"], 1600)
        self.assertEqual(plan["serving"]["sequence_pages"], 164)
        self.assertEqual(rows["target.full_attention.kv"]["serving_dtype"], "bfloat16")
        self.assertEqual(rows["mtp.full_attention.kv"]["serving_dtype"], "bfloat16")

    def test_align_mode_separates_logical_from_allocated_state(self) -> None:
        rows = {row["id"]: row for row in self.plan()["families"]}
        for name in ("target.linear.conv", "target.linear.recurrent", "target.ple.conv"):
            self.assertEqual(rows[name]["cuda_blocks_per_stream"], 2)
            self.assertGreater(rows[name]["cuda_allocated_bytes_per_stream"], rows[name]["logical_bytes_per_stream"])
        self.assertEqual(rows["target.linear.recurrent"]["serving_dtype"], "float32")
        self.assertEqual(rows["target.ple.conv"]["tp_ownership"], "replicated on both TP ranks (tp_replicated=True)")

    def test_totals_are_exact_sums_and_64k_padded(self) -> None:
        plan = self.plan()
        rows = plan["families"]
        totals = plan["totals_per_rank"]
        self.assertEqual(totals["logical_bytes_per_stream"], sum(x["logical_bytes_per_stream"] for x in rows))
        recurrent = next(x for x in rows if x["id"] == "target.linear.recurrent")
        self.assertEqual(totals["cuda_allocated_bytes_c16"], sum(x["cuda_allocated_bytes_c16"] for x in rows) - recurrent["cuda_allocated_bytes_c16"])
        self.assertFalse(recurrent["cuda_allocation_accounted"])
        self.assertEqual(recurrent["shares_cuda_allocation_with"], "target.linear.conv")
        self.assertTrue(all(x["nvme_padded_bytes_per_stream"] % 65536 == 0 for x in rows))

    def test_fails_closed_on_missing_dimension(self) -> None:
        config = json.loads(self.config)
        del config["text_config"]["linear_value_head_dim"]
        mutated = json.dumps(config).encode()
        with mock.patch.object(MODULE, "CONFIG_SHA256", MODULE.sha256(mutated)):
            with self.assertRaisesRegex(MODULE.PlanError, "linear_value_head_dim"):
                MODULE.build_plan(mutated, **self.kwargs)

    def test_fails_closed_on_config_hash_or_dtype_drift(self) -> None:
        with self.assertRaisesRegex(MODULE.PlanError, "config hash drift"):
            MODULE.build_plan(self.config + b"\n", **self.kwargs)
        bad = dict(self.kwargs, mamba_ssm_dtype="bfloat16")
        with self.assertRaisesRegex(MODULE.PlanError, "recurrent state"):
            MODULE.build_plan(self.config, **bad)
        bad = dict(self.kwargs, kv_dtype="fp8_e4m3")
        with self.assertRaisesRegex(MODULE.PlanError, "working QSA kernel"):
            MODULE.build_plan(self.config, **bad)

    def test_source_contract_checks_hashes_and_semantic_anchors(self) -> None:
        sources = {}
        for name, contract in MODULE.SOURCE_CONTRACT.items():
            sources[name] = "\n".join(contract["anchors"]).encode()
        with self.assertRaisesRegex(MODULE.PlanError, "hash drift"):
            MODULE.validate_sources(MODULE.IMAGE_ID, sources)
        with self.assertRaisesRegex(MODULE.PlanError, "image ID drift"):
            MODULE.validate_sources("sha256:wrong", sources)

    def test_original_lazy_allocation_artifact_is_rejected(self) -> None:
        old_proof = {
            "schema": "rocket.qwen38.state-capacity.cuda-allocation.v1",
            "requested_bytes": 33420083200,
            "storage_bytes": 33420083200,
            "free_bytes_before": 1846018048,
            "free_bytes_after": 1237516288,
        }
        with self.assertRaisesRegex(MODULE.PlanError, "touched-allocation schema v2"):
            MODULE.validate_cuda_proof(old_proof, 33420083200)
        deceptive_v2 = {
            "schema": "rocket.qwen38.state-capacity.cuda-allocation.v2",
            "status": "passed", "requested_bytes": 33420083200,
            "storage_bytes": 33420083200,
            "touched_bytes": 33420083200,
            "allocator_allocated_delta_bytes": 33420083200,
            "resident_delta_bytes": 1846018048 - 1237516288,
            "verified_pages": 509950, "expected_pages": 509950,
            "deterministic_readback": True,
        }
        with self.assertRaisesRegex(MODULE.PlanError, "resident-memory delta"):
            MODULE.validate_cuda_proof(deceptive_v2, 33420083200)

    def test_cuda_proof_requires_full_residency_and_readback(self) -> None:
        proof = {
            "schema": "rocket.qwen38.state-capacity.cuda-allocation.v2",
            "status": "passed", "requested_bytes": 100, "storage_bytes": 100,
            "touched_bytes": 100,
            "allocator_allocated_delta_bytes": 100, "resident_delta_bytes": 99,
            "verified_pages": 1, "expected_pages": 1,
            "deterministic_readback": True,
        }
        with self.assertRaisesRegex(MODULE.PlanError, "resident-memory delta"):
            MODULE.validate_cuda_proof(proof, 100)
        proof["resident_delta_bytes"] = 100
        proof["deterministic_readback"] = False
        with self.assertRaisesRegex(MODULE.PlanError, "deterministic readback"):
            MODULE.validate_cuda_proof(proof, 100)

    def test_page_readback_is_chunked_and_checks_tail(self) -> None:
        class FakeSlice:
            def __init__(self, values): self.values = values
            def cpu(self): return self
            def tolist(self): return self.values

        class FakeScalar:
            def __init__(self, value): self.value = value
            def cpu(self): return self
            def item(self): return self.value

        class FakeTensor:
            def __init__(self, size, pattern):
                self.size, self.pattern, self.slices = size, pattern, []
            def numel(self): return self.size
            def __getitem__(self, key):
                if key == -1: return FakeScalar(self.pattern)
                self.slices.append(key)
                return FakeSlice([self.pattern] * len(range(*key.indices(self.size))))

        tensor = FakeTensor(4 * 65536 + 7, 19)
        self.assertEqual(MODULE._readback_pages(tensor, 19, 65536, 2), 5)
        self.assertEqual(len(tensor.slices), 3)


if __name__ == "__main__":
    unittest.main()

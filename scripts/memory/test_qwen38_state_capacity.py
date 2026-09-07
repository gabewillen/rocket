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
            block_size=16, kernel_block_alignment=16, kv_dtype="fp8_e4m3",
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
        self.assertEqual(plan["serving"]["effective_block_size_tokens"], 3200)
        self.assertEqual(plan["serving"]["sequence_pages"], 82)

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

    def test_source_contract_checks_hashes_and_semantic_anchors(self) -> None:
        sources = {}
        for name, contract in MODULE.SOURCE_CONTRACT.items():
            sources[name] = "\n".join(contract["anchors"]).encode()
        with self.assertRaisesRegex(MODULE.PlanError, "hash drift"):
            MODULE.validate_sources(MODULE.IMAGE_ID, sources)
        with self.assertRaisesRegex(MODULE.PlanError, "image ID drift"):
            MODULE.validate_sources("sha256:wrong", sources)


if __name__ == "__main__":
    unittest.main()

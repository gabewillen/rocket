#!/usr/bin/env python3
"""Focused contract tests for the Qwen3.8 TP2 slab planner."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("qwen38-rank-slab-plan.py")
SPEC = importlib.util.spec_from_file_location("qwen38_rank_slab_plan", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

PINNED_CHECKPOINT = Path(
    "/home/glwillen/.cache/huggingface/hub/"
    "models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
    f"{planner.PINNED_REVISION}"
)


def header(name: str, shape: tuple[int, ...], dtype: str = "BF16"):
    byte_count = planner.DTYPE_BYTES[dtype]
    for extent in shape:
        byte_count *= extent
    return planner.TensorHeader(
        name=name,
        dtype=dtype,
        shape=shape,
        source_file="model.safetensors",
        source_blob="a" * 64,
        source_file_size_bytes=byte_count + 4096,
        source_header_sha256="b" * 64,
        source_payload_offset_bytes=4096,
        source_data_offsets=(0, byte_count),
    )


class RankSlabPlanTests(unittest.TestCase):
    def test_gdn_qkv_uses_three_exact_logical_slices(self) -> None:
        item = header(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            (10_240, 2560),
        )
        plan = planner.build_plan([item], Path("checkpoint"))
        rank0 = plan["slabs"]["rank0-target"]["entries"][0]
        rank1 = plan["slabs"]["rank1-target"]["entries"][0]
        self.assertEqual(
            [
                (part["logical_shard"], part["start"], part["length"])
                for part in rank0["source_slices"]
            ],
            [("q", 0, 1024), ("k", 2048, 1024), ("v", 4096, 3072)],
        )
        self.assertEqual(
            [
                (part["logical_shard"], part["start"], part["length"])
                for part in rank1["source_slices"]
            ],
            [("q", 1024, 1024), ("k", 3072, 1024), ("v", 7168, 3072)],
        )
        self.assertEqual(rank0["local_bytes"], 5120 * 2560 * 2)
        self.assertEqual(rank0["local_shape"], [5120, 2560])
        self.assertFalse(plan["payload_io_performed"])

    def test_column_row_vocab_and_replication_have_exact_dimensions(self) -> None:
        tensors = [
            header("model.language_model.embed_tokens.weight", (248_320, 2560)),
            header("model.language_model.layers.3.self_attn.q_proj.weight", (12_288, 2560)),
            header("model.language_model.layers.3.self_attn.o_proj.weight", (2560, 6144)),
            header("model.language_model.layers.3.self_attn.q_norm.weight", (256,)),
        ]
        plan = planner.build_plan(tensors, Path("checkpoint"))
        entries = {
            entry["name"]: entry
            for entry in plan["slabs"]["rank1-target"]["entries"]
        }
        self.assertEqual(entries[tensors[0].name]["source_slices"][0]["start"], 124_160)
        self.assertEqual(entries[tensors[1].name]["source_slices"][0]["start"], 6144)
        self.assertEqual(entries[tensors[2].name]["source_slices"][0]["dimension"], 1)
        self.assertIsNone(entries[tensors[3].name]["source_slices"][0]["dimension"])
        self.assertEqual(entries[tensors[0].name]["consumers"], ["target", "mtp"])

    def test_index_qk_projection_is_replicated_before_q_proj_suffix_match(self) -> None:
        item = header(
            "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
            (640, 2560),
        )
        plan = planner.build_plan([item], Path("checkpoint"))
        for rank in (0, 1):
            entry = plan["slabs"][f"rank{rank}-target"]["entries"][0]
            self.assertEqual(entry["family"], "self_attn.indexer")
            self.assertEqual(entry["state"], "replicated")
            self.assertEqual(entry["local_shape"], [640, 2560])
            self.assertIsNone(entry["source_slices"][0]["dimension"])

    def test_every_linear_attention_partition_rule_is_explicit(self) -> None:
        prefix = "model.language_model.layers.0.linear_attn."
        tensors = [
            header(prefix + "in_proj_a.weight", (48, 2560)),
            header(prefix + "in_proj_b.weight", (48, 2560)),
            header(prefix + "in_proj_z.weight", (6144, 2560)),
            header(prefix + "out_proj.weight", (2560, 6144)),
            header(prefix + "A_log", (48,)),
            header(prefix + "dt_bias", (48,)),
            header(prefix + "norm.weight", (128,)),
            header(prefix + "conv1d.weight", (10_240, 1, 4)),
        ]
        rank1 = planner.build_plan(tensors, Path("checkpoint"))["slabs"][
            "rank1-target"
        ]
        entries = {entry["name"]: entry for entry in rank1["entries"]}
        self.assertEqual(entries[prefix + "in_proj_a.weight"]["source_slices"][0]["start"], 24)
        self.assertEqual(entries[prefix + "in_proj_b.weight"]["source_slices"][0]["start"], 24)
        self.assertEqual(entries[prefix + "in_proj_z.weight"]["source_slices"][0]["start"], 3072)
        self.assertEqual(entries[prefix + "out_proj.weight"]["source_slices"][0]["dimension"], 1)
        self.assertEqual(entries[prefix + "A_log"]["source_slices"][0]["start"], 24)
        self.assertEqual(entries[prefix + "dt_bias"]["source_slices"][0]["start"], 24)
        self.assertIsNone(entries[prefix + "norm.weight"]["source_slices"][0]["dimension"])
        self.assertEqual(
            [part["start"] for part in entries[prefix + "conv1d.weight"]["source_slices"]],
            [1024, 3072, 7168],
        )

    def test_ple_checkpoint_shards_are_assigned_to_one_rank(self) -> None:
        tensors = [
            header(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_63.weight",
                (2_500_012, 160),
                "F8_E4M3",
            ),
            header(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_64.weight",
                (2_500_012, 160),
                "F8_E4M3",
            ),
            header(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
                (1,),
            ),
        ]
        plan = planner.build_plan(tensors, Path("checkpoint"))
        names0 = [
            entry["name"]
            for entry in plan["slabs"]["rank0-target"]["entries"]
        ]
        names1 = [
            entry["name"]
            for entry in plan["slabs"]["rank1-target"]["entries"]
        ]
        self.assertIn(tensors[0].name, names0)
        self.assertNotIn(tensors[0].name, names1)
        self.assertNotIn(tensors[1].name, names0)
        self.assertIn(tensors[1].name, names1)
        self.assertIn(tensors[2].name, names0)
        self.assertIn(tensors[2].name, names1)

    def test_ple_last_shard_copies_logical_rows_then_zero_fills_padding(self) -> None:
        item = header(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
            "shard_127.weight",
            (2_500_012, 160),
            "F8_E4M3",
        )
        plan = planner.build_plan([item], Path("checkpoint"))
        entry = plan["slabs"]["rank1-target"]["entries"][0]
        self.assertEqual(entry["source_slices"][0]["length"], 2_499_922)
        self.assertEqual(entry["transforms"][1]["operation"], "zero_fill")
        self.assertEqual(entry["transforms"][1]["length"], 90)
        self.assertEqual(entry["local_shape"], [2_500_012, 160])

    def test_ple_scale_shape_drift_fails_closed(self) -> None:
        item = header(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
            "weight_scale",
            (2,),
        )
        with self.assertRaisesRegex(planner.PlanError, "PLE FP8 scale ABI drift"):
            planner.build_plan([item], Path("checkpoint"))

    def test_mtp_assignment_precedes_layout_and_offsets_are_256b_aligned(self) -> None:
        tensors = [
            header("mtp.fc_embedding.weight", (2560, 2560)),
            header("mtp.pre_fc_norm_embedding.weight", (2560,)),
        ]
        plan = planner.build_plan(tensors, Path("checkpoint"))
        for key in ("rank0-mtp", "rank1-mtp"):
            slab = plan["slabs"][key]
            self.assertTrue(
                all(entry["consumers"] == ["mtp"] for entry in slab["entries"])
            )
            self.assertTrue(
                all(
                    entry["slab_offset_bytes"] % planner.TENSOR_ALIGNMENT_BYTES == 0
                    for entry in slab["entries"]
                )
            )

    def test_shared_weights_are_target_owned_and_mtp_referenced(self) -> None:
        item = header("model.language_model.embed_tokens.weight", (248_320, 2560))
        plan = planner.build_plan([item], Path("checkpoint"))
        for rank in range(2):
            target = plan["slabs"][f"rank{rank}-target"]["entries"][0]
            mtp = plan["slabs"][f"rank{rank}-mtp"]["entries"][0]
            self.assertEqual(target["entry_type"], "payload")
            self.assertEqual(target["physical_owner"], f"rank{rank}-target")
            self.assertEqual(mtp["entry_type"], "reference")
            self.assertEqual(mtp["physical_owner"], f"rank{rank}-target")
            self.assertFalse(mtp["payload_io"])

    def test_expert_ownership_preserves_auxiliary_with_abi(self) -> None:
        target_aux = header(
            "model.language_model.layers.0.mlp.experts.256.down_proj.weight_scale",
            (2560, 40),
            "F8_E4M3",
        )
        mtp_aux = header(
            "mtp.layers.0.mlp.experts.255.gate_proj.weight_scale_inv",
            (5, 20),
        )
        plan = planner.build_plan([target_aux, mtp_aux], Path("checkpoint"))
        self.assertEqual(plan["slabs"]["rank0-target"]["entries"], [])
        target_entry = plan["slabs"]["rank1-target"]["entries"][0]
        mtp_entry = plan["slabs"]["rank0-mtp"]["entries"][0]
        self.assertEqual(target_entry["source_slices"][0]["expert_id"], 256)
        self.assertEqual(
            target_entry["source_slices"][0]["expert_abi"],
            "modelopt_nvfp4_group16",
        )
        self.assertEqual(mtp_entry["source_slices"][0]["expert_id"], 255)
        self.assertEqual(
            mtp_entry["source_slices"][0]["expert_abi"],
            "fp8_e4m3_block_128x128",
        )

    def test_expert_name_and_shape_drift_fail_closed(self) -> None:
        with self.assertRaisesRegex(planner.PlanError, "unresolved text non-expert"):
            planner.build_plan(
                [
                    header(
                        "model.language_model.layers.0.mlp.experts.0."
                        "down_proj.unknown_scale",
                        (),
                        "F32",
                    )
                ],
                Path("checkpoint"),
            )
        with self.assertRaisesRegex(planner.PlanError, "expert tensor ABI drift"):
            planner.build_plan(
                [
                    header(
                        "mtp.layers.0.mlp.experts.0.down_proj.weight_scale_inv",
                        (5, 20),
                    )
                ],
                Path("checkpoint"),
            )

    def test_unknown_family_and_non_scalar_auxiliary_fail_closed(self) -> None:
        with self.assertRaisesRegex(planner.PlanError, "unresolved text non-expert"):
            planner.build_plan([header("model.language_model.mystery.weight", (2, 2))], Path("checkpoint"))
        with self.assertRaisesRegex(planner.PlanError, "unresolved quantization auxiliary"):
            planner.build_plan(
                [header("model.language_model.mystery.weight_scale", (1,))],
                Path("checkpoint"),
            )

    def test_visual_is_the_only_explicit_exclusion(self) -> None:
        tensors = [header("model.visual.patch_embed.proj.weight", (2, 2))]
        with self.assertRaisesRegex(planner.PlanError, "explicit --text-only"):
            planner.build_plan(tensors, Path("checkpoint"))
        plan = planner.build_plan(tensors, Path("checkpoint"), text_only=True)
        self.assertEqual(plan["excluded_tensor_counts"]["visual"]["tensors"], 1)
        self.assertEqual(
            plan["text_only_omission"]["loader_guard"],
            "skip model.visual.* before payload I/O",
        )
        self.assertEqual(plan["slabs"]["rank0-target"]["entries"], [])

    @unittest.skipUnless(PINNED_CHECKPOINT.is_dir(), "pinned checkpoint unavailable")
    def test_real_checkpoint_inventory_is_fully_accounted(self) -> None:
        plan = planner.build_plan(
            planner.load_pinned_checkpoint(PINNED_CHECKPOINT),
            PINNED_CHECKPOINT,
            text_only=True,
        )
        inventory = plan["inventory"]
        visual = plan["excluded_tensor_counts"]["visual"]
        self.assertEqual(inventory["classified_entries"] + visual["tensors"], 299_545)
        self.assertEqual(inventory["accounted_source_bytes"], 132_639_846_394)
        self.assertEqual(
            set(plan["slabs"]),
            {"rank0-target", "rank0-mtp", "rank1-target", "rank1-mtp"},
        )
        physical_names: set[str] = set()
        for rank in range(2):
            rank_payload_names: set[str] = set()
            for consumer in ("target", "mtp"):
                slab = plan["slabs"][f"rank{rank}-{consumer}"]
                for entry in slab["entries"]:
                    if entry["entry_type"] != "payload":
                        continue
                    self.assertNotIn(entry["name"], rank_payload_names)
                    rank_payload_names.add(entry["name"])
                    physical_names.add(entry["name"])
        self.assertEqual(len(physical_names), inventory["classified_entries"])

        expected_expert_counts = {
            "rank0-target": ("target_expert_nvfp4", 147_456, 0, 255),
            "rank1-target": ("target_expert_nvfp4", 147_456, 256, 511),
            "rank0-mtp": ("mtp_expert_fp8_block", 1_536, 0, 255),
            "rank1-mtp": ("mtp_expert_fp8_block", 1_536, 256, 511),
        }
        for key, (family, count, first, last) in expected_expert_counts.items():
            experts = [
                entry
                for entry in plan["slabs"][key]["entries"]
                if entry.get("family") == family
            ]
            ids = [entry["source_slices"][0]["expert_id"] for entry in experts]
            self.assertEqual(len(experts), count)
            self.assertEqual((min(ids), max(ids)), (first, last))
        for slab in plan["slabs"].values():
            payloads = [
                entry for entry in slab["entries"] if entry["entry_type"] == "payload"
            ]
            payload_bytes = sum(entry["local_bytes"] for entry in payloads)
            overhead_bytes = slab["slab_bytes"] - payload_bytes
            self.assertLessEqual(overhead_bytes, payload_bytes // 100)
            self.assertEqual(slab["payload_bytes"], payload_bytes)
            self.assertEqual(slab["tensor_alignment_bytes"], 256)
            self.assertLess(slab["io_tail_padding_bytes"], 65_536)
            self.assertTrue(
                all(entry["slab_offset_bytes"] % 256 == 0 for entry in payloads)
            )
            self.assertTrue(
                all(
                    chunk["start_bytes"] % 65_536 == 0
                    and chunk["end_bytes"] % 65_536 == 0
                    for chunk in slab["io_chunks"]
                )
            )
        cuda_usable_bytes = 106 * 1024**3
        for rank in range(2):
            resident_slab_bytes = sum(
                plan["slabs"][f"rank{rank}-{consumer}"]["slab_bytes"]
                for consumer in ("target", "mtp")
            )
            self.assertLessEqual(resident_slab_bytes, cuda_usable_bytes)


if __name__ == "__main__":
    unittest.main()

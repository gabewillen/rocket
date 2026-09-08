#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT / "engines/qwen38-flash-next-nvfp4-2b/src/moe/target_full_moe_c1.cc"
)


class TargetFullMoeC1SourceContract(unittest.TestCase):
    def test_enqueue_orders_real_router_routed_and_shared_launches(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        begin = source.index("TargetDenseOutcome TargetFullMoeC1::enqueue(")
        end = source.index("const TargetDenseIdentity&", begin)
        enqueue = source[begin:end]
        calls = (
            "enqueue_pending_evidence",
            "enqueue_target_router_c1",
            "enqueue_target_moe_c1_routes",
            "routed_stage->enqueue",
            "impl_->routed.enqueue",
            "enqueue_target_shared_c1",
        )
        offsets = [enqueue.index(call) for call in calls]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("launch.rank_local_partial_bf16", enqueue)
        self.assertIn("w.routed_stage.compact_expert_ids", enqueue)
        self.assertIn("w.routed_stage.compact_routing_weights", enqueue)
        routed = enqueue[enqueue.index("const TargetMoeB12xLaunch"):]
        self.assertNotIn("w.local_ids_i32", routed)
        self.assertNotIn("w.local_weights_f32", routed)

    def test_enqueue_owns_nothing_and_never_synchronizes(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        begin = source.index("TargetDenseOutcome TargetFullMoeC1::enqueue(")
        end = source.index("const TargetDenseIdentity&", begin)
        enqueue = source[begin:end]
        for forbidden in (
            "new ", "delete ", "cudaMalloc", "cudaFree", "cudaMemcpy",
            "cudaMemset", "cudaDeviceSynchronize", "cudaStreamSynchronize",
            "torch", "PyObject",
        ):
            self.assertNotIn(forbidden, enqueue)
        self.assertIn("launch.stream", enqueue)

    def test_no_fake_or_zero_shared_path_exists(self) -> None:
        source = SOURCE.read_text(encoding="utf-8").lower()
        self.assertNotIn("fallback", source)
        self.assertNotIn("synthetic", source)
        self.assertNotIn("zero shared", source)


if __name__ == "__main__":
    unittest.main()

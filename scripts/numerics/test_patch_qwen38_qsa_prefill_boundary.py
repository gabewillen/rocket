#!/usr/bin/env python3
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("patch-qwen38-qsa-prefill-boundary.py")


class TestQsaPrefillBoundaryPatcher(unittest.TestCase):
    def test_capture_is_authenticated_m35_and_reads_back_main_kv(self):
        source = SCRIPT.read_text()
        self.assertIn("owner.indexer.layer_id != 3", source)
        self.assertIn("not oracle.active_forward", source)
        self.assertIn('"query": ((35, 12, 256), torch.bfloat16)', source)
        self.assertIn('"written_key": ((35, 1, 256)', source)
        self.assertIn("main_key.index_select(0, main_slots)", source)
        self.assertIn('values["k_scale"] = owner._k_scale', source)
        self.assertIn('"written_key": ((35, 1, 256), owner.kv_cache.dtype)', source)
        self.assertIn('"schema": "rocket.qwen38.qsa-prefill-boundary.v1"', source)
        self.assertIn('"generation_index": 0', source)
        self.assertIn("hashlib.sha256(canonical).hexdigest()", source)
        self.assertNotIn('"source_sha256": PINNED_SOURCE_SHA256', source)
        self.assertNotIn("repeat(", source)

    def test_existing_native_boundary_remains_explicitly_c1(self):
        root = SCRIPT.parents[2] / "engines/qwen38-flash-next-nvfp4-2b"
        state = (root / "src/attention/qsa_target_state_view.h").read_text()
        graph = (root / "src/attention/native_qsa_graph.cc").read_text()
        self.assertIn("state.rows == 1", state)
        self.assertIn("qwen38_target_qsa_project_qkv_c1", graph)
        self.assertIn("qwen38_target_qsa_attention_c1", graph)


if __name__ == "__main__":
    unittest.main()

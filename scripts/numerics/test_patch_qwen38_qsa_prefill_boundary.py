#!/usr/bin/env python3
from pathlib import Path
import subprocess
import tempfile
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
        self.assertIn('"schema": "rocket.qwen38.qsa-prefill-boundary.v2"', source)
        self.assertIn('"raw_state": ((4, 1, 140), torch.bfloat16)', source)
        self.assertIn('"compressed_state_slots": ((8,), torch.int64)', source)
        self.assertIn("raw_cache.index_select(0, raw_state_slots)", source)
        self.assertIn(
            "compressed_cache.index_select(\n        0, compressed_state_slots",
            source,
        )
        self.assertIn("compressed_cache.dtype", source)
        self.assertNotIn("owner.indexer.indexer_dtype", source)
        self.assertIn('"generation_index": 0', source)
        self.assertIn("hashlib.sha256(canonical).hexdigest()", source)
        self.assertNotIn('"source_sha256": PINNED_SOURCE_SHA256', source)
        self.assertNotIn("repeat(", source)

    def test_existing_native_boundary_remains_explicitly_c1(self):
        root = SCRIPT.parents[2] / "engines/qwen38-flash-next-nvfp4-2b"
        state = (root / "src/attention/qsa_target_state_view.h").read_text()
        graph = (root / "src/attention/native_qsa_graph.cc").read_text()
        rope = (root / "src/attention/layer3_rope_owner.h").read_text()
        self.assertIn("state.rows == 1", state)
        self.assertIn("qwen38_target_qsa_project_qkv_c1", graph)
        self.assertIn("qwen38_target_qsa_attention_c1", graph)
        self.assertIn("kLayer3RopeRows = 35", rope)

    def test_v3_extends_v2_with_same_forward_c1_inputs(self):
        source = SCRIPT.read_text()
        self.assertIn('"schema": "rocket.qwen38.qsa-c1-input.v3"', source)
        self.assertIn('"parent_artifact_key": _ROCKET_QSA_PREFILL_V2_KEY', source)
        self.assertIn('"row35_hidden": ((1, 2560), torch.bfloat16)', source)
        self.assertIn('"row35_index_query": ((1, 4, 128), torch.bfloat16)', source)
        self.assertIn('torch.full((3, 1), 35', source)
        self.assertIn("authenticated QSA v2 payload changed", source)
        self.assertIn("self.indexer._rocket_c1_hidden = (", source)
        self.assertIn("self._rocket_c1_index_query = q.detach()", source)

    def test_patcher_rejects_unpaired_indexer_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    "python3", str(SCRIPT),
                    "--input", str(SCRIPT),
                    "--output", str(Path(directory) / "qsa.py"),
                    "--indexer-input", str(SCRIPT),
                ],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import unittest

ENGINE = Path(__file__).parents[1]


class TargetQsaPreprocessSourceTests(unittest.TestCase):
    def test_replay_path_is_c1_external_state_only(self):
        source = (ENGINE / "src/attention/qsa_target_preprocess.cu").read_text()
        header = (ENGINE / "src/attention/qsa_target_preprocess.h").read_text()
        launch = source.split("void launch_target_qsa_preprocess_c1", 1)[1]
        self.assertIn("state.main_key_cache", launch)
        self.assertIn("state.raw_key_cache", launch)
        self.assertIn("index_qk_gemv<<<kIndexOutputs", launch)
        self.assertNotIn("cudaMalloc", launch)
        self.assertNotIn("cudaMemcpy", launch)
        self.assertNotIn("cudaDeviceSynchronize", launch)
        self.assertNotIn("cudaStreamSynchronize", launch)
        self.assertIn("Caller-owned graph arena", header)

    def test_pinned_and_current_upstream_provenance_is_named(self):
        source = (ENGINE / "src/attention/qsa_target_preprocess.cu").read_text()
        self.assertIn("8e685d198", source)
        self.assertIn("9ea8f3ffc354901b740f0b31988900897b7221d7", source)

    def test_target_selection_uses_external_block64_c1_buffers(self):
        source = (ENGINE / "src/projection/cutlass_qkv.cu").read_text()
        launch = source.split('extern "C" int qwen38_target_qsa_select_c1', 1)[1]
        launch = launch.split('extern "C" int qwen38_target_qsa_projection_create_c1', 1)[0]
        self.assertIn("score_target_qsa_c1_block64", launch)
        self.assertIn("select_qsa_topk_radix512<<<1, 512", launch)
        self.assertIn("selected_tokens", launch)
        self.assertNotIn("cudaMalloc", launch)
        self.assertNotIn("cudaDeviceSynchronize", launch)

    def test_c1_projection_replay_borrows_graph_arena(self):
        source = (ENGINE / "src/projection/cutlass_qkv.cu").read_text()
        replay = source.split('extern "C" int qwen38_target_qsa_project_qkv_c1', 1)[1]
        replay = replay.split('extern "C" int qwen38_target_qsa_projection_destroy_c1', 1)[0]
        self.assertIn("<<<1, 256", replay)
        self.assertIn("apply_attention_gate_c1", replay)
        self.assertNotIn("cudaMalloc", replay)
        self.assertNotIn("cudaMemcpy", replay)
        self.assertNotIn("cudaDeviceSynchronize", replay)
        self.assertNotIn("cudaStreamSynchronize", replay)

    def test_c1_attention_reads_external_bf16_pages(self):
        source = (ENGINE / "src/projection/cutlass_qkv.cu").read_text()
        launch = source.split('extern "C" int qwen38_target_qsa_attention_c1', 1)[1]
        self.assertIn("kTargetMainPageSize = 1600", launch)
        self.assertIn("qsa_sparse_splitk_block16<<<dim3(1", launch)
        self.assertIn("main_block_table", launch)
        self.assertNotIn("cudaMalloc", launch)
        self.assertNotIn("cudaMemcpy", launch)
        self.assertNotIn("cudaDeviceSynchronize", launch)

    def test_concrete_graph_composes_the_full_c1_chain(self):
        source = (ENGINE / "src/attention/native_qsa_graph.cc").read_text()
        launch = source.split("void NativeQsaFullAttentionGraph::launch", 1)[1]
        for operation in (
            "qwen38_target_qsa_project_qkv_c1",
            "launch_target_qsa_preprocess_c1",
            "qwen38_target_qsa_select_c1",
            "qwen38_target_qsa_attention_c1",
            "qwen38_target_qsa_project_output_c1",
        ):
            self.assertIn(operation, launch)
        self.assertNotIn("cudaMalloc", launch)
        self.assertNotIn("cudaMemcpy", launch)
        self.assertNotIn("Synchronize", launch)
        self.assertIn("validate_target_qsa_state_view", launch)


if __name__ == "__main__":
    unittest.main()

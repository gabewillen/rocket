# SPDX-License-Identifier: Apache-2.0
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/decode/target_layer3_generation.cu"


class TargetLayer3GenerationSourceTests(unittest.TestCase):
    def test_single_kernel_updates_qsa_then_moe_generation(self):
        source = SOURCE.read_text()
        kernel = source.index("__global__ void prepare_layer3_generation")
        qsa = source.index("*positions = row", kernel)
        moe = source.index("*target_moe_requested_generation = generation", qsa)
        launch = source.index("prepare_layer3_generation<<<1, 1, 0, stream>>>")
        error = source.index("cudaPeekAtLastError()", launch)
        self.assertLess(qsa, moe)
        self.assertLess(launch, error)

    def test_replay_path_has_no_allocation_copy_or_sync(self):
        source = SOURCE.read_text()
        for forbidden in (
            "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaStreamSynchronize",
            "cudaDeviceSynchronize", "cudaMemcpyDeviceToHost",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("row != next_row_", source)
        self.assertIn("faulted_ = true", source)


if __name__ == "__main__":
    unittest.main()

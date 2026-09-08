#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import ctypes
import pathlib
import random
import struct
import unittest

from qwen38_slab.projection import _e2m1, _e4m3, _sfb_offset
from qwen38_slab.routed_moe import load_owner_local_moe


ROOT = pathlib.Path(__file__).resolve().parents[3]
ARTIFACT = pathlib.Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/"
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
)
NATIVE = (
    ROOT / "engines/qwen38-flash-next-nvfp4-2b/"
    "build-k0-target-moe-aot-link/libqwen38_target_router_shared_c1.so"
)
CUDA_SOURCE = (
    ROOT / "engines/qwen38-flash-next-nvfp4-2b/"
    "src/moe/target_router_shared_c1.cu"
)
HEADER = (
    ROOT / "engines/qwen38-flash-next-nvfp4-2b/"
    "src/moe/target_router_shared_c1.h"
)
HIDDEN = 2560


@unittest.skipUnless(ARTIFACT.is_dir() and NATIVE.is_file(),
                     "authenticated slab and native CPU decoder are required")
class TargetRouterNativeLayoutContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.native = ctypes.CDLL(NATIVE)
        cls.native.rocket_qwen38_target_nvfp4_e2m1_host.argtypes = [ctypes.c_uint8]
        cls.native.rocket_qwen38_target_nvfp4_e2m1_host.restype = ctypes.c_float
        cls.native.rocket_qwen38_target_nvfp4_packed_host.argtypes = [
            ctypes.c_uint8, ctypes.c_int,
        ]
        cls.native.rocket_qwen38_target_nvfp4_packed_host.restype = ctypes.c_float
        cls.native.rocket_qwen38_target_nvfp4_e4m3_host.argtypes = [ctypes.c_uint8]
        cls.native.rocket_qwen38_target_nvfp4_e4m3_host.restype = ctypes.c_float
        cls.native.rocket_qwen38_target_router_sfb_offset_host.argtypes = [
            ctypes.c_int, ctypes.c_int,
        ]
        cls.native.rocket_qwen38_target_router_sfb_offset_host.restype = ctypes.c_int

    def test_enqueue_is_borrowed_stream_and_allocation_free(self) -> None:
        source = CUDA_SOURCE.read_text(encoding="utf-8")
        for forbidden in (
            "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaMemset",
            "cudaDeviceSynchronize", "cudaStreamSynchronize", "torch", "PyObject",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("launch.stream>>>", source)
        self.assertIn("partial * shared_gate[0]", source)
        self.assertIn("__bfloat162float(output[column]) +", source)

    def test_failure_dimensions_and_fixed_shapes_are_bounded(self) -> None:
        header = HEADER.read_text(encoding="utf-8")
        self.assertEqual(header.count("  k", header.index("enum class TargetDenseFailure"),
                                      header.index("};", header.index("enum class TargetDenseFailure"))), 11)
        for contract in ("2'560", "512", "10", "320", "160"):
            self.assertIn(contract, header)

    def test_all_packed_nibbles_match_accepted_reconstruction(self) -> None:
        for pair in range(256):
            self.assertEqual(
                self.native.rocket_qwen38_target_nvfp4_packed_host(pair, 0),
                _e2m1(pair & 15),
            )
            self.assertEqual(
                self.native.rocket_qwen38_target_nvfp4_packed_host(pair, 1),
                _e2m1(pair >> 4),
            )

    def test_sfb_boundaries_match_cutlass_layout(self) -> None:
        for row in (0, 31, 32, 127, 128, 255, 256, 511):
            for group in (0, 1, 3, 4, 39, 40, 79, 80, 156, 159):
                self.assertEqual(
                    self.native.rocket_qwen38_target_router_sfb_offset_host(
                        row, group
                    ),
                    _sfb_offset(row, group, HIDDEN),
                )

    def test_real_layer3_router_samples_match_on_both_ranks(self) -> None:
        rng = random.Random(38)
        samples = {
            (row, column)
            for row in (0, 31, 32, 127, 128, 255, 256, 511)
            for column in (0, 1, 15, 16, 17, 639, 640, 2543, 2544, 2559)
        }
        samples.update((rng.randrange(512), rng.randrange(HIDDEN)) for _ in range(256))
        rank_payloads = []
        for rank in (0, 1):
            slab = load_owner_local_moe(ARTIFACT, rank, 3)
            extents = {item.name.rsplit(".gate.", 1)[-1]: item for item in slab.router}
            with slab.slab_path.open("rb", buffering=0) as stream:
                def read(name):
                    extent = extents[name]
                    stream.seek(extent.offset)
                    return stream.read(extent.length)
                packed = read("weight")
                scales = read("weight_scale")
                alpha = struct.unpack("<f", read("weight_scale_2"))[0]
            decoded = []
            for row, column in sorted(samples):
                pair = packed[row * (HIDDEN // 2) + column // 2]
                sf_index = self.native.rocket_qwen38_target_router_sfb_offset_host(
                    row, column // 16
                )
                sf = scales[sf_index]
                native = (
                    self.native.rocket_qwen38_target_nvfp4_packed_host(pair, column)
                    * self.native.rocket_qwen38_target_nvfp4_e4m3_host(sf)
                    * alpha
                )
                reference = (
                    _e2m1(pair >> 4 if column & 1 else pair & 15)
                    * _e4m3(scales[_sfb_offset(row, column // 16, HIDDEN)])
                    * alpha
                )
                self.assertAlmostEqual(native, reference, places=6)
                decoded.append(native)
            rank_payloads.append((packed, scales, alpha, tuple(decoded)))
        self.assertEqual(rank_payloads[0], rank_payloads[1])


if __name__ == "__main__":
    unittest.main()

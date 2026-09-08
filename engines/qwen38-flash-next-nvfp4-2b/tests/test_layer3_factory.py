# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import os
import tempfile
import unittest
import weakref
from pathlib import Path
from types import MappingProxyType

from qwen38_slab.layer3_factory import (
    CtypesNativeTargetSlabLeaseFactory,
    Layer3FactoryError,
    native_target_slab_handoff,
    prepare_layer3_physical_plan,
    native_rank_descriptor,
    public_plan,
)
from qwen38_slab.cuda_slab_loader import (
    ChunkTransferReceipt, LoadedRankSlabs, RankLoadReceipt,
    SlabTransferReceipt,
)

ARTIFACT = Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/"
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
)
SIDECAR = Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/qsa-indexer-sidecars/"
    "bdbebd4f45c398f090a41ab98cd3881b969d958d8ae0bc42f3411844d3262edd"
)
ORACLE = Path("/home/glwillen/calibration/qwen38-k0-oracle-a1794d5-01/capture")


class Span:
    def __init__(self):
        self.attributes = {}
        self.exception = None
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, exception): self.exception = exception


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, name):
        span = Span(); self.spans.append(span); return span


class TargetTensor:
    copies = 0
    dtype = "torch.uint8"
    device = "cuda:0"
    def __init__(self, pointer, elements):
        self.pointer = pointer; self.elements = elements
    def data_ptr(self): return self.pointer
    def numel(self): return self.elements
    def copy_(self, *args, **kwargs):
        type(self).copies += 1
        raise AssertionError("handoff must not copy")


class ReadyEvent:
    cuda_event = 0xABC0


class Layer3FactoryTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("ROCKET_QWEN38_TARGET_SLAB_OWNER_LIBRARY"),
        "native accepted-loader lease library is unavailable",
    )
    def test_concrete_native_lease_factory_retains_exact_rank1_handoff(self):
        plan = prepare_layer3_physical_plan(
            artifact=ARTIFACT, indexer_sidecar=SIDECAR,
            oracle_capture=ORACLE, tracer=Tracer(),
        )
        descriptor = native_rank_descriptor(plan, 1)
        size = descriptor["slab_bytes"]
        tensor = TargetTensor(0x2234_0000, size)
        chunks = tuple(
            ChunkTransferReceipt(index, size if index == 0 else 0, 1, 1, 1)
            for index in range(236)
        )
        target = SlabTransferReceipt(
            "rank1-target", size, size, 236, 236, 1, 2, chunks,
        )
        loaded = LoadedRankSlabs(
            MappingProxyType({"rank1-target": tensor, "rank1-mtp": object()}),
            RankLoadReceipt(
                1, target,
                SlabTransferReceipt(
                    "rank1-mtp", 1, 1, 1, 1, 1, 2,
                    (ChunkTransferReceipt(0, 1, 1, 1, 1),),
                ),
                1, 1, 2, 1,
            ),
            ReadyEvent(),
        )
        with self.assertRaisesRegex(Layer3FactoryError, "identity"):
            native_target_slab_handoff(descriptor, loaded)

    def test_missing_oracle_fails_closed_with_bounded_telemetry(self):
        tracer = Tracer()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(Layer3FactoryError, "oracle manifest"):
                prepare_layer3_physical_plan(
                    artifact=ARTIFACT, indexer_sidecar=SIDECAR,
                    oracle_capture=Path(directory), tracer=tracer,
                )
        self.assertEqual(tracer.spans[-1].attributes, {
            "phase": "prepare", "outcome": "failure",
            "failure.class": "contract",
        })

    @unittest.skipUnless(
        ARTIFACT.is_dir() and SIDECAR.is_dir() and ORACLE.is_dir(),
        "authenticated layer-3 artifacts are unavailable",
    )
    def test_real_two_rank_plan_authenticates_replicated_prefill_boundary(self):
        tracer = Tracer()
        plan = prepare_layer3_physical_plan(
            artifact=ARTIFACT, indexer_sidecar=SIDECAR,
            oracle_capture=ORACLE, tracer=tracer,
        )
        record = dict(public_plan(plan))
        self.assertEqual(record["ranks"], [0, 1])
        self.assertEqual(record["replicated_hc_shape"], [35, 10240])
        self.assertEqual(record["replay"], "sequential_rows_0_34")
        self.assertEqual(record["compare_row"], 34)
        self.assertEqual(record["pair_reduce"]["calls_per_rank"], 70)
        self.assertEqual(record["pair_reduce"]["session_sha256"],
                         "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b")
        self.assertEqual(tuple(item.rank for item in plan.ranks), (0, 1))
        descriptors = [dict(native_rank_descriptor(plan, rank)) for rank in (0, 1)]
        for rank, descriptor in enumerate(descriptors):
            self.assertEqual(descriptor["rank"], rank)
            self.assertEqual(len(descriptor["extents"]), 3_108)
            self.assertEqual(len(descriptor["descriptor_sha256"]), 64)
            self.assertEqual(len(descriptor["slab_publication_layout_sha256"]), 64)
            self.assertTrue(all(item["source_chunks"]
                                for item in descriptor["extents"]))
            self.assertEqual(descriptor["native_abis"]["pair_reduce_bootstrap"], 3)
            globals_ = descriptor["qsa_projection_globals"]
            self.assertEqual(set(globals_), {"q", "k", "v", "o"})
            self.assertTrue(all(value["dtype"] == "F32"
                                and len(value["value_le_hex"]) == 8
                                and len(value["source_chunk_sha256"]) == 64
                                for value in globals_.values()))
        self.assertNotEqual(descriptors[0]["descriptor_sha256"],
                            descriptors[1]["descriptor_sha256"])
        self.assertEqual(tracer.spans[-1].attributes, {
            "phase": "prepare", "outcome": "success",
            "failure.class": "none",
        })

    @unittest.skipUnless(
        ARTIFACT.is_dir() and SIDECAR.is_dir() and ORACLE.is_dir(),
        "authenticated layer-3 artifacts are unavailable",
    )
    def test_native_handoff_retains_owner_aliases_source_and_never_copies(self):
        plan = prepare_layer3_physical_plan(
            artifact=ARTIFACT, indexer_sidecar=SIDECAR,
            oracle_capture=ORACLE, tracer=Tracer(),
        )
        descriptor = native_rank_descriptor(plan, 0)
        size = descriptor["slab_bytes"]
        tensor = TargetTensor(0x1234_0000, size)
        chunks = tuple(
            ChunkTransferReceipt(index, size if index == 0 else 0, 1, 1, 1)
            for index in range(236)
        )
        target = SlabTransferReceipt(
            "rank0-target", size, size, 236, 236, 1, 2, chunks,
        )
        mtp = SlabTransferReceipt(
            "rank0-mtp", 1, 1, 1, 1, 1, 2,
            (ChunkTransferReceipt(0, 1, 1, 1, 1),),
        )
        loaded = LoadedRankSlabs(
            MappingProxyType({"rank0-target": tensor, "rank0-mtp": object()}),
            RankLoadReceipt(0, target, mtp, 1, 1, 2, 1), ReadyEvent(),
        )
        TargetTensor.copies = 0
        with self.assertRaisesRegex(Layer3FactoryError, "identity"):
            native_target_slab_handoff(descriptor, loaded)
        self.assertEqual(TargetTensor.copies, 0)


if __name__ == "__main__":
    unittest.main()

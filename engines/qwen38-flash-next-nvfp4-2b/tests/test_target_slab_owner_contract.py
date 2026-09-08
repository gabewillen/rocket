import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/runtime/qwen38-generate-native-target-slab-contract.py"
ARTIFACT = Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/"
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
)
INCLUDE = ROOT / (
    "engines/qwen38-flash-next-nvfp4-2b/src/model/target_slab_contract.inc"
)
HEADER = ROOT / (
    "engines/qwen38-flash-next-nvfp4-2b/src/model/target_slab_owner.h"
)

spec = importlib.util.spec_from_file_location("target_slab_generator", SCRIPT)
generator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(generator)


@unittest.skipUnless(ARTIFACT.is_dir(), "authenticated rank slab is unavailable")
class TargetSlabOwnerContractTests(unittest.TestCase):
    def test_generated_contract_matches_authenticated_manifest(self):
        generated = generator.generate(ARTIFACT)
        self.assertEqual(generated, INCLUDE.read_text())
        self.assertIn("kTargetSlabRingDepth = 4", HEADER.read_text())
        self.assertEqual(generated.count("ULL, \""), 472)

    def test_every_generated_chunk_digest_mutation_is_rejected(self):
        generated = generator.generate(ARTIFACT)
        manifest = json.loads((ARTIFACT / "manifest.json").read_text())
        for rank in (0, 1):
            for index, chunk in enumerate(
                manifest["slabs"][f"rank{rank}-target"]["chunks"]
            ):
                original = chunk["sha256"]
                replacement = ("0" if original[0] != "0" else "1") + original[1:]
                mutated = generated.replace(original, replacement, 1)
                with self.subTest(rank=rank, chunk=index):
                    self.assertNotEqual(mutated, generated)
                    self.assertNotEqual(mutated, INCLUDE.read_text())

    def test_real_manifest_mutation_is_rejected(self):
        manifest = json.loads((ARTIFACT / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / generator.ARTIFACT_KEY
            artifact.mkdir()
            chunk = manifest["slabs"]["rank1-target"]["chunks"][235]
            original = chunk["sha256"]
            chunk["sha256"] = ("0" if original[0] != "0" else "1") + original[1:]
            (artifact / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            with self.assertRaisesRegex(
                ValueError, "authenticated target-slab manifest changed"
            ):
                generator.generate(artifact)

    def test_generated_extent_order_and_layout_are_independently_recomputed(self):
        manifest = json.loads((ARTIFACT / "manifest.json").read_text())
        include = INCLUDE.read_text()
        for rank in (0, 1):
            slab = manifest["slabs"][f"rank{rank}-target"]
            offset = 0
            for chunk in slab["chunks"]:
                self.assertEqual(chunk["offset_bytes"], offset)
                offset += chunk["length_bytes"]
            self.assertEqual(offset, 63_212_748_800)
            selected = {key: slab[key] for key in ("file", "bytes", "chunks")}
            layout = hashlib.sha256(generator.canonical(selected)).hexdigest()
            self.assertIn(f'kRank{rank}LayoutSha256 = "{layout}"', include)

    def test_production_source_excludes_fallback_io_and_frameworks(self):
        source = (
            ROOT
            / "engines/qwen38-flash-next-nvfp4-2b/src/model/target_slab_owner.cc"
        ).read_text()
        self.assertIn("O_DIRECT", source)
        self.assertIn("kTargetSlabRingDepth", source)
        self.assertIn("emit(telemetry, TargetSlabLoadPhase::kPublish", source)
        self.assertIn("TargetSlabFailureClass::kNone, rank", source)
        self.assertIn("TargetSlabLoadPhase::kCleanup", source)
        for forbidden in ("mmap(", "cuFile", "GDS", "torch", "Python.h"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_physical_harness_is_guarded_and_samples_after_publication(self):
        harness = (
            ROOT
            / "engines/qwen38-flash-next-nvfp4-2b/bench/target_slab_load.cc"
        ).read_text()
        publication = harness.index("owner->publication()")
        sampling = harness.index("sample_matches(metadata.payload")
        cleanup = harness.index("owner.reset()")
        self.assertLess(publication, sampling)
        self.assertLess(sampling, cleanup)
        for field in (
            "chunks_authenticated",
            "peak_host_pinned_bytes",
            "gpu_allocation_delta_bytes",
            "gpu_cleanup_delta_bytes",
            "open_to_publish_ns",
            "cold_load_regression_guard_ns",
            "cold_load_regression_guard_passed",
            "publication_fence_completed",
            "bytes_per_second",
            "sample_offsets",
            "samples_match",
            "telemetry_overflow",
        ):
            with self.subTest(field=field):
                self.assertIn(field, harness)
        self.assertNotIn("cudaDeviceSynchronize", harness)
        self.assertIn("cudaEventQuery(view.ready_event)", harness)
        self.assertGreaterEqual(harness.count("telemetry.count != 1"), 2)
        self.assertIn("17'862'785'416ULL", harness)

        source = (
            ROOT
            / "engines/qwen38-flash-next-nvfp4-2b/src/model/target_slab_owner.cc"
        ).read_text()
        opened = source.index("const auto opened = Clock::now()")
        direct_open = source.index("O_RDONLY | O_DIRECT", opened)
        publication = source.index("owner->publication_ =")
        self.assertLess(opened, direct_open)
        self.assertLess(direct_open, publication)


if __name__ == "__main__":
    unittest.main()

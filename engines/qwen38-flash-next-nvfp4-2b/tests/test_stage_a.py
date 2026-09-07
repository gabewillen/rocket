from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qwen38_slab import DirectSlabLoader, SlabContract, SlabError, materialize
from qwen38_slab.contract import MODEL_NVFP4_ABI, MTP_FP8_ABI, PAGE_BYTES, SCHEMA, sf_swizzle


class Span:
    def __init__(self):
        self.attributes = {}
        self.exceptions = []
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, exception): self.exceptions.append(type(exception).__name__)


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, name):
        span = Span(); span.name = name; self.spans.append(span); return span


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]):
    header, offset = {}, 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(item[2] for item in tensors.values()))
    return hashlib.sha256(raw).hexdigest(), 8 + len(raw), header


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.checkpoint = root / "checkpoint"; self.checkpoint.mkdir()
        self.overlay = root / "overlay"; self.overlay.mkdir()
        self.output = root / "output"
        self.matrix_name = "model.language_model.layers.0.linear_attn.in_proj_z.weight"
        matrix = bytes(range(256)) * 32
        embed = bytes(range(32))
        mtp = bytes(range(16))
        source_tensors = {
            self.matrix_name: ("BF16", [128, 32], matrix),
            "model.language_model.embed_tokens.weight": ("BF16", [4, 4], embed),
            "mtp.layers.0.mlp.experts.0.gate_proj.weight": ("F8_E4M3", [4, 4], mtp),
        }
        header_hash, payload_start, header = write_safetensors(self.checkpoint / "model.safetensors", source_tensors)
        file_size = (self.checkpoint / "model.safetensors").stat().st_size
        self.sources = {}
        for name, (_, _, data) in source_tensors.items():
            offsets = header[name]["data_offsets"]
            self.sources[name] = {
                "file": "model.safetensors", "blob": header_hash, "file_size_bytes": file_size,
                "header_sha256": header_hash, "payload_offset_bytes": payload_start,
                "data_offsets": offsets, "absolute_offset_bytes": payload_start + offsets[0],
                "byte_count": len(data),
            }
        packed = bytes((index * 7) % 251 for index in range(128 * 16))
        scales = bytes((index * 11) % 251 for index in range(128 * 2))
        overlay_tensors = {
            self.matrix_name: ("U8", [128, 16], packed),
            self.matrix_name.removesuffix("weight") + "weight_scale": ("F8_E4M3", [128, 2], scales),
            self.matrix_name.removesuffix("weight") + "weight_scale_2": ("F32", [1], b"\0\0\x80?"),
            self.matrix_name.removesuffix("weight") + "input_scale": ("F32", [1], b"\0\0\x00?"),
        }
        _, _, _ = write_safetensors(self.overlay / "overlay.safetensors", overlay_tensors)
        overlay_sha = hashlib.sha256((self.overlay / "overlay.safetensors").read_bytes()).hexdigest()
        artifact_key = "a" * 64
        (self.overlay / "manifest.json").write_text(json.dumps({
            "schema": "rocket.qwen38.nvfp4-overlay.v2", "artifact_key": artifact_key,
            "source": {"revision": "synthetic", "families": ["base_ple", "base_routers", "full_attention", "linear_attention"],
                       "tensors": [{"name": self.matrix_name, "sha256": hashlib.sha256(matrix).hexdigest()}]},
            "overlay": {"file": "overlay.safetensors", "sha256": overlay_sha},
        }))
        self.contract = SlabContract("synthetic", artifact_key, overlay_sha, 1, chunk_bytes=PAGE_BYTES)
        self.plan = root / "plan.json"
        self.plan_data = self._plan()
        self.plan.write_text(json.dumps(self.plan_data))

    def entry(self, name, dtype, shape, local_shape, fragments, family):
        return {"entry_type": "payload", "name": name, "family": family, "dtype": dtype,
                "full_shape": shape, "local_shape": local_shape, "local_bytes": 1,
                "slab_offset_bytes": 0, "source": self.sources[name],
                "source_slices": fragments, "transforms": fragments,
                "physical_owner": "unused", "consumers": ["target"]}

    def _plan(self):
        slabs = {}
        for rank in range(2):
            matrix_fragment = [{"dimension": 1, "start": rank * 16, "length": 16, "shape": [128, 16]}]
            full = [{"dimension": None, "start": 0, "length": None, "shape": [4, 4]}]
            target_key, mtp_key = f"rank{rank}-target", f"rank{rank}-mtp"
            embed = self.entry("model.language_model.embed_tokens.weight", "BF16", [4, 4], [4, 4], full, "token_embedding_or_head")
            embed["physical_owner"] = target_key; embed["consumers"] = ["target", "mtp"]
            matrix = self.entry(self.matrix_name, "BF16", [128, 32], [128, 16], matrix_fragment, "linear_attn.in_proj_z")
            matrix["physical_owner"] = target_key
            slabs[target_key] = {"key": target_key, "entries": [matrix, embed]}
            mtp_fragment = [{"dimension": None, "start": 0, "length": None, "shape": [4, 4]}]
            mtp = self.entry("mtp.layers.0.mlp.experts.0.gate_proj.weight", "F8_E4M3", [4, 4], [4, 4], mtp_fragment, "mtp_expert_fp8_block")
            mtp["physical_owner"] = mtp_key
            slabs[mtp_key] = {"key": mtp_key, "entries": [
                {"entry_type": "reference", "name": embed["name"], "physical_owner": target_key, "payload_io": False}, mtp]}
        return {"schema": "rocket.qwen38-rank-slab-plan.v2", "checkpoint_revision": "synthetic",
                "tensor_parallel_size": 2, "tensor_alignment_bytes": 256,
                "io_alignment_bytes": PAGE_BYTES, "io_chunk_bytes": PAGE_BYTES,
                "payload_io_performed": False, "assignment_phase": "complete-before-payload-io",
                "slabs": slabs}

    def build(self):
        return materialize(self.plan, self.checkpoint, self.overlay, self.output, self.contract)


class StageAContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temp.name))
    def tearDown(self): self.temp.cleanup()

    def test_synthetic_end_to_end_inventory_overlay_alignment_and_abis(self):
        artifact = self.fixture.build()
        manifest = json.loads((artifact / "manifest.json").read_text())
        self.assertEqual(manifest["schema"], SCHEMA)
        self.assertEqual(manifest["overlay_substitutions"], 1)
        self.assertEqual(set(manifest["slabs"]), {"rank0-target", "rank0-mtp", "rank1-target", "rank1-mtp"})
        for slab in manifest["slabs"].values():
            self.assertEqual(slab["bytes"] % PAGE_BYTES, 0)
            self.assertTrue(all(entry["offset_bytes"] % 256 == 0 for entry in slab["entries"]))
            self.assertTrue(all(chunk["length_bytes"] % PAGE_BYTES == 0 for chunk in slab["chunks"]))
        target = manifest["slabs"]["rank0-target"]
        scale = next(item for item in target["entries"] if item["name"].endswith("weight_scale"))
        self.assertEqual(scale["abi"], MODEL_NVFP4_ABI)
        self.assertEqual(scale["layout"], "cutlass_sm121_sfb")
        mtp = manifest["slabs"]["rank0-mtp"]
        self.assertEqual(mtp["entries"][0]["abi"], MTP_FP8_ABI)
        self.assertEqual(mtp["references"], [{"name": "model.language_model.embed_tokens.weight", "physical_owner": "rank0-target"}])
        self.assertFalse(any(item["name"].endswith("in_proj_z.weight") and item["dtype"] == "BF16" for item in target["entries"]))

    def test_family_topology_page_and_inventory_drift_fail_closed(self):
        original = json.loads((self.fixture.overlay / "manifest.json").read_text())
        for mutation, message in (
            (lambda value: value["source"].update(families=["linear_attention"]), "family"),
            (lambda value: value["source"]["tensors"].append(value["source"]["tensors"][0]), "inventory"),
        ):
            value = json.loads(json.dumps(original)); mutation(value)
            (self.fixture.overlay / "manifest.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(SlabError, message): self.fixture.build()
        (self.fixture.overlay / "manifest.json").write_text(json.dumps(original))
        value = json.loads(self.fixture.plan.read_text()); value["tensor_parallel_size"] = 4
        self.fixture.plan.write_text(json.dumps(value))
        with self.assertRaisesRegex(SlabError, "contract drift"): self.fixture.build()
        value["tensor_parallel_size"] = 2; value["io_alignment_bytes"] = 4096
        self.fixture.plan.write_text(json.dumps(value))
        with self.assertRaisesRegex(SlabError, "contract drift"): self.fixture.build()

    def test_source_and_overlay_tamper_fail_closed(self):
        source = self.fixture.checkpoint / "model.safetensors"
        data = bytearray(source.read_bytes())
        data[self.fixture.sources[self.fixture.matrix_name]["absolute_offset_bytes"]] ^= 1
        source.write_bytes(data)
        with self.assertRaisesRegex(SlabError, "provenance drift"): self.fixture.build()
        self.fixture = Fixture(Path(self.temp.name) / "second")
        overlay = self.fixture.overlay / "overlay.safetensors"
        data = bytearray(overlay.read_bytes()); data[-1] ^= 1; overlay.write_bytes(data)
        with self.assertRaisesRegex(SlabError, "payload digest mismatch"): self.fixture.build()

    def test_chunk_digest_and_direct_io_are_enforced(self):
        artifact = self.fixture.build(); tracer = Tracer()
        loader = DirectSlabLoader(artifact, tracer, self.fixture.contract)
        seen = []
        try:
            count = loader.read("rank0-target", lambda offset, data: seen.append((offset, len(data))))
        except OSError as exc:
            if exc.errno in {22, 95}: self.skipTest(f"filesystem lacks O_DIRECT: {exc}")
            raise
        self.assertEqual(count, PAGE_BYTES); self.assertEqual(seen, [(0, PAGE_BYTES)])
        self.assertEqual(tracer.spans[0].attributes["slab.kind"], "rank0-target")
        slab = artifact / "rank0-target.slab"; os.chmod(slab, 0o644)
        data = bytearray(slab.read_bytes()); data[0] ^= 1; slab.write_bytes(data)
        with self.assertRaisesRegex(SlabError, "chunk digest mismatch"):
            loader.read("rank0-target", lambda _offset, _data: None)

    def test_direct_io_unavailable_and_open_failure_do_not_fallback(self):
        artifact = self.fixture.build(); loader = DirectSlabLoader(artifact, Tracer(), self.fixture.contract)
        with mock.patch("qwen38_slab.loader.os.open", side_effect=OSError(22, "direct rejected")) as opened:
            with self.assertRaisesRegex(SlabError, "O_DIRECT"): loader.read("rank0-target", lambda _offset, _data: None)
        self.assertTrue(opened.call_args.args[1] & os.O_DIRECT)

    def test_manifest_tamper_and_incomplete_slab_inventory_fail_closed(self):
        artifact = self.fixture.build()
        manifest_path = artifact / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["slabs"].pop("rank1-mtp")
        os.chmod(manifest_path, 0o644)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(SlabError, "content-address digest mismatch"):
            DirectSlabLoader(artifact, Tracer(), self.fixture.contract)

        value = json.loads(self.fixture.plan.read_text())
        value["slabs"].pop("rank1-mtp")
        self.fixture.plan.write_text(json.dumps(value))
        second = Fixture(Path(self.temp.name) / "incomplete")
        second.plan.write_text(json.dumps(value))
        with self.assertRaisesRegex(SlabError, "exactly four"):
            second.build()

    def test_swizzle_matches_documented_cutlass_atom(self):
        linear = bytes(range(128))
        got = sf_swizzle(linear, 128, 16)
        self.assertEqual(len(got), 512)
        self.assertEqual(got[0], 0); self.assertEqual(got[16], 1); self.assertEqual(got[4], 32)


if __name__ == "__main__": unittest.main()

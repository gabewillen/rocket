# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import unittest

from qwen38_slab.contract import PINNED_CONTRACT, SCHEMA, canonical_bytes
from qwen38_slab.mtp_source import (
    EXTERNAL_MTP_SOURCE_SCHEMA,
    MTP_NONEXPERT_TENSORS,
    MtpSourceError,
    inspect_external_mtp_source,
    inspect_native_mtp_source,
)


def native_manifest(rank: int = 0) -> dict[str, object]:
    first = rank * 256
    entries = []
    offset = 0
    for expert in range(first, first + 256):
        for projection, shape in (
            ("down_proj", [2560, 640]),
            ("gate_proj", [640, 2560]),
            ("up_proj", [640, 2560]),
        ):
            entries.append({
                "name": f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight",
                "offset_bytes": offset,
                "length_bytes": 1_638_400,
                "shape": shape,
                "dtype": "F8_E4M3",
                "layout": "checkpoint",
                "abi": "fp8_e4m3_block_128x128",
            })
            offset += 1_638_400
            scale_shape = [20, 5] if projection == "down_proj" else [5, 20]
            entries.append({
                "name": f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight_scale_inv",
                "offset_bytes": offset,
                "length_bytes": 200,
                "shape": scale_shape,
                "dtype": "BF16",
                "layout": "checkpoint",
                "abi": "fp8_e4m3_block_128x128",
            })
            offset += 256
    for name, spec in MTP_NONEXPERT_TENSORS.items():
        entries.append({
            "name": name,
            "offset_bytes": offset,
            "length_bytes": spec.length_bytes,
            "shape": list(spec.shape),
            "dtype": "BF16",
            "layout": "checkpoint",
            "abi": "native",
        })
        offset += (spec.length_bytes + 255) & ~255
    slab = {
        "bytes": offset,
        "file": f"rank{rank}-mtp.slab",
        "chunks": [{"offset_bytes": 0, "length_bytes": offset, "sha256": "b" * 64}],
        "entries": entries,
        "references": [
            {"name": "lm_head.weight", "physical_owner": f"rank{rank}-target"},
            {"name": "model.language_model.embed_tokens.weight", "physical_owner": f"rank{rank}-target"},
        ],
    }
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "revision": PINNED_CONTRACT.revision,
        "overlay_artifact_key": PINNED_CONTRACT.artifact_key,
        "overlay_sha256": PINNED_CONTRACT.overlay_sha256,
        "tp_size": 2,
        "tensor_alignment_bytes": 256,
        "shared_payload_policy": "target-owned-mtp-reference-read-once",
        "abis": {
            "mtp_experts": "fp8_e4m3_block_128x128",
            "target_nvfp4": "modelopt_nvfp4_group16_cutlass_sm121_sfb",
        },
        "slabs": {f"rank{rank}-mtp": slab},
    }
    manifest["artifact_key"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    return manifest


class MtpSourceContractTests(unittest.TestCase):
    def test_native_source_has_exact_29_nonexpert_and_fp8_e256_inventory(self):
        source = inspect_native_mtp_source(native_manifest(1), 1)
        self.assertEqual(source.rank, 1)
        self.assertEqual(source.expert_abi, "fp8_e4m3_block_128x128")
        self.assertEqual(source.local_experts, (256, 511))
        self.assertEqual(len(source.expert_extents), 1_536)
        self.assertEqual(len(source.nonexpert_extents), 29)
        self.assertEqual(
            source.target_references,
            ("lm_head.weight", "model.language_model.embed_tokens.weight"),
        )

    def test_missing_or_retyped_nonexpert_fails_closed(self):
        manifest = native_manifest()
        slab = manifest["slabs"]["rank0-mtp"]
        entry = next(item for item in slab["entries"] if item["name"] == "mtp.fc_hidden.weight")
        entry["dtype"] = "F8_E4M3"
        manifest.pop("artifact_key")
        manifest["artifact_key"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
        with self.assertRaisesRegex(MtpSourceError, "nonexpert extent contract"):
            inspect_native_mtp_source(manifest, 0)

    def test_external_source_requires_byte_identical_nonexpert_hashes(self):
        native = inspect_native_mtp_source(native_manifest(), 0)
        hashes = {name: "c" * 64 for name in MTP_NONEXPERT_TENSORS}
        record: dict[str, object] = {
            "schema": EXTERNAL_MTP_SOURCE_SCHEMA,
            "source_id": "d" * 64,
            "base_revision": PINNED_CONTRACT.revision,
            "rank": 0,
            "expert_abi": "modelopt_nvfp4_group16_cutlass_sm121_sfb",
            "local_experts": [0, 255],
            "expert_inventory_sha256": "e" * 64,
            "nonexpert_sha256": hashes,
            "nonexpert_contract_sha256": native.nonexpert_contract_sha256,
        }
        source = inspect_external_mtp_source(record, native, hashes)
        self.assertEqual(source.expert_abi, "modelopt_nvfp4_group16_cutlass_sm121_sfb")
        self.assertFalse(source.trained_head_distinct)
        self.assertEqual(source.nonexpert_differences, ())

        observed = dict(hashes)
        observed["mtp.fc_hidden.weight"] = "f" * 64
        with self.assertRaisesRegex(MtpSourceError, "29 nonexpert tensors"):
            inspect_external_mtp_source(record, native, observed)

    def test_external_source_cannot_claim_a_distinct_trained_head(self):
        native = inspect_native_mtp_source(native_manifest(), 0)
        hashes = {name: "c" * 64 for name in MTP_NONEXPERT_TENSORS}
        record = {
            "schema": EXTERNAL_MTP_SOURCE_SCHEMA,
            "source_id": "d" * 64,
            "base_revision": PINNED_CONTRACT.revision,
            "rank": 0,
            "expert_abi": "modelopt_nvfp4_group16_cutlass_sm121_sfb",
            "local_experts": [0, 255],
            "expert_inventory_sha256": "e" * 64,
            "nonexpert_sha256": hashes,
            "nonexpert_contract_sha256": native.nonexpert_contract_sha256,
            "trained_head_distinct": True,
        }
        with self.assertRaisesRegex(MtpSourceError, "trained-head"):
            inspect_external_mtp_source(record, native, hashes)


if __name__ == "__main__":
    unittest.main()

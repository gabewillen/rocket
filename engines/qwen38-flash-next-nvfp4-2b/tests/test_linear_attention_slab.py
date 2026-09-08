# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import math
import os
import struct
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from qwen38_slab.linear_attention import (
    GDN_COMPONENT_CONTRACTS,
    GDN_SCHEMA,
    GDN_STATE_FAMILIES,
    LinearAttentionSlabError,
    load_gdn_layer,
    validate_gdn_inventory,
)
from qwen38_slab.contract import PINNED_CONTRACT, SCHEMA
from qwen38_slab.state_txn import STATE_FAMILIES

REAL_SLAB = Path(os.environ.get("ROCKET_QWEN38_RANK_SLAB", "/nonexistent"))


class LinearAttentionSlabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        slabs = {}
        for rank in (0, 1):
            entries = []
            offset = 0
            for layer in range(48):
                if layer % 4 == 3:
                    continue
                prefix = f"model.language_model.layers.{layer}.linear_attn"
                for suffix, length, shape, dtype, layout, abi in GDN_COMPONENT_CONTRACTS:
                    entries.append({
                        "name": f"{prefix}.{suffix}", "offset_bytes": offset,
                        "length_bytes": length, "shape": list(shape),
                        "dtype": dtype, "layout": layout, "abi": abi,
                    })
                    offset += ((length + 255) // 256) * 256
            slabs[f"rank{rank}-target"] = {"entries": entries}
        cls.manifest = {
            "schema": SCHEMA,
            "revision": PINNED_CONTRACT.revision,
            "overlay_artifact_key": PINNED_CONTRACT.artifact_key,
            "overlay_sha256": PINNED_CONTRACT.overlay_sha256,
            "tp_size": 2,
            "tensor_alignment_bytes": PINNED_CONTRACT.tensor_alignment_bytes,
            "slabs": slabs,
        }

    def _entry(self, manifest, suffix, rank=0, layer=0):
        name = f"model.language_model.layers.{layer}.linear_attn.{suffix}"
        return next(
            x for x in manifest["slabs"][f"rank{rank}-target"]["entries"]
            if x["name"] == name
        )

    @unittest.skipUnless(REAL_SLAB.is_dir(), "production rank slab unavailable")
    def test_exact_production_descriptor_is_immutable(self):
        descriptor = load_gdn_layer(REAL_SLAB, 0, 0)
        self.assertEqual(descriptor.schema, GDN_SCHEMA)
        self.assertEqual((descriptor.rank, descriptor.layer), (0, 0))
        self.assertEqual(len(descriptor.components), 24)
        self.assertEqual(descriptor.components[0].offset, 1_297_735_680)
        self.assertEqual(descriptor.components[-1].offset, 1_314_102_272)
        with self.assertRaises(FrozenInstanceError):
            descriptor.slab_bytes = 0

    def test_wrong_a_log_dtype_fails_before_binding(self):
        manifest = copy.deepcopy(self.manifest)
        self._entry(manifest, "A_log")["dtype"] = "F32"
        with self.assertRaisesRegex(LinearAttentionSlabError, "A_log"):
            validate_gdn_inventory(manifest, 0, 0)

    def test_both_ranks_and_all_36_gdn_layers_bind(self):
        for rank in (0, 1):
            for layer in range(48):
                if layer % 4 != 3:
                    self.assertEqual(
                        len(validate_gdn_inventory(self.manifest, rank, layer)), 24
                    )

    def test_recurrent_families_remain_in_nine_family_transaction(self):
        self.assertEqual(GDN_STATE_FAMILIES, STATE_FAMILIES[3:5])

    def test_missing_family_fails_before_binding(self):
        manifest = copy.deepcopy(self.manifest)
        entries = manifest["slabs"]["rank0-target"]["entries"]
        entries.remove(self._entry(manifest, "conv1d.weight"))
        with self.assertRaisesRegex(LinearAttentionSlabError, "conv1d.weight"):
            validate_gdn_inventory(manifest, 0, 0)

    def test_wrong_offset_fails_before_binding(self):
        manifest = copy.deepcopy(self.manifest)
        self._entry(manifest, "dt_bias")["offset_bytes"] += 1
        with self.assertRaisesRegex(LinearAttentionSlabError, "dt_bias"):
            validate_gdn_inventory(manifest, 0, 0)

    def test_missing_projection_fails_before_binding(self):
        manifest = copy.deepcopy(self.manifest)
        entries = manifest["slabs"]["rank0-target"]["entries"]
        entries.remove(self._entry(manifest, "in_proj_qkv.weight"))
        with self.assertRaisesRegex(LinearAttentionSlabError, "in_proj_qkv.weight"):
            validate_gdn_inventory(manifest, 0, 0)

    def test_core_selects_owner_device_before_stream_tail_clear(self):
        source = (
            Path(__file__).parents[1] / "src" / "linear_attention" / "gdn_core.cu"
        ).read_text()
        launch = source[source.index("void CorePlan::launch(") :]
        self.assertLess(
            launch.index('cudaSetDevice(impl_->device)'),
            launch.index('"clear inactive GDN rows"'),
        )

    def test_recurrence_rounds_beta_to_bf16_like_pinned_vllm(self):
        source = (
            Path(__file__).parents[1] / "src" / "linear_attention" / "gdn_core.cu"
        ).read_text()
        helper = source[source.index("__device__ __forceinline__ float recurrent_beta") :]
        self.assertIn(
            "__bfloat162float(__float2bfloat16(sigmoid))",
            helper[:500],
        )
        self.assertEqual(source.count("recurrent_beta(ba["), 2)

    def test_causal_conv_matches_pinned_accumulation_and_silu(self):
        source = (
            Path(__file__).parents[1] / "src" / "linear_attention" / "gdn_core.cu"
        ).read_text()
        conv = source[source.index("__global__ void causal_conv_update") :]
        self.assertEqual(source.count("value = value / (1.0F + expf(-value));"), 2)
        self.assertEqual(
            source.count(
                "value += __bfloat162float(__float2bfloat16(product));"
            ),
            2,
        )
        self.assertNotIn(
            "value += inputs[index] * __bfloat162float(weight[w + index]);",
            conv,
        )
        self.assertNotIn("value *= 1.0F / (1.0F + __expf(-value))", conv)

        def bf16_to_float(bits: int) -> float:
            return struct.unpack("<f", struct.pack("<I", bits << 16))[0]

        def float_to_bf16(value: float) -> int:
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16

        x = bf16_to_float(0xBF31)  # -0.69140625, captured layer-0 row 0.
        weight = bf16_to_float(0xBC71)  # -0.01470947265625.
        full_product = x * weight
        rounded_product = bf16_to_float(float_to_bf16(full_product))
        full_result = float_to_bf16(
            full_product / (1.0 + math.exp(-full_product))
        )
        rounded_result = float_to_bf16(
            rounded_product / (1.0 + math.exp(-rounded_product))
        )
        self.assertEqual(full_result, 0x3BA7)
        self.assertEqual(rounded_result, 0x3BA8)

    def test_gdn_debug_capture_is_layer0_once_and_bounded(self):
        source = (
            Path(__file__).parents[1] / "src" / "linear_attention" / "gdn_cutlass.cu"
        ).read_text()
        body = source[source.index("void debug_dump(") : source.index("~Impl()")]
        self.assertIn("layer != 0", body)
        self.assertIn("debug_dumped[stage]", body)
        self.assertIn("bytes > kMaxDebugBytes", body)
        launch = source[source.index("void CutlassGdnGraph::launch(") :]
        self.assertLess(launch.index('"qkvz"'), launch.index('"ba"'))
        self.assertLess(launch.index('"ba"'), launch.index('"conv"'))
        self.assertLess(launch.index('"conv"'), launch.index('"recurrent"'))
        self.assertLess(launch.index('"recurrent"'), launch.index('"core"'))
        self.assertLess(launch.index('"core"'), launch.index('"projected"'))

    def test_decode_projection_uses_authenticated_input_scale_before_bf16(self):
        source = (
            Path(__file__).parents[1] / "src" / "linear_attention" / "gdn_cutlass.cu"
        ).read_text()
        launch = source[source.index("void CutlassGdnGraph::launch(") :]
        self.assertIn("gdn_quantizer_scale(impl_->input_global_scale)", launch)
        self.assertIn("gdn_quantizer_scale(impl_->output_global_scale)", launch)
        for family in ("qkv_gemms", "z_gemms", "b_gemms", "a_gemms"):
            self.assertIn(f"impl_->{family}[bucket].gemm.run(stream)", launch)
        before_verifier = launch[: launch.index("void CutlassGdnGraph::launch_verifier(")]
        self.assertNotIn("scale_projection<<<", before_verifier)

    def test_decode_owner_binds_every_authenticated_input_scale(self):
        source_root = Path(__file__).parents[1] / "src"
        owner = (source_root / "decode" / "target_gdn_layer_owner.cu").read_text()
        resolver = owner[owner.index("TargetGdnNativeWeightBindings resolve(") :]
        resolver = resolver[: resolver.index("void validate_publication(")]
        self.assertIn('std::string(root) + ".input_scale"', resolver)
        self.assertIn("address<float>(base, extent(", resolver)

        graph = (source_root / "linear_attention" / "gdn_cutlass.cu").read_text()
        constructor = graph[graph.index("CutlassGdnGraph::CutlassGdnGraph(") :]
        constructor = constructor[
            : constructor.index("CutlassGdnGraph::~CutlassGdnGraph()")
        ]
        self.assertIn("!matrix.input_scale", constructor)
        self.assertIn("matrices[index].input_scale", constructor)
        self.assertIn("input_scales[0] != input_scales[1]", constructor)
        self.assertIn("input_scales[0] != input_scales[2]", constructor)
        self.assertIn("input_scales[0] != input_scales[3]", constructor)
        self.assertIn("impl_->output_global_scale = input_scales[4]", constructor)

    def test_decode_quantizer_matches_pinned_vllm_nvfp4_rounding(self):
        source = (
            Path(__file__).parents[1] / "src" / "linear_attention" / "gdn_cutlass.cu"
        ).read_text()
        body = source[source.index("__global__ void quantize_fixed") :]
        body = body[: body.index("struct alignas(32) PackedBf16x16")]
        self.assertIn("float sf_scale)", body)
        self.assertNotIn("reciprocal_approximate_ftz(activation_global)", body)
        self.assertIn("amax * reciprocal_approximate_ftz(6.0F)", body)
        self.assertIn("reciprocal_approximate_ftz(sf_scale)", body)
        self.assertIn("pack_e2m1x16(converted)", body)
        self.assertNotIn("float_to_e2m1", body)


if __name__ == "__main__":
    unittest.main()

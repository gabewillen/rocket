#!/usr/bin/env python3
"""Focused CPU tests for the Qwen3.8 linear-attention FP8 materializer."""

import hashlib
import importlib.util
import json
import pathlib
import random
import struct
import subprocess
import tempfile
import time
import unittest


PATH = pathlib.Path(__file__).with_name("qwen38-materialize-linear-fp8.py")
SPEC = importlib.util.spec_from_file_location("materializer", PATH)
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)
REAL_SNAPSHOT = pathlib.Path(
    "/home/glwillen/.cache/huggingface/hub/"
    "models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
    + materializer.REVISION
)
REAL_TRACE = pathlib.Path(
    "/home/glwillen/calibration/qwen38-expanded-mtp3-20260907-03/"
    "combined-activation-summary.json"
)


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.pack("<H", bits >> 16)


def write_safetensors(path, tensors):
    header, payload, offset = {}, bytearray(), 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def fixture(root, missing=False):
    checkpoint = root / materializer.REVISION
    blobs = root / "blobs"
    checkpoint.mkdir()
    blobs.mkdir()
    tensors = {}
    names = []
    linear_layers = [layer for layer in range(48) if layer % 4 != 3]
    for layer in linear_layers:
        for projection in materializer.PROJECTIONS:
            name = f"model.language_model.layers.{layer}.linear_attn.{projection}.weight"
            if missing and layer == 46 and projection == "out_proj":
                continue
            names.append(name)
            tensors[name] = ("BF16", [2, 2], bf16(-2.0) + bf16(0.0) + bf16(1.0) + bf16(2.0))
    blob = blobs / "abc123"
    write_safetensors(blob, tensors)
    shard = checkpoint / "model-00001-of-00001.safetensors"
    shard.symlink_to(blob)
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: shard.name for name in names}}))
    telemetry = {}
    for layer in linear_layers:
        for projection in materializer.PROJECTIONS:
            telemetry[materializer.input_channel(layer, projection)] = {"absmax": 448.0}
    trace = root / "trace.json"
    trace.write_text(json.dumps({
        "schema": materializer.TRACE_SCHEMA,
        "gate": {"source": "v2_only"},
        "coverage": dict(materializer.EXPECTED_COVERAGE),
        "telemetry": telemetry,
    }, sort_keys=True))
    config = root / "hf_quant_config.json"
    config.write_text(json.dumps({"quantization": {
        "quant_algo": "MIXED_PRECISION",
        "exclude_modules": [*(f"model.language_model.layers.{layer}.linear_attn*" for layer in linear_layers), "lm_head"],
        "quantized_layers": {"existing": {"quant_algo": "NVFP4"}},
    }}))
    return checkpoint, trace, config


class MaterializerTest(unittest.TestCase):
    def test_fp8_saturation_ties_and_zero(self):
        self.assertEqual(materializer.fp8_encode(0.0), 0x00)
        self.assertEqual(materializer.fp8_encode(-0.0), 0x80)
        self.assertEqual(materializer.fp8_encode(448.0), 0x7E)
        self.assertEqual(materializer.fp8_encode(1000.0), 0x7E)
        self.assertEqual(materializer.fp8_encode(-1000.0), 0xFE)
        midpoint = (materializer.fp8_decode_positive(0x20) + materializer.fp8_decode_positive(0x21)) / 2
        self.assertEqual(materializer.fp8_encode(midpoint), 0x20)

    def test_host_materialization_fails_clearly_without_torch(self):
        if importlib.util.find_spec("torch") is None:
            with self.assertRaisesRegex(materializer.MaterializeError, "requires PyTorch"):
                materializer.load_torch()

    def test_plan_hashes_and_exact_family(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path.cwd()) as directory:
            checkpoint, trace, config = fixture(pathlib.Path(directory))
            plan = materializer.build_plan(checkpoint, trace, config)
            self.assertEqual(len(plan["tensors"]), 180)
            self.assertEqual(len(plan["shards"]), 1)
            shard = next(iter(plan["shards"].values()))
            self.assertEqual(set(shard), {"target", "size", "header_sha256"})
            first = plan["tensors"][0]
            self.assertEqual(first["sha256"], hashlib.sha256(bf16(-2.0) + bf16(0.0) + bf16(1.0) + bf16(2.0)).hexdigest())
            self.assertEqual(first["input_scale"], 1.0)
            self.assertLess(plan["bytes_hashed"], sum(item["source_bytes"] for item in plan["tensors"]) + 100_000)
            self.assertEqual(len(materializer.safetensors_header(plan["tensors"])), len(materializer.safetensors_header(plan["tensors"])))

    def test_vectorized_random_edges_and_chunk_boundaries_match_oracle(self):
        rng = random.Random(20260907)
        values = [
            0.0, -0.0, 2.0**-9, -(2.0**-9), 1.0, -1.0, 447.0, 448.0,
            449.0, 65504.0, -65504.0,
        ] + [rng.uniform(-1000.0, 1000.0) for _ in range(4096)]
        raw = b"".join(bf16(value) for value in values)
        scale = materializer.float32(997.0 / materializer.FP8_MAX)
        expected = materializer.scalar_quantize(raw, scale).hex()
        code = (
            "import importlib.util,json,pathlib;"
            f"p=pathlib.Path({str(PATH)!r});"
            "s=importlib.util.spec_from_file_location('m',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "t=m.load_torch();"
            f"raw=bytes.fromhex({raw.hex()!r});scale={scale!r};"
            "whole=m.vectorized_quantize(raw,scale,t);"
            "chunked=b''.join(m.vectorized_quantize(raw[i:i+14],scale,t) for i in range(0,len(raw),14));"
            "zero=m.vectorized_quantize(bytes(34),0.0,t);"
            "benchmark=m.benchmark_vectorized(32);"
            "print(whole.hex());print(chunked.hex());print(zero.hex());print(json.dumps(benchmark))"
        )
        output = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{pathlib.Path.cwd()}:{pathlib.Path.cwd()}:ro", "-w", str(pathlib.Path.cwd()),
             "--entrypoint", "python3", "vllm/vllm-openai:qwen38-flash-next", "-c", code],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()[-4:]
        self.assertEqual(output[:3], [expected, expected, bytes(17).hex()])
        benchmark = json.loads(output[3])
        self.assertEqual(benchmark["bf16_input_bytes"], 32 * 2**20)
        self.assertEqual(benchmark["fp8_output_bytes"], 16 * 2**20)
        self.assertGreater(benchmark["input_gb_per_second"], 0.1)

    def test_180_family_gate_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path.cwd()) as directory:
            checkpoint, trace, config = fixture(pathlib.Path(directory), missing=True)
            with self.assertRaisesRegex(materializer.MaterializeError, "179/180"):
                materializer.build_plan(checkpoint, trace, config)

    def test_interrupted_output_refusal(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path.cwd()) as directory:
            output = pathlib.Path(directory)
            staging = output / f".{materializer.REVISION}.linear-fp8.building"
            staging.mkdir()
            with self.assertRaisesRegex(materializer.MaterializeError, "interrupted output"):
                materializer.materialize({"tensors": []}, output)

    def test_real_plan_hashes_selected_ranges_not_whole_checkpoint(self):
        started = time.monotonic()
        plan = materializer.build_plan(
            REAL_SNAPSHOT, REAL_TRACE, REAL_SNAPSHOT / "hf_quant_config.json"
        )
        elapsed = time.monotonic() - started
        source_bytes = sum(item["source_bytes"] for item in plan["tensors"])
        checkpoint_bytes = sum(item["size"] for item in plan["shards"].values())
        self.assertEqual(source_bytes, 4_170_055_680)
        self.assertLessEqual(plan["bytes_hashed"], source_bytes + 64 * 2**20)
        self.assertLess(plan["bytes_hashed"], checkpoint_bytes // 10)
        self.assertLess(elapsed, 60.0)


if __name__ == "__main__":
    unittest.main()

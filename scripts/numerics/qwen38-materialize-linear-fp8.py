#!/usr/bin/env python3
"""Materialize the pinned Qwen3.8 linear-attention ModelOpt FP8 overlay."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path


REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
TRACE_SCHEMA = "rocket.qwen38.activation-summary.v2"
MANIFEST_SCHEMA = "rocket.qwen38.linear-fp8-overlay.v1"
FP8_MAX = 448.0
HEADER_LIMIT = 64 * 2**20
CHUNK_ELEMENTS = 4 * 2**20
PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
SELECTED = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.linear_attn\."
    r"(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)\.weight$"
)
EXPECTED_COVERAGE = {
    "linear_attention_layers": 36,
    "linear_projection_input_channels": 108,
    "linear_projection_output_channels": 108,
    "recurrent_state_layers": 36,
}


class MaterializeError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 2**20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def quant_config_module():
    path = Path(__file__).parents[1] / "runtime" / "patch-qwen38-fp8-quant-config.py"
    spec = importlib.util.spec_from_file_location("qwen38_fp8_quant_config", path)
    if spec is None or spec.loader is None:
        raise MaterializeError(f"cannot load accepted quant-config patcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def input_channel(layer: int, projection: str) -> str:
    packed = {
        "in_proj_qkv": "in_proj_qkvz",
        "in_proj_z": "in_proj_qkvz",
        "in_proj_a": "in_proj_ba",
        "in_proj_b": "in_proj_ba",
        "out_proj": "out_proj",
    }[projection]
    return f"layer.{layer}.linear_attn.{packed}.input"


def read_checkpoint(checkpoint: Path) -> tuple[dict, dict, int]:
    if checkpoint.absolute().name != REVISION:
        raise MaterializeError(f"checkpoint lexical directory must be revision {REVISION}")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise MaterializeError("checkpoint index needs a non-empty weight_map")
    shards, tensors, bytes_hashed = {}, {}, 0
    for filename in sorted(set(weight_map.values())):
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise MaterializeError("unsafe checkpoint shard path")
        lexical = checkpoint.absolute() / filename
        if not lexical.is_symlink():
            raise MaterializeError(f"checkpoint shard must be a snapshot symlink: {filename}")
        target = lexical.resolve(strict=True)
        with target.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise MaterializeError(f"truncated safetensors prefix: {filename}")
            header_size = int.from_bytes(prefix, "little")
            if not 0 < header_size <= HEADER_LIMIT:
                raise MaterializeError(f"invalid safetensors header size: {filename}")
            raw = stream.read(header_size)
        if len(raw) != header_size:
            raise MaterializeError(f"truncated safetensors header: {filename}")
        header_digest_input = prefix + raw
        shards[filename] = {
            "target": target.name,
            "size": target.stat().st_size,
            "header_sha256": hashlib.sha256(header_digest_input).hexdigest(),
        }
        bytes_hashed += len(header_digest_input)
        header = json.loads(raw)
        header.pop("__metadata__", None)
        for name, meta in header.items():
            if name in tensors or not isinstance(meta, dict):
                raise MaterializeError(f"duplicate or invalid tensor: {name}")
            tensors[name] = (target, 8 + header_size, meta)
    if set(tensors) != set(weight_map):
        raise MaterializeError("checkpoint index/header tensor mismatch")
    return shards, tensors, bytes_hashed


def range_sha256(path: Path, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            chunk = stream.read(min(8 * 2**20, remaining))
            if not chunk:
                raise MaterializeError(f"short source tensor range: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def validate_trace(path: Path) -> tuple[dict, str]:
    trace = json.loads(path.read_text())
    if trace.get("schema") != TRACE_SCHEMA:
        raise MaterializeError(f"trace schema must be {TRACE_SCHEMA}")
    if trace.get("gate", {}).get("source") != "v2_only":
        raise MaterializeError("trace must be gated from v2_only telemetry")
    coverage = trace.get("coverage")
    telemetry = trace.get("telemetry")
    if not isinstance(coverage, dict) or not isinstance(telemetry, dict):
        raise MaterializeError("trace coverage or telemetry missing")
    failures = [
        f"{key}={coverage.get(key)!r}/{expected}"
        for key, expected in EXPECTED_COVERAGE.items()
        if coverage.get(key) != expected
    ]
    if failures:
        raise MaterializeError("incomplete trace coverage: " + ", ".join(failures))
    return trace, sha256_file(path)


def build_plan(checkpoint: Path, trace_path: Path, source_quant_config: Path) -> dict:
    shards, headers, bytes_hashed = read_checkpoint(checkpoint)
    trace, trace_hash = validate_trace(trace_path)
    telemetry = trace["telemetry"]
    selected = []
    for name, (path, data_start, meta) in headers.items():
        match = SELECTED.fullmatch(name)
        if match is None:
            continue
        dtype, shape, offsets = meta.get("dtype"), meta.get("shape"), meta.get("data_offsets")
        if dtype != "BF16" or not isinstance(shape, list) or len(shape) != 2:
            raise MaterializeError(f"selected tensor dtype/shape drift: {name}")
        if not all(isinstance(value, int) and value > 0 for value in shape):
            raise MaterializeError(f"selected tensor shape drift: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise MaterializeError(f"selected tensor offsets missing: {name}")
        size = offsets[1] - offsets[0]
        if size != math.prod(shape) * 2:
            raise MaterializeError(f"selected tensor byte/shape drift: {name}")
        layer, projection = int(match.group(1)), match.group(2)
        channel = input_channel(layer, projection)
        record = telemetry.get(channel)
        activation_amax = record.get("absmax") if isinstance(record, dict) else None
        if not isinstance(activation_amax, (int, float)) or not math.isfinite(activation_amax) or activation_amax < 0:
            raise MaterializeError(f"missing or invalid calibrated activation: {channel}")
        absolute = data_start + offsets[0]
        selected.append({
            "name": name,
            "layer": layer,
            "projection": projection,
            "shape": shape,
            "source_bytes": size,
            "source_path": str(path),
            "source_offset": absolute,
            "sha256": range_sha256(path, absolute, size),
            "input_scale": activation_amax / FP8_MAX,
        })
        bytes_hashed += size
    selected.sort(key=lambda item: (item["layer"], PROJECTIONS.index(item["projection"])))
    layer_projections = {}
    for item in selected:
        layer_projections.setdefault(item["layer"], set()).add(item["projection"])
    if len(selected) != 180 or len(layer_projections) != 36 or any(
        projections != set(PROJECTIONS) for projections in layer_projections.values()
    ):
        raise MaterializeError(
            f"selected family coverage is {len(layer_projections)}/36 layers, {len(selected)}/180 matrices"
        )
    config_patch = quant_config_module()
    quant_config = config_patch.patched(json.loads(source_quant_config.read_text()))
    quant_bytes = config_patch.canonical_bytes(quant_config)
    quant_hash = hashlib.sha256(quant_bytes).hexdigest()
    bytes_hashed += trace_path.stat().st_size + len(quant_bytes)
    payload_bytes = sum(item["source_bytes"] // 2 + 8 for item in selected)
    return {
        "checkpoint": str(checkpoint.absolute()),
        "revision": REVISION,
        "trace_sha256": trace_hash,
        "shards": shards,
        "tensors": selected,
        "quant_config": quant_config,
        "quant_config_bytes": quant_bytes,
        "quant_config_sha256": quant_hash,
        "payload_bytes": payload_bytes,
        "bytes_hashed": bytes_hashed,
    }


def bf16_values(raw: bytes):
    if len(raw) % 2:
        raise MaterializeError("odd BF16 byte count")
    for (bits,) in struct.iter_unpack("<H", raw):
        yield struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fp8_decode_positive(code: int) -> float:
    exponent, mantissa = (code >> 3) & 0xF, code & 7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


FP8_POSITIVE = [(fp8_decode_positive(code), code) for code in range(0x7F)]
FP8_VALUES = [value for value, _ in FP8_POSITIVE]


def fp8_encode(value: float) -> int:
    """Scalar E4M3FN oracle. Production conversion uses PyTorch."""
    if math.isnan(value):
        raise MaterializeError("NaN cannot be quantized")
    negative = math.copysign(1.0, value) < 0
    magnitude = min(abs(value), FP8_MAX)
    index = bisect.bisect_left(FP8_VALUES, magnitude)
    if index == 0:
        code = 0
    elif index == len(FP8_VALUES):
        code = FP8_POSITIVE[-1][1]
    else:
        low_value, low_code = FP8_POSITIVE[index - 1]
        high_value, high_code = FP8_POSITIVE[index]
        low_delta, high_delta = magnitude - low_value, high_value - magnitude
        if low_delta < high_delta:
            code = low_code
        elif high_delta < low_delta:
            code = high_code
        else:
            code = low_code if low_code % 2 == 0 else high_code
    return code | (0x80 if negative else 0)


def load_torch():
    try:
        import torch
    except ImportError as error:
        raise MaterializeError(
            "materialization requires PyTorch with torch.float8_e4m3fn; "
            "run this script in vllm/vllm-openai:qwen38-flash-next"
        ) from error
    if not hasattr(torch, "float8_e4m3fn"):
        raise MaterializeError("PyTorch lacks torch.float8_e4m3fn support")
    try:
        torch.zeros(1, dtype=torch.float32).to(torch.float8_e4m3fn)
    except Exception as error:
        raise MaterializeError("PyTorch float8_e4m3fn conversion is unavailable") from error
    return torch


def torch_bf16(raw: bytes, torch):
    if len(raw) % 2:
        raise MaterializeError("odd BF16 byte count")
    # bytearray owns writable storage; clone detaches tensor lifetime from it.
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()


def vectorized_quantize(raw: bytes, scale: float, torch) -> bytes:
    values = torch_bf16(raw, torch).to(torch.float32)
    if not bool(torch.isfinite(values).all().item()):
        raise MaterializeError("non-finite BF16 input")
    if scale == 0.0:
        if bool((values != 0).any().item()):
            raise MaterializeError("zero scale with nonzero weights")
        return bytes(values.numel())
    encoded = (
        values.div(torch.tensor(scale, dtype=torch.float32))
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    return encoded.numpy().tobytes()


def scalar_quantize(raw: bytes, scale: float) -> bytes:
    """Test oracle matching float32 division before E4M3FN rounding."""
    if scale == 0.0:
        values = list(bf16_values(raw))
        if any(value != 0 for value in values):
            raise MaterializeError("zero scale with nonzero weights")
        return bytes(len(values))
    return bytes(fp8_encode(float32(value / scale)) for value in bf16_values(raw))


def tensor_absmax(item: dict, torch, chunk_elements: int = CHUNK_ELEMENTS) -> float:
    maximum = 0.0
    with Path(item["source_path"]).open("rb") as stream:
        stream.seek(item["source_offset"])
        remaining = item["source_bytes"]
        while remaining:
            raw = stream.read(min(chunk_elements * 2, remaining))
            if not raw:
                raise MaterializeError(f"short source tensor: {item['name']}")
            values = torch_bf16(raw, torch).to(torch.float32)
            if not bool(torch.isfinite(values).all().item()):
                raise MaterializeError(f"non-finite source tensor: {item['name']}")
            maximum = max(maximum, float(values.abs().max().item()))
            remaining -= len(raw)
    return maximum


def safetensors_header(items: list[dict]) -> bytes:
    header, offset = {}, 0
    for item in items:
        name, shape = item["name"], item["shape"]
        entries = (
            (name, "F8_E4M3", shape, math.prod(shape)),
            (name.removesuffix(".weight") + ".weight_scale", "F32", [1], 4),
            (name.removesuffix(".weight") + ".input_scale", "F32", [1], 4),
        )
        for tensor_name, dtype, tensor_shape, size in entries:
            if tensor_name in header:
                raise MaterializeError(f"duplicate output tensor: {tensor_name}")
            header[tensor_name] = {"dtype": dtype, "shape": tensor_shape, "data_offsets": [offset, offset + size]}
            offset += size
    if len(header) != 540:
        raise MaterializeError(f"overlay tensor count is {len(header)}/540")
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return raw + b" " * ((-len(raw)) % 8)


def write_overlay(path: Path, items: list[dict], chunk_elements: int = CHUNK_ELEMENTS) -> None:
    torch = load_torch()
    header = safetensors_header(items)
    with path.open("xb") as output:
        output.write(len(header).to_bytes(8, "little"))
        output.write(header)
        for item in items:
            weight_amax = tensor_absmax(item, torch, chunk_elements)
            scale = float32(weight_amax / FP8_MAX)
            with Path(item["source_path"]).open("rb") as source:
                source.seek(item["source_offset"])
                remaining = item["source_bytes"]
                while remaining:
                    raw = source.read(min(chunk_elements * 2, remaining))
                    if not raw:
                        raise MaterializeError(f"short source tensor: {item['name']}")
                    output.write(vectorized_quantize(raw, scale, torch))
                    remaining -= len(raw)
            output.write(struct.pack("<f", scale))
            output.write(struct.pack("<f", float32(item["input_scale"])))


def estimated_overlay_bytes(items: list[dict]) -> int:
    return 8 + len(safetensors_header(items)) + sum(item["source_bytes"] // 2 + 8 for item in items)


def materialize(plan: dict, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{REVISION}.linear-fp8.building"
    if staging.exists():
        raise MaterializeError(f"refusing interrupted output: {staging}")
    staging.mkdir()
    try:
        overlay_path = staging / "linear-attention-fp8.safetensors"
        write_overlay(overlay_path, plan["tensors"])
        overlay_hash = sha256_file(overlay_path)
        quant_path = staging / "hf_quant_config.json"
        with quant_path.open("xb") as stream:
            stream.write(plan["quant_config_bytes"])
        source = {
            "revision": REVISION,
            "trace_sha256": plan["trace_sha256"],
            "shards": plan["shards"],
            "tensors": [
                {"name": item["name"], "shape": item["shape"], "sha256": item["sha256"]}
                for item in plan["tensors"]
            ],
        }
        overlay = {"file": overlay_path.name, "sha256": overlay_hash}
        quant_contract = {"file": quant_path.name, "sha256": plan["quant_config_sha256"]}
        key = canonical_hash({"source": source, "overlay": overlay, "quant_config": quant_contract})
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "artifact_key": key,
            "source": source,
            "overlay": overlay,
            "quant_config": quant_contract,
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        destination = output_root / key
        if destination.exists():
            raise MaterializeError(f"refusing existing immutable artifact: {destination}")
        staging.rename(destination)
        return destination
    except Exception:
        raise


def benchmark_vectorized(size_mib: int) -> dict:
    torch = load_torch()
    raw = (struct.pack("<4H", 0x3F80, 0xBF80, 0x4000, 0x0000) * (size_mib * 2**20 // 8))
    scale = float32(2.0 / FP8_MAX)
    vectorized_quantize(raw[:4096], scale, torch)
    started = time.perf_counter()
    encoded = vectorized_quantize(raw, scale, torch)
    elapsed = time.perf_counter() - started
    return {
        "bf16_input_bytes": len(raw),
        "fp8_output_bytes": len(encoded),
        "elapsed_seconds": elapsed,
        "input_gb_per_second": len(raw) / elapsed / 1e9,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--source-quant-config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--benchmark-vectorized-mib", type=int)
    args = parser.parse_args()
    try:
        if args.benchmark_vectorized_mib is not None:
            if args.benchmark_vectorized_mib <= 0:
                raise MaterializeError("benchmark size must be positive")
            print(json.dumps(benchmark_vectorized(args.benchmark_vectorized_mib), indent=2, sort_keys=True))
            return 0
        if None in (args.checkpoint, args.trace, args.source_quant_config):
            raise MaterializeError("--checkpoint, --trace, and --source-quant-config are required")
        if args.plan_only == (args.output_root is not None):
            raise MaterializeError("choose exactly one of --plan-only or --output-root")
        plan = build_plan(args.checkpoint, args.trace, args.source_quant_config)
        summary = {
            "revision": plan["revision"],
            "trace_sha256": plan["trace_sha256"],
            "source_tensors": len(plan["tensors"]),
            "output_tensors": len(plan["tensors"]) * 3,
            "source_bytes": sum(item["source_bytes"] for item in plan["tensors"]),
            "estimated_overlay_bytes": estimated_overlay_bytes(plan["tensors"]),
            "quant_config_sha256": plan["quant_config_sha256"],
            "bytes_hashed": plan["bytes_hashed"],
        }
        if args.plan_only:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            destination = materialize(plan, args.output_root)
            summary["artifact"] = str(destination)
            print(json.dumps(summary, indent=2, sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

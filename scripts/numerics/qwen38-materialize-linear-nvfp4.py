#!/usr/bin/env python3
"""Materialize the selected Qwen linear-attention family as ModelOpt NVFP4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path


MANIFEST_SCHEMA = "rocket.qwen38.linear-nvfp4-overlay.v1"
NVFP4_DENOMINATOR = 6.0 * 448.0


class MaterializeError(ValueError):
    pass


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MaterializeError(f"cannot load dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parents[1]
FP8 = load_module(Path(__file__).with_name("qwen38-materialize-linear-fp8.py"), "qwen38_fp8_base")
CONFIG = load_module(ROOT / "runtime" / "patch-qwen38-nvfp4-quant-config.py", "qwen38_nvfp4_config")
REVISION = FP8.REVISION


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_plan(checkpoint: Path, trace_path: Path, source_quant_config: Path) -> dict:
    plan = FP8.build_plan(checkpoint, trace_path, source_quant_config)
    config = CONFIG.patched(json.loads(source_quant_config.read_text()))
    config_bytes = CONFIG.canonical_bytes(config)
    for item in plan["tensors"]:
        n, k = item["shape"]
        if k % 16:
            raise MaterializeError(f"selected tensor K is not divisible by 16: {item['name']}")
        item["input_scale"] = FP8.float32(item["input_scale"] / 6.0)
        item["encoded_bytes"] = n * (k // 2 + k // 16) + 8
    plan["quant_config"] = config
    plan["quant_config_bytes"] = config_bytes
    plan["quant_config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    plan["payload_bytes"] = sum(item["encoded_bytes"] for item in plan["tensors"])
    return plan


def output_entries(item: dict):
    name, (n, k) = item["name"], item["shape"]
    prefix = name.removesuffix(".weight")
    return (
        (name, "U8", [n, k // 2], n * k // 2),
        (prefix + ".weight_scale", "F8_E4M3", [n, k // 16], n * k // 16),
        (prefix + ".weight_scale_2", "F32", [1], 4),
        (prefix + ".input_scale", "F32", [1], 4),
    )


def safetensors_header(items: list[dict]) -> bytes:
    header, offset = {}, 0
    for item in items:
        for name, dtype, shape, size in output_entries(item):
            if name in header:
                raise MaterializeError(f"duplicate output tensor: {name}")
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
            offset += size
    if len(header) != 720:
        raise MaterializeError(f"overlay tensor count is {len(header)}/720")
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return raw + b" " * ((-len(raw)) % 8)


def load_runtime():
    try:
        import torch
        from vllm._custom_ops import scaled_fp4_quant
    except ImportError as error:
        raise MaterializeError("run materialization in the pinned Qwen vLLM image") from error
    if not torch.cuda.is_available():
        raise MaterializeError("NVFP4 materialization requires a CUDA device")
    return torch, scaled_fp4_quant


def quantize_item(item: dict, torch, scaled_fp4_quant):
    with Path(item["source_path"]).open("rb") as source:
        source.seek(item["source_offset"])
        raw = source.read(item["source_bytes"])
    if len(raw) != item["source_bytes"]:
        raise MaterializeError(f"short source tensor: {item['name']}")
    n, k = item["shape"]
    cpu = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(n, k)
    if not bool(torch.isfinite(cpu).all().item()):
        raise MaterializeError(f"non-finite source tensor: {item['name']}")
    weight = cpu.cuda()
    amax = weight.abs().max().to(torch.float32).clamp_min(1e-8)
    inverse_global_scale = torch.tensor(NVFP4_DENOMINATOR, device=weight.device) / amax
    packed, block_scale = scaled_fp4_quant(
        weight, inverse_global_scale, is_sf_swizzled_layout=False
    )
    global_scale = (1.0 / inverse_global_scale).to(torch.float32)
    return (
        packed.view(torch.uint8).cpu().contiguous().numpy().tobytes(),
        block_scale.view(torch.uint8).cpu().contiguous().numpy().tobytes(),
        FP8.float32(float(global_scale.item())),
    )


def write_overlay(path: Path, items: list[dict]) -> None:
    torch, quantizer = load_runtime()
    header = safetensors_header(items)
    with path.open("xb") as output:
        output.write(len(header).to_bytes(8, "little"))
        output.write(header)
        for item in items:
            packed, scales, global_scale = quantize_item(item, torch, quantizer)
            expected = list(output_entries(item))
            if len(packed) != expected[0][3] or len(scales) != expected[1][3]:
                raise MaterializeError(f"quantizer shape/ABI drift: {item['name']}")
            output.write(packed)
            output.write(scales)
            output.write(struct.pack("<f", global_scale))
            output.write(struct.pack("<f", item["input_scale"]))


def estimated_overlay_bytes(items: list[dict]) -> int:
    return 8 + len(safetensors_header(items)) + sum(item["encoded_bytes"] for item in items)


def materialize(plan: dict, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{REVISION}.linear-nvfp4.building"
    if staging.exists():
        raise MaterializeError(f"refusing interrupted output: {staging}")
    staging.mkdir()
    try:
        overlay_path = staging / "linear-attention-nvfp4.safetensors"
        write_overlay(overlay_path, plan["tensors"])
        overlay_hash = FP8.sha256_file(overlay_path)
        config_path = staging / "hf_quant_config.json"
        with config_path.open("xb") as stream:
            stream.write(plan["quant_config_bytes"])
        source = {
            "revision": REVISION,
            "trace_sha256": plan["trace_sha256"],
            "shards": plan["shards"],
            "tensors": [{"name": x["name"], "shape": x["shape"], "sha256": x["sha256"]} for x in plan["tensors"]],
        }
        overlay = {"file": overlay_path.name, "sha256": overlay_hash}
        quant = {"file": config_path.name, "sha256": plan["quant_config_sha256"]}
        key = canonical_hash({"source": source, "overlay": overlay, "quant_config": quant})
        manifest = {"schema": MANIFEST_SCHEMA, "artifact_key": key, "source": source, "overlay": overlay, "quant_config": quant}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        destination = output_root / key
        if destination.exists():
            raise MaterializeError(f"refusing existing immutable artifact: {destination}")
        staging.rename(destination)
        return destination
    except Exception:
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--source-quant-config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.plan_only == (args.output_root is not None):
            raise MaterializeError("choose exactly one of --plan-only or --output-root")
        plan = build_plan(args.checkpoint, args.trace, args.source_quant_config)
        summary = {
            "revision": REVISION,
            "source_tensors": len(plan["tensors"]),
            "output_tensors": 4 * len(plan["tensors"]),
            "source_bytes": sum(x["source_bytes"] for x in plan["tensors"]),
            "estimated_overlay_bytes": estimated_overlay_bytes(plan["tensors"]),
            "bytes_removed_per_c16_step": sum(x["source_bytes"] - x["encoded_bytes"] for x in plan["tensors"]),
            "quant_config_sha256": plan["quant_config_sha256"],
            "bytes_hashed": plan["bytes_hashed"],
        }
        if args.plan_only:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            summary["artifact"] = str(materialize(plan, args.output_root))
            print(json.dumps(summary, indent=2, sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

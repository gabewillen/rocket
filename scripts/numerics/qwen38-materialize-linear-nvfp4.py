#!/usr/bin/env python3
"""Materialize complete Qwen attention families as immutable ModelOpt NVFP4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import struct
import sys
from pathlib import Path


LINEAR_MANIFEST_SCHEMA = "rocket.qwen38.linear-nvfp4-overlay.v1"
MANIFEST_SCHEMA = "rocket.qwen38.nvfp4-overlay.v2"
NVFP4_DENOMINATOR = 6.0 * 448.0
FAMILY_SPECS = {
    "linear_attention": {
        "module": "linear_attn",
        "layers": set(range(48)) - set(range(3, 48, 4)),
        "projections": ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"),
    },
    "full_attention": {
        "module": "self_attn",
        "layers": set(range(3, 48, 4)),
        "projections": ("q_proj", "k_proj", "v_proj", "o_proj"),
    },
    "base_routers": {
        "module": "mlp",
        "layers": set(range(48)),
        "projections": ("gate",),
    },
    "base_ple": {
        "module": "ple",
        "layers": {1},
        "projections": ("key_proj", "value_proj"),
    },
}


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


def input_channel(family: str, layer: int, projection: str) -> str:
    """Map a source projection to its accepted v2 activation-input channel."""
    if family == "linear_attention":
        return FP8.input_channel(layer, projection)
    if family == "base_routers":
        return f"layer.{layer}.router.gate.input"
    if family == "base_ple":
        return f"layer.{layer}.ple.embedding.output"
    packed = "o_proj" if projection == "o_proj" else "qkv_proj"
    return f"layer.{layer}.full_attn.{packed}.input"


def selected_family(name: str, families: tuple[str, ...]):
    for family in families:
        spec = FAMILY_SPECS[family]
        prefix = r"^model\.language_model\.layers\.(\d+)\." + spec["module"] + r"\."
        projections = "|".join(spec["projections"])
        match = re.fullmatch(prefix + rf"({projections})\.weight$", name)
        if match:
            return family, int(match.group(1)), match.group(2)
    return None


def build_plan(
    checkpoint: Path,
    trace_path: Path,
    source_quant_config: Path,
    families=("linear_attention",),
) -> dict:
    """Build a bounded, read-only plan; all family invariants hold on return."""
    families = CONFIG.normalize_families(families)
    shards, headers, bytes_hashed = FP8.read_checkpoint(checkpoint)
    trace, trace_hash = FP8.validate_trace(trace_path)
    coverage = trace["coverage"]
    if "full_attention" in families:
        expected = {
            "full_attention_layers": 12,
            "full_qkv_projection_layers": 12,
            "full_output_projection_layers": 12,
        }
        failures = [
            f"{key}={coverage.get(key)!r}/{value}"
            for key, value in expected.items()
            if coverage.get(key) != value
        ]
        if failures:
            raise MaterializeError("incomplete full-attention trace coverage: " + ", ".join(failures))
    auxiliary_coverage = {
        "base_routers": ("router_layers", 48),
        "base_ple": ("ple_layers", 1),
    }
    for family in families:
        contract = auxiliary_coverage.get(family)
        if contract is not None and coverage.get(contract[0]) != contract[1]:
            raise MaterializeError(
                f"incomplete {family} trace coverage: "
                f"{contract[0]}={coverage.get(contract[0])!r}/{contract[1]}"
            )
    tensors = []
    for name, (path, data_start, meta) in headers.items():
        selected = selected_family(name, families)
        if selected is None:
            continue
        family, layer, projection = selected
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
        channel = input_channel(family, layer, projection)
        record = trace["telemetry"].get(channel)
        activation_amax = record.get("absmax") if isinstance(record, dict) else None
        if (
            not isinstance(activation_amax, (int, float))
            or not math.isfinite(activation_amax)
            or activation_amax < 0
        ):
            raise MaterializeError(f"missing or invalid calibrated activation: {channel}")
        absolute = data_start + offsets[0]
        n, k = shape
        if k % 16:
            raise MaterializeError(f"selected tensor K is not divisible by 16: {name}")
        source_hash = FP8.range_sha256(path, absolute, size)
        bytes_hashed += size
        tensors.append(
            {
                "name": name,
                "family": family,
                "layer": layer,
                "projection": projection,
                "shape": shape,
                "source_bytes": size,
                "source_path": str(path),
                "source_offset": absolute,
                "sha256": source_hash,
                "input_scale": FP8.float32(activation_amax / NVFP4_DENOMINATOR),
                "encoded_bytes": n * (k // 2 + k // 16) + 8,
            }
        )
    family_order = {family: index for index, family in enumerate(families)}
    tensors.sort(
        key=lambda item: (
            family_order[item["family"]],
            item["layer"],
            FAMILY_SPECS[item["family"]]["projections"].index(item["projection"]),
        )
    )
    for family in families:
        spec = FAMILY_SPECS[family]
        selected = [item for item in tensors if item["family"] == family]
        by_layer = {}
        for item in selected:
            by_layer.setdefault(item["layer"], set()).add(item["projection"])
        expected_count = len(spec["layers"]) * len(spec["projections"])
        if (
            set(by_layer) != spec["layers"]
            or len(selected) != expected_count
            or any(value != set(spec["projections"]) for value in by_layer.values())
        ):
            raise MaterializeError(
                f"{family} coverage is {len(by_layer)}/{len(spec['layers'])} layers, "
                f"{len(selected)}/{expected_count} matrices"
            )
    config = CONFIG.patched(json.loads(source_quant_config.read_text()), families)
    config_bytes = CONFIG.canonical_bytes(config)
    bytes_hashed += trace_path.stat().st_size + len(config_bytes)
    return {
        "checkpoint": str(checkpoint.absolute()),
        "revision": REVISION,
        "families": families,
        "trace_sha256": trace_hash,
        "shards": shards,
        "tensors": tensors,
        "quant_config": config,
        "quant_config_bytes": config_bytes,
        "quant_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "payload_bytes": sum(item["encoded_bytes"] for item in tensors),
        "bytes_hashed": bytes_hashed,
    }


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
    expected = 4 * len(items)
    if not items or len(header) != expected:
        raise MaterializeError(f"overlay tensor count is {len(header)}/{expected}")
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
    families = tuple(plan["families"])
    slug = "linear-nvfp4" if families == ("linear_attention",) else "-".join(families) + "-nvfp4"
    staging = output_root / f".{REVISION}.{slug}.building"
    if staging.exists():
        raise MaterializeError(f"refusing interrupted output: {staging}")
    staging.mkdir()
    try:
        overlay_name = "linear-attention-nvfp4.safetensors" if families == ("linear_attention",) else "attention-nvfp4.safetensors"
        overlay_path = staging / overlay_name
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
        schema = LINEAR_MANIFEST_SCHEMA
        if families != ("linear_attention",):
            schema = MANIFEST_SCHEMA
            source["families"] = list(families)
        overlay = {"file": overlay_path.name, "sha256": overlay_hash}
        quant = {"file": config_path.name, "sha256": plan["quant_config_sha256"]}
        key = canonical_hash({"source": source, "overlay": overlay, "quant_config": quant})
        manifest = {"schema": schema, "artifact_key": key, "source": source, "overlay": overlay, "quant_config": quant}
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
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(FAMILY_SPECS),
        default=None,
        help="attention family to materialize; repeat for a combined immutable overlay",
    )
    args = parser.parse_args()
    try:
        if args.plan_only == (args.output_root is not None):
            raise MaterializeError("choose exactly one of --plan-only or --output-root")
        plan = build_plan(
            args.checkpoint,
            args.trace,
            args.source_quant_config,
            args.family or ("linear_attention",),
        )
        summary = {
            "revision": REVISION,
            "families": list(plan["families"]),
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

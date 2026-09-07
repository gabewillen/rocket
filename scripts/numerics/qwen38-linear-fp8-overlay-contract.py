#!/usr/bin/env python3
"""Prove the Qwen3.8 linear-attention FP8 artifact and loader contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Mapping


SCHEMA = "rocket.qwen38.linear-fp8-overlay-contract.v1"
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
TRACE_SCHEMA = "rocket.qwen38.activation-summary.v2"
IMAGE_ID = "sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a"
FP8_MAX = 448.0
HEADER_LIMIT = 64 * 2**20
SELECTED = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.linear_attn\."
    r"(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)\.weight$"
)
PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
EXPECTED_COVERAGE = {
    "linear_attention_layers": 36,
    "linear_projection_input_channels": 108,
    "linear_projection_output_channels": 108,
    "recurrent_state_layers": 36,
}
DENSE_FP8_ABI_MARKERS = (
    'if quant_algo == "FP8":',
    "ModelOptFp8LinearMethod(self.fp8_config)",
    "torch.float8_e4m3fn",
    "PerTensorScaleParameter",
    'layer.register_parameter("weight_scale", weight_scale)',
    'layer.register_parameter("input_scale", scale)',
    "kFp8StaticTensorSym",
)
OVERLAY_ABI_MARKER = "ROCKET_QWEN38_FP8_OVERLAY_V1"


class ContractError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 2**20):
            digest.update(chunk)
    return digest.hexdigest()


def load_headers(root: Path) -> dict[str, dict[str, object]]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ContractError("checkpoint index needs a non-empty weight_map")
    headers: dict[str, dict[str, object]] = {}
    for filename in sorted(set(weight_map.values())):
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ContractError("unsafe checkpoint shard path")
        path = root / filename
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ContractError(f"truncated header prefix: {filename}")
            size = struct.unpack("<Q", prefix)[0]
            if not 0 < size <= HEADER_LIMIT:
                raise ContractError(f"header size outside bound: {filename}")
            raw = stream.read(size)
        if len(raw) != size:
            raise ContractError(f"truncated header: {filename}")
        shard = json.loads(raw)
        shard.pop("__metadata__", None)
        for name, meta in shard.items():
            if name in headers or not isinstance(meta, dict):
                raise ContractError(f"invalid or duplicate tensor: {name}")
            headers[name] = meta
    if set(headers) != set(weight_map):
        raise ContractError("checkpoint index/header tensor mismatch")
    return headers


def tensor_contract(name: str, meta: Mapping[str, object]) -> dict[str, object]:
    match = SELECTED.match(name)
    if match is None:
        raise ContractError(f"tensor is outside selected family: {name}")
    dtype, shape, offsets = meta.get("dtype"), meta.get("shape"), meta.get("data_offsets")
    if dtype != "BF16":
        raise ContractError(f"selected tensor dtype drift: {name}={dtype!r}")
    if (
        not isinstance(shape, list) or len(shape) != 2
        or not all(isinstance(x, int) and x > 0 for x in shape)
    ):
        raise ContractError(f"selected tensor shape drift: {name}={shape!r}")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ContractError(f"selected tensor offsets missing: {name}")
    source_bytes = offsets[1] - offsets[0]
    if source_bytes != math.prod(shape) * 2:
        raise ContractError(f"selected tensor byte/shape drift: {name}")
    return {
        "name": name,
        "layer": int(match.group(1)),
        "projection": match.group(2),
        "source_dtype": "BF16",
        "shape": shape,
        "source_bytes": source_bytes,
        "artifact": {
            "weight_dtype": "F8_E4M3",
            "weight_shape": shape,
            "weight_scale_dtype": "F32",
            "weight_scale_shape": [1],
            "input_scale_dtype": "F32",
            "input_scale_shape": [1],
            "scale_convention": "amax/448",
        },
    }


def input_channel(layer: int, projection: str) -> str:
    packed = {
        "in_proj_qkv": "in_proj_qkvz",
        "in_proj_z": "in_proj_qkvz",
        "in_proj_a": "in_proj_ba",
        "in_proj_b": "in_proj_ba",
        "out_proj": "out_proj",
    }[projection]
    return f"layer.{layer}.linear_attn.{packed}.input"


def build_contract(
    headers: Mapping[str, Mapping[str, object]],
    trace: Mapping[str, object],
    modelopt_source: str,
    loader_source: str,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    if trace.get("schema") != TRACE_SCHEMA:
        raise ContractError(f"calibration schema must be {TRACE_SCHEMA}")
    coverage = trace.get("coverage")
    telemetry = trace.get("telemetry")
    if not isinstance(coverage, Mapping) or not isinstance(telemetry, Mapping):
        raise ContractError("calibration coverage or telemetry missing")
    failures = [
        f"{key}={coverage.get(key)!r}/{value}"
        for key, value in EXPECTED_COVERAGE.items()
        if coverage.get(key) != value
    ]
    if failures:
        raise ContractError("incomplete calibration: " + ", ".join(failures))

    selected = [tensor_contract(name, meta) for name, meta in headers.items() if SELECTED.match(name)]
    selected.sort(key=lambda item: (item["layer"], PROJECTIONS.index(item["projection"])))
    layers = {item["layer"] for item in selected}
    if len(selected) != 180 or len(layers) != 36:
        raise ContractError(f"selected family coverage is {len(layers)}/36 layers, {len(selected)}/180 matrices")
    for item in selected:
        channel = input_channel(item["layer"], item["projection"])
        record = telemetry.get(channel)
        if not isinstance(record, Mapping):
            raise ContractError(f"missing calibrated activation channel: {channel}")
        value = record.get("absmax")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ContractError(f"invalid calibrated activation amax: {channel}")
        item["calibration_channel"] = channel
        item["activation_amax"] = value
        item["input_scale"] = value / FP8_MAX
        if source_hashes is not None:
            digest = source_hashes.get(item["name"])
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ContractError(f"missing source hash: {item['name']}")
            item["source_sha256"] = digest

    missing_dense = [marker for marker in DENSE_FP8_ABI_MARKERS if marker not in modelopt_source]
    dense_valid = not missing_dense
    overlay_valid = OVERLAY_ABI_MARKER in loader_source
    source_bytes = sum(int(item["source_bytes"]) for item in selected)
    immutable_key_input = {
        "schema": SCHEMA,
        "revision": REVISION,
        "trace_sha256": trace.get("input_sha256"),
        "matrices": [
            {key: item[key] for key in ("name", "shape", "source_bytes", "input_scale")}
            | ({"source_sha256": item["source_sha256"]} if "source_sha256" in item else {})
            for item in selected
        ],
    }
    artifact_key = hashlib.sha256(
        json.dumps(immutable_key_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    status = "ready" if dense_valid and overlay_valid and source_hashes is not None else "blocked"
    blocker = None
    if not dense_valid:
        blocker = "pinned ModelOpt dense FP8 ABI drift: " + ", ".join(missing_dense)
    elif not overlay_valid:
        blocker = "pinned loader cannot replace immutable-base tensors from a layered overlay"
    elif source_hashes is None:
        blocker = "source tensor hashes were not supplied; materialization is disabled"
    return {
        "schema": SCHEMA,
        "status": status,
        "blocker": blocker,
        "source": {"revision": REVISION, "image_id": IMAGE_ID},
        "artifact": {
            "key": artifact_key,
            "path_contract": "OUTPUT/<artifact-key>/",
            "overwrite": "forbidden",
            "base_checkpoint_mutation": "forbidden",
            "format": "ModelOpt FP8 serialized dense linear",
            "selected_layers": len(layers),
            "selected_matrices": len(selected),
            "source_bytes": source_bytes,
            "encoded_weight_bytes": source_bytes // 2,
        },
        "loader_abi": {
            "dense_fp8_valid": dense_valid,
            "overlay_valid": overlay_valid,
            "required_marker": OVERLAY_ABI_MARKER,
            "smallest_patch": [
                "add one read-only overlay manifest argument to safetensors_weights_iterator",
                "verify revision, source tensor SHA-256, artifact key, dtype, and shape before first yield",
                "replace exactly the 180 named BF16 tensors and inject each .weight_scale and .input_scale",
                "reject duplicate, missing, extra, or partially replaced selected tensors",
            ],
        },
        "tensors": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--modelopt", required=True, type=Path)
    parser.add_argument("--loader", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.checkpoint.resolve().name != REVISION:
            raise ContractError(f"checkpoint must resolve to pinned revision {REVISION}")
        trace = json.loads(args.trace.read_text())
        trace["input_sha256"] = sha256(args.trace)
        result = build_contract(
            load_headers(args.checkpoint), trace,
            args.modelopt.read_text(), args.loader.read_text(),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())

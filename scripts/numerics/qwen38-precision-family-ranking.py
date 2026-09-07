#!/usr/bin/env python3
"""Rank whole Qwen3.8 projection families for one precision experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping


TRACE_SCHEMA = "rocket.qwen38.activation-summary.v2"
OUTPUT_SCHEMA = "rocket.qwen38.precision-family-ranking.v1"
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
MAX_HEADER_BYTES = 64 * 2**20
EXPECTED_COVERAGE = {
    "linear_attention_layers": 36,
    "linear_projection_input_channels": 108,
    "linear_projection_output_channels": 108,
    "full_attention_layers": 12,
    "full_qkv_projection_layers": 12,
    "full_output_projection_layers": 12,
    "ple_layers": 1,
    "router_layers": 48,
    "recurrent_state_layers": 36,
}
MTP_MARKER = "SpecDecoding metrics:"
MTP_COUNTS = re.compile(r"Accepted:\s*(\d+) tokens,\s*Drafted:\s*(\d+) tokens")


def tensor_bytes(meta: Mapping[str, object]) -> int:
    offsets = meta.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("tensor needs two data_offsets")
    start, end = offsets
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise ValueError("tensor data_offsets must be ordered non-negative integers")
    return end - start


def load_headers(root: Path) -> dict[str, dict[str, object]]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index needs a non-empty weight_map")
    headers: dict[str, dict[str, object]] = {}
    for filename in sorted(set(weight_map.values())):
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("checkpoint index contains an unsafe shard path")
        path = root / filename
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated header prefix: {filename}")
            size = struct.unpack("<Q", prefix)[0]
            if not 0 < size <= MAX_HEADER_BYTES:
                raise ValueError(f"header size outside bound: {filename}")
            raw = stream.read(size)
        if len(raw) != size:
            raise ValueError(f"truncated header: {filename}")
        shard = json.loads(raw)
        shard.pop("__metadata__", None)
        overlap = headers.keys() & shard.keys()
        if overlap:
            raise ValueError(f"duplicate checkpoint tensors: {sorted(overlap)[:3]}")
        headers.update(shard)
    if set(headers) != set(weight_map):
        raise ValueError("checkpoint index/header tensor mismatch")
    return headers


def classify_tensor(name: str) -> tuple[str, int] | None:
    base = r"^model\.language_model\.layers\.(\d+)\."
    patterns = (
        (base + r"linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)\.weight$", "base_linear_attention"),
        (base + r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$", "base_full_attention"),
        (base + r"ple\.(key_proj|value_proj)\.weight$", "base_ple"),
        (base + r"mlp\.gate\.weight$", "base_routers"),
    )
    for pattern, family in patterns:
        match = re.match(pattern, name)
        if match:
            return family, int(match.group(1))
    mtp_patterns = (
        (r"^mtp\.layers\.0\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$", "mtp_attention"),
        (r"^mtp\.layers\.0\.mlp\.gate\.weight$", "mtp_routers"),
        (r"^mtp\.fc_(embedding|hidden)\.weight$", "mtp_input_projection"),
    )
    for pattern, family in mtp_patterns:
        if re.match(pattern, name):
            return family, 0
    return None


def validate_trace(trace: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    if trace.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"trace schema must be {TRACE_SCHEMA}")
    gate = trace.get("gate")
    if not isinstance(gate, Mapping) or gate.get("source") != "v2_only":
        raise ValueError("trace must be gated from v2_only telemetry")
    coverage = trace.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("trace coverage is missing")
    missing = [
        f"{name}={coverage.get(name)!r}/{expected}"
        for name, expected in EXPECTED_COVERAGE.items()
        if coverage.get(name) != expected
    ]
    if missing:
        raise ValueError("incomplete required coverage: " + ", ".join(missing))
    telemetry = trace.get("telemetry")
    if not isinstance(telemetry, Mapping) or not telemetry:
        raise ValueError("trace telemetry is missing")
    for channel, record in telemetry.items():
        if not isinstance(channel, str) or not isinstance(record, Mapping):
            raise ValueError("trace telemetry entries must be objects")
        for field in ("absmax", "abs_p99", "rms", "sample_numel", "source_numel"):
            value = record.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"telemetry {channel} has invalid {field}")
    return telemetry


def mtp_evidence(log: Path) -> dict[str, object]:
    records = accepted = drafted = 0
    for line in log.read_text(errors="replace").splitlines():
        if MTP_MARKER not in line:
            continue
        match = MTP_COUNTS.search(line)
        if match is None:
            raise ValueError("malformed MTP runtime metric")
        records += 1
        accepted += int(match.group(1))
        drafted += int(match.group(2))
    if records == 0 or drafted == 0 or accepted == 0:
        raise ValueError("MTP runtime evidence has no accepted draft tokens")
    return {
        "source": str(log),
        "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "records": records,
        "accepted_tokens": accepted,
        "drafted_tokens": drafted,
        "acceptance_rate": accepted / drafted,
    }


def interaction_channels(family: str, layers: set[int], telemetry: Mapping[str, object]) -> list[str]:
    channels: set[str] = set()
    for layer in layers:
        prefix = f"layer.{layer}."
        if family == "base_linear_attention":
            wanted = tuple(
                f"{prefix}linear_attn.{projection}.{end}"
                for projection in ("in_proj_qkvz", "in_proj_ba", "out_proj")
                for end in ("input", "output")
            ) + (
                f"{prefix}linear_attn.output",
                f"{prefix}linear_attn.recurrent_state.output",
            )
        elif family == "base_full_attention":
            wanted = tuple(
                f"{prefix}full_attn.{projection}.{end}"
                for projection in ("qkv_proj", "o_proj")
                for end in ("input", "output")
            ) + (
                f"{prefix}full_attn.output",
            )
        elif family == "base_ple":
            wanted = (f"{prefix}ple.embedding.output", f"{prefix}ple.output")
        elif family == "base_routers":
            wanted = (
                f"{prefix}router.gate.input",
                f"{prefix}router.gate.output",
                f"{prefix}router.topk.output",
            )
        else:
            wanted = ()
        missing = [channel for channel in wanted if channel not in telemetry]
        if missing:
            raise ValueError(
                f"missing required telemetry interactions for {family}: "
                + ", ".join(missing)
            )
        channels.update(wanted)
    return sorted(channels)


def risk_summary(channels: list[str], telemetry: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    if not channels:
        return {"status": "unavailable", "reason": "no activation telemetry"}
    absmax = max(float(telemetry[name]["absmax"]) for name in channels)
    p99 = max(float(telemetry[name]["abs_p99"]) for name in channels)
    rms = max(float(telemetry[name]["rms"]) for name in channels)
    ratios = [
        float(telemetry[name]["absmax"]) / max(float(telemetry[name]["abs_p99"]), 1e-30)
        for name in channels
    ]
    ratio = max(ratios)
    risk = "low" if ratio <= 4 else "moderate" if ratio <= 8 else "high"
    return {
        "status": "measured",
        "risk": risk,
        "absmax": absmax,
        "max_abs_p99": p99,
        "max_rms": rms,
        "max_absmax_to_p99": ratio,
    }


def rank(
    headers: Mapping[str, Mapping[str, object]],
    trace: Mapping[str, object],
    mtp: Mapping[str, object],
    concurrency: int = 16,
) -> dict[str, object]:
    telemetry = validate_trace(trace)
    grouped: defaultdict[str, list[tuple[str, int, str, int]]] = defaultdict(list)
    for name, meta in headers.items():
        classified = classify_tensor(name)
        if classified is None:
            continue
        family, layer = classified
        dtype = meta.get("dtype")
        if not isinstance(dtype, str):
            raise ValueError(f"tensor {name} has no dtype")
        grouped[family].append((name, tensor_bytes(meta), dtype, layer))
    required = {
        "base_linear_attention", "base_full_attention", "base_ple",
        "base_routers", "mtp_attention", "mtp_routers",
        "mtp_input_projection",
    }
    absent = sorted(required - grouped.keys())
    if absent:
        raise ValueError("missing required checkpoint families: " + ", ".join(absent))

    families = []
    for family, tensors in grouped.items():
        source_bytes = sum(item[1] for item in tensors)
        dtypes = sorted({item[2] for item in tensors})
        layers = {item[3] for item in tensors}
        channels = interaction_channels(family, layers, telemetry)
        risk = risk_summary(channels, telemetry)
        eligible = dtypes == ["BF16"] and risk["status"] == "measured"
        fp8_removed = source_bytes // 2 if eligible else 0
        nvfp4_removed = int(source_bytes * (1.0 - 0.28125)) if eligible else 0
        families.append({
            "family": family,
            "tensor_count": len(tensors),
            "layers": len(layers),
            "source_dtypes": dtypes,
            "source_bytes": source_bytes,
            "c16_decode": {
                "concurrency": concurrency,
                "weight_reuse": "one family read per batched decode step",
                "fp8_bytes_removed_per_step": fp8_removed,
                "nvfp4_bytes_removed_per_step": nvfp4_removed,
            },
            "telemetry_interactions": channels,
            "calibration": risk,
            "eligible": eligible,
            "ineligible_reason": None if eligible else (
                "activation range unavailable" if risk["status"] != "measured"
                else "source family is not wholly BF16"
            ),
        })
    families.sort(key=lambda item: (-int(item["c16_decode"]["fp8_bytes_removed_per_step"]), item["family"]))
    for index, family in enumerate(families, 1):
        family["rank"] = index
    candidates = [item for item in families if item["eligible"]]
    if not candidates:
        raise ValueError("no fully covered BF16 projection family is eligible")
    recommendation = candidates[0]
    return {
        "schema": OUTPUT_SCHEMA,
        "checkpoint_revision": REVISION,
        "decision": {
            "family": recommendation["family"],
            "first_experiment": "BF16_to_FP8",
            "bytes_removed_per_c16_decode_step": recommendation["c16_decode"]["fp8_bytes_removed_per_step"],
            "reason": "largest wholly BF16 family with complete measured interactions",
        },
        "coverage": dict(trace["coverage"]),
        "mtp_runtime_evidence": dict(mtp),
        "families": families,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mtp-log", required=True, type=Path)
    parser.add_argument("--format", choices=("json", "table"), default="table")
    args = parser.parse_args()
    try:
        if args.checkpoint.resolve().name != REVISION:
            raise ValueError(f"checkpoint path must resolve to pinned revision {REVISION}")
        trace = json.loads(args.trace.read_text())
        headers = load_headers(args.checkpoint)
        result = rank(headers, trace, mtp_evidence(args.mtp_log))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.format == "json":
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print("rank\tfamily\tsource_GiB\tFP8_removed_GiB/c16_step\tNVFP4_removed_GiB/c16_step\trisk\teligible")
        for item in result["families"]:
            calibration = item["calibration"]
            print(
                f"{item['rank']}\t{item['family']}\t{item['source_bytes']/2**30:.6f}\t"
                f"{item['c16_decode']['fp8_bytes_removed_per_step']/2**30:.6f}\t"
                f"{item['c16_decode']['nvfp4_bytes_removed_per_step']/2**30:.6f}\t"
                f"{calibration.get('risk', calibration['status'])}\t{str(item['eligible']).lower()}"
            )
        print(
            "decision\t" + result["decision"]["family"] + "\tBF16_to_FP8\t" +
            str(result["decision"]["bytes_removed_per_c16_decode_step"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

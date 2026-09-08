#!/usr/bin/env python3
"""Validate the 51-artifact oracle and compare Rocket's layer-3 output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

ORACLE_SCHEMA = "rocket.qwen38.k0-target-oracle.v1"
SLICE_SCHEMA = "rocket.qwen38.k0-layer3-slice.v1"
RESULT_SCHEMA = "rocket.qwen38.k0-layer3-comparison.v1"
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
HIDDEN = 2560
HC_STREAMS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(phase: str, reason: str) -> None:
    record = {
        "schema": RESULT_SCHEMA,
        "valid": False,
        "complete": False,
        "phase": phase,
        "reason": reason[:512],
    }
    print(json.dumps(record, sort_keys=True))
    raise SystemExit(1)


def artifact(manifest: dict, name: str) -> dict:
    matches = [
        item
        for item in manifest.get("artifacts", [])
        if item.get("name") == name
    ]
    if len(matches) != 1:
        fail("oracle", f"oracle requires exactly one {name} artifact")
    return matches[0]


def prepare(capture_dir: Path, output: Path) -> None:
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.is_file():
        fail("oracle", "51-artifact oracle manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    names = [
        "embedding",
        *[f"layer.{i:02d}" for i in range(48)],
        "final_norm",
        "logits",
    ]
    identity = manifest.get("identity", {})
    if (
        manifest.get("schema") != ORACLE_SCHEMA
        or manifest.get("valid") is not True
        or manifest.get("complete") is not True
        or [item.get("name") for item in manifest.get("artifacts", [])] != names
        or identity.get("model_revision") != REVISION
    ):
        fail("oracle", "51-artifact oracle identity or sequence changed")
    token_ids = manifest.get("input_token_ids")
    expected_files = {"manifest.json"}
    for item in manifest["artifacts"]:
        path = capture_dir / item.get("file", "")
        expected_files.add(item.get("file", ""))
        if (
            item.get("dtype") not in ("bfloat16", "float16", "float32")
            or not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or sha256(path) != item.get("sha256")
        ):
            fail("oracle", f"oracle artifact identity changed: {item.get('name')}")
    actual_files = {path.name for path in capture_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        fail("oracle", "51-artifact oracle has missing or extra files")
    before, expected = artifact(manifest, "layer.02"), artifact(manifest, "layer.03")
    shape = [len(token_ids), HC_STREAMS * HIDDEN] if isinstance(token_ids, list) else None
    if (
        not token_ids
        or before.get("dtype") != "bfloat16"
        or expected.get("dtype") != "bfloat16"
        or before.get("shape") != shape
        or expected.get("shape") != shape
    ):
        fail("oracle", "layer-3 HC extent or dtype changed")
    selected = {}
    for name, item in (("input", before), ("expected", expected)):
        path = capture_dir / item["file"]
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or sha256(path) != item.get("sha256")
        ):
            fail("oracle", f"{name} layer artifact identity changed")
        selected[name] = {
            "file": str(path.resolve()),
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
    record = {
        "schema": SLICE_SCHEMA,
        "oracle_manifest_sha256": sha256(manifest_path),
        "model_revision": REVISION,
        "layer": 3,
        "token_count": len(token_ids),
        "shape": shape,
        "dtype": "bfloat16",
        "entry_state": "materialized_post_layer_02",
        "input_token_ids": token_ids,
        **selected,
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def bf16_values(payload: bytes):
    for (bits,) in struct.iter_unpack("<H", payload):
        yield struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def compare(contract_path: Path, observed: Path, atol: float, rtol: float) -> None:
    contract = json.loads(contract_path.read_text())
    expected_spec = contract.get("expected", {})
    expected = Path(expected_spec.get("file", ""))
    if (
        contract.get("schema") != SLICE_SCHEMA
        or contract.get("model_revision") != REVISION
        or contract.get("layer") != 3
        or contract.get("dtype") != "bfloat16"
        or contract.get("entry_state") != "materialized_post_layer_02"
    ):
        fail("contract", "layer-3 slice contract identity changed")
    if atol < 0.0 or rtol < 0.0 or not math.isfinite(atol + rtol):
        fail("contract", "comparison tolerances are invalid")
    size = expected_spec.get("bytes")
    if (
        not expected.is_file()
        or expected.stat().st_size != size
        or sha256(expected) != expected_spec.get("sha256")
    ):
        fail("oracle", "expected layer.03 artifact identity changed")
    if not observed.is_file() or observed.stat().st_size != size:
        fail("rocket", "Rocket post-layer3 BF16 tensor is absent or has wrong extent")
    max_abs = 0.0
    max_rel = 0.0
    mismatches = 0
    count = 0
    pairs = zip(
        bf16_values(observed.read_bytes()), bf16_values(expected.read_bytes())
    )
    for actual, target in pairs:
        if not math.isfinite(actual) or not math.isfinite(target):
            fail("compare", "nonfinite BF16 value encountered")
        absolute = abs(actual - target)
        relative = absolute / max(abs(target), 1.0e-12)
        max_abs = max(max_abs, absolute)
        max_rel = max(max_rel, relative)
        mismatches += absolute > atol + rtol * abs(target)
        count += 1
    result = {
        "schema": RESULT_SCHEMA,
        "valid": mismatches == 0,
        "complete": True,
        "layer": 3,
        "elements": count,
        "mismatches": mismatches,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "atol": atol,
        "rtol": rtol,
        "observed_sha256": sha256(observed),
        "expected_sha256": expected_spec["sha256"],
    }
    print(json.dumps(result, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    c = commands.add_parser("compare")
    c.add_argument("--contract", type=Path, required=True)
    c.add_argument("--observed", type=Path, required=True)
    c.add_argument("--atol", type=float, default=0.08)
    c.add_argument("--rtol", type=float, default=0.08)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.capture_dir, args.output)
    else:
        compare(args.contract, args.observed, args.atol, args.rtol)


if __name__ == "__main__":
    main()

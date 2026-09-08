#!/usr/bin/env python3
"""Materialize the replicated QSA index projection omitted by old rank slabs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

SCHEMA = "qwen3.8-flash-next:qsa-indexer-replica-sidecar:v1"
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
BASE_ARTIFACT = "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
LAYERS = tuple(range(3, 48, 4))
SHAPE = [640, 2560]
BYTES = 640 * 2560 * 2
ALIGNMENT = 256
PAGE = 4096


class SidecarError(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def align(value: int, boundary: int) -> int:
    return (value + boundary - 1) // boundary * boundary


def _source_entries(plan: dict) -> list[dict]:
    if plan.get("checkpoint_revision") != REVISION:
        raise SidecarError("rank-slab plan revision drift")
    wanted = {
        f"model.language_model.layers.{layer}.self_attn.indexer.index_qk_proj.weight"
        for layer in LAYERS
    }
    candidates = {}
    for slab in plan.get("slabs", {}).values():
        for entry in slab.get("entries", []):
            if entry.get("name") in wanted:
                candidates.setdefault(entry["name"], entry)
    if set(candidates) != wanted:
        raise SidecarError("rank-slab plan lacks all 12 QSA index projections")
    result = []
    for name in sorted(candidates, key=lambda item: int(item.split("layers.")[1].split(".")[0])):
        entry = candidates[name]
        source = entry.get("source")
        if entry.get("dtype") != "BF16" or entry.get("full_shape") != SHAPE or not isinstance(source, dict) or source.get("byte_count") != BYTES:
            raise SidecarError(f"QSA index source contract drift: {name}")
        result.append(entry)
    return result


def materialize(plan_path: Path, base_artifact: Path, output_root: Path) -> Path:
    try:
        plan_raw = plan_path.read_bytes()
        plan = json.loads(plan_raw)
        base_manifest_raw = (base_artifact / "manifest.json").read_bytes()
        base_manifest = json.loads(base_manifest_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SidecarError("cannot load plan or base manifest") from exc
    if base_artifact.name != BASE_ARTIFACT or base_manifest.get("artifact_key") != BASE_ARTIFACT:
        raise SidecarError("base rank-slab identity drift")
    checkpoint = Path(plan.get("checkpoint", ""))
    entries = _source_entries(plan)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".qsa-indexer-sidecar.{os.getpid()}.building"
    if staging.exists():
        raise SidecarError("refusing interrupted sidecar owned by this process")
    staging.mkdir()
    try:
        payload_path = staging / "qsa-indexer-replicated.slab"
        records = []
        offset = 0
        validated_files = set()
        with payload_path.open("xb") as output:
            for entry in entries:
                source = entry["source"]
                source_path = checkpoint / source["file"]
                stat = source_path.stat()
                if stat.st_size != source.get("file_size_bytes"):
                    raise SidecarError(f"source file size drift: {source_path.name}")
                if source_path not in validated_files:
                    with source_path.open("rb") as stream:
                        header_length = int.from_bytes(stream.read(8), "little")
                        header = stream.read(header_length)
                    if hashlib.sha256(header).hexdigest() != source.get("header_sha256") or source_path.resolve().name != source.get("blob"):
                        raise SidecarError(f"source identity drift: {source_path.name}")
                    validated_files.add(source_path)
                with source_path.open("rb", buffering=0) as stream:
                    stream.seek(source["absolute_offset_bytes"])
                    tensor = stream.read(BYTES)
                if len(tensor) != BYTES:
                    raise SidecarError(f"short QSA index source read: {entry['name']}")
                aligned = align(offset, ALIGNMENT)
                output.write(b"\0" * (aligned - offset))
                output.write(tensor)
                records.append({
                    "name": entry["name"], "dtype": "BF16", "shape": SHAPE,
                    "layout": "checkpoint", "abi": "native-replicated",
                    "offset_bytes": aligned, "length_bytes": BYTES,
                    "sha256": hashlib.sha256(tensor).hexdigest(),
                })
                offset = aligned + BYTES
            final_size = align(offset, PAGE)
            output.write(b"\0" * (final_size - offset))
        payload_sha = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        manifest = {
            "schema": SCHEMA,
            "revision": REVISION,
            "base_artifact_key": BASE_ARTIFACT,
            "base_manifest_sha256": hashlib.sha256(base_manifest_raw).hexdigest(),
            "source_plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
            "payload": {"file": payload_path.name, "bytes": final_size, "sha256": payload_sha},
            "components": records,
            "rank_bindings": {
                "rank0-target": [item["name"] for item in records],
                "rank1-target": [item["name"] for item in records],
            },
            "old_sharded_index_qk_proj": "rejected",
        }
        artifact_key = hashlib.sha256(canonical(manifest)).hexdigest()
        manifest["artifact_key"] = artifact_key
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.chmod(payload_path, 0o444)
        os.chmod(staging / "manifest.json", 0o444)
        destination = output_root / artifact_key
        if destination.exists():
            raise SidecarError("refusing existing immutable sidecar")
        os.replace(staging, destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(materialize(args.plan, args.base_artifact, args.output_root))
    except (SidecarError, OSError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
